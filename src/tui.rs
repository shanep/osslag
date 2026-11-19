//! `osslag dataset tui`: a full-screen view of the pipeline with a stage list, a worker panel
//! and a log. Press r to run every step, 1 to 8 to run one, s to stop and q to quit.
//!
//! The engine runs on a background thread and sends its events over a channel. The screen is
//! drawn from `App`, which only changes when an event or a key arrives, so tests can build an
//! `App`, feed it events and render it to a `TestBackend` without a terminal.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::Local;
use ratatui::Frame;
use ratatui::crossterm::event::{self, Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};

use crate::engine::{Config, Engine, Level, Observer, Step, StepSummary};
use crate::github::GitHubApi;
use crate::progress::format_elapsed;

const LOG_LINES: usize = 2000;
const BAR_WIDTH: usize = 12;
const TICK: Duration = Duration::from_millis(250);

/// What the engine thread tells the screen.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    StepBegin(Step),
    StepStart {
        step: Step,
        tasks: usize,
        skipped: usize,
    },
    TaskStart {
        worker: usize,
        task: String,
    },
    TaskDone {
        worker: usize,
        task: String,
        error: Option<String>,
    },
    StepDone {
        step: Step,
        summary: StepSummary,
    },
    Log {
        level: Level,
        message: String,
    },
    /// The run ended, with the error that stopped it if any.
    Finished(Option<String>),
}

/// Sends engine events to the screen.
pub struct ChannelObserver(Mutex<Sender<Event>>);

impl ChannelObserver {
    pub fn new(tx: Sender<Event>) -> Self {
        Self(Mutex::new(tx))
    }

    fn send(&self, event: Event) {
        // The screen is gone when this fails, and then nobody is left to tell.
        let _ = self.0.lock().unwrap_or_else(|e| e.into_inner()).send(event);
    }
}

impl Observer for ChannelObserver {
    fn step_begin(&self, step: Step) {
        self.send(Event::StepBegin(step));
    }

    fn step_start(&self, step: Step, tasks: usize, skipped: usize, _workers: usize) {
        self.send(Event::StepStart { step, tasks, skipped });
    }

    fn task_start(&self, worker: usize, task: &str) {
        self.send(Event::TaskStart { worker, task: task.to_string() });
    }

    fn task_done(&self, worker: usize, task: &str, result: &Result<(), String>) {
        self.send(Event::TaskDone { worker, task: task.to_string(), error: result.clone().err() });
    }

    fn step_done(&self, step: Step, summary: &StepSummary) {
        self.send(Event::StepDone { step, summary: *summary });
    }

    fn log(&self, level: Level, message: &str) {
        self.send(Event::Log { level, message: message.to_string() });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Status {
    #[default]
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
    Stopped,
}

impl Status {
    fn icon(self) -> (&'static str, Style) {
        match self {
            Status::Pending => ("○", Style::new().dim()),
            Status::Running => ("▸", Style::new().yellow().bold()),
            Status::Done => ("✓", Style::new().green().bold()),
            Status::Failed => ("✗", Style::new().red().bold()),
            Status::Skipped => ("–", Style::new().dim()),
            Status::Stopped => ("■", Style::new().yellow()),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Stage {
    pub status: Status,
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub skipped: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Worker {
    pub task: Option<String>,
    pub done: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Info,
    Warn,
    Error,
    Ok,
    Fail,
    Step,
}

#[derive(Debug, Clone)]
struct LogLine {
    time: String,
    kind: Kind,
    text: String,
}

/// Everything on the screen.
pub struct App {
    pub config: Config,
    pub stages: [Stage; 8],
    pub workers: Vec<Worker>,
    /// The step that is running now, if any.
    current: Option<Step>,
    step_started: Option<Instant>,
    log: VecDeque<LogLine>,
    /// Lines scrolled up from the bottom of the log. 0 follows new lines.
    scroll: usize,
    pub running: bool,
    pub stopping: bool,
    /// Quit once the run ends.
    pub quitting: bool,
}

impl App {
    pub fn new(config: Config) -> Self {
        let workers = vec![Worker::default(); config.max_workers.max(1)];
        let mut app = Self {
            config,
            stages: [Stage::default(); 8],
            workers,
            current: None,
            step_started: None,
            log: VecDeque::new(),
            scroll: 0,
            running: false,
            stopping: false,
            quitting: false,
        };
        app.push("", Kind::Info, "Press r to run every step, or 1 to 8 to run one step.".into());
        app
    }

    fn stage(&mut self, step: Step) -> &mut Stage {
        &mut self.stages[step.number() - 1]
    }

    fn push(&mut self, time: &str, kind: Kind, text: String) {
        if self.log.len() == LOG_LINES {
            self.log.pop_front();
        }
        self.log.push_back(LogLine { time: time.to_string(), kind, text });
        if self.scroll > 0 {
            // Keep the same lines in view while scrolled up.
            self.scroll += 1;
        }
    }

    /// A run is starting.
    pub fn start(&mut self, steps: &[Step], time: &str) {
        for step in steps {
            *self.stage(*step) = Stage::default();
        }
        self.running = true;
        self.stopping = false;
        let what = match steps {
            [one] => format!("step {}", one.number()),
            _ => "every step".into(),
        };
        self.push(time, Kind::Step, format!("Running {what}"));
    }

    pub fn request_stop(&mut self, time: &str) {
        if self.running && !self.stopping {
            self.stopping = true;
            self.push(time, Kind::Warn, "Stop requested, finishing the tasks that are running".into());
        }
    }

    pub fn scroll_by(&mut self, lines: isize) {
        let max = self.log.len().saturating_sub(1);
        self.scroll = self.scroll.saturating_add_signed(lines).min(max);
    }

    /// Apply one engine event. `time` labels log lines and `now` times the step.
    pub fn apply(&mut self, event: Event, time: &str, now: Instant) {
        match event {
            Event::StepBegin(step) => {
                *self.stage(step) = Stage { status: Status::Running, ..Stage::default() };
                self.current = Some(step);
                self.step_started = Some(now);
                for w in &mut self.workers {
                    *w = Worker::default();
                }
                self.push(time, Kind::Step, format!("Step {}: {}", step.number(), step.title()));
            }
            Event::StepStart { step, tasks, skipped } => {
                let stage = self.stage(step);
                stage.total = tasks;
                stage.skipped = skipped;
                if tasks > 0 || skipped > 0 {
                    self.push(time, Kind::Info, format!("{tasks} tasks, {skipped} skipped"));
                }
            }
            Event::TaskStart { worker, task } => {
                if let Some(w) = self.workers.get_mut(worker) {
                    w.task = Some(task);
                }
            }
            Event::TaskDone { worker, task, error } => {
                if let Some(w) = self.workers.get_mut(worker) {
                    w.task = None;
                    if error.is_some() { w.failed += 1 } else { w.done += 1 }
                }
                if let Some(step) = self.current {
                    let stage = self.stage(step);
                    if error.is_some() { stage.failed += 1 } else { stage.completed += 1 }
                }
                match error {
                    None => self.push(time, Kind::Ok, task),
                    Some(e) => self.push(time, Kind::Fail, format!("{task}: {e}")),
                }
            }
            Event::StepDone { step, summary } => {
                let elapsed = self.step_started.take().map(|s| now.saturating_duration_since(s)).unwrap_or_default();
                let ran = summary.completed + summary.failed;
                let stage = self.stage(step);
                stage.completed = summary.completed;
                stage.failed = summary.failed;
                stage.skipped = summary.skipped;
                stage.status = if summary.stopped {
                    Status::Stopped
                } else if summary.failed > 0 {
                    Status::Failed
                } else if ran == 0 {
                    Status::Skipped
                } else {
                    Status::Done
                };
                self.current = None;
                for w in &mut self.workers {
                    w.task = None;
                }
                let kind = if summary.failed > 0 || summary.stopped { Kind::Warn } else { Kind::Step };
                let mut text = format!(
                    "Step {} done: {} completed, {} failed, {} skipped in {}",
                    step.number(),
                    summary.completed,
                    summary.failed,
                    summary.skipped,
                    format_elapsed(elapsed)
                );
                if summary.stopped {
                    text.push_str(" (stopped)");
                }
                self.push(time, kind, text);
            }
            Event::Log { level, message } => {
                let kind = match level {
                    Level::Info => Kind::Info,
                    Level::Warn => Kind::Warn,
                    Level::Error => Kind::Error,
                };
                self.push(time, kind, message);
            }
            Event::Finished(error) => {
                if let Some(step) = self.current.take() {
                    self.stage(step).status = Status::Failed;
                }
                self.step_started = None;
                for w in &mut self.workers {
                    w.task = None;
                }
                match error {
                    Some(e) => self.push(time, Kind::Error, format!("Run failed: {e}")),
                    None if self.stopping => self.push(time, Kind::Warn, "Run stopped".into()),
                    None => self.push(time, Kind::Step, "Run finished".into()),
                }
                self.running = false;
                self.stopping = false;
            }
        }
    }
}

fn kind_style(kind: Kind) -> Style {
    match kind {
        Kind::Info => Style::new(),
        Kind::Warn => Style::new().yellow(),
        Kind::Error | Kind::Fail => Style::new().red(),
        Kind::Ok => Style::new().green(),
        Kind::Step => Style::new().cyan().bold(),
    }
}

fn bar(done: usize, total: usize) -> String {
    let filled = (BAR_WIDTH * done).checked_div(total).unwrap_or(0).min(BAR_WIDTH);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(BAR_WIDTH - filled))
}

fn stage_lines(app: &App) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for (i, stage) in app.stages.iter().enumerate() {
        let step = Step::ALL[i];
        let (icon, style) = stage.status.icon();
        let mut line = vec![Span::styled(format!(" {icon} {}. {}", step.number(), step.title()), style)];
        if matches!(stage.status, Status::Done | Status::Failed | Status::Stopped | Status::Skipped) {
            let mut parts = Vec::new();
            if stage.completed > 0 {
                parts.push(format!("✓{}", stage.completed));
            }
            if stage.failed > 0 {
                parts.push(format!("✗{}", stage.failed));
            }
            if stage.skipped > 0 {
                parts.push(format!("–{}", stage.skipped));
            }
            if !parts.is_empty() {
                line.push(Span::styled(format!("  ({})", parts.join(", ")), Style::new().dim()));
            }
        }
        lines.push(Line::from(line));
        if stage.status == Status::Running && stage.total > 0 {
            let done = stage.completed + stage.failed;
            lines.push(Line::from(vec![
                Span::styled(format!("     {} {}%", bar(done, stage.total), done * 100 / stage.total), Color::Cyan),
                Span::styled(format!("  {done}/{}", stage.total), Style::new().dim()),
            ]));
        }
    }
    let sum = |f: fn(&Stage) -> usize| app.stages.iter().map(f).sum::<usize>();
    lines.push(Line::default());
    lines.push(Line::from(vec![
        Span::raw(" Total: "),
        Span::styled(format!("{} done", sum(|s| s.completed)), Color::Green),
        Span::raw(", "),
        Span::styled(format!("{} failed", sum(|s| s.failed)), Color::Red),
        Span::raw(", "),
        Span::styled(format!("{} skipped", sum(|s| s.skipped)), Style::new().dim()),
    ]));
    lines
}

fn worker_lines(app: &App, now: Instant) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for (i, w) in app.workers.iter().enumerate() {
        let counts = Span::styled(format!("  ✓{} ✗{}", w.done, w.failed), Style::new().dim());
        let line = match &w.task {
            Some(task) => vec![
                Span::styled(format!(" #{} ", i + 1), Color::Cyan),
                Span::styled("● ", Style::new().green().bold()),
                Span::styled(task.clone(), Color::Yellow),
                counts,
            ],
            None => vec![
                Span::styled(format!(" #{} ", i + 1), Color::Cyan),
                Span::styled("○ idle", Style::new().dim()),
                counts,
            ],
        };
        lines.push(Line::from(line));
    }
    let (total, completed, failed, skipped) = match app.current {
        Some(step) => {
            let s = app.stages[step.number() - 1];
            (s.total + s.skipped, s.completed, s.failed, s.skipped)
        }
        None => (0, 0, 0, 0),
    };
    let busy = app.workers.iter().filter(|w| w.task.is_some()).count();
    let stat = |name: &str, value: String, style: Style| {
        Line::from(vec![Span::raw(format!(" {name:<13}")), Span::styled(value, style)])
    };
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(" This step", Style::new().add_modifier(Modifier::UNDERLINED))));
    lines.push(stat("Total:", total.to_string(), Style::new()));
    lines.push(stat("Completed:", completed.to_string(), Style::new().green()));
    lines.push(stat("Failed:", failed.to_string(), Style::new().red()));
    lines.push(stat("Skipped:", skipped.to_string(), Style::new().dim()));
    lines.push(stat("In progress:", busy.to_string(), Style::new().yellow()));
    if let Some(started) = app.step_started {
        lines.push(stat("Elapsed:", format_elapsed(now.saturating_duration_since(started)), Style::new().cyan()));
    }
    lines
}

fn log_lines(app: &App, height: usize) -> Vec<Line<'static>> {
    let end = app.log.len().saturating_sub(app.scroll);
    let start = end.saturating_sub(height);
    app.log
        .range(start..end)
        .map(|l| {
            let icon = match l.kind {
                Kind::Ok => "✓ ",
                Kind::Fail => "✗ ",
                _ => "",
            };
            let mut spans = Vec::new();
            if !l.time.is_empty() {
                spans.push(Span::styled(format!("{} ", l.time), Style::new().dim()));
            }
            spans.push(Span::styled(format!("{icon}{}", l.text), kind_style(l.kind)));
            Line::from(spans)
        })
        .collect()
}

fn config_line(config: &Config) -> Line<'static> {
    let mut spans = vec![
        Span::raw(" distro="),
        Span::styled(config.distro.clone(), Style::new().bold()),
        Span::raw("  releases="),
        Span::styled(config.releases.join(", "), Style::new().bold()),
        Span::raw("  workers="),
        Span::styled(config.max_workers.to_string(), Style::new().bold()),
        Span::raw("  cache="),
        Span::styled(config.cache_dir.display().to_string(), Style::new().bold()),
    ];
    if let Some(n) = config.limit {
        spans.push(Span::raw("  limit="));
        spans.push(Span::styled(n.to_string(), Style::new().bold()));
    }
    if config.force {
        spans.push(Span::styled("  --force", Style::new().yellow().bold()));
    }
    Line::from(spans)
}

fn footer(app: &App) -> Line<'static> {
    let key = |k: &str| Span::styled(k.to_string(), Style::new().cyan().bold());
    let state = if app.quitting {
        Span::styled("quitting after the running tasks", Style::new().yellow())
    } else if app.stopping {
        Span::styled("stopping", Style::new().yellow())
    } else if app.running {
        Span::styled("running", Style::new().green())
    } else {
        Span::styled("idle", Style::new().dim())
    };
    Line::from(vec![
        Span::raw(" "),
        key("r"),
        Span::raw(" run all  "),
        key("1-8"),
        Span::raw(" run one  "),
        key("s"),
        Span::raw(" stop  "),
        key("↑↓"),
        Span::raw(" scroll  "),
        key("q"),
        Span::raw(" quit   "),
        state,
    ])
}

/// Draw the whole screen. `clock` is the time shown in the header and `now` times the step.
pub fn draw(frame: &mut Frame, app: &App, clock: &str, now: Instant) {
    let [header, config, main, log, keys] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(13),
        Constraint::Percentage(40),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    let [title, time] = Layout::horizontal([Constraint::Min(1), Constraint::Length(10)]).areas(header);
    let style = Style::new().bg(Color::Blue).fg(Color::White);
    frame.render_widget(Paragraph::new(" OSS Lag Pipeline").style(style.bold()), title);
    frame.render_widget(Paragraph::new(format!("{clock} ")).style(style).right_aligned(), time);
    frame.render_widget(Paragraph::new(config_line(&app.config)), config);

    let [stages, workers] = Layout::horizontal([Constraint::Length(46), Constraint::Min(30)]).areas(main);
    frame.render_widget(Paragraph::new(stage_lines(app)).block(Block::bordered().title(" Pipeline stages ")), stages);
    frame.render_widget(Paragraph::new(worker_lines(app, now)).block(Block::bordered().title(" Workers ")), workers);

    let title = if app.scroll > 0 { format!(" Log (scrolled up {}) ", app.scroll) } else { " Log ".into() };
    let inner = Block::bordered().inner(log);
    frame.render_widget(
        Paragraph::new(log_lines(app, inner.height as usize)).block(Block::bordered().title(title)),
        log,
    );
    frame.render_widget(Paragraph::new(footer(app)), keys);
}

/// Owns the engine thread for the screen.
struct Runner {
    config: Config,
    github: Arc<dyn GitHubApi + Send>,
    tx: Sender<Event>,
    stop: Option<Arc<AtomicBool>>,
}

impl Runner {
    fn run(&mut self, app: &mut App, steps: Vec<Step>, time: &str) {
        if app.running {
            app.push(time, Kind::Warn, "A run is already going. Press s to stop it first.".into());
            return;
        }
        let observer = Arc::new(ChannelObserver::new(self.tx.clone()));
        let engine = match Engine::new(self.config.clone(), self.github.clone(), observer) {
            Ok(engine) => engine,
            Err(e) => {
                app.push(time, Kind::Error, format!("{e:#}"));
                return;
            }
        };
        self.stop = Some(engine.stop_flag());
        app.start(&steps, time);
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let result = if steps.len() == Step::ALL.len() {
                engine.run_all().map(|_| ())
            } else {
                steps.iter().try_for_each(|s| engine.run_step(*s).map(|_| ()))
            };
            let _ = tx.send(Event::Finished(result.err().map(|e| format!("{e:#}"))));
        });
    }

    fn stop(&self, app: &mut App, time: &str) {
        if let Some(flag) = &self.stop {
            flag.store(true, Ordering::Relaxed);
        }
        app.request_stop(time);
    }
}

/// Handle one key. Returns true when the screen should close now.
fn key(app: &mut App, runner: &mut Runner, code: KeyCode, modifiers: KeyModifiers, time: &str) -> bool {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => {
            if !app.running || app.quitting {
                return true;
            }
            app.quitting = true;
            runner.stop(app, time);
        }
        KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => return true,
        KeyCode::Char('r') => runner.run(app, Step::ALL.to_vec(), time),
        KeyCode::Char('s') => runner.stop(app, time),
        KeyCode::Char(c @ '1'..='8') => {
            if let Some(step) = c.to_digit(10).and_then(|n| Step::from_number(n as usize)) {
                runner.run(app, vec![step], time);
            }
        }
        KeyCode::Up => app.scroll_by(1),
        KeyCode::Down => app.scroll_by(-1),
        KeyCode::PageUp => app.scroll_by(10),
        KeyCode::PageDown => app.scroll_by(-10),
        KeyCode::End => app.scroll = 0,
        _ => {}
    }
    false
}

fn clock() -> String {
    Local::now().format("%H:%M:%S").to_string()
}

/// Run the screen until q. The terminal is restored on exit and on panic.
pub fn run(config: Config, github: Arc<dyn GitHubApi + Send>) -> Result<()> {
    let (tx, rx): (Sender<Event>, Receiver<Event>) = channel();
    let mut app = App::new(config.clone());
    let mut runner = Runner { config, github, tx, stop: None };
    let mut terminal = ratatui::init();
    let result = (|| -> Result<()> {
        loop {
            let now = Instant::now();
            let time = clock();
            while let Ok(event) = rx.try_recv() {
                app.apply(event, &time, now);
            }
            if app.quitting && !app.running {
                return Ok(());
            }
            terminal.draw(|f| draw(f, &app, &time, now))?;
            if event::poll(TICK)?
                && let TermEvent::Key(k) = event::read()?
                && k.kind == KeyEventKind::Press
                && key(&mut app, &mut runner, k.code, k.modifiers, &time)
            {
                return Ok(());
            }
        }
    })();
    ratatui::restore();
    result
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use insta::assert_snapshot;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;

    fn config() -> Config {
        Config {
            releases: vec!["bookworm".into(), "trixie".into()],
            cache_dir: PathBuf::from("./cache"),
            max_workers: 3,
            limit: Some(5),
            ..Config::default()
        }
    }

    fn render(app: &App, now: Instant) -> String {
        let mut terminal = Terminal::new(TestBackend::new(100, 32)).unwrap();
        terminal.draw(|f| draw(f, app, "12:00:00", now)).unwrap();
        terminal.backend().to_string()
    }

    fn summary(completed: usize, failed: usize, skipped: usize) -> StepSummary {
        StepSummary { total: completed + failed + skipped, skipped, completed, failed, stopped: false }
    }

    /// Steps 1 to 4 done, step 5 halfway with two workers busy.
    fn mid_run() -> (App, Instant) {
        let t0 = Instant::now();
        let mut app = App::new(config());
        app.start(&Step::ALL, "12:00:00");
        for step in &Step::ALL[..4] {
            app.apply(Event::StepBegin(*step), "12:00:00", t0);
            app.apply(Event::StepStart { step: *step, tasks: 2, skipped: 0 }, "12:00:00", t0);
            app.apply(Event::StepDone { step: *step, summary: summary(2, 0, 0) }, "12:00:01", t0);
        }
        app.apply(Event::StepBegin(Step::CloneRepos), "12:00:01", t0);
        app.apply(Event::StepStart { step: Step::CloneRepos, tasks: 4, skipped: 1 }, "12:00:01", t0);
        let task =
            |worker: usize, name: &str| Event::TaskStart { worker, task: format!("https://github.com/o/{name}") };
        app.apply(task(0, "a"), "12:00:01", t0);
        app.apply(task(1, "b"), "12:00:01", t0);
        app.apply(Event::TaskDone { worker: 0, task: "https://github.com/o/a".into(), error: None }, "12:00:02", t0);
        app.apply(
            Event::TaskDone { worker: 1, task: "https://github.com/o/b".into(), error: Some("404 Not Found".into()) },
            "12:00:03",
            t0,
        );
        app.apply(task(0, "c"), "12:00:03", t0);
        app.apply(task(2, "d"), "12:00:03", t0);
        (app, t0 + Duration::from_secs(65))
    }

    #[test]
    fn snapshot_idle() {
        assert_snapshot!(render(&App::new(config()), Instant::now()));
    }

    #[test]
    fn snapshot_mid_run() {
        let (app, now) = mid_run();
        assert_snapshot!(render(&app, now));
    }

    #[test]
    fn step_counts_follow_tasks() {
        let (app, _) = mid_run();
        let clone = app.stages[4];
        assert_eq!(
            (clone.status, clone.total, clone.completed, clone.failed, clone.skipped),
            (Status::Running, 4, 1, 1, 1)
        );
        assert_eq!(app.stages[0].status, Status::Done);
        assert_eq!(app.workers[1].failed, 1);
        assert_eq!(app.workers.iter().filter(|w| w.task.is_some()).count(), 2);
    }

    #[test]
    fn stop_and_failure_end_the_run() {
        let (mut app, now) = mid_run();
        app.request_stop("12:00:04");
        assert!(app.stopping);
        let stopped = StepSummary { stopped: true, ..summary(2, 1, 1) };
        app.apply(Event::StepDone { step: Step::CloneRepos, summary: stopped }, "12:00:05", now);
        app.apply(Event::Finished(None), "12:00:05", now);
        assert_eq!(app.stages[4].status, Status::Stopped);
        assert!(!app.running && !app.stopping);
        assert!(app.workers.iter().all(|w| w.task.is_none()));

        app.start(&[Step::LoadCommits], "12:00:06");
        app.apply(Event::StepBegin(Step::LoadCommits), "12:00:06", now);
        app.apply(Event::Finished(Some("Required file x does not exist.".into())), "12:00:06", now);
        assert_eq!(app.stages[5].status, Status::Failed);
        assert!(render(&app, now).contains("Run failed: Required file x does not exist."));
    }

    #[test]
    fn a_step_with_nothing_to_do_is_skipped() {
        let t0 = Instant::now();
        let mut app = App::new(config());
        app.apply(Event::StepBegin(Step::GitHubMetadata), "t", t0);
        app.apply(Event::StepDone { step: Step::GitHubMetadata, summary: summary(0, 0, 3) }, "t", t0);
        assert_eq!(app.stages[6].status, Status::Skipped);
    }

    #[test]
    fn scrolling_holds_the_view() {
        let mut app = App::new(config());
        for i in 0..50 {
            app.push("t", Kind::Info, format!("line {i}"));
        }
        app.scroll_by(5);
        app.push("t", Kind::Info, "new".into());
        assert_eq!(app.scroll, 6);
        let lines = log_lines(&app, 3);
        assert!(lines.last().unwrap().to_string().ends_with("line 44"), "{:?}", lines.last());
        app.scroll_by(-100);
        assert_eq!(app.scroll, 0);
        assert!(log_lines(&app, 3).last().unwrap().to_string().ends_with("new"));
    }

    #[test]
    fn channel_observer_sends_events() {
        let (tx, rx) = channel();
        let observer = ChannelObserver::new(tx);
        observer.step_begin(Step::LoadCommits);
        observer.task_done(1, "x", &Err("boom".into()));
        let events: Vec<Event> = rx.try_iter().collect();
        assert_eq!(
            events,
            vec![
                Event::StepBegin(Step::LoadCommits),
                Event::TaskDone { worker: 1, task: "x".into(), error: Some("boom".into()) }
            ]
        );
    }

    #[test]
    fn bar_fills() {
        assert_eq!(bar(0, 4), "[░░░░░░░░░░░░]");
        assert_eq!(bar(2, 4), "[██████░░░░░░]");
        assert_eq!(bar(5, 0), "[░░░░░░░░░░░░]");
    }
}
