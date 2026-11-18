//! Line-based progress for `dataset run`: a bar per step, a line per worker, failures as they
//! happen and a summary when the step ends.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::engine::{Level, Observer, Step, StepSummary};

/// `1m 05s`, or `42s` under a minute.
pub fn format_elapsed(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 { format!("{secs}s") } else { format!("{}m {:02}s", secs / 60, secs % 60) }
}

#[derive(Default)]
struct State {
    bar: Option<ProgressBar>,
    workers: Vec<ProgressBar>,
    started: Option<Instant>,
}

pub struct ProgressObserver {
    multi: MultiProgress,
    state: Mutex<State>,
}

impl Default for ProgressObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressObserver {
    pub fn new() -> Self {
        Self { multi: MultiProgress::new(), state: Mutex::new(State::default()) }
    }

    /// Print a line above the bars, or plainly when there is no terminal to draw on.
    fn say(&self, line: &str) {
        if self.multi.is_hidden() || self.multi.println(line).is_err() {
            eprintln!("{line}");
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Observer for ProgressObserver {
    fn step_begin(&self, step: Step) {
        self.say(&format!("Step {}/8: {}", step.number(), step.title()));
        self.state().started = Some(Instant::now());
    }

    fn step_start(&self, _step: Step, tasks: usize, skipped: usize, workers: usize) {
        let mut state = self.state();
        if tasks == 0 {
            return;
        }
        let bar = self.multi.add(ProgressBar::new(tasks as u64));
        bar.set_style(
            ProgressStyle::with_template("  [{bar:30.cyan/blue}] {pos}/{len} {percent}% {elapsed} eta {eta} {msg}")
                .expect("valid template")
                .progress_chars("=> "),
        );
        if skipped > 0 {
            bar.set_message(format!("({skipped} skipped)"));
        }
        state.workers = (0..workers.min(tasks))
            .map(|w| {
                let line = self.multi.add(ProgressBar::new_spinner());
                line.set_style(
                    ProgressStyle::with_template("  {spinner} {prefix} {wide_msg}").expect("valid template"),
                );
                line.set_prefix(format!("worker {}", w + 1));
                line.set_message("idle");
                line.enable_steady_tick(Duration::from_millis(120));
                line
            })
            .collect();
        state.bar = Some(bar);
    }

    fn task_start(&self, worker: usize, task: &str) {
        if let Some(line) = self.state().workers.get(worker) {
            line.set_message(task.to_string());
        }
    }

    fn task_done(&self, worker: usize, task: &str, result: &Result<(), String>) {
        {
            let state = self.state();
            if let Some(line) = state.workers.get(worker) {
                line.set_message("idle");
            }
            if let Some(bar) = &state.bar {
                bar.inc(1);
                if result.is_ok() {
                    bar.set_message(format!("done {task}"));
                }
            }
        }
        if let Err(e) = result {
            self.say(&format!("  failed {task}: {e}"));
        }
    }

    fn step_done(&self, step: Step, summary: &StepSummary) {
        let elapsed = {
            let mut state = self.state();
            for line in state.workers.drain(..) {
                line.finish_and_clear();
            }
            if let Some(bar) = state.bar.take() {
                bar.finish_and_clear();
            }
            state.started.take().map(|s| s.elapsed()).unwrap_or_default()
        };
        let mark = if summary.failed > 0 || summary.stopped { "!" } else { "ok" };
        let mut line = format!(
            "{mark} Step {}/8: {}: {} completed, {} failed, {} skipped in {}",
            step.number(),
            step.title(),
            summary.completed,
            summary.failed,
            summary.skipped,
            format_elapsed(elapsed)
        );
        if summary.stopped {
            line.push_str(" (stopped)");
        }
        self.say(&line);
    }

    fn log(&self, level: Level, message: &str) {
        match level {
            Level::Info => self.say(&format!("  {message}")),
            Level::Warn => self.say(&format!("  warning: {message}")),
            Level::Error => self.say(&format!("  error: {message}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed() {
        assert_eq!(format_elapsed(Duration::from_secs(42)), "42s");
        assert_eq!(format_elapsed(Duration::from_secs(65)), "1m 05s");
    }
}
