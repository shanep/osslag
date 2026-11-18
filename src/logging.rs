//! Log to a size-rotating file in the same format as the Python version:
//! `time | target | file:line | LEVEL | message`.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::Local;
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::filter::{LevelFilter, filter_fn};
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{Layer, fmt};

const MAX_BYTES: u64 = 5 * 1024 * 1024;
const BACKUPS: usize = 3;

/// A log file that moves to `.1`, `.2`, ... when it would pass `max_bytes`, keeping `backups` old files.
pub struct RotatingFile {
    path: PathBuf,
    max_bytes: u64,
    backups: usize,
    file: File,
    size: u64,
}

impl RotatingFile {
    pub fn open(path: &Path, max_bytes: u64, backups: usize) -> io::Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let size = file.metadata()?.len();
        Ok(Self { path: path.to_path_buf(), max_bytes, backups, file, size })
    }

    fn backup(&self, n: usize) -> PathBuf {
        let mut name = self.path.as_os_str().to_owned();
        name.push(format!(".{n}"));
        PathBuf::from(name)
    }

    fn rotate(&mut self) -> io::Result<()> {
        self.file.flush()?;
        if self.backups == 0 {
            self.file = File::create(&self.path)?;
        } else {
            for n in (1..self.backups).rev() {
                let from = self.backup(n);
                if from.exists() {
                    std::fs::rename(&from, self.backup(n + 1))?;
                }
            }
            std::fs::rename(&self.path, self.backup(1))?;
            self.file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        }
        self.size = 0;
        Ok(())
    }
}

impl Write for RotatingFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.size > 0 && self.size + buf.len() as u64 > self.max_bytes {
            self.rotate()?;
        }
        let n = self.file.write(buf)?;
        self.size += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

/// Python's log line layout.
struct PythonFormat;

impl<S, N> FormatEvent<S, N> for PythonFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(&self, ctx: &FmtContext<'_, S, N>, mut writer: Writer<'_>, event: &Event<'_>) -> std::fmt::Result {
        let meta = event.metadata();
        let file = meta.file().and_then(|f| Path::new(f).file_stem()).map(|f| f.to_string_lossy()).unwrap_or_default();
        let level = match *meta.level() {
            Level::ERROR => "ERROR",
            Level::WARN => "WARNING",
            Level::INFO => "INFO",
            Level::DEBUG | Level::TRACE => "DEBUG",
        };
        write!(
            writer,
            "{} | {:<20} | {:<20}:{:<4} | {:<8} | ",
            Local::now().format("%Y-%m-%d %H:%M:%S,%3f"),
            meta.target(),
            file,
            meta.line().unwrap_or(0),
            level
        )?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

/// Python's level names (and `WARN`), any case. Unknown names mean INFO.
pub fn parse_level(name: &str) -> LevelFilter {
    match name.trim().to_uppercase().as_str() {
        "DEBUG" | "TRACE" => LevelFilter::DEBUG,
        "WARNING" | "WARN" => LevelFilter::WARN,
        "ERROR" | "CRITICAL" | "FATAL" => LevelFilter::ERROR,
        "NOTSET" => LevelFilter::TRACE,
        _ => LevelFilter::INFO,
    }
}

/// Send osslag's logs to `LOG_FILE` (default `osslag.log`) at `LOG_LEVEL` (default INFO).
pub fn init() -> io::Result<()> {
    let level = parse_level(&std::env::var("LOG_LEVEL").unwrap_or_default());
    let path = std::env::var("LOG_FILE").ok().filter(|p| !p.is_empty()).unwrap_or_else(|| "osslag.log".into());
    let file = RotatingFile::open(Path::new(&path), MAX_BYTES, BACKUPS)?;
    let layer = fmt::layer()
        .event_format(PythonFormat)
        .with_writer(Mutex::new(file))
        .with_ansi(false)
        .with_filter(level)
        .with_filter(filter_fn(|meta| meta.target().starts_with("osslag")));
    let subscriber = tracing_subscriber::registry().with(layer);
    // Only fails if a subscriber is already set, as in tests.
    let _ = tracing::subscriber::set_global_default(subscriber);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotates_and_keeps_backups() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.log");
        let mut f = RotatingFile::open(&path, 10, 2).unwrap();
        for _ in 0..4 {
            f.write_all(b"12345678\n").unwrap();
        }
        assert!(path.exists());
        assert!(dir.path().join("x.log.1").exists());
        assert!(dir.path().join("x.log.2").exists());
        assert!(!dir.path().join("x.log.3").exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "12345678\n");
    }

    #[test]
    fn level_names() {
        assert_eq!(parse_level("warning"), LevelFilter::WARN);
        assert_eq!(parse_level(""), LevelFilter::INFO);
        assert_eq!(parse_level("CRITICAL"), LevelFilter::ERROR);
    }
}
