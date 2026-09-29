//! Human-readable download progress for Hub resolution.
//!
//! Implements [`hf_hub::progress::ProgressHandler`] so a `serve --model ...`
//! that pulls a fresh model package shows the user *something* is happening
//! rather than silently blocking. Without this, a first-time resolution can
//! hang with no output and the user may mistake it for a ready-to-serve model.
//!
//! Progress is always drawn as a **single, self-updating line** on stderr,
//! whether or not stderr is a TTY. The line is prefixed with `\r` and cleared
//! to end-of-line (`\x1b[K`) before each redraw, and the label is truncated,
//! so a long transfer never scrolls or wraps and "clogs" the terminal.

use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use hf_hub::progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler};

/// Maximum characters shown for the file label before it is truncated.
const LABEL_MAX: usize = 18;
/// Width of the block-bar indicator (in cells).
const BAR_WIDTH: usize = 10;
const BLOCK: char = '\u{2588}'; // █
const EMPTY: char = '\u{2591}'; // ░

/// Renders progress for a single `download_file` call (one repo path).
pub struct FileDownloadProgress {
    filename: String,
    start: Instant,
    tty: bool,
    last_print_ms: AtomicU64,
    done: AtomicBool,
}

impl FileDownloadProgress {
    pub fn new(repo: impl Into<String>, filename: impl Into<String>) -> Self {
        let repo = repo.into();
        let filename = filename.into();
        let tty = io::stderr().is_terminal();
        // Diagnostic aid: `HUNCHO_PROGRESS_DEBUG=1 huncho serve ...` prints the
        // detected rendering mode so a silent hang can be attributed to a
        // non-TTY stderr or to a download not being attempted at all.
        if std::env::var_os("HUNCHO_PROGRESS_DEBUG").is_some() {
            eprintln!(
                "[huncho-hub] progress {repo}:{filename} mode={} (stderr {})",
                if tty { "bar" } else { "line" },
                if tty { "is a TTY" } else { "is not a TTY" }
            );
        }
        Self {
            filename,
            start: Instant::now(),
            tty,
            last_print_ms: AtomicU64::new(0),
            done: AtomicBool::new(false),
        }
    }

    /// Redraw the single progress line on stderr.
    fn render(&self, bytes: u64, total: u64, rate_bps: Option<f64>) {
        let now_ms = self.start.elapsed().as_millis() as u64;
        let last_ms = self.last_print_ms.load(Ordering::Relaxed);
        // Both modes draw a single in-place line; the interval only controls
        // how often it is refreshed (TTY is snappier, non-TTY is calmer).
        let interval_ms = if self.tty { 150 } else { 500 };
        if now_ms.saturating_sub(last_ms) < interval_ms {
            return;
        }
        self.last_print_ms.store(now_ms, Ordering::Relaxed);

        let line = progress_line(&self.filename, bytes, total, rate_bps);
        // `\r` returns to the start of the line, `\x1b[K` clears anything the
        // previous, longer render left behind. Written to stderr (not stdout)
        // so the in-place bar is not trapped in line-buffered stdout.
        eprint!("\r\x1b[K{line}");
        let _ = io::stderr().flush();
    }

    /// Render the initial line (0% / known size).
    pub fn begin(&self, total: u64) {
        self.render(0, total, None);
    }

    /// Report bytes transferred so far (throttled to the configured interval).
    pub fn report(&self, bytes: u64, total: u64, rate_bps: Option<f64>) {
        self.render(bytes, total, rate_bps);
    }

    /// Force a final 100% render and terminate the progress line.
    pub fn finish(&self, bytes: u64, total: u64) {
        // Reset the throttle so the final 100% is always drawn, then terminate.
        self.last_print_ms.store(0, Ordering::Relaxed);
        self.render(bytes, total, None);
        self.done();
    }

    fn done(&self) {
        if self.done.swap(true, Ordering::Relaxed) {
            return;
        }
        let label = truncate_label(&self.filename, LABEL_MAX);
        eprintln!("\r\x1b[K  \u{2713} {label}");
    }
}

impl Drop for FileDownloadProgress {
    fn drop(&mut self) {
        // If a download fails, `done` is never called; leave a terminating
        // newline so the error message (or next download) starts on a fresh
        // line rather than being appended to the progress line.
        if !self.done.load(Ordering::Relaxed) {
            eprintln!();
        }
    }
}

impl ProgressHandler for FileDownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Download(DownloadEvent::Start { total_bytes, .. }) => {
                self.begin(*total_bytes);
            }
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                if let Some(f) = files.last() {
                    self.render(f.bytes_completed, f.total_bytes, None);
                    if f.status == FileStatus::Complete {
                        self.done();
                    }
                }
            }
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            }) => {
                self.render(*bytes_completed, *total_bytes, *bytes_per_sec);
            }
            ProgressEvent::Download(DownloadEvent::Complete) => self.done(),
            _ => {}
        }
    }
}

/// Build the single progress line:
/// `  model.safetensors [██████░░░░] 55.0% 421.5/766.4 MB @5.8 MB/s ETA 1m12s`
/// Kept under ~75 chars so it never wraps on an 80-column terminal.
fn progress_line(filename: &str, bytes: u64, total: u64, rate_bps: Option<f64>) -> String {
    let label = truncate_label(filename, LABEL_MAX);
    let pct = if total > 0 {
        (bytes as f64 / total as f64).min(1.0)
    } else {
        0.0
    };

    let mut line = format!("  {label}");

    if total > 0 {
        let filled = (pct * BAR_WIDTH as f64).round() as usize;
        let bar = format!(
            "[{}{}]",
            BLOCK.to_string().repeat(filled),
            EMPTY.to_string().repeat(BAR_WIDTH - filled),
        );
        line.push_str(&format!(" {bar} {:>5.1}% {}", pct * 100.0, human_pair(bytes, total)));
    } else {
        line.push_str(&format!(" {}", human_bytes(bytes)));
    }

    if let Some(r) = rate_bps {
        line.push_str(&format!(" @{}", human_rate(r)));
    }
    if let Some(eta) = eta_secs(bytes, total, rate_bps) {
        line.push_str(&format!(" ETA {}", fmt_eta(eta)));
    }

    line
}

/// Format a transferred/total byte pair with a single shared unit, e.g.
/// `421.5/766.4 MB`. Uses the larger magnitude to pick the unit.
fn human_pair(bytes: u64, total: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let max = bytes.max(total);
    let mut unit = 0usize;
    let mut vmax = max as f64;
    while vmax >= 1024.0 && unit < UNITS.len() - 1 {
        vmax /= 1024.0;
        unit += 1;
    }
    let scale = 1024f64.powi(unit as i32);
    let fa = bytes as f64 / scale;
    let fb = total as f64 / scale;
    let one = |v: f64| {
        if unit == 0 {
            format!("{v:.0}")
        } else {
            format!("{v:.1}")
        }
    };
    format!("{}/{} {}", one(fa), one(fb), UNITS[unit])
}

/// Truncate to `max` characters, replacing the tail with a single ellipsis.
fn truncate_label(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('\u{2026}'); // …
    out
}

/// Seconds to completion, if the size and instant rate are known.
fn eta_secs(bytes: u64, total: u64, rate_bps: Option<f64>) -> Option<u64> {
    if total > bytes && total > 0 {
        rate_bps
            .filter(|r| *r > 0.0)
            .map(|r| ((total - bytes) as f64 / r).round() as u64)
    } else {
        None
    }
}

/// Format a whole number of seconds as `1m12s` / `42s`.
fn fmt_eta(secs: u64) -> String {
    let m = secs / 60;
    let s = secs % 60;
    if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = n as f64;
    let mut idx = 0;
    while value >= 1024.0 && idx < UNITS.len() - 1 {
        value /= 1024.0;
        idx += 1;
    }
    if idx == 0 {
        format!("{n} B")
    } else {
        format!("{value:.1} {}", UNITS[idx])
    }
}

fn human_rate(bps: f64) -> String {
    format!("{}/s", human_bytes(bps as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_formats() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KB");
        assert_eq!(human_bytes(3 * 1024 * 1024), "3.0 MB");
        assert_eq!(human_bytes(2 * 1024 * 1024 * 1024), "2.0 GB");
    }

    #[test]
    fn rate_uses_bytes_per_sec() {
        assert_eq!(human_rate(1024.0), "1.0 KB/s");
    }

    #[test]
    fn truncate_short_label_unchanged() {
        assert_eq!(truncate_label("config.json", 18), "config.json");
    }

    #[test]
    fn truncate_long_label_adds_ellipsis() {
        let s = truncate_label("tokenizer/tokenizer.json", 18);
        assert_eq!(s.chars().count(), 18);
        assert!(s.ends_with('\u{2026}'));
        assert_eq!(s, "tokenizer/tokeniz\u{2026}");
    }

    #[test]
    fn eta_formats_minutes_and_seconds() {
        assert_eq!(fmt_eta(42), "42s");
        assert_eq!(fmt_eta(72), "1m12s");
    }

    #[test]
    fn eta_requires_progress_and_rate() {
        assert_eq!(eta_secs(0, 100, Some(10.0)), Some(10));
        assert_eq!(eta_secs(0, 100, None), None);
        assert_eq!(eta_secs(100, 100, Some(10.0)), None); // already complete
        assert_eq!(eta_secs(0, 0, Some(10.0)), None); // unknown size
    }

    #[test]
    fn human_pair_shares_a_unit() {
        assert_eq!(human_pair(363_434_000, 803_600_000), "346.6/766.4 MB");
        assert_eq!(human_pair(0, 2048), "0.0/2.0 KB");
    }

    #[test]
    fn progress_line_includes_bar_rate_and_eta() {
        let line = progress_line("model.safetensors", 363_434_000, 803_600_000, Some(6_100_000.0));
        assert!(line.starts_with("  model.safetensors ["));
        assert!(line.contains('%'));
        assert!(line.contains("346.6/766.4 MB"));
        assert!(line.contains("@5.8 MB/s"));
        assert!(line.contains("ETA "));
    }

    #[test]
    fn progress_line_handles_unknown_size() {
        let line = progress_line("x.bin", 1234, 0, None);
        assert!(!line.contains('['));
        assert!(line.contains("1.2 KB"));
    }

    #[test]
    fn handler_survives_a_download_lifecycle() {
        use hf_hub::progress::FileProgress;
        let h = FileDownloadProgress::new("org/repo", "model.onnx");
        h.on_progress(&ProgressEvent::Download(DownloadEvent::Start {
            total_files: 1,
            total_bytes: 12_345_678,
        }));
        h.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "model.onnx".into(),
                bytes_completed: 6_172_839,
                total_bytes: 12_345_678,
                status: FileStatus::InProgress,
            }],
        }));
        h.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "model.onnx".into(),
                bytes_completed: 12_345_678,
                total_bytes: 12_345_678,
                status: FileStatus::Complete,
            }],
        }));
        h.on_progress(&ProgressEvent::Download(DownloadEvent::Complete));
    }
}
