//! Human-readable download progress for Hub resolution.
//!
//! Implements [`hf_hub::progress::ProgressHandler`] so a `serve --model ...`
//! that pulls a fresh model package shows the user *something* is happening
//! rather than silently blocking. Without this, a first-time resolution can
//! hang with no output and the user may mistake it for a ready-to-serve model.

use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use hf_hub::progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler};

/// Renders progress for a single `download_file` call (one repo path).
///
/// When stderr is a TTY it draws an in-place percentage bar; otherwise it
/// prints one `downloading ...` line and one `done` line so piped logs stay
/// clean. Events are throttled to ~5Hz so a fast transfer does not flood the
/// terminal.
pub struct FileDownloadProgress {
    repo: String,
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
                "[huncho-hub] progress {repo}:{filename} mode={} (stderr tty)",
                if tty { "live-bar" } else { "one-line" }
            );
        }
        Self {
            repo,
            filename,
            start: Instant::now(),
            tty,
            last_print_ms: AtomicU64::new(0),
            done: AtomicBool::new(false),
        }
    }

    fn render(&self, bytes: u64, total: u64, rate_bps: Option<f64>) {
        let now_ms = self.start.elapsed().as_millis() as u64;
        let last_ms = self.last_print_ms.load(Ordering::Relaxed);
        // TTY: fast in-place bar (~6 Hz). Non-TTY: a one-line log line ~1 Hz
        // so a long transfer is never silent even when stderr is not a TTY
        // (piped logs, capture, or an awkward terminal configuration).
        let interval_ms = if self.tty { 150 } else { 1000 };
        if now_ms.saturating_sub(last_ms) < interval_ms {
            return;
        }
        self.last_print_ms.store(now_ms, Ordering::Relaxed);

        let pct = if total > 0 {
            (bytes as f64 / total as f64 * 100.0).min(100.0)
        } else {
            0.0
        };
        let rate = rate_bps.map(|r| format!(" @ {}", human_rate(r))).unwrap_or_default();

        if self.tty {
            // The terminal bar is written to stderr (not stdout) so an
            // interactive `\r` bar is not trapped in line-buffered stdout.
            eprint!(
                "\r  {}/{}  {:>5.1}%  {} / {} {rate}",
                self.filename,
                self.repo,
                pct,
                human_bytes(bytes),
                human_bytes(total)
            );
            let _ = io::stderr().flush();
        } else {
            eprintln!(
                "  downloading {}:{}  {:>5.1}%  {} / {} {rate}",
                self.repo,
                self.filename,
                pct,
                human_bytes(bytes),
                human_bytes(total)
            );
        }
    }

    fn done(&self) {
        if self.done.swap(true, Ordering::Relaxed) {
            return;
        }
        if self.tty {
            eprintln!();
        } else {
            eprintln!("  downloaded {}:{}", self.repo, self.filename);
        }
    }
}

impl Drop for FileDownloadProgress {
    fn drop(&mut self) {
        // If a download fails, `done` is never called; leave a terminating
        // newline so the error message is not appended to the progress line.
        if !self.done.load(Ordering::Relaxed) && self.tty {
            eprintln!();
        }
    }
}

impl ProgressHandler for FileDownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Download(DownloadEvent::Start {
                total_bytes, ..
            }) => {
                // Only print the headline for non-TTY; TTY draws the bar below.
                if !self.tty {
                    eprintln!(
                        "downloading {}:{} ({})",
                        self.repo,
                        self.filename,
                        human_bytes(*total_bytes)
                    );
                } else {
                    eprint!("\r  {}/{}  ({} bytes)", self.filename, self.repo, human_bytes(*total_bytes));
                    let _ = io::stderr().flush();
                }
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
    fn handler_survives_a_download_lifecycle() {
        use super::*;
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
