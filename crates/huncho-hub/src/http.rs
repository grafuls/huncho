//! Plain-HTTP Hub downloads for xet-backed artifacts.
//!
//! `hf-hub` fetches large weights through the Xet CAS (a content-addressed
//! store). For some artifacts the Xet session stalls silently after the CAS
//! token handshake, so a first-time `serve --model owner/repo` shows no
//! progress and appears hung.
//!
//! This module bypasses Xet entirely: it downloads the raw bytes over plain
//! HTTPS from the Hub's `/resolve/<revision>/<filename>` endpoint (following
//! the CDN redirect), writing the file directly into the HF snapshot directory
//! (`<cache>/<type>s--<owner>--<name>/snapshots/<commit>/<filename>`) as a
//! regular file. That is exactly where `huncho-model.json` and its artifacts
//! are read from, so the rest of the resolver and the engine loader work
//! unchanged.
//!
//! Downloads are pinned to the resolved commit (from the `x-repo-commit`
//! header) so a moving branch cannot scatter a package across snapshots, and
//! the snapshot path is reused as the cache: if it already exists for the
//! resolved commit the transfer is skipped on subsequent runs.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use reqwest::blocking::{Client, Response};
use reqwest::header::LOCATION;
use reqwest::redirect::Policy;
use reqwest::StatusCode;

use crate::error::{HubError, Result};
use crate::progress::FileDownloadProgress;
use crate::resolver::ResolveOptions;

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_REVISION: &str = "main";
const HEADER_X_REPO_COMMIT: &str = "x-repo-commit";
const HEADER_X_LINKED_SIZE: &str = "x-linked-size";
const CHUNK: usize = 64 * 1024;

/// Download a single repo file over plain HTTP into the HF snapshot directory.
///
/// Returns the absolute path of the file in the snapshot dir (a regular file).
/// A `404` produces [`HubError::NotFound`] so callers can distinguish "this
/// repo has no such file" from a transient failure.
///
/// `reqwest::blocking` owns a Tokio runtime, and creating or dropping that
/// runtime inside an async `#[tokio::main]` context (as `serve` uses) is
/// forbidden by Tokio. The transfer therefore runs on a dedicated OS thread so
/// the runtime lives and dies there, not in the async context.
pub fn download_file(
    repo: &str,
    filename: &str,
    revision: Option<String>,
    opts: &ResolveOptions,
) -> Result<PathBuf> {
    let cache = cache_dir(opts);
    let rev = revision.unwrap_or_else(|| DEFAULT_REVISION.to_string());

    if opts.local_files_only {
        return local_cached_file(&cache, repo, &rev, filename);
    }

    let repo = repo.to_string();
    let filename = filename.to_string();
    let opts = opts.clone();
    let what = format!("download of `{filename}` from `{repo}` panicked");
    let handle = std::thread::spawn(move || blocking_download(&repo, &filename, &rev, &opts, &cache));
    handle.join().map_err(|_| HubError::Package(what))?
}

/// The blocking reqwest transfer. Runs on a dedicated OS thread (see
/// [`download_file`]) so its Tokio runtime is never created or dropped inside
/// the caller's async context.
fn blocking_download(
    repo: &str,
    filename: &str,
    rev: &str,
    opts: &ResolveOptions,
    cache: &Path,
) -> Result<PathBuf> {
    let client = Client::builder()
        // We read the commit / redirect from the resolve response ourselves, so
        // do not let reqwest follow redirects for us.
        .redirect(Policy::none())
        .build()
        .map_err(|e| HubError::Hf(format!("building the HTTP client: {e}")))?;
    let token = opts.token.clone().or_else(|| std::env::var("HF_TOKEN").ok());
    let url = resolve_url(repo, filename, rev);
    let resp0 = send_get(&client, &url, token.as_deref())?;
    let status = resp0.status();

    if status == StatusCode::NOT_FOUND {
        return Err(HubError::NotFound {
            repo: repo.to_string(),
            filename: filename.to_string(),
        });
    }
    if !status.is_success() && !status.is_redirection() {
        return Err(HubError::Hf(format!(
            "fetching `{filename}` from `{repo}`: HTTP {status}"
        )));
    }

    let headers = resp0.headers().clone();
    let commit = commit(&headers, Some(rev)).ok_or_else(|| {
        HubError::Hf(format!("could not resolve a commit for `{filename}` in `{repo}`"))
    })?;
    let linked_size = headers
        .get(HEADER_X_LINKED_SIZE)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok());

    let dest = snapshot_dir(cache, repo, &commit).join(filename);
    // Reuse the snapshot path as the cache: if the resolved commit already has
    // this file, there is nothing to download.
    if dest.exists() {
        return Ok(dest);
    }

    let (mut resp, total) = if status.is_redirection() {
        let loc = headers
            .get(LOCATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                HubError::Hf(format!("redirect without a Location for `{filename}` in `{repo}`"))
            })?;
        let r = send_get(&client, loc, token.as_deref())?;
        if !r.status().is_success() {
            return Err(HubError::Hf(format!(
                "fetching `{filename}` from `{repo}`: HTTP {}",
                r.status()
            )));
        }
        let t = r.content_length().or(linked_size).unwrap_or(0);
        (r, t)
    } else {
        let t = resp0.content_length().or(linked_size).unwrap_or(0);
        (resp0, t)
    };

    let parent = dest.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;

    let progress = opts
        .show_progress
        .then(|| FileDownloadProgress::new(repo.to_string(), filename.to_string()));
    if let Some(p) = &progress {
        p.begin(total);
    }

    let tmp = temp_path(&dest);
    let start = Instant::now();
    let mut bytes: u64 = 0;
    {
        let mut file = std::fs::File::create(&tmp)?;
        let mut buffer = vec![0u8; CHUNK];
        loop {
            let n = resp.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            file.write_all(&buffer[..n])?;
            bytes += n as u64;
            if let Some(p) = &progress {
                let rate = (bytes as f64) / start.elapsed().as_secs_f64().max(0.001);
                p.report(bytes, total, Some(rate));
            }
        }
        file.flush()?;
    }

    std::fs::rename(&tmp, &dest)?;

    if let Some(p) = &progress {
        p.finish(bytes, total);
    }
    Ok(dest)
}

/// Return the path of a cached snapshot file when `local_files_only` is set,
/// without touching the network. Resolves the commit from the revision itself
/// (a SHA) or from the repo's `refs/<revision>` pointer.
fn local_cached_file(cache: &Path, repo: &str, rev: &str, filename: &str) -> Result<PathBuf> {
    let commit = resolve_commit_local(cache, repo, rev).ok_or_else(|| {
        HubError::Package(format!(
            "`{repo}` is not cached locally (local_files_only), cannot resolve `{filename}`"
        ))
    })?;
    let dest = snapshot_dir(cache, repo, &commit).join(filename);
    if dest.exists() {
        Ok(dest)
    } else {
        Err(HubError::Package(format!(
            "`{filename}` is not cached locally for `{repo}` at `{commit}` (local_files_only)"
        )))
    }
}

fn resolve_commit_local(cache: &Path, repo: &str, rev: &str) -> Option<String> {
    if is_commit_hash(rev) {
        return Some(rev.to_string());
    }
    let ref_path = cache.join(repo_folder(repo)).join("refs").join(rev);
    std::fs::read_to_string(&ref_path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| is_commit_hash(s))
}

/// Extract the resolved commit from the resolve response headers, falling back
/// to the requested revision when it is already a full commit SHA.
fn commit(headers: &reqwest::header::HeaderMap, revision: Option<&str>) -> Option<String> {
    headers
        .get(HEADER_X_REPO_COMMIT)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| is_commit_hash(s))
        .or_else(|| revision.filter(|s| is_commit_hash(s)).map(String::from))
}

fn send_get(client: &Client, url: &str, token: Option<&str>) -> Result<Response> {
    let mut req = client.get(url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    req.send()
        .map_err(|e| HubError::Hf(format!("requesting `{url}`: {e}")))
}

fn resolve_url(repo: &str, filename: &str, revision: &str) -> String {
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
    format!("{endpoint}/{repo}/resolve/{revision}/{filename}")
}

fn cache_dir(opts: &ResolveOptions) -> PathBuf {
    if let Some(c) = &opts.cache_dir {
        c.clone()
    } else {
        hf_hub::resolve_cache_dir()
    }
}

fn repo_folder(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

fn snapshot_dir(cache: &Path, repo: &str, commit: &str) -> PathBuf {
    cache.join(repo_folder(repo)).join("snapshots").join(commit)
}

fn temp_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    dest.parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{name}.incomplete"))
}

fn is_commit_hash(s: &str) -> bool {
    s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_full_commit_from_headers() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            HEADER_X_REPO_COMMIT,
            reqwest::header::HeaderValue::from_static(
                "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
            ),
        );
        assert_eq!(
            commit(&headers, Some("main")).as_deref(),
            Some("55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851")
        );
    }

    #[test]
    fn falls_back_to_revision_when_it_is_a_sha() {
        let headers = reqwest::header::HeaderMap::new();
        let sha = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851";
        assert_eq!(commit(&headers, Some(sha)).as_deref(), Some(sha));
    }

    #[test]
    fn snapshot_dir_matches_hf_layout() {
        let dir = Path::new("/cache");
        let p = snapshot_dir(dir, "convaiinnovations/laya", "deadbeef");
        assert_eq!(
            p,
            Path::new("/cache/models--convaiinnovations--laya/snapshots/deadbeef")
        );
    }

    #[test]
    fn repo_folder_replaces_slash() {
        assert_eq!(repo_folder("org/repo"), "models--org--repo");
    }

    #[test]
    fn resolve_url_builds_endpoint_and_revision() {
        let url = resolve_url("org/repo", "model.safetensors", "main");
        assert_eq!(
            url,
            "https://huggingface.co/org/repo/resolve/main/model.safetensors"
        );
    }

    #[test]
    fn temp_path_sits_next_to_dest() {
        let dest = Path::new("/cache/models--o--r/snapshots/abc/model.safetensors");
        assert_eq!(
            temp_path(dest),
            Path::new("/cache/models--o--r/snapshots/abc/model.safetensors.incomplete")
        );
    }

    #[test]
    fn commit_hash_check() {
        assert!(is_commit_hash("55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"));
        assert!(!is_commit_hash("main"));
        assert!(!is_commit_hash("deadbeef"));
    }
}
