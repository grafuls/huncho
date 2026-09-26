//! Resolve a model package reference to a local `huncho-model.json`.
//!
//! A model reference is either a local package (a directory containing
//! `huncho-model.json`, or a path to the manifest itself) or a Hugging Face
//! model repository id (`owner/repo`). In the latter case the package manifest
//! and every artifact it references are downloaded from the Hub into the HF
//! cache (honoring `HF_TOKEN`/`HF_HOME`/`HF_HUB_CACHE`), and the manifest's
//! snapshot directory is returned.

use std::path::{Path, PathBuf};

use huncho_core::manifest::{BackendId, ModelManifest};

use crate::error::{HubError, Result};

/// Where a model package lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    /// A local package directory or a path to a `huncho-model.json`.
    Local(PathBuf),
    /// A Hugging Face repository id (`owner/repo`), possibly with a revision.
    HuggingFace { repo: String, revision: Option<String> },
}

impl ModelRef {
    /// Parse a user-supplied model reference.
    ///
    /// If the string names an existing file or directory, it is treated as a
    /// local package; otherwise it is treated as a Hugging Face repo id. An
    /// explicit `revision` is attached to the HF form.
    pub fn parse(input: &str, revision: Option<String>) -> ModelRef {
        let trimmed = input.trim();
        let path = Path::new(trimmed);
        if path.exists() {
            ModelRef::Local(path.to_path_buf())
        } else {
            ModelRef::HuggingFace {
                repo: trimmed.to_string(),
                revision,
            }
        }
    }
}

/// Options controlling Hub resolution.
#[derive(Debug, Clone, Default)]
pub struct ResolveOptions {
    /// A git revision (branch, tag, or commit SHA). Defaults to the repo's
    /// default branch when `None`.
    pub revision: Option<String>,
    /// An explicit HF token; defaults to the `HF_TOKEN` env var / cached login.
    pub token: Option<String>,
    /// Override the HF cache directory (defaults to `HF_HUB_CACHE`).
    pub cache_dir: Option<PathBuf>,
    /// Only use the local HF cache; fail if a file is not already present.
    pub local_files_only: bool,
    /// Also fetch the golden conformance vectors referenced by the manifest.
    pub fetch_golden: bool,
}

/// A resolved package: the local path to its `huncho-model.json`.
#[derive(Debug, Clone)]
pub struct ResolvedPackage {
    /// Absolute or relative path to the package manifest.
    pub manifest_path: PathBuf,
}

/// Resolve a model reference to a local package manifest.
///
/// `backend` and `dtype` select which backbone artifact(s) to fetch from the
/// Hub; pass `backend = None` when only the manifest is needed (e.g. for the
/// offline mock backend or for `calibrate`).
pub fn resolve(
    model: &str,
    backend: Option<BackendId>,
    dtype: &str,
    opts: &ResolveOptions,
) -> Result<ResolvedPackage> {
    let mref = ModelRef::parse(model, opts.revision.clone());
    match mref {
        ModelRef::Local(path) => resolve_local(&path),
        ModelRef::HuggingFace { repo, revision } => {
            resolve_hf(&repo, revision.as_deref(), backend, dtype, opts)
        }
    }
}

/// A convenience helper returning only the manifest path.
pub fn resolve_manifest_path(
    model: &str,
    backend: Option<BackendId>,
    dtype: &str,
    opts: &ResolveOptions,
) -> Result<PathBuf> {
    Ok(resolve(model, backend, dtype, opts)?.manifest_path)
}

// ---------------------------------------------------------------------------
// Local resolution
// ---------------------------------------------------------------------------

fn resolve_local(path: &Path) -> Result<ResolvedPackage> {
    let manifest = local_manifest_path(path)?;
    // Validate now so a malformed package yields a clear error at resolve time.
    ModelManifest::load(&manifest).map_err(|e| HubError::Package(e.to_string()))?;
    Ok(ResolvedPackage { manifest_path: manifest })
}

fn local_manifest_path(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        let manifest = path.join("huncho-model.json");
        if !manifest.is_file() {
            return Err(HubError::PackageNotFound(manifest));
        }
        Ok(manifest)
    } else if path.is_file() {
        Ok(path.to_path_buf())
    } else {
        Err(HubError::InvalidRef {
            ref_: path.display().to_string(),
            reason: "not an existing file or directory".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Hugging Face resolution
// ---------------------------------------------------------------------------

fn resolve_hf(
    repo: &str,
    revision: Option<&str>,
    backend: Option<BackendId>,
    dtype: &str,
    opts: &ResolveOptions,
) -> Result<ResolvedPackage> {
    let (owner, name) = hf_hub::split_id(repo);
    let client = build_client(opts)?;
    let sync = hf_hub::HFClientSync::from_inner(client)
        .map_err(|e| HubError::hf("creating the Hugging Face client", e))?;
    let model_repo = sync.model(owner, name);

    let rev: Option<String> = revision.map(|s| s.to_string());

    // Fetch the manifest first so we know which artifacts it references.
    let manifest_path = model_repo
        .download_file()
        .filename("huncho-model.json".to_string())
        .maybe_revision(rev.clone())
        .local_files_only(opts.local_files_only)
        .force_download(false)
        .send()
        .map_err(|e| HubError::hf(format!("fetching `huncho-model.json` from `{repo}`"), e))?;

    let manifest = ModelManifest::load(&manifest_path).map_err(|e| {
        HubError::Package(format!("`{repo}` is not a Huncho model package: {e}"))
    })?;

    // Resolve the snapshot directory (the manifest's parent) and pin all
    // artifact downloads to the same resolved commit so a moving branch cannot
    // scatter files across snapshots.
    let package_root = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            HubError::Package(format!(
                "manifest path has no parent directory: {}",
                manifest_path.display()
            ))
        })?;
    let commit = package_root
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let pinned: Option<String> = if commit.is_empty() { rev.clone() } else { Some(commit) };

    for file in required_files(&manifest, backend, dtype, opts.fetch_golden)? {
        model_repo
            .download_file()
            .filename(file.clone())
            .maybe_revision(pinned.clone())
            .local_files_only(opts.local_files_only)
            .force_download(false)
            .send()
            .map_err(|e| HubError::hf(format!("fetching `{file}` from `{repo}`"), e))?;
    }

    Ok(ResolvedPackage { manifest_path })
}

fn build_client(opts: &ResolveOptions) -> Result<hf_hub::HFClient> {
    let mut builder = hf_hub::HFClient::builder();
    if let Some(token) = &opts.token {
        builder = builder.token(token.clone());
    }
    if let Some(cache) = &opts.cache_dir {
        builder = builder.cache_dir(cache.clone());
    }
    builder
        .build()
        .map_err(|e| HubError::hf("configuring the Hugging Face client", e))
}

/// The set of files that must be fetched from the Hub for the requested
/// backend/dtype.
fn required_files(
    manifest: &ModelManifest,
    backend: Option<BackendId>,
    dtype: &str,
    fetch_golden: bool,
) -> Result<Vec<String>> {
    let mut files = Vec::new();

    if let Some(backend) = backend {
        let artifact = manifest.find_artifact(backend, dtype).ok_or_else(|| {
            HubError::MissingArtifact {
                package: manifest.name.clone(),
                backend: backend.to_string(),
                dtype: dtype.to_string(),
            }
        })?;
        files.push(artifact.path.clone());
    }

    if let Some(tokenizer) = &manifest.backbone.tokenizer {
        files.push(tokenizer.clone());
    }
    if !manifest.head.weights.is_empty() {
        files.push(manifest.head.weights.clone());
    }
    if fetch_golden {
        if let Some(reference) = &manifest.reference {
            files.push(reference.golden.clone());
        }
    }

    files.sort();
    files.dedup();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_manifest_json(name: &str) -> serde_json::Value {
        serde_json::json!({
            "schema_version": "1.0",
            "name": name,
            "family": "F1",
            "backbone": {
                "source": { "kind": "hf", "repo": "org/repo", "revision": "deadbeef" },
                "artifacts": {
                    "onnx": [{ "path": "model.onnx", "dtype": "fp32" }]
                },
                "hidden_size": 1024,
                "max_context": 4096,
                "tokenizer": "tokenizer.json"
            },
            "head": { "kind": "option-marker", "weights": "head.safetensors", "width": 1 },
            "prompt_contract": {
                "template": "laya-v1",
                "option_marker_tokens": ["<option:0>"],
                "state_budget": 3072,
                "head_budget": 512,
                "max_options": 255,
                "contract_hash": "abc123"
            },
            "calibration": {
                "default": { "temperature": 1.0, "confidence": "peak" }
            },
            "reference": { "family_impl": "laya-serve", "revision": "x", "golden": "golden.json" }
        })
    }

    #[test]
    fn parses_existing_path_as_local() {
        let dir = tempfile::tempdir().unwrap();
        let mref = ModelRef::parse(dir.path().to_str().unwrap(), None);
        assert!(matches!(mref, ModelRef::Local(_)));
    }

    #[test]
    fn parses_hf_repo_when_not_a_path() {
        let mref = ModelRef::parse("org/repo", Some("main".into()));
        assert_eq!(
            mref,
            ModelRef::HuggingFace {
                repo: "org/repo".into(),
                revision: Some("main".into())
            }
        );
    }

    #[test]
    fn resolves_local_directory() {
        let dir = tempfile::tempdir().unwrap();
        let manifest_path = dir.path().join("huncho-model.json");
        std::fs::write(&manifest_path, serde_json::to_vec(&minimal_manifest_json("local-test")).unwrap())
            .unwrap();

        let opts = ResolveOptions::default();
        let pkg = resolve(dir.path().to_str().unwrap(), None, "fp32", &opts).unwrap();
        assert_eq!(pkg.manifest_path, manifest_path);
    }

    #[test]
    fn local_directory_without_manifest_fails() {
        let dir = tempfile::tempdir().unwrap();
        let opts = ResolveOptions::default();
        let err = resolve(dir.path().to_str().unwrap(), None, "fp32", &opts).unwrap_err();
        assert!(matches!(err, HubError::PackageNotFound(_)));
    }

    #[test]
    fn required_files_select_artifact_and_head() {
        let m: ModelManifest =
            serde_json::from_value(minimal_manifest_json("m")).unwrap();
        let files = required_files(&m, Some(BackendId::Onnx), "fp32", true).unwrap();
        assert!(files.contains(&"model.onnx".to_string()));
        assert!(files.contains(&"head.safetensors".to_string()));
        assert!(files.contains(&"tokenizer.json".to_string()));
        assert!(files.contains(&"golden.json".to_string()));
    }

    #[test]
    fn required_files_omits_backend_artifact_for_mock() {
        let m: ModelManifest =
            serde_json::from_value(minimal_manifest_json("m")).unwrap();
        // For None (mock / no backbone needed), the ONNX artifact must not be required.
        let files = required_files(&m, None, "fp32", false).unwrap();
        assert!(!files.contains(&"model.onnx".to_string()));
        assert!(files.contains(&"head.safetensors".to_string()));
    }
}
