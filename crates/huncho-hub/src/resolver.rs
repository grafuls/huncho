//! Resolve a model package reference to a local `huncho-model.json`.
//!
//! A model reference is either a local package (a directory containing
//! `huncho-model.json`, or a path to the manifest itself) or a Hugging Face
//! model repository id (`owner/repo`). In the latter case the package manifest
//! and every artifact it references are downloaded from the Hub into the HF
//! cache (honoring `HF_TOKEN`/`HF_HOME`/`HF_HUB_CACHE`), and the manifest's
//! snapshot directory is returned.

use std::path::{Path, PathBuf};

use huncho_core::manifest::{
    self, ArtifactRef, Backbone, BackboneSource, BackendId, CalibrationConfig, CalibrationEntry,
    CalibrationStatus, ConfidenceDef, Family, HeadConfig, ModelCapabilities, ModelManifest,
    PromptContract,
};

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

    // Fetch the manifest first so we know which artifacts it references. If the
    // repo has no `huncho-model.json`, fall back to synthesizing a servable
    // package from a raw decision-model checkpoint (e.g. Laya's ModernBERT).
    let manifest_path = match model_repo
        .download_file()
        .filename("huncho-model.json".to_string())
        .maybe_revision(rev.clone())
        .local_files_only(opts.local_files_only)
        .force_download(false)
        .send()
    {
        Ok(p) => p,
        Err(_) => {
            return synthesize_checkpoint(
                repo, &model_repo, rev.as_deref(), backend, dtype, opts,
            );
        }
    };

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

// ---------------------------------------------------------------------------
// Checkpoint synthesis
// ---------------------------------------------------------------------------

/// A raw decision-model checkpoint repository that has no explicit
/// `huncho-model.json`. When `serve --model owner/repo --backend candle` points
/// at such a repo (e.g. [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)),
/// synthesize a servable F1 package manifest on the fly from its ModernBERT
/// `encoder/config.json` + `model.safetensors`.
///
/// Only the `candle` backend can be served this way (it loads `.safetensors`
/// directly). The decision head uses the engine's deterministic fallback
/// projection, because the trained head is not extracted from the checkpoint.
fn synthesize_checkpoint(
    repo: &str,
    model_repo: &hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>,
    revision: Option<&str>,
    backend: Option<BackendId>,
    dtype: &str,
    opts: &ResolveOptions,
) -> Result<ResolvedPackage> {
    let Some(backend) = backend else {
        return Err(HubError::Package(format!(
            "`{repo}` has no `huncho-model.json`; a raw checkpoint needs an explicit backend"
        )));
    };
    if backend != BackendId::Candle {
        return Err(HubError::Package(format!(
            "no `huncho-model.json` in `{repo}`; only the `candle` backend can be synthesized from a raw checkpoint"
        )));
    }
    if dtype != "fp32" {
        return Err(HubError::Package(format!(
            "synthesized `{repo}` only supports dtype `fp32` (requested `{dtype}`)"
        )));
    }

    // Fetch the encoder config (Laya keeps it under `encoder/`, HF models keep
    // it at the root).
    let config_path = download_config(model_repo, revision, opts)?;
    let config_bytes = std::fs::read(&config_path)?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)?;

    let package_root = snapshot_root(&config_path).ok_or_else(|| {
        HubError::Package("could not locate the HF snapshot directory for the checkpoint".into())
    })?;

    // Laya ships a `rl_agent_config.json` (calibration + prompt budgets). When
    // present, synthesize as `laya-v1`; otherwise fall back to a generic F1.
    let rl_config = download_first(model_repo, revision, opts, &["rl_agent_config.json"])
        .ok()
        .and_then(|path| std::fs::read(&path).ok())
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());

    // The real ModernBERT tokenizer lives under `tokenizer/` in Laya. Record
    // the repo-relative path so the loader can read it back from the snapshot.
    let tokenizer_path = download_first(
        model_repo,
        revision,
        opts,
        &["tokenizer/tokenizer.json", "tokenizer.json"],
    )
    .ok()
    .and_then(|path| {
        path.strip_prefix(&package_root)
            .ok()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
    });

    let rev = revision.unwrap_or("main");
    let manifest = build_synth_manifest(
        &config,
        repo,
        rev,
        rl_config.as_ref(),
        tokenizer_path.as_deref(),
    )?;

    // The candle loader reads `config.json` next to the manifest, so mirror the
    // encoder config to the snapshot root regardless of the repo's layout
    // (Laya keeps it under `encoder/`, HF models keep it at the root).
    let config_target = package_root.join("config.json");
    if config_path != config_target {
        std::fs::copy(&config_path, &config_target)?;
    }

    // Persist the synthesized manifest in the resolved snapshot directory so a
    // later `serve --model` is deterministic and offline-resolvable.
    let manifest_path = package_root.join("huncho-model.json");
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    // Pin artifact downloads to the resolved snapshot commit.
    let commit = package_root
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let pinned: Option<String> = if commit.is_empty() {
        revision.map(|s| s.to_string())
    } else {
        Some(commit)
    };

    for file in required_files(&manifest, Some(BackendId::Candle), dtype, false)? {
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

/// Find the HF snapshot root (`.../snapshots/<commit>`) that contains a
/// downloaded file, regardless of how deeply it is nested (e.g. Laya keeps
/// `encoder/config.json` under a subdirectory).
fn snapshot_root(path: &Path) -> Option<PathBuf> {
    let mut dir = path.parent()?.to_path_buf();
    loop {
        let parent = dir.parent()?;
        if parent.file_name().and_then(|s| s.to_str()) == Some("snapshots") {
            return Some(dir);
        }
        dir = parent.to_path_buf();
    }
}

/// Download the first of the candidate config filenames that exists in the repo.
fn download_config(
    model_repo: &hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>,
    revision: Option<&str>,
    opts: &ResolveOptions,
) -> Result<PathBuf> {
    let mut last_err: Option<hf_hub::HFError> = None;
    for filename in ["encoder/config.json", "config.json"] {
        match model_repo
            .download_file()
            .filename(filename.to_string())
            .maybe_revision(revision.map(|s| s.to_string()))
            .local_files_only(opts.local_files_only)
            .force_download(false)
            .send()
        {
            Ok(path) => return Ok(path),
            Err(e) => last_err = Some(e),
        }
    }
    match last_err {
        Some(e) => Err(HubError::hf(
            "finding an encoder config (encoder/config.json or config.json)",
            e,
        )),
        None => Err(HubError::Package(
            "no encoder config found in the checkpoint repo".into(),
        )),
    }
}

/// Download the first of a list of candidate filenames that exists in the repo,
/// returning the first successful download. Errors if all candidates fail.
fn download_first(
    model_repo: &hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>,
    revision: Option<&str>,
    opts: &ResolveOptions,
    candidates: &[&str],
) -> Result<PathBuf> {
    let mut last_err: Option<hf_hub::HFError> = None;
    for filename in candidates {
        match model_repo
            .download_file()
            .filename(filename.to_string())
            .maybe_revision(revision.map(|s| s.to_string()))
            .local_files_only(opts.local_files_only)
            .force_download(false)
            .send()
        {
            Ok(path) => return Ok(path),
            Err(e) => last_err = Some(e),
        }
    }
    match last_err {
        Some(e) => Err(HubError::hf("finding a required repo file", e)),
        None => Err(HubError::Package("no matching file found in the repo".into())),
    }
}

/// Build an F1 (ModernBERT) `ModelManifest` from a raw checkpoint config.
/// Pure: performs no I/O, so it is unit-testable against a sampled config.
fn build_synth_manifest(
    config: &serde_json::Value,
    repo: &str,
    revision: &str,
    rl_config: Option<&serde_json::Value>,
    tokenizer_path: Option<&str>,
) -> Result<ModelManifest> {
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if !model_type.to_lowercase().contains("modernbert") {
        return Err(HubError::Package(format!(
            "`{repo}` is a `{model_type}` checkpoint; only ModernBERT (F1) checkpoints can be synthesized"
        )));
    }

    let hidden_size = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024) as usize;
    let max_context = config
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as usize;

    let name = repo.rsplit('/').next().unwrap_or("model").to_string();
    let is_laya = rl_config.is_some();
    let template = if is_laya {
        "laya-v1".to_string()
    } else {
        "f1-v1".to_string()
    };

    let (max_len, head_max_len) = laya_budgets(rl_config);
    let calibration = laya_calibration(rl_config);
    let contract_hash = fnv1a(&template, repo);

    let mut artifacts = std::collections::BTreeMap::new();
    artifacts.insert(
        BackendId::Candle,
        vec![ArtifactRef {
            path: "model.safetensors".to_string(),
            dtype: "fp32".to_string(),
            quantization: None,
        }],
    );

    let manifest = ModelManifest {
        schema_version: manifest::MANIFEST_SCHEMA_VERSION.into(),
        name,
        family: Family::F1,
        backbone: Backbone {
            source: BackboneSource::Hf {
                repo: repo.to_string(),
                revision: revision.to_string(),
            },
            artifacts,
            hidden_size,
            max_context,
            // Laya scores candidates at real `[MASK]` positions, so we must
            // bundle the actual ModernBERT tokenizer rather than synthesize
            // `<option:N>` marker ids.
            tokenizer: tokenizer_path.map(|s| s.to_string()),
        },
        adapter: None,
        head: HeadConfig {
            kind: manifest::family_kind(Family::F1),
            // The decision-head tensors live inside `model.safetensors`; the
            // candle backend detects them and runs the typed option-marker head.
            weights: String::new(),
            width: 1,
            pointer_offset: None,
        },
        prompt_contract: PromptContract {
            template,
            option_marker_tokens: if is_laya {
                vec!["[MASK]".to_string()]
            } else {
                vec!["<option:0>".to_string()]
            },
            state_budget: head_max_len,
            head_budget: head_max_len,
            max_options: 255,
            contract_hash,
            max_len,
            head_max_len,
        },
        calibration,
        reference: None,
        capabilities: ModelCapabilities::default(),
    };
    manifest.validate().map_err(|e| HubError::Package(e.to_string()))?;
    Ok(manifest)
}

/// Laya's `max_len` / `head_max_len` budgets, falling back to reference defaults.
fn laya_budgets(rl_config: Option<&serde_json::Value>) -> (usize, usize) {
    let r = rl_config.unwrap_or(&serde_json::Value::Null);
    let max_len = r.get("max_len").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
    let head_max_len = r.get("head_max_len").and_then(|v| v.as_u64()).unwrap_or(192) as usize;
    (max_len, head_max_len)
}

/// Build calibration for a synthesized Laya package from `rl_agent_config.json`,
/// or a generic F1 calibration when no RL config is present.
fn laya_calibration(rl_config: Option<&serde_json::Value>) -> CalibrationConfig {
    let Some(rl) = rl_config else {
        return CalibrationConfig {
            default: CalibrationEntry {
                temperature: 1.0,
                per_type_temperatures: None,
                temperature_by_options: None,
                confidence: ConfidenceDef::Peak,
                status: CalibrationStatus::Pending,
            },
            entries: std::collections::BTreeMap::new(),
            eval_set_hash: None,
        };
    };

    // Base per-type temperatures `[choice, score, noul]`.
    let mut per_type = std::collections::BTreeMap::new();
    if let Some(arr) = rl.get("temperature").and_then(|v| v.as_array()) {
        let names = ["choice", "score", "noul"];
        for (i, name) in names.iter().enumerate() {
            if let Some(t) = arr.get(i).and_then(|v| v.as_f64()) {
                per_type.insert(name.to_string(), clamp_temp(t));
            }
        }
    }

    // Per `{type}:{bucket}` temperatures.
    let mut tbo = std::collections::BTreeMap::new();
    if let Some(map) = rl.get("temperature_by_options").and_then(|v| v.as_object()) {
        for (k, v) in map {
            if let Some(t) = v.as_f64() {
                tbo.insert(k.clone(), clamp_temp(t));
            }
        }
    }

    CalibrationConfig {
        default: CalibrationEntry {
            temperature: 1.0,
            per_type_temperatures: if per_type.is_empty() {
                None
            } else {
                Some(per_type)
            },
            temperature_by_options: if tbo.is_empty() {
                None
            } else {
                Some(tbo)
            },
            confidence: ConfidenceDef::Entropy,
            status: CalibrationStatus::Fit,
        },
        entries: std::collections::BTreeMap::new(),
        eval_set_hash: None,
    }
}

/// Clamp a temperature to Laya's usable `[0.5, 5.0]` range.
fn clamp_temp(t: f64) -> f32 {
    (t.max(0.5).min(5.0)) as f32
}

/// Stable FNV-1a hash (matches the converter's `contract_hash` scheme).
fn fnv1a(parts: &str, extra: &str) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for b in parts.as_bytes().iter().chain(extra.as_bytes()) {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
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

    #[test]
    fn synthesize_modernbert_manifest() {
        // Shape of `convaiinnovations/laya`'s `encoder/config.json`.
        let config = serde_json::json!({
            "model_type": "modernbert",
            "architectures": ["ModernBertForMaskedLM"],
            "hidden_size": 1024,
            "max_position_embeddings": 8192,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "vocab_size": 50368,
            "rope_parameters": {
                "full_attention": { "rope_theta": 160000.0 },
                "sliding_attention": { "rope_theta": 10000.0 }
            }
        });

        let m = build_synth_manifest(&config, "convaiinnovations/laya", "main", None, None).unwrap();
        assert_eq!(m.name, "laya");
        assert_eq!(m.family, Family::F1);
        assert_eq!(m.backbone.hidden_size, 1024);
        assert_eq!(m.backbone.max_context, 8192);
        assert_eq!(m.backbone.tokenizer, None);
        assert_eq!(m.head.kind, manifest::family_kind(Family::F1));
        assert_eq!(m.head.weights, "");
        assert_eq!(
            m.find_artifact(BackendId::Candle, "fp32").unwrap().path,
            "model.safetensors"
        );

        // The synthesized candle package only needs the backbone weights.
        let files = required_files(&m, Some(BackendId::Candle), "fp32", false).unwrap();
        assert_eq!(files, vec!["model.safetensors".to_string()]);
    }

    #[test]
    fn synthesize_laya_sets_laya_contract_and_calibration() {
        let config = serde_json::json!({ "model_type": "modernbert", "hidden_size": 1024 });
        let rl = serde_json::json!({
            "temperature": [1.6369, 1.2514, 1.9834],
            "temperature_by_options": {
                "choice:2": 1.9064,
                "choice:3-5": 1.7602,
                "choice:6-10": 1.0,
                "choice:11+": 0.1006,
                "score:3-5": 1.2514,
                "noul:2": 1.9834
            },
            "max_len": 512,
            "head_max_len": 192
        });
        let m = build_synth_manifest(
            &config,
            "convaiinnovations/laya",
            "main",
            Some(&rl),
            Some("tokenizer/tokenizer.json"),
        )
        .unwrap();
        assert_eq!(m.prompt_contract.template, "laya-v1");
        assert_eq!(m.prompt_contract.max_len, 512);
        assert_eq!(m.prompt_contract.head_max_len, 192);
        assert_eq!(m.backbone.tokenizer.as_deref(), Some("tokenizer/tokenizer.json"));
        assert_eq!(m.prompt_contract.option_marker_tokens, vec!["[MASK]"]);
        let c = &m.calibration.default;
        assert_eq!(c.confidence, ConfidenceDef::Entropy);
        assert_eq!(
            c.per_type_temperatures.as_ref().unwrap().get("choice").copied().unwrap(),
            1.6369
        );
        assert_eq!(
            c.temperature_by_options.as_ref().unwrap().get("choice:11+").copied().unwrap(),
            0.5
        );
        // Bundling the tokenizer means the candle package must fetch it too.
        let files = required_files(&m, Some(BackendId::Candle), "fp32", false).unwrap();
        assert!(files.contains(&"tokenizer/tokenizer.json".to_string()));
        assert!(files.contains(&"model.safetensors".to_string()));
    }

    #[test]
    fn synthesize_rejects_non_modernbert() {
        let config = serde_json::json!({ "model_type": "qwen3", "hidden_size": 4096 });
        let err = build_synth_manifest(&config, "org/qwen", "main", None, None).unwrap_err();
        assert!(matches!(err, HubError::Package(_)));
    }

    #[test]
    fn snapshot_root_finds_commit_dir_when_config_is_nested() {
        // Laya keeps `encoder/config.json` under a subdirectory; the snapshot
        // root is the `.../snapshots/<commit>` dir, not the `encoder/` dir.
        let dir = tempfile::tempdir().unwrap();
        let root = dir
            .path()
            .join("hub")
            .join("models--convaiinnovations--laya")
            .join("snapshots")
            .join("55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851");
        let encoder = root.join("encoder");
        std::fs::create_dir_all(&encoder).unwrap();
        let config = encoder.join("config.json");
        std::fs::write(&config, "{}").unwrap();
        assert_eq!(snapshot_root(&config).unwrap(), root);
    }

    #[test]
    fn snapshot_root_handles_config_at_the_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("hub").join("snapshots").join("abc123");
        std::fs::create_dir_all(&root).unwrap();
        let config = root.join("config.json");
        std::fs::write(&config, "{}").unwrap();
        assert_eq!(snapshot_root(&config).unwrap(), root);
    }
}
