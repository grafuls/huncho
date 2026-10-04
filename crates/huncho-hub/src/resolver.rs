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
    self, Adapter, ArtifactRef, Backbone, BackboneSource, BackendId, CalibrationConfig,
    CalibrationEntry, CalibrationStatus, ConfidenceDef, F3Config, Family, HeadConfig,
    ModelCapabilities, ModelManifest, PromptContract,
};

use crate::error::{HubError, Result};
use crate::http;

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
#[derive(Debug, Clone)]
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
    /// Render per-file download progress to stderr while fetching from the Hub.
    pub show_progress: bool,
}

impl Default for ResolveOptions {
    fn default() -> Self {
        Self {
            revision: None,
            token: None,
            cache_dir: None,
            local_files_only: false,
            fetch_golden: false,
            show_progress: true,
        }
    }
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
    if backend == Some(BackendId::Clef) {
        return crate::clef::resolve(model, dtype, opts);
    }
    let mref = ModelRef::parse(model, opts.revision.clone());
    match mref {
        ModelRef::Local(path) => resolve_local(&path),
        ModelRef::HuggingFace { repo, revision } => {
            resolve_hf(&repo, revision.as_deref(), backend, Some(dtype), None, opts)
        }
    }
}

/// Resolve a model and fetch artifacts for an automatically selected runtime.
/// `available` describes the caller's compiled runtimes, not the Hub client's.
pub fn resolve_auto(
    model: &str,
    dtype: Option<&str>,
    available: &[BackendId],
    opts: &ResolveOptions,
) -> Result<ResolvedPackage> {
    match ModelRef::parse(model, opts.revision.clone()) {
        ModelRef::Local(path) => {
            if path.is_dir()
                && !path.join("huncho-model.json").exists()
                && path.join("joint_head_config.json").is_file()
            {
                BackendId::Clef
                    .require_available(available)
                    .map_err(|e| HubError::Package(e.to_string()))?;
                return crate::clef::resolve(model, dtype.unwrap_or("fp32"), opts);
            }
            let package = resolve_local(&path)?;
            let manifest = ModelManifest::load(&package.manifest_path)
                .map_err(|e| HubError::Package(e.to_string()))?;
            manifest
                .select_backend(available, dtype)
                .map_err(|e| HubError::Package(e.to_string()))?;
            Ok(package)
        }
        ModelRef::HuggingFace { repo, revision } => resolve_hf(
            &repo,
            revision.as_deref(),
            None,
            dtype,
            Some(available),
            opts,
        ),
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
    dtype: Option<&str>,
    auto_available: Option<&[BackendId]>,
    opts: &ResolveOptions,
) -> Result<ResolvedPackage> {
    let rev: Option<String> = revision.map(|s| s.to_string());

    // Fetch the manifest first so we know which artifacts it references. If the
    // repo has no `huncho-model.json`, fall back to synthesizing a servable
    // package from a raw decision-model checkpoint (e.g. Laya's ModernBERT).
    // Only a genuine 404 (or a cache miss in offline mode) triggers synthesis;
    // transient network errors are surfaced to the user.
    let manifest_path = match http::download_file(repo, "huncho-model.json", rev.clone(), opts) {
        Ok(p) => p,
        Err(e) if matches!(e, HubError::NotFound { .. }) || opts.local_files_only => {
            let backend = if let Some(available) = auto_available {
                // Detect release layout rather than hard-coding repository names,
                // so forks and private releases use the same routing.
                match http::download_file(repo, "joint_head_config.json", rev.clone(), opts) {
                    Ok(head) => {
                        BackendId::Clef
                            .require_available(available)
                            .map_err(|e| HubError::Package(e.to_string()))?;
                        let mut pinned = opts.clone();
                        pinned.revision = head
                            .parent()
                            .and_then(|p| p.file_name())
                            .map(|s| s.to_string_lossy().into_owned());
                        return crate::clef::resolve(repo, dtype.unwrap_or("fp32"), &pinned);
                    }
                    Err(e) if matches!(e, HubError::NotFound { .. }) || opts.local_files_only => {}
                    Err(e) => return Err(e),
                }
                Some(
                    BackendId::Candle
                        .require_available(available)
                        .map_err(|e| HubError::Package(e.to_string()))?,
                )
            } else {
                backend
            };
            return synthesize_checkpoint(
                repo,
                rev.as_deref(),
                backend,
                dtype.unwrap_or("fp32"),
                opts,
            );
        }
        Err(e) => return Err(e),
    };

    let manifest = ModelManifest::load(&manifest_path)
        .map_err(|e| HubError::Package(format!("`{repo}` is not a Huncho model package: {e}")))?;
    let backend = match auto_available {
        Some(available) => Some(
            manifest
                .select_backend(available, dtype)
                .map_err(|e| HubError::Package(e.to_string()))?,
        ),
        None => backend,
    };
    let dtype = dtype.unwrap_or_else(|| manifest.default_dtype(backend.unwrap_or_default()));

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
    let pinned: Option<String> = if commit.is_empty() {
        rev.clone()
    } else {
        Some(commit)
    };

    if backend == Some(BackendId::Clef) {
        let mut pinned_opts = opts.clone();
        pinned_opts.revision = pinned.clone();
        let package = crate::clef::resolve(repo, dtype, &pinned_opts)?;
        if opts.fetch_golden {
            if let Some(reference) = &manifest.reference {
                http::download_file(repo, &reference.golden, pinned, opts)?;
            }
        }
        return Ok(package);
    }

    if backend == Some(BackendId::Candle) {
        if manifest.adapter.is_some() {
            http::download_file(repo, "adapter_config.json", pinned.clone(), opts)?;
            if let BackboneSource::Hf {
                repo: base,
                revision,
            } = &manifest.backbone.source
            {
                http::download_file(base, "config.json", Some(revision.clone()), opts)?;
                download_base_weights(base, revision, opts)?;
            }
        } else {
            let config = download_config(repo, pinned.as_deref(), opts)?;
            let target = package_root.join("config.json");
            if config != target {
                std::fs::copy(config, target)?;
            }
        }
    }

    for file in required_files(&manifest, backend, dtype, opts.fetch_golden)? {
        http::download_file(repo, &file, pinned.clone(), opts)?;
    }

    Ok(ResolvedPackage { manifest_path })
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
    // Base models are large; the Qwen3.5+LoRA (F3) path can build in fp32/fp16/
    // bf16. The ModernBERT (F1) candle backend always builds in fp32, so the
    // fp32-only restriction is enforced below only for that path.
    if !matches!(dtype, "fp32" | "fp16" | "f16" | "bf16" | "bfloat16") {
        return Err(HubError::Package(format!(
            "unsupported dtype `{dtype}` for `{repo}` (expected one of `fp32`, `fp16`, `bf16`)"
        )));
    }

    // PEFT is shared by multiple decision families. Kev extracts features for
    // its pointer head; Nimble uses the causal LM's candidate-token logits.
    if let Some(adapter_path) = probe_adapter(repo, revision, opts)? {
        let adapter = read_json(&adapter_path)?;
        if adapter.get("task_type").and_then(|v| v.as_str()) == Some("FEATURE_EXTRACTION") {
            if !matches!(dtype, "fp32" | "fp16" | "f16") {
                return Err(HubError::Package(format!(
                    "Kev's Candle CPU backend supports fp32 or fp16, not `{dtype}`"
                )));
            }
            return synthesize_kev(repo, opts, &adapter_path, &adapter);
        }
        return synthesize_f3(repo, revision, opts, &adapter_path);
    }

    // The ModernBERT (F1) candle backend always builds in fp32.
    if dtype != "fp32" {
        return Err(HubError::Package(format!(
            "synthesized `{repo}` (ModernBERT) only supports dtype `fp32` (requested `{dtype}`)"
        )));
    }

    // Fetch the encoder config (Laya keeps it under `encoder/`, HF models keep
    // it at the root).
    let config_path = download_config(repo, revision, opts)?;
    let config_bytes = std::fs::read(&config_path)?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)?;

    let package_root = snapshot_root(&config_path).ok_or_else(|| {
        HubError::Package("could not locate the HF snapshot directory for the checkpoint".into())
    })?;

    // Laya ships a `rl_agent_config.json` (calibration + prompt budgets). When
    // present, synthesize as `laya-v1`; otherwise fall back to a generic F1.
    let rl_config = download_first(repo, revision, opts, &["rl_agent_config.json"])
        .ok()
        .and_then(|path| std::fs::read(&path).ok())
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());

    // The real ModernBERT tokenizer lives under `tokenizer/` in Laya. Record
    // the repo-relative path so the loader can read it back from the snapshot.
    let tokenizer_path = download_first(
        repo,
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
        http::download_file(repo, &file, pinned.clone(), opts)?;
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
    repo: &str,
    revision: Option<&str>,
    opts: &ResolveOptions,
) -> Result<PathBuf> {
    let mut last_err: Option<HubError> = None;
    for filename in ["encoder/config.json", "config.json"] {
        match http::download_file(repo, filename, revision.map(|s| s.to_string()), opts) {
            Ok(path) => return Ok(path),
            Err(e) => last_err = Some(e),
        }
    }
    match last_err {
        Some(e) => Err(e),
        None => Err(HubError::Package(
            "no encoder config found in the checkpoint repo".into(),
        )),
    }
}

/// Download the first of a list of candidate filenames that exists in the repo,
/// returning the first successful download. Errors if all candidates fail.
fn download_first(
    repo: &str,
    revision: Option<&str>,
    opts: &ResolveOptions,
    candidates: &[&str],
) -> Result<PathBuf> {
    let mut last_err: Option<HubError> = None;
    for filename in candidates {
        match http::download_file(repo, filename, revision.map(|s| s.to_string()), opts) {
            Ok(path) => return Ok(path),
            Err(e) => last_err = Some(e),
        }
    }
    match last_err {
        Some(e) => Err(e),
        None => Err(HubError::Package("no matching file found in the repo".into())),
    }
}

/// Read a JSON file from a downloaded snapshot path.
fn read_json(path: &Path) -> Result<serde_json::Value> {
    let bytes = std::fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Probe a raw checkpoint repo for a PEFT adapter (not a decision family).
///
/// Returns `Ok(Some(adapter_config_path))` when `adapter_config.json` is
/// present, `Ok(None)` when the file is genuinely missing (404), and propagates
/// any other (e.g. network) error.
fn probe_adapter(
    repo: &str,
    revision: Option<&str>,
    opts: &ResolveOptions,
) -> Result<Option<PathBuf>> {
    match http::download_file(repo, "adapter_config.json", revision.map(|s| s.to_string()), opts) {
        Ok(path) => Ok(Some(path)),
        Err(e) if matches!(e, HubError::NotFound { .. }) || opts.local_files_only => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(not(feature = "candle"))]
fn synthesize_kev(
    repo: &str,
    _opts: &ResolveOptions,
    _adapter_path: &Path,
    _adapter: &serde_json::Value,
) -> Result<ResolvedPackage> {
    Err(HubError::Package(format!("`{repo}` is a feature-extraction adapter; Kev/F2 loading requires rebuilding with `--features hf,candle,tokenizers`")))
}

#[cfg(feature = "candle")]
fn synthesize_kev(
    repo: &str,
    opts: &ResolveOptions,
    adapter_path: &Path,
    adapter: &serde_json::Value,
) -> Result<ResolvedPackage> {
    use huncho_backend::kev::KevMetadata;

    let package_root = snapshot_root(adapter_path)
        .ok_or_else(|| HubError::Package("could not locate the Kev snapshot".into()))?;
    let commit = package_root
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| HubError::Package("invalid Kev snapshot path".into()))?;
    let pinned = Some(commit.to_string());
    // Read the head metadata before downloading any base weights: it pins the
    // trained base revision and temperature, which adapter_config alone cannot.
    let head_path = http::download_file(repo, "head.pt", pinned.clone(), opts)?;
    let metadata = KevMetadata::load(&head_path).map_err(|e| HubError::Package(e.to_string()))?;
    if adapter.get("peft_type").and_then(|v| v.as_str()) != Some("LORA")
        || adapter.get("r").and_then(|v| v.as_u64()) != Some(metadata.lora_rank as u64)
        || adapter
            .get("base_model_name_or_path")
            .and_then(|v| v.as_str())
            != Some(metadata.base.as_str())
    {
        return Err(HubError::Package(
            "Kev head.pt and adapter_config.json disagree about the base model or LoRA rank".into(),
        ));
    }
    for flag in ["use_dora", "use_rslora", "fan_in_fan_out"] {
        if adapter.get(flag).and_then(|v| v.as_bool()) == Some(true) {
            return Err(HubError::Package(format!(
                "Kev adapter option `{flag}=true` is not supported"
            )));
        }
    }
    for key in [
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "trainable_token_indices",
    ] {
        if let Some(v) = adapter.get(key) {
            let configured = match v {
                serde_json::Value::Null => false,
                serde_json::Value::Object(fields) => !fields.is_empty(),
                serde_json::Value::Array(items) => !items.is_empty(),
                _ => true,
            };
            if configured {
                return Err(HubError::Package(format!(
                    "Kev adapter option `{key}` is not supported"
                )));
            }
        }
    }
    if adapter
        .get("bias")
        .and_then(|v| v.as_str())
        .unwrap_or("none")
        != "none"
    {
        return Err(HubError::Package(
            "Kev adapter bias updates are not supported".into(),
        ));
    }
    let base_config_path = http::download_file(
        &metadata.base,
        "config.json",
        metadata.base_revision.clone(),
        opts,
    )?;
    let base_config = read_json(&base_config_path)?;
    let text = base_config.get("text_config").unwrap_or(&base_config);
    if !matches!(
        text.get("model_type").and_then(|v| v.as_str()),
        Some("qwen3_5_text" | "qwen3_5")
    ) {
        return Err(HubError::Package(
            "native Kev Candle loading currently requires a Qwen3.5 backbone".into(),
        ));
    }
    let hidden_size = text
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .filter(|n| *n > 0)
        .ok_or_else(|| HubError::Package("Kev base config has no hidden_size".into()))?
        as usize;
    // CPU attention materializes the causal mask. Use an 8k
    // window, rather than claiming the upstream GPU server's 64k state limit.
    let max_context = text
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(8192)
        .min(8192) as usize;
    let base_root = snapshot_root(&base_config_path)
        .ok_or_else(|| HubError::Package("could not locate Kev base snapshot".into()))?;
    let base_commit = base_root
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| HubError::Package("invalid Kev base snapshot".into()))?;
    let manifest = ModelManifest {
        schema_version: manifest::MANIFEST_SCHEMA_VERSION.into(),
        name: repo.rsplit('/').next().unwrap_or("kev").into(),
        family: Family::F2,
        backbone: Backbone {
            source: BackboneSource::Hf {
                repo: metadata.base.clone(),
                revision: base_commit.into(),
            },
            artifacts: [(
                BackendId::Candle,
                ["fp32", "fp16", "f16"]
                    .into_iter()
                    .map(|dtype| ArtifactRef {
                        path: "adapter_model.safetensors".into(),
                        dtype: dtype.into(),
                        quantization: None,
                    })
                    .collect(),
            )]
            .into(),
            hidden_size,
            max_context,
            tokenizer: Some("tokenizer.json".into()),
        },
        adapter: Some(Adapter {
            repo: repo.into(),
            revision: commit.into(),
            rank: metadata.lora_rank,
        }),
        f3: None,
        head: HeadConfig {
            kind: manifest::family_kind(Family::F2),
            weights: "head.pt".into(),
            width: metadata.head_dim,
            pointer_offset: None,
        },
        prompt_contract: PromptContract {
            template: "kev-v1".into(),
            option_marker_tokens: vec!["<|box_end|>".into()],
            state_budget: max_context,
            head_budget: max_context,
            max_options: 255,
            contract_hash: fnv1a("kev-v1", commit),
            max_len: max_context,
            head_max_len: max_context,
        },
        calibration: CalibrationConfig {
            default: CalibrationEntry {
                temperature: metadata.temperature,
                per_type_temperatures: None,
                temperature_by_options: None,
                confidence: ConfidenceDef::Peak,
                status: CalibrationStatus::Fit,
            },
            entries: Default::default(),
            eval_set_hash: None,
        },
        reference: None,
        capabilities: ModelCapabilities::default(),
    };
    manifest
        .validate()
        .map_err(|e| HubError::Package(e.to_string()))?;
    for file in required_files(&manifest, Some(BackendId::Candle), "fp32", false)? {
        http::download_file(repo, &file, pinned.clone(), opts)?;
    }
    download_base_weights(&metadata.base, base_commit, opts)?;
    let manifest_path = package_root.join("huncho-model.json");
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(ResolvedPackage { manifest_path })
}

/// Download the base-language-model weights for a Kev or Nimble adapter into the
/// base repo's snapshot directory (pinned to `rev`), so the Qwen3.5 candle
/// backend can build the model at serve time.
///
/// Prefers the sharded layout via `model.safetensors.index.json`, falling back
/// to a single `model.safetensors`. Reuses the snapshot path as a cache, so
/// already-downloaded shards are skipped. A repo with no reachable weights is
/// not an error here (the backend reports a clear error later); genuine
/// transport failures are surfaced.
fn download_base_weights(repo: &str, rev: &str, opts: &ResolveOptions) -> Result<()> {
    let rev_opt = Some(rev.to_string());

    let shards = match http::download_file(repo, "model.safetensors.index.json", rev_opt.clone(), opts)
    {
        Ok(index_path) => {
            let bytes = std::fs::read(&index_path)
                .map_err(|e| HubError::Package(format!("reading `{}`: {e}", index_path.display())))?;
            let idx: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
                HubError::Package(format!(
                    "parsing `model.safetensors.index.json` for `{repo}`: {e}"
                ))
            })?;
            let mut set = std::collections::BTreeSet::new();
            if let Some(wm) = idx.get("weight_map").and_then(|v| v.as_object()) {
                for p in wm.values().filter_map(|v| v.as_str()) {
                    if p.ends_with(".safetensors") {
                        set.insert(p.to_string());
                    }
                }
            }
            set.into_iter().collect::<Vec<_>>()
        }
        Err(HubError::NotFound { .. }) => Vec::new(),
        Err(e) => return Err(e),
    };

    if shards.is_empty() {
        // Single-file model, or no weights at all.
        match http::download_file(repo, "model.safetensors", rev_opt, opts) {
            Ok(_) => {}
            Err(HubError::NotFound { .. }) => {
                log::warn!(
                    "no base weights found for `{repo}`; the Qwen3.5 candle backend \
                     will fail at serve time until the base weights are available"
                );
            }
            Err(e) => return Err(e),
        }
        return Ok(());
    }

    for shard in &shards {
        http::download_file(repo, shard, rev_opt.clone(), opts)?;
    }
    Ok(())
}

/// Synthesize an F3 (Nimble) `ModelManifest` from a Bespoke-Nimble LoRA repo.
///
/// F3 packages are PEFT adapters over a base language model (e.g. a
/// `Qwen/Qwen3.5-*` backbone). The adapter repo carries an `adapter_config.json`
/// (LoRA rank), `schema_config.json` / `serving_config.json` (the candidate
/// codebook + system prompt), and `temperature_config.json` (calibration), but
/// no `config.json`. The base model reference and the architecture dimensions
/// are read from the base model's `config.json`, which is fetched (best-effort).
fn synthesize_f3(
    repo: &str,
    revision: Option<&str>,
    opts: &ResolveOptions,
    adapter_path: &Path,
) -> Result<ResolvedPackage> {
    let rev = revision.unwrap_or("main");
    let package_root = snapshot_root(adapter_path).ok_or_else(|| {
        HubError::Package("could not locate the HF snapshot directory for the adapter".into())
    })?;
    let commit = package_root
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    let schema_path = download_first(repo, revision, opts, &["schema_config.json"])?;
    let serving_path = download_first(repo, revision, opts, &["serving_config.json"])?;
    let temp_path = download_first(repo, revision, opts, &["temperature_config.json"]).ok();
    let tokenizer_path = download_first(repo, revision, opts, &["tokenizer.json"])
        .ok()
        .and_then(|p| {
            p.strip_prefix(&package_root)
                .ok()
                .map(|s| s.to_string_lossy().replace('\\', "/"))
        });

    let schema = read_json(&schema_path)?;
    let serving = read_json(&serving_path)?;
    let temp = match &temp_path {
        Some(p) => read_json(p)?,
        None => serde_json::Value::Null,
    };
    let adapter = read_json(adapter_path)?;

    let base_repo = schema
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("Qwen/Qwen3.5-9B")
        .to_string();
    let base_rev = schema
        .get("revision")
        .and_then(|v| v.as_str())
        .unwrap_or("main")
        .to_string();

    // The adapter repo has no `config.json`, so fetch the base model's tiny
    // `config.json` to derive the architecture dimensions. The resolved commit
    // (the snapshot dir name) is captured so base weights are downloaded and
    // later located deterministically at serve time.
    let base_config_path =
        http::download_file(&base_repo, "config.json", Some(base_rev.clone()), opts).ok();
    let base_config = base_config_path
        .as_deref()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok());
    let base_pin = base_config_path
        .as_deref()
        .and_then(snapshot_root)
        .and_then(|r| r.file_name().map(|s| s.to_string_lossy().to_string()))
        .unwrap_or_else(|| base_rev.clone());

    // Auto-fetch the (large) base weights into the base repo's snapshot dir so
    // the Qwen3.5 candle backend can build the model at serve time. Only when
    // the base config was reachable (the repo exists).
    if base_config.is_some() {
        download_base_weights(&base_repo, &base_pin, opts)?;
    }

    let manifest = build_f3_manifest(
        &schema,
        &serving,
        &temp,
        &adapter,
        base_config.as_ref(),
        repo,
        if commit.is_empty() { rev } else { &commit },
        &base_repo,
        &base_pin,
        tokenizer_path.as_deref(),
    )?;

    // Mirror the base `config.json` next to the adapter as a fallback. The
    // Qwen3.5 candle backend locates the base weights/config via the manifest's
    // `backbone.source` (the base repo snapshot), so this copy is only used when
    // that base-snapshot resolution cannot be derived (e.g. a local manifest).
    if let Some(cfg) = &base_config {
        if let Ok(bytes) = serde_json::to_vec_pretty(cfg) {
            let _ = std::fs::write(package_root.join("config.json"), bytes);
        }
    }

    // Persist the synthesized manifest in the resolved snapshot directory so a
    // later `serve --model` is deterministic and offline-resolvable.
    let manifest_path = package_root.join("huncho-model.json");
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    // Pin artifact downloads to the resolved snapshot commit.
    let pinned: Option<String> = if commit.is_empty() {
        revision.map(|s| s.to_string())
    } else {
        Some(commit.clone())
    };
    for file in required_files(&manifest, Some(BackendId::Candle), "fp32", false)? {
        http::download_file(repo, &file, pinned.clone(), opts)?;
    }

    Ok(ResolvedPackage { manifest_path })
}

/// Build an F3 (Nimble) `ModelManifest` from the adapter repo's service configs.
/// Pure: performs no I/O, so it is unit-testable against sampled configs.
#[allow(clippy::too_many_arguments)]
fn build_f3_manifest(
    schema: &serde_json::Value,
    serving: &serde_json::Value,
    temp: &serde_json::Value,
    adapter: &serde_json::Value,
    base_config: Option<&serde_json::Value>,
    repo: &str,
    adapter_revision: &str,
    base_repo: &str,
    base_rev: &str,
    tokenizer_path: Option<&str>,
) -> Result<ModelManifest> {
    let name = repo.rsplit('/').next().unwrap_or("model").to_string();

    let hidden_size = base_config
        .and_then(|c| c.get("hidden_size").and_then(|v| v.as_u64()))
        .unwrap_or(4096) as usize;
    let max_context = schema
        .get("max_length")
        .and_then(|v| v.as_u64())
        .unwrap_or(8192) as usize;

    let system_prompt = schema
        .get("system_prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let prompt_code_sha256 = schema
        .get("prompt_code_sha256")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let max_length = max_context;
    let lora_rank = schema
        .get("lora_rank")
        .and_then(|v| v.as_u64())
        .or_else(|| adapter.get("r").and_then(|v| v.as_u64()))
        .unwrap_or(16) as usize;

    let max_choices = serving
        .get("max_choices")
        .and_then(|v| v.as_u64())
        .or_else(|| schema.get("max_choices").and_then(|v| v.as_u64()))
        .unwrap_or(255) as usize;

    let candidate_codes: Vec<String> = serving
        .get("candidate_codes")
        .and_then(|v| v.as_array())
        .or_else(|| schema.get("candidate_codes").and_then(|v| v.as_array()))
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let candidate_token_ids: Vec<u32> = serving
        .get("candidate_token_ids")
        .and_then(|v| v.as_array())
        .or_else(|| schema.get("candidate_token_ids").and_then(|v| v.as_array()))
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default();

    let temperature = temp
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    let effective_prompt_hash = if prompt_code_sha256.is_empty() {
        fnv1a("nimble-v1", repo)
    } else {
        prompt_code_sha256.clone()
    };

    let mut artifacts = std::collections::BTreeMap::new();
    artifacts.insert(
        BackendId::Candle,
        ["fp32", "fp16", "f16"]
            .into_iter()
            .map(|dtype| ArtifactRef {
                path: "adapter_model.safetensors".to_string(),
                dtype: dtype.to_string(),
                quantization: None,
            })
            .collect(),
    );

    let manifest = ModelManifest {
        schema_version: manifest::MANIFEST_SCHEMA_VERSION.into(),
        name,
        family: Family::F3,
        backbone: Backbone {
            source: BackboneSource::Hf {
                repo: base_repo.to_string(),
                revision: base_rev.to_string(),
            },
            artifacts,
            hidden_size,
            max_context,
            // The Qwen tokenizer (for the base model) ships in the adapter repo.
            tokenizer: tokenizer_path.map(|s| s.to_string()),
        },
        adapter: Some(Adapter {
            repo: repo.to_string(),
            revision: adapter_revision.to_string(),
            rank: lora_rank,
        }),
        f3: Some(F3Config {
            candidate_codes,
            candidate_token_ids,
            system_prompt,
            prompt_code_sha256: effective_prompt_hash.clone(),
            max_input_tokens: max_length,
        }),
        head: HeadConfig {
            kind: manifest::family_kind(Family::F3),
            weights: String::new(),
            width: 1,
            pointer_offset: None,
        },
        prompt_contract: PromptContract {
            template: "nimble-v1".to_string(),
            option_marker_tokens: vec![],
            state_budget: max_length,
            head_budget: 0,
            max_options: max_choices,
            contract_hash: effective_prompt_hash,
            max_len: max_length,
            head_max_len: 0,
        },
        calibration: CalibrationConfig {
            default: CalibrationEntry {
                temperature,
                per_type_temperatures: None,
                temperature_by_options: None,
                confidence: ConfidenceDef::Peak,
                status: CalibrationStatus::Pending,
            },
            entries: std::collections::BTreeMap::new(),
            eval_set_hash: None,
        },
        reference: None,
        capabilities: ModelCapabilities::default(),
    };
    manifest.validate().map_err(|e| HubError::Package(e.to_string()))?;
    Ok(manifest)
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
        f3: None,
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
    (t.clamp(0.5, 5.0)) as f32
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
    fn auto_hub_resolution_fetches_only_the_selected_backend_and_dtype() {
        let cache = tempfile::tempdir().unwrap();
        let commit = "4444444444444444444444444444444444444444";
        let snapshot = cache
            .path()
            .join("models--fixture--package/snapshots")
            .join(commit);
        std::fs::create_dir_all(&snapshot).unwrap();
        let mut value = minimal_manifest_json("auto-package");
        value["backbone"]["artifacts"]["candle"] = serde_json::json!([
            {"path":"model.safetensors", "dtype":"fp16"}
        ]);
        std::fs::write(
            snapshot.join("huncho-model.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        // No Candle files in the cache: requesting fp32 must select ONNX.
        for file in [
            "model.onnx",
            "tokenizer.json",
            "head.safetensors",
            "golden.json",
        ] {
            std::fs::write(snapshot.join(file), []).unwrap();
        }
        let opts = ResolveOptions {
            revision: Some(commit.into()),
            cache_dir: Some(cache.path().into()),
            local_files_only: true,
            fetch_golden: true,
            show_progress: false,
            ..Default::default()
        };
        let available = [BackendId::Candle, BackendId::Onnx];
        let resolved = resolve_auto("fixture/package", Some("fp32"), &available, &opts).unwrap();
        assert_eq!(resolved.manifest_path, snapshot.join("huncho-model.json"));
        assert!(
            resolve_auto("fixture/package", Some("fp16"), &[BackendId::Onnx], &opts)
                .unwrap_err()
                .to_string()
                .contains("--features candle")
        );
        std::fs::remove_file(snapshot.join("model.onnx")).unwrap();
        assert!(resolve_auto("fixture/package", Some("fp32"), &available, &opts).is_err());
    }

    #[test]
    fn auto_raw_modernbert_resolves_and_reopens_from_cache() {
        let cache = tempfile::tempdir().unwrap();
        let commit = "5555555555555555555555555555555555555555";
        let snapshot = cache
            .path()
            .join("models--fixture--modernbert/snapshots")
            .join(commit);
        std::fs::create_dir_all(&snapshot).unwrap();
        let fixture = Path::new("../huncho-backend/tests/fixtures/tiny_modernbert");
        for file in ["config.json", "model.safetensors"] {
            std::fs::copy(fixture.join(file), snapshot.join(file)).unwrap();
        }
        let opts = ResolveOptions {
            revision: Some(commit.into()),
            cache_dir: Some(cache.path().into()),
            local_files_only: true,
            show_progress: false,
            ..Default::default()
        };
        assert!(resolve_auto("fixture/modernbert", None, &[], &opts)
            .unwrap_err()
            .to_string()
            .contains("--features candle"));
        for _ in 0..2 {
            let package =
                resolve_auto("fixture/modernbert", None, &[BackendId::Candle], &opts).unwrap();
            let manifest = ModelManifest::load(package.manifest_path).unwrap();
            assert_eq!(manifest.family, Family::F1);
            assert_eq!(
                manifest.select_backend(&[BackendId::Candle], None).unwrap(),
                BackendId::Candle
            );
        }
    }

    #[cfg(feature = "candle")]
    #[test]
    fn kev_adapter_is_not_treated_as_nimble() {
        // The real raw-checkpoint dispatcher, using a tiny upstream-generated
        // PEFT + head.pt snapshot with no Nimble schema/serving configs.
        let dir = tempfile::tempdir().unwrap();
        let fixture = Path::new("../huncho-backend/tests/fixtures/tiny_kev");
        let revision = "2222222222222222222222222222222222222222";
        let snapshot = dir
            .path()
            .join("models--fixture--kev/snapshots")
            .join(revision);
        let base = dir
            .path()
            .join("models--fixture--qwen3.5/snapshots/1111111111111111111111111111111111111111");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(&base).unwrap();
        for file in [
            "adapter_config.json",
            "adapter_model.safetensors",
            "head.pt",
            "tokenizer.json",
        ] {
            std::fs::copy(fixture.join(file), snapshot.join(file)).unwrap();
        }
        for file in ["config.json", "model.safetensors"] {
            std::fs::copy(fixture.join(file), base.join(file)).unwrap();
        }
        std::fs::write(
            base.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"model.safetensors"}}"#,
        )
        .unwrap();
        let opts = ResolveOptions {
            revision: Some(revision.into()),
            cache_dir: Some(dir.path().to_path_buf()),
            local_files_only: true,
            show_progress: false,
            ..Default::default()
        };
        let resolved = resolve_auto("fixture/kev", None, &[BackendId::Candle], &opts).unwrap();
        // A second resolution must fetch the adapter and pinned base package too.
        resolve_auto("fixture/kev", None, &[BackendId::Candle], &opts).unwrap();
        let manifest = ModelManifest::load(resolved.manifest_path).unwrap();
        assert_eq!(manifest.family, Family::F2);
        assert_eq!(manifest.prompt_contract.template, "kev-v1");
        assert_eq!(manifest.head.weights, "head.pt");
        assert_eq!(manifest.backbone.hidden_size, 16);
        assert_eq!(manifest.backbone.max_context, 512);
        assert_eq!(manifest.adapter.unwrap().revision, revision);
        assert!((manifest.calibration.default.temperature - 2.40605).abs() < 1e-5);
        assert!(
            matches!(manifest.backbone.source, BackboneSource::Hf { ref repo, ref revision }
            if repo == "fixture/qwen3.5" && revision == "1111111111111111111111111111111111111111")
        );
        assert!(!snapshot.join("schema_config.json").exists());
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
    fn synthesize_f3_manifest() {
        let schema = serde_json::json!({
            "model": "Qwen/Qwen3.5-9B",
            "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "system_prompt": "Classify using the schema; answer with one letter.",
            "prompt_code_sha256": "deadbeef",
            "max_length": 8192,
            "lora_rank": 16,
            "max_choices": 255
        });
        let serving = serde_json::json!({
            "max_choices": 255,
            "candidate_codes": ["A", "B", "C"],
            "candidate_token_ids": [1001, 1002, 1003]
        });
        let temp = serde_json::json!({"temperature": 1.0});
        let adapter = serde_json::json!({"peft_type": "LORA", "r": 16, "lora_alpha": 32});
        let base_config =
            serde_json::json!({"model_type": "qwen3", "hidden_size": 4096, "max_position_embeddings": 262144});

        let m = build_f3_manifest(
            &schema, &serving, &temp, &adapter, Some(&base_config),
            "bespokelabs/Bespoke-Nimble-9B", "c2022",
            "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            Some("tokenizer.json"),
        )
        .unwrap();

        assert_eq!(m.family, Family::F3);
        assert_eq!(m.default_dtype(BackendId::Candle), "fp16");
        assert_eq!(
            m.select_backend(&[BackendId::Candle], Some("fp16")).unwrap(),
            BackendId::Candle
        );
        assert_eq!(m.name, "Bespoke-Nimble-9B");
        assert_eq!(m.backbone.hidden_size, 4096);
        assert_eq!(m.backbone.max_context, 8192);
        assert_eq!(m.backbone.tokenizer.as_deref(), Some("tokenizer.json"));
        assert_eq!(m.head.kind, manifest::family_kind(Family::F3));

        // Base model reference + adapter bookkeeping.
        match &m.backbone.source {
            BackboneSource::Hf { repo, revision } => {
                assert_eq!(repo, "Qwen/Qwen3.5-9B");
                assert_eq!(revision, "c202236235762e1c871ad0ccb60c8ee5ba337b9a");
            }
            other => panic!("expected an Hf base source, got {other:?}"),
        }
        let adapter = m.adapter.as_ref().unwrap();
        assert_eq!(adapter.repo, "bespokelabs/Bespoke-Nimble-9B");
        assert_eq!(adapter.revision, "c2022");
        assert_eq!(adapter.rank, 16);

        // F3 codebook + system prompt.
        let f3 = m.f3.as_ref().unwrap();
        assert_eq!(f3.candidate_codes, vec!["A", "B", "C"]);
        assert_eq!(f3.candidate_token_ids, vec![1001, 1002, 1003]);
        assert_eq!(f3.max_input_tokens, 8192);
        assert_eq!(f3.prompt_code_sha256, "deadbeef");

        assert_eq!(m.prompt_contract.template, "nimble-v1");
        assert_eq!(m.prompt_contract.contract_hash, "deadbeef");
        assert_eq!(m.calibration.default.temperature, 1.0);

        // The candle artifact is the LoRA adapter; the tokenizer is bundled.
        let files = required_files(&m, Some(BackendId::Candle), "fp32", false).unwrap();
        assert!(files.contains(&"adapter_model.safetensors".to_string()));
        assert!(files.contains(&"tokenizer.json".to_string()));
    }

    #[test]
    fn synthesize_f3_rejects_mismatched_codebook() {
        let schema = serde_json::json!({
            "model": "Qwen/Qwen3.5-9B", "revision": "c2022",
            "system_prompt": "s", "prompt_code_sha256": "h", "max_length": 4096
        });
        let serving = serde_json::json!({
            "max_choices": 3,
            "candidate_codes": ["A", "B"],
            "candidate_token_ids": [1001, 1002, 1003]
        });
        let temp = serde_json::json!({});
        let adapter = serde_json::json!({"r": 16});

        let err = build_f3_manifest(
            &schema, &serving, &temp, &adapter, None,
            "bespokelabs/Bespoke-Nimble-9B", "c2022",
            "Qwen/Qwen3.5-9B", "c2022", None,
        )
        .unwrap_err();
        assert!(matches!(err, HubError::Package(_)));
    }

    #[test]
    fn synthesize_f3_defaults_temperature_and_contract_hash() {
        let schema = serde_json::json!({
            "model": "Qwen/Qwen3.5-9B", "revision": "c2022",
            "system_prompt": "s", "max_length": 4096
        });
        let serving = serde_json::json!({
            "max_choices": 2,
            "candidate_codes": ["A", "B"],
            "candidate_token_ids": [1001, 1002]
        });
        let temp = serde_json::json!({});
        let adapter = serde_json::json!({"r": 16});

        let m = build_f3_manifest(
            &schema, &serving, &temp, &adapter, None,
            "bespokelabs/Bespoke-Nimble-9B", "c2022",
            "Qwen/Qwen3.5-9B", "c2022", None,
        )
        .unwrap();
        // No temperature config -> T = 1.0; no prompt hash -> FNV fallback.
        assert_eq!(m.calibration.default.temperature, 1.0);
        assert!(!m.prompt_contract.contract_hash.is_empty());
        assert_eq!(m.f3.as_ref().unwrap().candidate_codes, vec!["A", "B"]);
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
