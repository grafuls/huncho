//! Helpers to build a serving [`Engine`] from a manifest or an in-memory mock.

use std::path::{Path, PathBuf};

use huncho_backend::MockBackend;
#[cfg(feature = "onnx")]
use huncho_backend::OnnxBackend;
use huncho_core::backend::Backend;
use huncho_core::engine::Engine;
use huncho_core::error::{Error, Result};
use huncho_core::head::HeadParams;
use huncho_core::manifest::{
    Backbone, BackboneSource, BackendId, CalibrationConfig, CalibrationEntry, CalibrationStatus,
    ConfidenceDef, Family, HeadConfig, ModelManifest, PromptContract,
};
#[cfg(feature = "tokenizers")]
use huncho_core::tokenizer::HfTokenizer;
use huncho_core::tokenizer::{SimpleTokenizer, Tokenizer};

/// Build the tokenizer for a manifest (CORE-02).
///
/// When the manifest declares a bundled `backbone.tokenizer` and the
/// `tokenizers` feature is enabled, the official Hugging Face `tokenizers`
/// crate is used so prompt tokenization is byte-identical to the reference
/// implementation. Otherwise the offline `SimpleTokenizer` is used (the
/// reference for the mock backend and the conformance fixtures).
#[allow(unused_variables)]
fn load_tokenizer(manifest: &ModelManifest, dir: &Path) -> Result<Box<dyn Tokenizer>> {
    match &manifest.backbone.tokenizer {
        Some(path) => {
            #[cfg(feature = "tokenizers")]
            {
                let p = dir.join(path);
                log::info!("loading tokenizer from {}", p.display());
                let tk = HfTokenizer::from_file(&p)
                    .map_err(|e| Error::Package(format!("failed to load tokenizer: {e}")))?;
                Ok(Box::new(tk) as Box<dyn Tokenizer>)
            }
            #[cfg(not(feature = "tokenizers"))]
            {
                log::warn!(
                    "manifest declares tokenizer `{path}` but the `tokenizers` feature is off; \
                     using the offline SimpleTokenizer (not reference-accurate)"
                );
                Ok(Box::new(SimpleTokenizer::new(32768)) as Box<dyn Tokenizer>)
            }
        }
        None => Ok(Box::new(SimpleTokenizer::new(32768)) as Box<dyn Tokenizer>),
    }
}

/// Build an in-memory mock model package and engine for offline/demo serving.
pub fn mock_engine(
    name: &str,
    family: Family,
    backend_id: BackendId,
    dtype: &str,
    temperature: f32,
) -> Result<Engine> {
    let manifest = mock_manifest(name, family, dtype, temperature)?;
    let backend: Box<dyn Backend> = Box::new(
        MockBackend::with_vocab(4096)
            .with_backend(backend_id)
            .with_dtype(dtype),
    );
    let tokenizer: Box<dyn Tokenizer> = Box::new(SimpleTokenizer::new(32768));
    Engine::new(manifest, backend, tokenizer, HeadParams::default(), backend_id, dtype)
}

/// Load a manifest and drive it with the mock backend (no weights required).
/// Useful for `huncho serve --manifest ...` demos where no ONNX model is present.
pub fn mock_engine_from_manifest(path: impl AsRef<Path>) -> Result<Engine> {
    let manifest = ModelManifest::load(path.as_ref())?;
    let name = manifest.name.clone();
    let family = manifest.family;
    log::info!("loading `{name}` ({family}) with the mock backend");
    let backend: Box<dyn Backend> = Box::new(
        MockBackend::with_vocab(4096)
            .with_backend(BackendId::Onnx)
            .with_dtype("fp32"),
    );
    let tokenizer: Box<dyn Tokenizer> = Box::new(SimpleTokenizer::new(32768));
    Engine::new(manifest, backend, tokenizer, HeadParams::default(), BackendId::Onnx, "fp32")
}

/// Build a mock [`ModelManifest`].
pub fn mock_manifest(name: &str, family: Family, _dtype: &str, temperature: f32) -> Result<ModelManifest> {
    let head_kind = huncho_core::manifest::family_kind(family);
    let manifest = ModelManifest {
        schema_version: huncho_core::manifest::MANIFEST_SCHEMA_VERSION.into(),
        name: name.into(),
        family,
        backbone: Backbone {
            source: BackboneSource::Hf {
                repo: "mock/placeholder".into(),
                revision: "main".into(),
            },
            artifacts: Default::default(),
            hidden_size: 1024,
            max_context: 8192,
            tokenizer: None,
        },
        adapter: None,
        head: HeadConfig {
            kind: head_kind,
            weights: "mock-head.safetensors".into(),
            width: 1,
            pointer_offset: None,
        },
        prompt_contract: PromptContract {
            template: format!("{family}-mock-v1"),
            option_marker_tokens: vec!["<option:0>".into()],
            state_budget: 3072,
            head_budget: 1024,
            max_options: 255,
            contract_hash: "mock-hash".into(),
        },
        calibration: CalibrationConfig {
            default: CalibrationEntry {
                temperature,
                per_type_temperatures: None,
                confidence: ConfidenceDef::Peak,
                status: CalibrationStatus::Fit,
            },
            entries: Default::default(),
            eval_set_hash: Some("none".into()),
        },
        reference: None,
        capabilities: Default::default(),
    };
    manifest.validate()?;
    Ok(manifest)
}

/// Load a serving [`Engine`] from a manifest on disk, using a real backend.
pub fn engine_from_manifest(
    path: impl AsRef<Path>,
    backend_id: BackendId,
    dtype: Option<&str>,
) -> Result<Engine> {
    let manifest = ModelManifest::load(path.as_ref())?;
    let dtype = dtype
        .map(|s| s.to_string())
        .unwrap_or_else(|| "fp32".to_string());

    // Artifacts and the tokenizer are declared relative to the manifest's
    // directory, not the manifest file itself.
    let dir = path.as_ref().parent().unwrap_or_else(|| Path::new("."));
    let tokenizer = load_tokenizer(&manifest, dir)?;
    let backend = load_backend(&manifest, backend_id, &dtype, dir)?;
    Engine::new(manifest, backend, tokenizer, HeadParams::default(), backend_id, dtype)
}

fn load_backend(
    manifest: &ModelManifest,
    backend_id: BackendId,
    dtype: &str,
    dir: &Path,
) -> Result<Box<dyn Backend>> {
    match backend_id {
        BackendId::Onnx => load_onnx(manifest, dtype, dir),
        other => Err(Error::Unsupported(format!(
            "backend `{other}` is not available in this build"
        ))),
    }
}

#[cfg(feature = "onnx")]
fn load_onnx(manifest: &ModelManifest, dtype: &str, dir: &Path) -> Result<Box<dyn Backend>> {
    let artifact = manifest
        .find_artifact(BackendId::Onnx, dtype)
        .ok_or_else(|| {
            Error::Package(format!("no ONNX artifact for dtype `{dtype}`"))
        })?;
    let onnx_path = dir.join(&artifact.path);
    let backend = OnnxBackend::load(
        onnx_path,
        manifest.backbone.hidden_size,
        manifest.backbone.max_context,
        dtype.to_string(),
    )
    .map_err(|e| Error::Package(format!("failed to load ONNX backend: {e}")))?;
    Ok(Box::new(backend) as Box<dyn Backend>)
}

#[cfg(not(feature = "onnx"))]
fn load_onnx(_manifest: &ModelManifest, _dtype: &str, _dir: &Path) -> Result<Box<dyn Backend>> {
    Err(Error::Unsupported(
        "the `onnx` feature is not enabled; rebuild with `--features onnx`".into(),
    ))
}

// ---------------------------------------------------------------------------
// Hugging Face Hub resolution (feature-gated)
// ---------------------------------------------------------------------------

/// Build an [`Engine`] from a resolved manifest path, honoring the mock
/// selection implied by `backend = None`.
pub fn engine_from_resolved_manifest(
    manifest_path: &Path,
    backend: Option<BackendId>,
    dtype: Option<&str>,
) -> Result<Engine> {
    match backend {
        None => mock_engine_from_manifest(manifest_path),
        Some(backend_id) => engine_from_manifest(manifest_path, backend_id, dtype),
    }
}

/// Resolve a model reference (local path or `owner/repo`) to a local package
/// manifest path, fetching missing artifacts from the Hub.
#[cfg(feature = "hf")]
pub fn resolve_model(
    model: &str,
    backend: Option<BackendId>,
    dtype: Option<&str>,
    revision: Option<String>,
    token: Option<String>,
    cache_dir: Option<String>,
    fetch_golden: bool,
) -> Result<PathBuf> {
    use huncho_hub::{ResolveOptions, resolve_manifest_path};

    let opts = ResolveOptions {
        revision,
        token,
        cache_dir: cache_dir.map(PathBuf::from),
        local_files_only: false,
        fetch_golden,
    };
    resolve_manifest_path(model, backend, dtype.unwrap_or("fp32"), &opts)
        .map_err(|e| Error::Package(e.to_string()))
}

#[cfg(not(feature = "hf"))]
pub fn resolve_model(
    _model: &str,
    _backend: Option<BackendId>,
    _dtype: Option<&str>,
    _revision: Option<String>,
    _token: Option<String>,
    _cache_dir: Option<String>,
    _fetch_golden: bool,
) -> Result<PathBuf> {
    Err(Error::Unsupported(
        "Hugging Face Hub resolution requires building huncho with `--features hf`".into(),
    ))
}

/// Build a serving [`Engine`] from a model reference that resolves against the
/// Hub (or a local package). `backend = None` selects the offline mock backend.
#[cfg(feature = "hf")]
pub fn engine_from_ref(
    model: &str,
    backend: Option<BackendId>,
    dtype: Option<&str>,
    revision: Option<String>,
    token: Option<String>,
    cache_dir: Option<String>,
) -> Result<Engine> {
    let manifest_path = resolve_model(model, backend, dtype, revision, token, cache_dir, false)?;
    engine_from_resolved_manifest(&manifest_path, backend, dtype)
}

#[cfg(not(feature = "hf"))]
pub fn engine_from_ref(
    _model: &str,
    _backend: Option<BackendId>,
    _dtype: Option<&str>,
    _revision: Option<String>,
    _token: Option<String>,
    _cache_dir: Option<String>,
) -> Result<Engine> {
    Err(Error::Unsupported(
        "Hugging Face Hub resolution requires building huncho with `--features hf`".into(),
    ))
}
