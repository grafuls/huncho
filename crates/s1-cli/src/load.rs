//! Helpers to build a serving [`Engine`] from a manifest or an in-memory mock.

use std::path::Path;

use s1_backend::MockBackend;
#[cfg(feature = "onnx")]
use s1_backend::OnnxBackend;
use s1_core::backend::Backend;
use s1_core::engine::Engine;
use s1_core::error::{Error, Result};
use s1_core::head::HeadParams;
use s1_core::manifest::{
    Backbone, BackboneSource, BackendId, CalibrationConfig, CalibrationEntry, CalibrationStatus,
    ConfidenceDef, Family, HeadConfig, ModelManifest, PromptContract,
};
use s1_core::tokenizer::{SimpleTokenizer, Tokenizer};

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
/// Useful for `s1 serve --manifest ...` demos where no ONNX model is present.
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
    let head_kind = s1_core::manifest::family_kind(family);
    let manifest = ModelManifest {
        schema_version: s1_core::manifest::MANIFEST_SCHEMA_VERSION.into(),
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

    let tokenizer: Box<dyn Tokenizer> = Box::new(SimpleTokenizer::new(32768));
    let backend = load_backend(&manifest, backend_id, &dtype, path.as_ref())?;
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
