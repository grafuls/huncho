//! Helpers to build a serving [`Engine`] from a manifest or an in-memory mock.

use std::path::{Path, PathBuf};

use huncho_backend::MockBackend;
#[cfg(feature = "candle")]
use huncho_backend::CandleBackend;
#[cfg(feature = "candle")]
use huncho_backend::Qwen3_5Backend;
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
use huncho_core::tokenizer::{CachedTokenizer, SimpleTokenizer, Tokenizer};

/// User intent is separate from the runtime id used for inference/calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Auto,
    Mock,
    Explicit(BackendId),
}

impl BackendChoice {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "mock" => Ok(Self::Mock),
            _ => BackendId::parse(value).map(Self::Explicit),
        }
    }
}

fn available_backends() -> Vec<BackendId> {
    [
        (BackendId::Clef, cfg!(feature = "clef")),
        (BackendId::Candle, cfg!(feature = "candle")),
        (BackendId::Onnx, cfg!(feature = "onnx")),
    ]
    .into_iter()
    .filter_map(|(backend, enabled)| enabled.then_some(backend))
    .collect()
}

/// Build the tokenizer for a manifest (CORE-02).
///
/// When the manifest declares a bundled `backbone.tokenizer` and the
/// `tokenizers` feature is enabled, the official Hugging Face `tokenizers`
/// crate is used so prompt tokenization is byte-identical to the reference
/// implementation. Otherwise the offline `SimpleTokenizer` is used (the
/// reference for the mock backend and the conformance fixtures).
#[allow(unused_variables)]
fn load_tokenizer(manifest: &ModelManifest, dir: &Path) -> Result<Box<dyn Tokenizer>> {
    let tokenizer: Result<Box<dyn Tokenizer>> = match &manifest.backbone.tokenizer {
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
                // A real model that declares a tokenizer path needs byte-identical
                // Hugging Face tokenization. Falling back to the offline
                // `SimpleTokenizer` silently feeds the model garbage token ids.
                return Err(Error::Package(format!(
                    "manifest declares tokenizer `{path}` but the `tokenizers` feature is off; \
                     rebuild with `--features tokenizers` (or `--features hf,candle,tokenizers` \
                     for Hub models) so real models get byte-identical HF tokenization"
                )));
            }
        }
        None => Ok(Box::new(SimpleTokenizer::new(32768)) as Box<dyn Tokenizer>),
    };
    let tokenizer = tokenizer?;
    let cache_bytes = match std::env::var("HUNCHO_TOKEN_CACHE_BYTES") {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            Error::Package("HUNCHO_TOKEN_CACHE_BYTES must be a nonnegative integer".into())
        })?,
        Err(std::env::VarError::NotPresent) => 0,
        Err(_) => return Err(Error::Package("invalid HUNCHO_TOKEN_CACHE_BYTES value".into())),
    };
    if cache_bytes == 0 {
        Ok(tokenizer)
    } else {
        log::info!("retaining exact tokenizer encodings up to {cache_bytes} charged bytes");
        Ok(Box::new(CachedTokenizer::new(tokenizer, cache_bytes)))
    }
}

fn configure_prompt_cache(engine: Engine) -> Result<Engine> {
    let bytes = match std::env::var("HUNCHO_PROMPT_CACHE_BYTES") {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            Error::Package("HUNCHO_PROMPT_CACHE_BYTES must be a nonnegative integer".into())
        })?,
        Err(std::env::VarError::NotPresent) => 0,
        Err(_) => return Err(Error::Package("invalid HUNCHO_PROMPT_CACHE_BYTES value".into())),
    };
    Ok(engine.with_prompt_cache(bytes))
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
        .and_then(configure_prompt_cache)
}

/// Load a manifest and drive it with the mock backend (no weights required).
/// Useful for explicit `--backend mock` demos where no weights are present.
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
        .and_then(configure_prompt_cache)
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
        f3: None,
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
            max_len: 512,
            head_max_len: 192,
        },
        calibration: CalibrationConfig {
            default: CalibrationEntry {
                temperature,
                per_type_temperatures: None,
                temperature_by_options: None,
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
    backend_id.require_available(&available_backends())?;
    // F3 (Qwen3.5+LoRA) backbones are large (9B+); default them to fp16 so a
    // plain `serve --backend candle` does not try to materialize a multi-GB
    // checkpoint in fp32. Candle's CPU matmul supports fp16 (but not bf16), so
    // fp16 is the right reduced-precision default here. Explicit `--dtype`
    // always wins.
    #[cfg(feature = "clef")]
    let clef_dtype = if backend_id == BackendId::Clef && dtype.is_none() {
        Some(huncho_backend::clef::default_dtype()?)
    } else {
        None
    };
    #[cfg(not(feature = "clef"))]
    let clef_dtype: Option<&str> = None;
    #[cfg(feature = "candle")]
    let kev_gpu = backend_id == BackendId::Candle
        && manifest.family == Family::F2
        && manifest.prompt_contract.template == "kev-v1"
        && dtype.is_none()
        && manifest.find_artifact(backend_id, "fp16").is_some()
        && huncho_backend::device::device_from_env()?.is_cuda();
    #[cfg(not(feature = "candle"))]
    let kev_gpu = false;
    let dtype = dtype.map(|s| s.to_string()).unwrap_or_else(|| {
        if backend_id == BackendId::Clef {
            clef_dtype.unwrap_or("fp16").to_string()
        } else if manifest.family == Family::F3 || kev_gpu {
            "fp16".to_string()
        } else {
            manifest.default_dtype(backend_id).to_string()
        }
    });
    tracing::info!(model = %manifest.name, backend = %backend_id, dtype = %dtype, "selected model runtime");

    // Artifacts and the tokenizer are declared relative to the manifest's
    // directory, not the manifest file itself.
    let dir = path.as_ref().parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    // The native Clef backend owns tokenization of the complete schema.
    let tokenizer: Box<dyn Tokenizer> = if backend_id == BackendId::Clef {
        Box::new(SimpleTokenizer::new(32768))
    } else {
        load_tokenizer(&manifest, dir)?
    };
    let backend = load_backend(&manifest, backend_id, &dtype, dir)?;
    Engine::new(manifest, backend, tokenizer, HeadParams::default(), backend_id, dtype)
        .and_then(configure_prompt_cache)
}

fn load_backend(
    manifest: &ModelManifest,
    backend_id: BackendId,
    dtype: &str,
    dir: &Path,
) -> Result<Box<dyn Backend>> {
    match backend_id {
        BackendId::Onnx => load_onnx(manifest, dtype, dir),
        BackendId::Candle => load_candle(manifest, dtype, dir),
        BackendId::Clef => load_clef(manifest, dtype, dir),
        other => Err(Error::Unsupported(format!(
            "backend `{other}` is not available in this build"
        ))),
    }
}

#[cfg(feature = "clef")]
fn load_clef(manifest: &ModelManifest, dtype: &str, dir: &Path) -> Result<Box<dyn Backend>> {
    Ok(Box::new(huncho_backend::ClefBackend::load(
        dir, manifest, dtype, huncho_backend::clef::device_from_env()?,
    )?))
}

#[cfg(not(feature = "clef"))]
fn load_clef(_manifest: &ModelManifest, _dtype: &str, _dir: &Path) -> Result<Box<dyn Backend>> {
    Err(Error::Unsupported("native Clef requires building huncho with --features clef".into()))
}

#[cfg(feature = "candle")]
fn load_candle(manifest: &ModelManifest, dtype: &str, dir: &Path) -> Result<Box<dyn Backend>> {
    if manifest.family == Family::F2 && manifest.prompt_contract.template == "kev-v1" {
        let backend = Qwen3_5Backend::load_kev_on_device(
            &adapter_base_dir(manifest, dir),
            dir,
            &dir.join(&manifest.head.weights),
            manifest.backbone.max_context,
            dtype,
            huncho_backend::device::device_from_env()?,
        )?
        .with_projection_chunk_rows(projection_chunk_rows_from_env()?)?
        .with_fp32_attention(fp32_attention_from_env()?)?;
        return Ok(Box::new(backend));
    }
    // F3 (Bespoke-Nimble) packages are candidate-logit PEFT adapters over a
    // Qwen3.5 hybrid backbone. Build the from-scratch Qwen3.5+LoRA candle
    // backend from the package directory (which holds `config.json`, the
    // `adapter_model.safetensors`/`adapter_config.json`, and the base weights
    // when the user has fetched them).
    if manifest.family == Family::F3 {
        // The adapter lives in the package dir; the (large) base weights are
        // downloaded into the base repo's own snapshot dir (same cache).
        let adapter_dir = dir;
        let base_dir = adapter_base_dir(manifest, adapter_dir);
        let backend = Qwen3_5Backend::load_on_device(
            &base_dir,
            Some(adapter_dir),
            manifest.backbone.max_context,
            dtype.to_string(),
            huncho_backend::device::opt_in_device_from_env()?,
        )
        .map_err(|e| {
            Error::Package(format!(
                "failed to load the F3 (Qwen3.5+LoRA) candle backend for `{}`: {e}",
                manifest.name
            ))
        })?;
        let backend = backend
            .with_projection_chunk_rows(projection_chunk_rows_from_env()?)?
            .with_fp32_attention(fp32_attention_from_env()?)?;
        return Ok(Box::new(backend) as Box<dyn Backend>);
    }
    let artifact = manifest
        .find_artifact(BackendId::Candle, dtype)
        .ok_or_else(|| Error::Package(format!("no candle artifact for dtype `{dtype}`")))?;
    let weights = dir.join(&artifact.path);
    // The ModernBERT config is stored next to the weights as `config.json`.
    let config = dir.join("config.json");
    let backend = CandleBackend::load_on_device(
        &config,
        &weights,
        manifest.backbone.max_context,
        dtype.to_string(),
        huncho_backend::device::opt_in_device_from_env()?,
    )
    .map_err(|e| Error::Package(format!("failed to load candle backend: {e}")))?;
    Ok(Box::new(backend) as Box<dyn Backend>)
}

#[cfg(feature = "candle")]
fn projection_chunk_rows_from_env() -> Result<usize> {
    match std::env::var("HUNCHO_PROJECTION_CHUNK_ROWS") {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            Error::Request("HUNCHO_PROJECTION_CHUNK_ROWS must be an integer from 0 to 4096".into())
        }),
        Err(std::env::VarError::NotPresent) => Ok(0),
        Err(_) => Err(Error::Request(
            "HUNCHO_PROJECTION_CHUNK_ROWS must be valid UTF-8".into(),
        )),
    }
}

#[cfg(feature = "candle")]
fn fp32_attention_from_env() -> Result<bool> {
    match std::env::var("HUNCHO_ATTENTION_FP32") {
        Ok(value) if matches!(value.as_str(), "1" | "true") => Ok(true),
        Ok(value) if matches!(value.as_str(), "0" | "false") => Ok(false),
        Err(std::env::VarError::NotPresent) => Ok(false),
        _ => Err(Error::Request(
            "HUNCHO_ATTENTION_FP32 must be 0, 1, false or true".into(),
        )),
    }
}

/// Locate the base-language-model directory for a Kev or Nimble manifest from the
/// adapter's resolved snapshot dir.
///
/// The adapter `dir` lives at `<cache>/models--<adapter>/snapshots/<commit>`.
/// The base weights are downloaded into the base repo's own snapshot dir in the
/// same cache, so we walk up to the cache root and rebuild the base path from
/// `backbone.source`. Falls back to `adapter_dir` when the layout does not match
/// the HF cache (e.g. a locally-authored manifest).
#[cfg(feature = "candle")]
fn adapter_base_dir(manifest: &ModelManifest, adapter_dir: &Path) -> PathBuf {
    let BackboneSource::Hf { repo, revision } = &manifest.backbone.source else {
        return adapter_dir.to_path_buf();
    };
    if adapter_dir.parent().and_then(|p| p.file_name()) != Some(std::ffi::OsStr::new("snapshots")) {
        return adapter_dir.to_path_buf();
    }
    let Some(cache) = adapter_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
    else {
        return adapter_dir.to_path_buf();
    };
    let revision = if revision.is_empty() { "main" } else { revision };
    cache
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(revision)
}

#[cfg(not(feature = "candle"))]
fn load_candle(_manifest: &ModelManifest, _dtype: &str, _dir: &Path) -> Result<Box<dyn Backend>> {
    Err(Error::Unsupported(
        "the `candle` feature is not enabled; rebuild with `--features candle`".into(),
    ))
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

/// Build an [`Engine`] from a manifest, choosing a runtime per model by default.
pub fn engine_from_resolved_manifest(
    manifest_path: &Path,
    backend: BackendChoice,
    dtype: Option<&str>,
) -> Result<Engine> {
    match backend {
        BackendChoice::Mock => mock_engine_from_manifest(manifest_path),
        BackendChoice::Explicit(backend_id) => {
            engine_from_manifest(manifest_path, backend_id, dtype)
        }
        BackendChoice::Auto => {
            let manifest = ModelManifest::load(manifest_path)?;
            let backend_id = manifest.select_backend(&available_backends(), dtype)?;
            engine_from_manifest(manifest_path, backend_id, dtype)
        }
    }
}

/// Resolve a model reference (local path or `owner/repo`) to a local package
/// manifest path, fetching missing artifacts from the Hub.
pub fn resolve_model(
    model: &str,
    backend: BackendChoice,
    dtype: Option<&str>,
    revision: Option<String>,
    token: Option<String>,
    cache_dir: Option<String>,
    fetch_golden: bool,
) -> Result<PathBuf> {
    if let BackendChoice::Explicit(id) = backend {
        id.require_available(&available_backends())?;
    }
    if backend == BackendChoice::Explicit(BackendId::Clef) {
        return resolve_hub_model(
            model,
            backend,
            dtype,
            revision,
            token,
            cache_dir,
            fetch_golden,
        );
    }
    // Local manifests do not need either Hub client.
    let path = Path::new(model);
    if path.exists() {
        if backend == BackendChoice::Auto
            && path.is_dir()
            && !path.join("huncho-model.json").exists()
            && path.join("joint_head_config.json").is_file()
        {
            BackendId::Clef.require_available(&available_backends())?;
            return resolve_hub_model(
                model,
                backend,
                dtype,
                revision,
                token,
                cache_dir,
                fetch_golden,
            );
        }
        let manifest = if path.is_dir() {
            path.join("huncho-model.json")
        } else {
            path.to_path_buf()
        };
        ModelManifest::load(&manifest)?;
        return Ok(manifest);
    }
    resolve_hub_model(model, backend, dtype, revision, token, cache_dir, fetch_golden)
}

#[cfg(feature = "hf")]
fn resolve_hub_model(
    model: &str,
    backend: BackendChoice,
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
        show_progress: true,
    };
    match backend {
        BackendChoice::Auto => huncho_hub::resolve_auto(model, dtype, &available_backends(), &opts)
            .map(|package| package.manifest_path),
        BackendChoice::Mock => resolve_manifest_path(model, None, dtype.unwrap_or("fp32"), &opts),
        BackendChoice::Explicit(id) => {
            resolve_manifest_path(model, Some(id), dtype.unwrap_or("fp32"), &opts)
        }
    }
    .map_err(|e| Error::Package(e.to_string()))
}

#[cfg(not(feature = "hf"))]
fn resolve_hub_model(
    _model: &str,
    _backend: BackendChoice,
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
/// Hub (or a local package).
pub fn engine_from_ref(
    model: &str,
    backend: BackendChoice,
    dtype: Option<&str>,
    revision: Option<String>,
    token: Option<String>,
    cache_dir: Option<String>,
) -> Result<Engine> {
    let manifest_path = resolve_model(model, backend, dtype, revision, token, cache_dir, false)?;
    engine_from_resolved_manifest(&manifest_path, backend, dtype)
}

#[cfg(all(test, feature = "candle"))]
mod candle_tests {
    use super::*;
    use crate::convert::{ConvertArgs, build_manifest};
    use huncho_core::backend::ForwardInput;
    use std::fs;
    use tempfile::tempdir;

    // Relative to `crates/huncho-cli` (the unit-test cwd).
    const FIXTURE: &str = "../huncho-backend/tests/fixtures/tiny_modernbert";

    #[cfg(feature = "tokenizers")]
    #[test]
    fn kev_prompts_and_answers_match_upstream() {
        use huncho_core::prompt::{formatter_for, PromptFormatter};
        let fixture = Path::new("../huncho-backend/tests/fixtures/tiny_kev");
        let dir = tempdir().unwrap();
        for file in [
            "config.json",
            "model.safetensors",
            "adapter_config.json",
            "adapter_model.safetensors",
            "head.pt",
            "tokenizer.json",
        ] {
            fs::copy(fixture.join(file), dir.path().join(file)).unwrap();
        }
        let mut manifest = mock_manifest("tiny-kev", Family::F2, "fp32", 2.4060501).unwrap();
        manifest.backbone.source = BackboneSource::Hf { repo: "fixture/qwen3.5".into(), revision: "1111111111111111111111111111111111111111".into() };
        manifest.backbone.hidden_size = 16;
        manifest.backbone.artifacts.insert(
            BackendId::Candle,
            vec![huncho_core::manifest::ArtifactRef {
                path: "adapter_model.safetensors".into(),
                dtype: "fp32".into(),
                quantization: None,
            }],
        );
        manifest.backbone.max_context = 512;
        manifest.backbone.tokenizer = Some("tokenizer.json".into());
        manifest.head.weights = "head.pt".into();
        manifest.head.width = 4;
        manifest.prompt_contract.template = "kev-v1".into();
        manifest.prompt_contract.state_budget = 512;
        let path = dir.path().join("huncho-model.json");
        fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
        let engine = engine_from_resolved_manifest(&path, BackendChoice::Auto, None).unwrap();
        assert_eq!(engine.backend_id(), BackendId::Candle);
        let formatter = formatter_for(&manifest);
        let tokenizer = load_tokenizer(&manifest, dir.path()).unwrap();
        let golden: serde_json::Value =
            serde_json::from_slice(&fs::read(fixture.join("golden.json")).unwrap()).unwrap();
        for case in golden["cases"].as_array().unwrap() {
            let req: huncho_core::contract::SystemOneRequest =
                serde_json::from_value(case["request"].clone()).unwrap();
            for ((id, _), row) in case["request"]["questions"]
                .as_object()
                .unwrap()
                .iter()
                .zip(case["rows"].as_array().unwrap())
            {
                let prompt = formatter
                    .build(&req.state, &req.questions[id], tokenizer.as_ref())
                    .unwrap();
                assert_eq!(
                    serde_json::to_value(&prompt.tokens).unwrap(),
                    row["tokens"],
                    "{id}: tokens"
                );
                assert_eq!(
                    serde_json::to_value(
                        prompt
                            .candidates
                            .iter()
                            .map(|c| c.position)
                            .collect::<Vec<_>>()
                    )
                    .unwrap(),
                    row["positions"],
                    "{id}: readout positions"
                );
                assert_eq!(
                    prompt.prefix_len,
                    row["prefix_len"].as_u64().unwrap() as usize
                );
            }
            let response = engine.eval(&req, &Default::default()).unwrap();
            for (id, expected) in case["answers"].as_object().unwrap() {
                let actual = serde_json::to_value(&response.answers[id]).unwrap();
                for (key, value) in expected.as_object().unwrap() {
                    if let Some(n) = value.as_f64() {
                        assert!(
                            (actual[key].as_f64().unwrap() - n).abs() < 0.0001,
                            "{id} {key}: {actual} vs {expected}"
                        );
                    } else if key == "probabilities" {
                        for (label, p) in value.as_object().unwrap() {
                            assert!(
                                (actual[key][label].as_f64().unwrap() - p.as_f64().unwrap()).abs()
                                    < 0.0001,
                                "{id}/{label}: {actual} vs {expected}"
                            );
                        }
                    } else {
                        assert_eq!(actual[key], *value, "{id}/{key}");
                    }
                }
            }
        }
        let req: huncho_core::contract::SystemOneRequest =
            serde_json::from_value(golden["cases"][0]["request"].clone()).unwrap();
        let limited = huncho_core::prompt::KevFormatter {
            max_state: 1,
            max_row: 512,
        };
        assert!(limited
            .build(&req.state, &req.questions["team"], tokenizer.as_ref())
            .unwrap_err()
            .to_string()
            .contains("state requires"));
        let limited = huncho_core::prompt::KevFormatter {
            max_state: 512,
            max_row: 2,
        };
        assert!(limited
            .build(&req.state, &req.questions["team"], tokenizer.as_ref())
            .unwrap_err()
            .to_string()
            .contains("question row requires"));
        // A package declaring FP16 uses it automatically only on CUDA.
        // Explicit dtype requests remain authoritative on either device.
        manifest.backbone.artifacts.get_mut(&BackendId::Candle).unwrap().push(
            huncho_core::manifest::ArtifactRef {
                path: "adapter_model.safetensors".into(),
                dtype: "fp16".into(),
                quantization: None,
            },
        );
        fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
        let automatic = engine_from_resolved_manifest(&path, BackendChoice::Auto, None).unwrap();
        assert_eq!(automatic.dtype(), if automatic.device().starts_with("GPU") { "fp16" } else { "fp32" });
        for dtype in ["fp16", "fp32"] {
            let explicit = engine_from_resolved_manifest(&path, BackendChoice::Auto, Some(dtype)).unwrap();
            assert_eq!(explicit.dtype(), dtype);
            explicit.eval(&req, &Default::default()).unwrap();
        }
    }

    #[test]
    fn load_candle_from_manifest_builds_backend() {
        let dir = tempdir().unwrap();
        fs::copy(Path::new(FIXTURE).join("config.json"), dir.path().join("config.json")).unwrap();
        fs::copy(Path::new(FIXTURE).join("model.safetensors"), dir.path().join("model.safetensors")).unwrap();

        let manifest = build_manifest(&ConvertArgs {
            hf_repo: "example/tiny".into(),
            revision: "main".into(),
            family: "F1".into(),
            backend: "candle".into(),
            dtype: "fp32".into(),
            out: dir.path().display().to_string(),
            hidden_size: Some(8),
            max_context: Some(16),
            tokenizer: None,
            runner: None,
            source: None,
            name: Some("tiny".into()),
        })
        .unwrap();

        let mut backend = load_candle(&manifest, "fp32", dir.path()).unwrap();
        assert_eq!(backend.id(), BackendId::Candle);

        let out = backend.forward(ForwardInput::new(vec![1, 2, 3, 4], vec![1])).unwrap();
        assert_eq!(out.values().shape(), &[1, 8]);
    }

    #[test]
    fn load_candle_f3_routes_to_qwen35_backend() {
        // An F3 (Bespoke-Nimble) package must route to the Qwen3.5+LoRA candle
        // backend (not be misinterpreted as ModernBERT). With no base weights in
        // the directory it must surface a clear package error rather than
        // silently loading the LoRA adapter as ModernBERT.
        let manifest: ModelManifest = serde_json::from_value(serde_json::json!({
            "schema_version": "1.0",
            "name": "nimble",
            "family": "F3",
            "backbone": {
                "source": { "kind": "hf", "repo": "Qwen/Qwen3.5-9B", "revision": "c2022" },
                "artifacts": { "candle": [{ "path": "adapter_model.safetensors", "dtype": "fp32", "quantization": null }] },
                "hidden_size": 4096,
                "max_context": 8192,
                "tokenizer": null
            },
            "f3": {
                "candidate_codes": ["A", "B"],
                "candidate_token_ids": [1001, 1002],
                "system_prompt": "Classify.",
                "prompt_code_sha256": "deadbeef",
                "max_input_tokens": 8192
            },
            "head": { "kind": "candidate-logit", "weights": "", "width": 1, "pointer_offset": null },
            "prompt_contract": {
                "template": "nimble-v1", "option_marker_tokens": [], "state_budget": 8192,
                "head_budget": 0, "max_options": 255, "contract_hash": "deadbeef",
                "max_len": 8192, "head_max_len": 0
            },
            "calibration": { "default": { "temperature": 1.0, "confidence": "peak" } }
        }))
        .unwrap();

        let dir = tempdir().unwrap();
        let err = match load_candle(&manifest, "fp32", dir.path()) {
            Ok(_) => panic!("expected an F3 load error without base weights"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("F3"),
            "expected the error to reference the F3 backend, got: {msg}"
        );
        assert!(
            !msg.contains("ModernBERT"),
            "F3 adapters must not be interpreted as ModernBERT: {msg}"
        );
    }

    #[test]
    fn engine_eval_end_to_end_via_candle() {
        let dir = tempdir().unwrap();
        fs::copy(Path::new(FIXTURE).join("config.json"), dir.path().join("config.json")).unwrap();
        fs::copy(Path::new(FIXTURE).join("model.safetensors"), dir.path().join("model.safetensors")).unwrap();

        let manifest = build_manifest(&ConvertArgs {
            hf_repo: "example/tiny".into(),
            revision: "main".into(),
            family: "F1".into(),
            backend: "candle".into(),
            dtype: "fp32".into(),
            out: dir.path().display().to_string(),
            hidden_size: Some(8),
            max_context: Some(128),
            tokenizer: None,
            runner: None,
            source: None,
            name: Some("tiny".into()),
        })
        .unwrap();

        let manifest_path = dir.path().join("huncho-model.json");
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

        // Full serving pipeline: manifest -> tokenizer -> candle backend -> F1
        // head -> calibration -> response.
        let engine =
            engine_from_resolved_manifest(&manifest_path, BackendChoice::Auto, None).unwrap();
        assert_eq!(engine.backend_id(), BackendId::Candle);
        let req: huncho_core::contract::SystemOneRequest = serde_json::from_value(serde_json::json!({
            "model": "tiny",
            "state": "short state text",
            "questions": {
                "q1": { "type": "choice", "instructions": "pick one", "criteria": { "a": "option A", "b": "option B" } }
            }
        }))
        .unwrap();

        let resp = engine.eval(&req, &huncho_core::engine::EvalOptions::default()).unwrap();
        assert!(resp.usage.input_tokens > 0);

        match resp.answers.get("q1").expect("answer for q1") {
            huncho_core::contract::Answer::Choice {
                probabilities,
                choice,
                confidence,
            } => {
                assert_eq!(probabilities.len(), 2);
                let sum: f32 = probabilities.values().sum();
                assert!(sum.is_finite());
                assert!((sum - 1.0).abs() < 1e-3, "probabilities must sum to 1, got {sum}");
                assert!(probabilities.contains_key(choice));
                assert!(*confidence >= 0.0 && *confidence <= 1.0);
            }
            other => panic!("expected a Choice answer, got {other:?}"),
        }
    }
}
