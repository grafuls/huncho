//! `huncho serve` — run the HTTP API.

use std::sync::Arc;

use clap::Args;
use huncho_api::{AppState, Metrics, ModelRegistry, ServerConfig};
use huncho_core::manifest::{BackendId, Family};

use crate::load::{engine_from_ref, engine_from_resolved_manifest, mock_engine, BackendChoice};

#[derive(Args)]
pub struct ServeArgs {
    /// Address to bind (host:port). Also read from `HUNCHO_BIND`.
    #[arg(long, default_value = "127.0.0.1:8080", env = "HUNCHO_BIND")]
    pub bind: String,

    /// Require this bearer token on requests (API-03). Also read from the
    /// `HUNCHO_AUTH_TOKEN` env var, so secrets can be supplied via an
    /// `EnvironmentFile=` (systemd) or a `.env` without appearing in the
    /// command line.
    #[arg(long, env = "HUNCHO_AUTH_TOKEN")]
    pub auth_token: Option<String>,

    /// Serve a built-in mock model (no weights required).
    #[arg(long, default_value_t = false)]
    pub mock: bool,

    /// Name(s) to register the mock model under (repeatable). If omitted, the
    /// mock is registered under both `mock` and `jev-latest` so that clients
    /// pointing at `model="jev-latest"` work with only a base-URL change.
    #[arg(long = "mock-model")]
    pub mock_model: Vec<String>,

    /// Additional model manifest paths (huncho-model.json) to serve.
    #[arg(long)]
    pub manifest: Vec<String>,

    /// Model reference(s) to serve: a local package dir/path or an HF repo id
    /// (`owner/repo`). Hub references require `hf` (included by `clef`).
    #[arg(long)]
    pub model: Vec<String>,

    /// Git revision to resolve HF model references at (default: repo default branch).
    #[arg(long)]
    pub revision: Option<String>,

    /// Hugging Face access token (defaults to HF_TOKEN / login cache).
    #[arg(long)]
    pub token: Option<String>,

    /// Backend override (auto|onnx|candle|clef|mock). Auto selects a runtime per model.
    #[arg(long, default_value = "auto", env = "HUNCHO_BACKEND")]
    pub backend: String,

    /// Override the dtype for manifest/manually-loaded models. Also read from `HUNCHO_DTYPE`.
    #[arg(long, env = "HUNCHO_DTYPE")]
    pub dtype: Option<String>,

    /// Serve engine extensions by default (API-05).
    #[arg(long, default_value_t = false)]
    pub extensions: bool,

    /// Model cache directory (also used by HF resolution; OPS-04). Also read from `HUNCHO_CACHE_DIR`.
    #[arg(long, env = "HUNCHO_CACHE_DIR")]
    pub cache_dir: Option<String>,
}

fn load_models(args: &ServeArgs) -> anyhow::Result<ModelRegistry> {
    let mut registry = ModelRegistry::new();

    if args.mock {
        let names = if args.mock_model.is_empty() {
            vec!["mock".to_string(), "jev-latest".to_string()]
        } else {
            args.mock_model.clone()
        };
        for name in &names {
            let engine = mock_engine(name, Family::F1, BackendId::Onnx, "fp32", 1.0)?;
            registry.insert(name.clone(), engine);
            tracing::info!("registered mock model `{name}` (F1 / onnx / fp32)");
        }
    }

    let backend = BackendChoice::parse(&args.backend)?;

    for path in &args.manifest {
        tracing::info!("loading model manifest {path}");
        let engine = engine_from_resolved_manifest(
            std::path::Path::new(path),
            backend,
            args.dtype.as_deref(),
        )?;
        let name = engine.manifest().name.clone();
        registry.insert(name.clone(), engine);
        tracing::info!("registered model `{name}`");
    }

    if !args.model.is_empty() {
        // The serve listener binds only after every model is loaded, so make it
        // clear that a first-time Hub resolution may be downloading weights.
        tracing::info!(
            "resolving {} model reference(s) from the Hugging Face Hub; the serve listener comes up only after they are ready",
            args.model.len()
        );
    }

    for model in &args.model {
        tracing::info!(
            "loading model `{model}` (fetching base weights if needed, then materializing; first load can take a couple of minutes)..."
        );
        let engine = engine_from_ref(
            model,
            backend,
            args.dtype.as_deref(),
            args.revision.clone(),
            args.token.clone(),
            args.cache_dir.clone(),
        )?;
        let name = engine.manifest().name.clone();
        registry.insert(name.clone(), engine);
        tracing::info!("registered model `{name}` (from `{model}`)");
    }

    if registry.is_empty() {
        tracing::warn!("no models registered; /v1/systemone will return 422 for every model");
    }
    Ok(registry)
}

pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
    let registry = load_models(&args)?;
    let config = ServerConfig {
        bind: args.bind.clone(),
        auth_token: args.auth_token.clone(),
        metrics: true,
        default_extensions: args.extensions,
    };

    let state = AppState::new(config, registry, Metrics::new());
    tracing::info!(
        "huncho starting (bind={}, mock={})",
        args.bind,
        args.mock,
    );

    huncho_api::serve(Arc::new(state)).await?;
    Ok(())
}

#[cfg(all(test, feature = "clef", feature = "onnx"))]
mod tests {
    use super::*;
    use clap::Parser;
    use huncho_core::manifest::ArtifactRef;
    use std::{fs, path::Path};

    #[test]
    fn one_server_selects_a_different_runtime_for_each_model() {
        let root = tempfile::tempdir().unwrap();
        let fixture = Path::new("../huncho-backend/tests/fixtures/tiny_modernbert");
        for file in ["config.json", "model.safetensors"] {
            fs::copy(fixture.join(file), root.path().join(file)).unwrap();
        }
        let mut manifest =
            crate::load::mock_manifest("tiny-candle", Family::F1, "fp32", 1.0).unwrap();
        manifest.backbone.hidden_size = 8;
        manifest.backbone.artifacts.insert(
            BackendId::Candle,
            vec![ArtifactRef {
                path: "model.safetensors".into(),
                dtype: "fp32".into(),
                quantization: None,
            }],
        );
        fs::write(
            root.path().join("huncho-model.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let crate::Command::Serve(args) = crate::Cli::try_parse_from([
            "huncho",
            "serve",
            "--backend",
            "auto",
            "--dtype",
            "fp32",
            "--model",
            root.path().to_str().unwrap(),
            "--manifest",
            "../../examples/mock-model/huncho-model.json",
            "--manifest",
            "../huncho-backend/tests/fixtures/tiny_clef/huncho-model.json",
        ])
        .unwrap()
        .command
        else {
            panic!("expected serve arguments")
        };
        let registry = load_models(&args).unwrap();
        assert_eq!(registry.len(), 3);
        assert_eq!(
            registry.get("tiny-candle").unwrap().backend_id(),
            BackendId::Candle
        );
        assert_eq!(
            registry.get("mock-laya").unwrap().backend_id(),
            BackendId::Onnx
        );
        assert_eq!(
            registry.get("tiny-clef").unwrap().backend_id(),
            BackendId::Clef
        );
    }
}
