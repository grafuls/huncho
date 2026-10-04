//! `huncho serve` — run the HTTP API.

use std::sync::Arc;

use clap::Args;
use huncho_api::{AppState, Metrics, ModelRegistry, ServerConfig};
use huncho_core::manifest::{BackendId, Family};

use crate::load::{engine_from_manifest, engine_from_ref, mock_engine, mock_engine_from_manifest};

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

    /// Backend to use for manifest/models (onnx|candle|clef|mock). Also read from `HUNCHO_BACKEND`.
    #[arg(long, default_value = "mock", env = "HUNCHO_BACKEND")]
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

pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
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

    let is_mock = args.backend.eq_ignore_ascii_case("mock");
    let backend_id = if is_mock {
        None
    } else {
        Some(BackendId::parse(&args.backend)?)
    };

    for path in &args.manifest {
        tracing::info!("loading model manifest {path}");
        let engine = if is_mock {
            mock_engine_from_manifest(path)?
        } else {
            engine_from_manifest(path, backend_id.expect("non-mock backend"), args.dtype.as_deref())?
        };
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
            backend_id,
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
