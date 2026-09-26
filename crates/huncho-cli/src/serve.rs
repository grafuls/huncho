//! `huncho serve` — run the HTTP API.

use std::sync::Arc;

use clap::Args;
use huncho_api::{AppState, Metrics, ModelRegistry, ServerConfig};
use huncho_core::manifest::{BackendId, Family};

use crate::load::{engine_from_manifest, mock_engine, mock_engine_from_manifest};

#[derive(Args)]
pub struct ServeArgs {
    /// Address to bind (host:port).
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: String,

    /// Require this bearer token on requests (API-03).
    #[arg(long)]
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

    /// Backend to use for manifest-loaded models (onnx|mock).
    #[arg(long, default_value = "mock")]
    pub backend: String,

    /// Override the dtype for manifest-loaded models.
    #[arg(long)]
    pub dtype: Option<String>,

    /// Serve engine extensions by default (API-05).
    #[arg(long, default_value_t = false)]
    pub extensions: bool,

    /// Model cache directory (OPS-04).
    #[arg(long)]
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

    for path in &args.manifest {
        tracing::info!("loading model manifest {path}");
        let engine = if args.backend.eq_ignore_ascii_case("mock") {
            mock_engine_from_manifest(path)?
        } else {
            engine_from_manifest(path, BackendId::parse(&args.backend)?, args.dtype.as_deref())?
        };
        let name = engine.manifest().name.clone();
        registry.insert(name.clone(), engine);
        tracing::info!("registered model `{name}`");
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
