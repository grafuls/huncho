//! HTTP routes for the `huncho` API.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;

use huncho_core::contract::SystemOneRequest;
use huncho_core::engine::EvalOptions;
use huncho_core::error::{Error, ErrorBody};

use crate::auth::{check_auth, wants_extensions};
use crate::state::AppState;

/// Build the API router. The state type is `Arc<AppState>`.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/metrics", get(metrics))
}

// ---------------------------------------------------------------------------
// POST /v1/systemone
// ---------------------------------------------------------------------------

async fn systemone(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<SystemOneRequest>,
) -> Response {
    if let Err(status) = check_auth(&headers, &state.config) {
        return error_response(status, "unauthorized", "invalid or missing Authorization header");
    }

    let model = req.model.clone();
    state.metrics.queue_depth.inc();
    let start = Instant::now();

    // Resolve the engine.
    let result = {
        let registry = state.registry.read().await;
        let engine = match registry.get(&model) {
            Some(e) => e,
            None => {
                state.metrics.queue_depth.dec();
                return error_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "model_not_found",
                    &format!("unknown model `{model}`"),
                );
            }
        };
        let opts = EvalOptions {
            extensions: wants_extensions(&headers) || state.config.default_extensions,
            ..Default::default()
        };
        engine.eval(&req, &opts)
    };

    let elapsed = start.elapsed().as_secs_f64();
    state
        .metrics
        .request_latency
        .with_label_values(&[&model])
        .observe(elapsed);
    state.metrics.queue_depth.dec();

    match result {
        Ok(resp) => {
            state
                .metrics
                .requests_total
                .with_label_values(&[&model])
                .inc();
            state
                .metrics
                .tokens_prefilled
                .inc_by(resp.usage.input_tokens);
            state
                .metrics
                .model_tokens
                .with_label_values(&[&model])
                .inc_by(resp.usage.input_tokens);
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => map_error(&e),
    }
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    uptime_secs: u64,
    models: usize,
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    let registry = state.registry.read().await;
    let body = HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: state.started.elapsed().as_secs(),
        models: registry.len(),
    };
    (StatusCode::OK, Json(body)).into_response()
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ModelsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    family: String,
    backend: String,
    dtype: String,
    max_context: usize,
}

async fn list_models(State(state): State<Arc<AppState>>) -> Response {
    let registry = state.registry.read().await;
    let mut models = Vec::new();
    for (name, engine) in registry
        .models()
    {
        models.push(ModelInfo {
            name: name.clone(),
            family: engine.family().to_string(),
            backend: engine.backend_id().to_string(),
            dtype: engine.dtype().to_string(),
            max_context: engine.manifest().backbone.max_context,
        });
    }
    (StatusCode::OK, Json(ModelsResponse { models })).into_response()
}

// ---------------------------------------------------------------------------
// GET /metrics
// ---------------------------------------------------------------------------

async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let body = state.metrics.render();
    (
        StatusCode::OK,
        [(http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        body,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn error_response(status: StatusCode, code: &str, message: &str) -> Response {
    let body = ErrorBody {
        error: huncho_core::error::ErrorDetail {
            code: code.to_string(),
            message: message.to_string(),
            details: None,
        },
    };
    (status, Json(body)).into_response()
}

fn map_error(e: &Error) -> Response {
    let (status, code) = match e {
        Error::Request(_) => (StatusCode::BAD_REQUEST, e.code()),
        Error::ModelNotFound(_) => (StatusCode::UNPROCESSABLE_ENTITY, e.code()),
        Error::Package(_) => (StatusCode::UNPROCESSABLE_ENTITY, e.code()),
        Error::Backend(_) => (StatusCode::INTERNAL_SERVER_ERROR, e.code()),
        Error::Unsupported(_) => (StatusCode::BAD_REQUEST, e.code()),
        Error::Calibration(_) => (StatusCode::INTERNAL_SERVER_ERROR, e.code()),
        Error::Conformance(_) => (StatusCode::INTERNAL_SERVER_ERROR, e.code()),
        Error::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, e.code()),
        Error::Json(_) => (StatusCode::BAD_REQUEST, e.code()),
    };
    error_response(status, code, &e.to_string())
}
