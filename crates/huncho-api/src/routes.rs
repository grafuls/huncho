//! HTTP routes for the `huncho` API.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use huncho_core::contract::SystemOneRequest;
use huncho_core::engine::{EvalOptions, EvalStats, PreparedEvaluation};
use huncho_core::error::{Error, ErrorBody};

use crate::auth::{check_auth, wants_extensions};
use crate::coalesce::{self, JobResult, Join};
use crate::metrics::{GaugeGuard, Metrics};
use crate::state::{AppState, ModelHandle};

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

/// The current serving contract is text/JSON only. Detect media explicitly so
/// a multimodal model request never silently loses its images or video.
#[derive(Deserialize)]
struct InferenceRequest {
    #[serde(flatten)]
    request: SystemOneRequest,
    images: Option<serde_json::Value>,
    videos: Option<serde_json::Value>,
}

async fn systemone(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<InferenceRequest>,
) -> Response {
    if let Err(status) = check_auth(&headers, &state.config) {
        return error_response(
            status,
            "unauthorized",
            "invalid or missing Authorization header",
        );
    }

    if body.images.is_some() || body.videos.is_some() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_media",
            "Huncho currently accepts text/JSON state only; images and videos are not supported",
        );
    }
    let req = body.request;

    let model = req.model.clone();
    let start = Instant::now();
    // Clone a model handle, then release the registry before waiting or running
    // inference. Health, metadata and model management remain responsive.
    let engine = {
        let registry = state.registry.read().await;
        registry.get(&model)
    };
    let Some(engine) = engine else {
        return error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "model_not_found",
            &format!("unknown model `{model}`"),
        );
    };
    if let Err(error) = req.validate() {
        return map_error(&error);
    }
    let admission = match engine.admission.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "queue_full",
                "model inference queue is full",
            )
        }
    };
    let admitted = GaugeGuard::new(&state.metrics.queue_depth);
    let opts = EvalOptions {
        extensions: wants_extensions(&headers) || state.config.default_extensions,
        prefix_cache: state.config.prefix_cache,
        max_batch_tokens: state.config.max_batch_tokens,
        reference_readout: !state.config.candidate_readout,
        prepare_all: engine.preparation.is_some(),
        ..Default::default()
    };
    let flight = if engine.flights.enabled() {
        match serde_json::to_vec(&(&req, &opts)) {
            Ok(key) => engine.flights.join(key),
            Err(error) => return map_error(&Error::Json(error)),
        }
    } else {
        None
    };
    let result = match flight {
        Some(Join::Follower(receiver)) => {
            // Followers remain authenticated, independently admitted callers.
            // Their cancellation releases capacity without affecting the job.
            let (_admission, _admitted) = (admission, admitted);
            let _waiting = GaugeGuard::new(&state.metrics.coalesced_waiting);
            state.metrics.requests_coalesced.inc();
            coalesce::wait(receiver).await
        }
        Some(Join::Leader(owner, receiver)) => {
            let metrics = state.metrics.clone();
            tokio::spawn(async move {
                let result = tokio::select! {
                    // If all callers cancel before execution, abandon the
                    // queued job. A running blocking job still owns permits.
                    biased;
                    _ = owner.sender.closed() => return,
                    result = evaluate(engine, req, opts, metrics, admission, admitted) => result,
                };
                owner.complete(result);
            });
            coalesce::wait(receiver).await
        }
        None => Arc::new(
            evaluate(
                engine,
                req,
                opts,
                state.metrics.clone(),
                admission,
                admitted,
            )
            .await,
        ),
    };

    state
        .metrics
        .request_latency
        .with_label_values(&[&model])
        .observe(start.elapsed().as_secs_f64());
    match result.as_ref() {
        JobResult::Finished(Ok(resp)) => {
            state
                .metrics
                .requests_total
                .with_label_values(&[&model])
                .inc();
            (StatusCode::OK, Json(resp)).into_response()
        }
        JobResult::Finished(Err(error)) => map_error(error),
        JobResult::Unavailable => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_unavailable",
            "model inference worker is unavailable",
        ),
        JobResult::WorkerFailed => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference_worker_failed",
            "model inference worker failed",
        ),
    }
}

async fn evaluate(
    engine: ModelHandle,
    req: SystemOneRequest,
    opts: EvalOptions,
    metrics: Arc<Metrics>,
    admission: tokio::sync::OwnedSemaphorePermit,
    admitted: GaugeGuard,
) -> JobResult {
    let model = req.model.clone();
    let (input, preparation_slot, admission, admitted) = if let Some(slots) = &engine.preparation {
        let start = Instant::now();
        let slot = match slots.clone().acquire_owned().await {
            Ok(slot) => slot,
            Err(_) => return JobResult::Unavailable,
        };
        metrics
            .preparation_wait
            .with_label_values(&[&model])
            .observe(start.elapsed().as_secs_f64());
        let backend = engine.engine.clone();
        let preparation_metrics = metrics.clone();
        let preparation_model = model.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            // This job owns admission and preparation capacity through HTTP
            // cancellation. A canceled preparation cannot submit model work.
            let _preparing = GaugeGuard::new(&preparation_metrics.requests_preparing);
            let start = Instant::now();
            let mut stats = EvalStats::default();
            let result = backend.prepare_eval_with_stats(req, opts, &mut stats);
            record_preparation(&preparation_metrics, &stats);
            preparation_metrics
                .preparation_latency
                .with_label_values(&[&preparation_model])
                .observe(start.elapsed().as_secs_f64());
            (result, slot, admission, admitted)
        })
        .await;
        match prepared {
            Ok((Ok(prepared), slot, admission, admitted)) => (
                EvaluationInput::Prepared(prepared),
                Some(slot),
                admission,
                admitted,
            ),
            Ok((Err(error), _, _, _)) => return JobResult::Finished(Err(error)),
            Err(_) => return JobResult::WorkerFailed,
        }
    } else {
        (EvaluationInput::Raw(req, opts), None, admission, admitted)
    };
    let prepared_waiting = preparation_slot
        .as_ref()
        .map(|_| GaugeGuard::new(&metrics.prepared_waiting));
    let waiting = GaugeGuard::new(&metrics.requests_waiting);
    let queue_start = Instant::now();
    let execution = match engine.execution.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => {
            return JobResult::Unavailable;
        }
    };
    metrics
        .queue_wait
        .with_label_values(&[&model])
        .observe(queue_start.elapsed().as_secs_f64());
    drop(waiting);
    drop(prepared_waiting);
    let job_model = model.clone();
    let result = tokio::task::spawn_blocking(move || {
        // Permits belong to the job, not the HTTP future. Cancellation cannot
        // admit another evaluation while this one is still using the backend.
        let (_admission, _execution, _admitted) = (admission, execution, admitted);
        // Free preparation capacity when execution begins, so the next bounded
        // CPU preparation can overlap this request's unchanged device work.
        drop(preparation_slot);
        let start = Instant::now();
        let mut stats = EvalStats::default();
        let result = match input {
            EvaluationInput::Raw(req, opts) => {
                let result = engine.engine.eval_with_stats(&req, &opts, &mut stats);
                record_preparation(&metrics, &stats);
                result
            }
            EvaluationInput::Prepared(prepared) => {
                engine.engine.eval_prepared_with_stats(prepared, &mut stats)
            }
        };
        metrics.tokens_prefilled.inc_by(stats.processed_tokens);
        metrics
            .model_tokens
            .with_label_values(&[&job_model])
            .inc_by(stats.processed_tokens);
        metrics.fork_count.inc_by(stats.cache_forks);
        metrics.batch_count.inc_by(stats.batch_calls);
        metrics
            .reused_prefix_tokens
            .inc_by(stats.reused_prefix_tokens);
        metrics
            .evaluation_latency
            .with_label_values(&[&job_model])
            .observe(start.elapsed().as_secs_f64());
        result
    })
    .await;

    match result {
        Ok(result) => JobResult::Finished(result),
        Err(_) => JobResult::WorkerFailed,
    }
}

// Jobs consume their inputs once. Keeping the owned packet inline avoids a
// transport-only allocation; admitted/preparing job counts bound retention.
#[allow(clippy::large_enum_variant)]
enum EvaluationInput {
    Raw(SystemOneRequest, EvalOptions),
    Prepared(PreparedEvaluation),
}

fn record_preparation(metrics: &Metrics, stats: &EvalStats) {
    metrics.result_cache_hits.inc_by(stats.result_cache_hits);
    metrics.prompt_cache_hits.inc_by(stats.prompt_cache_hits);
    metrics.questions_prepared.inc_by(stats.prepared_questions);
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
    for (name, engine) in registry.models() {
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
