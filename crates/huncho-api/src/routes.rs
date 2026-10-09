//! HTTP routes for the `huncho` API.

use std::sync::Arc;
use std::time::Instant;

use axum::body::HttpBody;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use huncho_core::contract::SystemOneRequest;
use huncho_core::engine::{EvalOptions, EvalStats, PreparedEvaluation};
use huncho_core::error::{Error, ErrorBody};

use crate::auth::{check_auth, wants_extensions};
use crate::coalesce::{self, JobResult, Join};
use crate::config::ServerConfig;
use crate::metrics::{GaugeGuard, Metrics};
use crate::state::{AppState, ModelHandle};

/// Build the API router for `state`, with bearer auth enforced on every route.
///
/// The API is served as the returned router's fallback so auth runs before
/// routing. Adding routes or a fallback to it would bypass auth or replace the
/// API: register routes here, and `nest` or `merge` the result elsewhere.
pub fn router(state: Arc<AppState>) -> Router {
    let config = state.config.clone();
    let routes = Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/metrics", get(metrics))
        .with_state(state);
    // Auth wraps the routed service as a whole rather than each route, so it
    // runs before routing and body extraction: rejections are identical for
    // every path and method (no `Allow` header reveals which routes exist).
    Router::new()
        .fallback_service(routes)
        .layer(middleware::from_fn_with_state(config, require_auth))
}

/// Reject requests that fail the configured bearer-token check (API-03).
async fn require_auth(
    State(config): State<Arc<ServerConfig>>,
    request: Request,
    next: Next,
) -> Response {
    if let Err(status) = check_auth(request.headers(), &config) {
        let mut response = error_response(
            status,
            "unauthorized",
            "invalid or missing Authorization header",
        );
        let headers = response.headers_mut();
        headers.insert(
            http::header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer"),
        );
        // The body is never read, so keep-alive clients must not reuse the
        // connection hyper closes after this response.
        if !request.body().is_end_stream() {
            headers.insert(http::header::CONNECTION, HeaderValue::from_static("close"));
        }
        return response;
    }
    next.run(request).await
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
    if let Err(error) = req.validate() {
        return map_error(&error);
    }
    // Clone a model handle, then release the registry before waiting or running
    // inference. Health, metadata and model management remain responsive.
    let engine = match state.resolve_model(&model).await {
        Ok(engine) => engine,
        Err(error) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "model_unavailable",
                &error.to_string(),
            )
        }
    };
    let Some(engine) = engine else {
        return error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "model_not_found",
            &format!("unknown model `{model}`"),
        );
    };
    if state.config.cooperative_prefill
        && (!state.config.prefix_cache
            || !engine.supports_resumable_prefill()
            || engine.replica_engines().len() != 1
            || state.config.max_queued_per_model > 62
            || engine.batch.is_some())
    {
        return map_error(&Error::Unsupported("cooperative prefill requires CPU Kev, prefix reuse, configured chunks, one context, no cross-request collation and at most 62 queued requests".into()));
    }
    if state.config.prefix_cache
        && state.config.max_batch_tokens.is_some()
        && (!engine.supports_fork_batch()
            || (state.config.max_batch_padding_percent > 0 && !engine.supports_padded_fork_batch())
            || engine.batch.is_some())
    {
        return map_error(&Error::Unsupported(
            "cached-branch batching requires CPU Kev, supported suffix padding and no cross-request collation"
                .into(),
        ));
    }
    let opts = engine.serving_options(
        &state.config,
        wants_extensions(&headers) || state.config.default_extensions,
    );
    let qualification_tokens = match engine.serving_qualification_tokens(&state.config) {
        Ok(tokens) => tokens,
        Err(error) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "model_unqualified",
                &error.to_string(),
            )
        }
    };
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
    let qualification = engine.clone();
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
        None => {
            evaluate(
                engine,
                req,
                opts,
                state.metrics.clone(),
                admission,
                admitted,
            )
            .await
        }
    };

    // Qualification can be revoked while work/coalesced callers are waiting.
    // Never publish a response under a changed context/profile/environment.
    if let Err(error) =
        qualification.validate_serving_qualification_tokens(&state.config, &qualification_tokens)
    {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_unqualified",
            &error.to_string(),
        );
    }
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
) -> Arc<JobResult> {
    let model = req.model.clone();
    let cooperative = opts.cooperative_prefill;
    let (input, preparation_slot, admission, admitted) = if let Some(slots) = &engine.preparation {
        let start = Instant::now();
        let slot = match slots.clone().acquire_owned().await {
            Ok(slot) => slot,
            Err(_) => return Arc::new(JobResult::Unavailable),
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
            Ok((Err(error), _, _, _)) => return Arc::new(JobResult::Finished(Err(error))),
            Err(_) => return Arc::new(JobResult::WorkerFailed),
        }
    } else {
        (EvaluationInput::Raw(req, opts), None, admission, admitted)
    };
    let prepared_waiting = preparation_slot
        .as_ref()
        .map(|_| GaugeGuard::new(&metrics.prepared_waiting));
    if cooperative {
        return evaluate_resumable(
            engine,
            input,
            model,
            metrics,
            admission,
            admitted,
            preparation_slot,
            prepared_waiting,
        )
        .await;
    }
    let waiting = GaugeGuard::new(&metrics.requests_waiting);
    let queue_start = Instant::now();
    if let Some(queue) = &engine.batch {
        let EvaluationInput::Prepared(prepared) = input else {
            return Arc::new(JobResult::WorkerFailed);
        };
        return queue
            .submit(
                crate::batch::BatchJob {
                    prepared,
                    model,
                    admission,
                    admitted,
                    waiting,
                    prepared_waiting,
                    preparation_slot,
                    queued: queue_start,
                },
                engine.pool.clone(),
                metrics,
            )
            .await;
    }
    let execution = match engine.pool.acquire().await {
        Ok(permit) => permit,
        Err(_) => {
            return Arc::new(JobResult::Unavailable);
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
        let (_admission, _admitted) = (admission, admitted);
        // Free preparation capacity when execution begins, so the next bounded
        // CPU preparation can overlap this request's unchanged device work.
        drop(preparation_slot);
        let start = Instant::now();
        let mut stats = EvalStats::default();
        let result = match input {
            EvaluationInput::Raw(req, opts) => {
                let result = execution.eval_with_stats(&req, &opts, &mut stats);
                record_preparation(&metrics, &stats);
                result
            }
            EvaluationInput::Prepared(prepared) => {
                execution.eval_prepared_with_stats(prepared, &mut stats)
            }
        };
        record_execution(&metrics, &job_model, &stats);
        metrics
            .evaluation_latency
            .with_label_values(&[&job_model])
            .observe(start.elapsed().as_secs_f64());
        result
    })
    .await;

    Arc::new(match result {
        Ok(result) => JobResult::Finished(result),
        Err(_) => JobResult::WorkerFailed,
    })
}

#[allow(clippy::too_many_arguments)]
async fn evaluate_resumable(
    engine: ModelHandle,
    input: EvaluationInput,
    model: String,
    metrics: Arc<Metrics>,
    mut admission: tokio::sync::OwnedSemaphorePermit,
    mut admitted: GaugeGuard,
    mut preparation_slot: Option<tokio::sync::OwnedSemaphorePermit>,
    mut prepared_waiting: Option<GaugeGuard>,
) -> Arc<JobResult> {
    if engine.replica_engines().len() != 1 || engine.batch.is_some() {
        return Arc::new(JobResult::Finished(Err(Error::Unsupported(
            "cooperative prefill requires one execution context without collation".into(),
        ))));
    }
    let EvaluationInput::Prepared(prepared) = input else {
        return Arc::new(JobResult::Finished(Err(Error::Unsupported(
            "cooperative prefill requires upfront preparation".into(),
        ))));
    };
    let mut cursor = match engine.begin_resumable_evaluation(prepared) {
        Ok(cursor) => cursor,
        Err(error) => return Arc::new(JobResult::Finished(Err(error))),
    };
    let identity = Arc::new(());
    loop {
        let waiting = GaugeGuard::new(&metrics.requests_waiting);
        let queued = Instant::now();
        let execution = match engine.pool.acquire().await {
            Ok(execution) => execution,
            Err(_) => return Arc::new(JobResult::Unavailable),
        };
        metrics
            .queue_wait
            .with_label_values(&[&model])
            .observe(queued.elapsed().as_secs_f64());
        drop(waiting);
        drop(preparation_slot.take());
        drop(prepared_waiting.take());
        let job_metrics = metrics.clone();
        let job_model = model.clone();
        let last_prefill = engine.last_prefill.clone();
        let job_identity = identity.clone();
        // A canceled HTTP future abandons the returned cursor. The current
        // native call still owns every permit; its cursor is dropped only
        // after the kernel completes, releasing partial cache state safely.
        let step = tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            let mut stats = EvalStats::default();
            let result = execution.advance_resumable_evaluation(&mut cursor, &mut stats);
            if stats.prefill_calls > 0 {
                if let Ok(mut last) = last_prefill.lock() {
                    if last.as_ref().is_some_and(|(previous, pending)| {
                        *pending
                            && previous
                                .upgrade()
                                .is_some_and(|previous| !Arc::ptr_eq(&previous, &job_identity))
                    }) {
                        stats.prefill_interleaves += 1;
                    }
                    *last = Some((Arc::downgrade(&job_identity), stats.prefill_yields > 0));
                }
            }
            record_execution(&job_metrics, &job_model, &stats);
            job_metrics
                .evaluation_latency
                .with_label_values(&[&job_model])
                .observe(started.elapsed().as_secs_f64());
            // Explicitly release this step's execution lease before returning.
            // FIFO semaphore waiters can run before this request's next chunk.
            drop(execution);
            (cursor, result, admission, admitted)
        })
        .await;
        match step {
            Ok((next, Ok(None), permit, gauge)) => {
                cursor = next;
                admission = permit;
                admitted = gauge;
            }
            Ok((_, result, _, _)) => {
                return Arc::new(JobResult::Finished(
                    result.map(|response| response.unwrap()),
                ))
            }
            Err(_) => return Arc::new(JobResult::WorkerFailed),
        }
    }
}

pub(crate) fn record_execution(metrics: &Metrics, model: &str, stats: &EvalStats) {
    metrics.tokens_prefilled.inc_by(stats.processed_tokens);
    metrics.prefill_calls.inc_by(stats.prefill_calls);
    metrics.chunked_prefills.inc_by(stats.chunked_prefills);
    metrics.prefill_yields.inc_by(stats.prefill_yields);
    metrics
        .prefill_interleaves
        .inc_by(stats.prefill_interleaves);
    metrics
        .model_tokens
        .with_label_values(&[model])
        .inc_by(stats.processed_tokens);
    metrics.fork_count.inc_by(stats.cache_forks);
    metrics
        .persistent_prefix_hits
        .inc_by(stats.persistent_prefix_hits);
    metrics.batch_count.inc_by(stats.batch_calls);
    metrics
        .cross_request_batch_count
        .inc_by(stats.cross_request_batches);
    metrics.padded_batch_count.inc_by(stats.padded_batch_calls);
    metrics.fork_batch_count.inc_by(stats.fork_batch_calls);
    metrics
        .fork_padded_batch_count
        .inc_by(stats.fork_padded_batch_calls);
    metrics.padded_tokens.inc_by(stats.padded_tokens);
    metrics
        .reused_prefix_tokens
        .inc_by(stats.reused_prefix_tokens);
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
    replicas: usize,
    residency: String,
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
            replicas: engine.replica_engines().len(),
            residency: "eager".into(),
        });
    }
    models.extend(
        registry
            .lazy_descriptions()
            .into_iter()
            .map(|model| ModelInfo {
                name: model.name,
                family: model.family.to_string(),
                backend: model.backend.to_string(),
                dtype: model.dtype,
                max_context: model.max_context,
                replicas: model.replicas,
                residency: model.residency,
            }),
    );
    models.sort_by(|a, b| a.name.cmp(&b.name));
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
