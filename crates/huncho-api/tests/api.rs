//! HTTP integration tests for the Jev `/v1/systemone` API (API-01/02/03/05).
//!
//! These drive the [`router()`] directly with `tower::ServiceExt::oneshot` (no
//! real socket) and assert the wire contract: typed answers, keys, health,
//! model listing, 422 on unknown model, 401 on missing/wrong auth, and opt-in
//! engine extensions.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use huncho_api::{AppState, Metrics, ModelRegistry, ServerConfig};
use huncho_backend::MockBackend;
use huncho_core::backend::Backend;
use huncho_core::engine::Engine;
use huncho_core::head::HeadParams;
use huncho_core::manifest::{
    self, Backbone, BackboneSource, BackendId, CalibrationConfig, CalibrationEntry,
    CalibrationStatus, ConfidenceDef, Family, HeadConfig, HeadKind, ModelManifest, PromptContract,
};
use huncho_core::tokenizer::{SimpleTokenizer, Tokenizer};
use serde_json::{json, Value};
use tower::ServiceExt;

fn mock_engine(name: &str) -> Engine {
    engine_with_backend(
        name,
        Box::new(
            MockBackend::with_vocab(4096)
                .with_backend(BackendId::Onnx)
                .with_dtype("fp32"),
        ),
    )
}

fn engine_with_backend(name: &str, backend: Box<dyn Backend>) -> Engine {
    engine_with_parts(name, backend, Box::new(SimpleTokenizer::new(32768)))
}

fn engine_with_parts(
    name: &str,
    backend: Box<dyn Backend>,
    tokenizer: Box<dyn Tokenizer>,
) -> Engine {
    let manifest = ModelManifest {
        schema_version: manifest::MANIFEST_SCHEMA_VERSION.into(),
        name: name.into(),
        family: Family::F1,
        backbone: Backbone {
            source: BackboneSource::Hf {
                repo: "mock/placeholder".into(),
                revision: "main".into(),
            },
            artifacts: Default::default(),
            hidden_size: 64,
            max_context: 512,
            tokenizer: None,
        },
        adapter: None,
        f3: None,
        head: HeadConfig {
            kind: HeadKind::OptionMarker,
            weights: "head.safetensors".into(),
            width: 1,
            pointer_offset: None,
        },
        prompt_contract: PromptContract {
            template: "f1-v1".into(),
            option_marker_tokens: vec!["<option:0>".into()],
            state_budget: 256,
            head_budget: 256,
            max_options: 255,
            contract_hash: "test-hash".into(),
            max_len: 512,
            head_max_len: 192,
        },
        calibration: CalibrationConfig {
            default: CalibrationEntry {
                temperature: 1.0,
                per_type_temperatures: None,
                temperature_by_options: None,
                confidence: ConfidenceDef::Peak,
                status: CalibrationStatus::Fit,
            },
            entries: Default::default(),
            eval_set_hash: None,
        },
        reference: None,
        capabilities: Default::default(),
    };
    manifest.validate().unwrap();

    Engine::new(
        manifest,
        backend,
        tokenizer,
        HeadParams::default(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap()
}

fn state(auth_token: Option<&str>) -> Arc<AppState> {
    let mut registry = ModelRegistry::new();
    registry.insert("mock-laya", mock_engine("mock-laya"));
    let config = ServerConfig {
        bind: "127.0.0.1:0".into(),
        auth_token: auth_token.map(|s| s.to_string()),
        metrics: true,
        default_extensions: false,
        max_queued_per_model: 32,
        max_prepared_per_model: 0,
        coalesce_bytes: 0,
        prefix_cache: false,
        cooperative_prefill: false,
        persistent_prefix_bytes: 0,
        max_batch_tokens: None,
        max_batch_padding_percent: 0,
        batch_max_requests: None,
        batch_wait_ms: 2,
        candidate_readout: false,
    };
    Arc::new(AppState::new(config, registry, Metrics::new()))
}

async fn send(
    state: Arc<AppState>,
    method: Method,
    uri: &str,
    body: Option<Value>,
    auth: Option<&str>,
    ext: bool,
) -> (StatusCode, Value) {
    let app = router_with_state(state);
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(tok) = auth {
        builder = builder.header("authorization", format!("Bearer {tok}"));
    }
    if ext {
        builder = builder.header("x-huncho-extensions", "1");
    }
    let req = match body {
        Some(b) => builder
            .header("content-type", "application/json")
            .body(Body::from(b.to_string()))
            .unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    };
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, v)
}

fn router_with_state(state: Arc<AppState>) -> axum::Router {
    huncho_api::router().with_state(state)
}

async fn send_raw(
    state: Arc<AppState>,
    method: Method,
    uri: &str,
    auth: Option<&str>,
) -> (StatusCode, String) {
    let app = router_with_state(state);
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(tok) = auth {
        builder = builder.header("authorization", format!("Bearer {tok}"));
    }
    let req = builder.body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

fn choice_request() -> Value {
    json!({
        "state": "The customer wants a refund because the shoes are too small.",
        "model": "mock-laya",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team handles this?",
                "criteria": { "returns": "The customer wants money back", "billing": "Charge problem" }
            }
        }
    })
}

#[tokio::test]
async fn rejects_media_instead_of_silently_discarding_it() {
    for field in ["images", "videos"] {
        let mut request = choice_request();
        request[field] = json!(["media"]);
        let (status, body) = send(
            state(None),
            Method::POST,
            "/v1/systemone",
            Some(request),
            None,
            false,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "unsupported_media");
    }
}

#[tokio::test]
async fn metrics_exposes_prometheus_registry() {
    // Issue a request first so the metrics counters are non-empty.
    let s = state(None);
    let (_, _) = send(
        s.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    let (status, text) = send_raw(s, Method::GET, "/metrics", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(text.contains("huncho_requests_total"));
    assert!(text.contains("huncho_request_latency_seconds"));
    assert!(text.contains("# TYPE huncho_requests_total counter"));
}

#[tokio::test]
async fn exact_result_cache_preserves_extensions_auth_and_physical_metrics() {
    let mut registry = ModelRegistry::new();
    registry.insert(
        "mock-laya",
        mock_engine("mock-laya").with_result_cache(1024 * 1024),
    );
    let s = Arc::new(AppState::new(
        ServerConfig {
            auth_token: Some("secret".into()),
            ..Default::default()
        },
        registry,
        Metrics::new(),
    ));
    let (status, first) = send(
        s.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("secret"),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let physical_tokens = s.metrics.tokens_prefilled.get();
    assert!(physical_tokens > 0);
    let (status, second) = send(
        s.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("secret"),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(first, second);
    assert_eq!(s.metrics.tokens_prefilled.get(), physical_tokens);
    assert_eq!(s.metrics.result_cache_hits.get(), 1);
    let (status, _) = send(
        s.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(s.metrics.result_cache_hits.get(), 1);
    let (_, extended) = send(
        s.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("secret"),
        true,
    )
    .await;
    assert!(extended["extensions"]["raw_logits"].is_object());
    assert_eq!(s.metrics.result_cache_hits.get(), 1);
    let (_, repeated) = send(
        s.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("secret"),
        true,
    )
    .await;
    assert_eq!(extended, repeated);
    assert_eq!(s.metrics.result_cache_hits.get(), 2);
    assert_eq!(s.metrics.tokens_prefilled.get(), 2 * physical_tokens);
}

#[tokio::test]
async fn health_reports_ok_and_model_count() {
    let (status, body) = send(state(None), Method::GET, "/health", None, None, false).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["models"], 1);
}

#[tokio::test]
async fn prepared_prompt_hits_keep_physical_metrics_and_authentication() {
    let mut registry = ModelRegistry::new();
    registry.insert(
        "mock-laya",
        mock_engine("mock-laya").with_prompt_cache(1024 * 1024),
    );
    let state = Arc::new(AppState::new(
        ServerConfig {
            auth_token: Some("secret".into()),
            ..Default::default()
        },
        registry,
        Metrics::new(),
    ));
    let mut first = None;
    let mut tokens = 0;
    for iteration in 0..2 {
        let (status, response) = send(
            state.clone(),
            Method::POST,
            "/v1/systemone",
            Some(choice_request()),
            Some("secret"),
            false,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        if let Some(first) = &first {
            assert_eq!(&response, first);
        } else {
            first = Some(response);
            tokens = state.metrics.tokens_prefilled.get();
        }
        assert!(tokens > 0);
        assert_eq!(
            state.metrics.tokens_prefilled.get(),
            tokens * (iteration + 1)
        );
        assert_eq!(state.metrics.prompt_cache_hits.get(), iteration);
        assert_eq!(state.metrics.result_cache_hits.get(), 0);
    }
    let (status, _) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(state.metrics.tokens_prefilled.get(), 2 * tokens);
    assert_eq!(state.metrics.prompt_cache_hits.get(), 1);
    let (status, response) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("secret"),
        true,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(response["extensions"]["raw_logits"].is_object());
    assert_eq!(state.metrics.tokens_prefilled.get(), 3 * tokens);
    assert_eq!(state.metrics.prompt_cache_hits.get(), 2);
    assert!(state
        .metrics
        .render()
        .contains("huncho_prompt_cache_hits 2"));
}

#[tokio::test]
async fn list_models_reports_registered_model() {
    let (status, body) = send(state(None), Method::GET, "/v1/models", None, None, false).await;
    assert_eq!(status, StatusCode::OK);
    let models = body["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["name"], "mock-laya");
    assert_eq!(models[0]["family"], "F1");
    assert_eq!(models[0]["backend"], "onnx");
    assert_eq!(models[0]["dtype"], "fp32");
}

#[tokio::test]
async fn choice_returns_summing_distribution() {
    let (status, body) = send(
        state(None),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let probs = &body["answers"]["department"]["probabilities"];
    let p_billing = probs["billing"].as_f64().unwrap();
    let p_returns = probs["returns"].as_f64().unwrap();
    assert!((p_billing + p_returns - 1.0).abs() < 1e-5);
    let choice = body["answers"]["department"]["choice"].as_str().unwrap();
    assert!(choice == "billing" || choice == "returns");
    assert!(
        body["answers"]["department"]["confidence"]
            .as_f64()
            .unwrap()
            >= 0.0
    );
    assert!(
        body["answers"]["department"]["confidence"]
            .as_f64()
            .unwrap()
            <= 1.0
    );
    assert!(body["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(body["usage"]["output_tokens"].as_u64().unwrap(), 0);
}

#[tokio::test]
async fn noul_and_score_questions_serve() {
    let req = json!({
        "state": "A user asks whether their account was charged twice.",
        "model": "mock-laya",
        "questions": {
            "is_refund": {
                "type": "noul",
                "instructions": "Is the customer requesting a refund?",
                "criteria": { "true": "Asks for money back", "false": "Does not ask" }
            },
            "severity": {
                "type": "score",
                "instructions": "Rate severity",
                "criteria": ["Low", "Moderate", "High"]
            }
        }
    });
    let (status, body) = send(
        state(None),
        Method::POST,
        "/v1/systemone",
        Some(req),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let noul = body["answers"]["is_refund"]["noul"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&noul));
    assert_eq!(body["answers"]["is_refund"]["type"], "noul");
    let sev = &body["answers"]["severity"];
    assert_eq!(sev["type"], "score");
    let score = sev["score"].as_f64().unwrap();
    assert!(score >= 0.0 && score <= 2.0);
    assert!(sev["legend"].is_object());
}

#[tokio::test]
async fn unknown_model_returns_422() {
    let mut req = choice_request();
    req["model"] = json!("does-not-exist");
    let (status, body) = send(
        state(None),
        Method::POST,
        "/v1/systemone",
        Some(req),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn missing_auth_token_returns_401() {
    let s = state(Some("secret"));
    // no Authorization header
    let (status, _) = send(
        s.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    // wrong token
    let (status, _) = send(
        s.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("wrong"),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    // correct token
    let (status, body) = send(
        s,
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("secret"),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["answers"].is_object());
}

#[tokio::test]
async fn extensions_returned_only_when_requested() {
    // Default: no extensions.
    let (_, body) = send(
        state(None),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert!(body.get("extensions").is_none());

    // Opt in via header.
    let (_, body) = send(
        state(None),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        true,
    )
    .await;
    let ext = body["extensions"].as_object().unwrap();
    assert_eq!(ext["backend"], "onnx");
    assert_eq!(ext["dtype"], "fp32");
    assert_eq!(ext["calibration_status"], "fit");
    assert!(ext["prompt_contract_hash"]
        .as_str()
        .unwrap()
        .contains("test-hash"));
    let raw = ext["raw_logits"].as_object().unwrap();
    assert!(raw["department"].as_array().unwrap().len() >= 2);
}

type SlowGate = Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>;

struct ReleaseSlowJob(SlowGate);
impl Drop for ReleaseSlowJob {
    fn drop(&mut self) {
        *self.0 .0.lock().unwrap() = true;
        self.0 .1.notify_all();
    }
}

fn slow_state(
    max_queued: u16,
) -> (
    Arc<AppState>,
    tokio::sync::oneshot::Receiver<()>,
    ReleaseSlowJob,
) {
    slow_state_with_config(ServerConfig {
        max_queued_per_model: max_queued,
        ..Default::default()
    })
}

fn slow_state_with_config(
    config: ServerConfig,
) -> (
    Arc<AppState>,
    tokio::sync::oneshot::Receiver<()>,
    ReleaseSlowJob,
) {
    struct SlowBackend {
        inner: MockBackend,
        gate: SlowGate,
        started: Option<tokio::sync::oneshot::Sender<()>>,
    }
    impl Backend for SlowBackend {
        fn id(&self) -> BackendId {
            self.inner.id()
        }
        fn capabilities(&self) -> huncho_core::backend::Capabilities {
            self.inner.capabilities()
        }
        fn fork(
            &mut self,
            handle: huncho_core::backend::CacheHandle,
        ) -> huncho_core::error::Result<huncho_core::backend::CacheHandle> {
            self.inner.fork(handle)
        }
        fn forward(
            &mut self,
            input: huncho_core::backend::ForwardInput,
        ) -> huncho_core::error::Result<huncho_core::backend::ForwardOutput> {
            if let Some(started) = self.started.take() {
                let _ = started.send(());
                let mut released = self.gate.0.lock().unwrap();
                while !*released {
                    let (guard, deadline) = self
                        .gate
                        .1
                        .wait_timeout(released, std::time::Duration::from_secs(2))
                        .unwrap();
                    released = guard;
                    if deadline.timed_out() {
                        break;
                    }
                }
            }
            self.inner.forward(input)
        }
    }
    let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let (started, receiver) = tokio::sync::oneshot::channel();
    let backend = SlowBackend {
        inner: MockBackend::with_vocab(4096),
        gate: gate.clone(),
        started: Some(started),
    };
    let mut registry = ModelRegistry::new();
    registry.insert(
        "mock-laya",
        engine_with_backend("mock-laya", Box::new(backend)),
    );
    (
        Arc::new(AppState::new(config, registry, Metrics::new())),
        receiver,
        ReleaseSlowJob(gate),
    )
}

fn batch_state(
    config: ServerConfig,
    slow: bool,
) -> (
    Arc<AppState>,
    tokio::sync::oneshot::Receiver<()>,
    ReleaseSlowJob,
    Arc<std::sync::Mutex<Vec<usize>>>,
) {
    struct Batched {
        inner: MockBackend,
        rows: Arc<std::sync::Mutex<Vec<usize>>>,
        gate: SlowGate,
        started: Option<tokio::sync::oneshot::Sender<()>>,
    }
    impl Backend for Batched {
        fn id(&self) -> BackendId {
            self.inner.id()
        }
        fn capabilities(&self) -> huncho_core::backend::Capabilities {
            self.inner.capabilities()
        }
        fn supports_batch(&self) -> bool {
            true
        }
        fn supports_padded_batch(&self) -> bool {
            true
        }
        fn forward_padded_batch(
            &mut self,
            inputs: Vec<huncho_core::backend::ForwardInput>,
        ) -> huncho_core::Result<Vec<huncho_core::backend::ForwardOutput>> {
            // Transport/scheduler fixture only: native padded arithmetic is
            // verified with actual Qwen tensors in backend/padded_batch.rs.
            self.forward_batch(inputs)
        }
        fn fork(
            &mut self,
            h: huncho_core::backend::CacheHandle,
        ) -> huncho_core::Result<huncho_core::backend::CacheHandle> {
            self.inner.fork(h)
        }
        fn forward(
            &mut self,
            input: huncho_core::backend::ForwardInput,
        ) -> huncho_core::Result<huncho_core::backend::ForwardOutput> {
            self.forward_batch(vec![input])
                .map(|mut outputs| outputs.remove(0))
        }
        fn forward_batch(
            &mut self,
            inputs: Vec<huncho_core::backend::ForwardInput>,
        ) -> huncho_core::Result<Vec<huncho_core::backend::ForwardOutput>> {
            self.rows.lock().unwrap().push(inputs.len());
            if let Some(started) = self.started.take() {
                let _ = started.send(());
                let released = self.gate.0.lock().unwrap();
                let _ = self
                    .gate
                    .1
                    .wait_timeout_while(released, std::time::Duration::from_secs(2), |released| {
                        !*released
                    })
                    .unwrap();
            }
            inputs
                .into_iter()
                .map(|input| self.inner.forward(input))
                .collect()
        }
    }
    let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let rows = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (started, receiver) = tokio::sync::oneshot::channel();
    let mut registry = ModelRegistry::new();
    registry.insert(
        "mock-laya",
        engine_with_backend(
            "mock-laya",
            Box::new(Batched {
                inner: MockBackend::with_vocab(4096),
                rows: rows.clone(),
                gate: gate.clone(),
                started: slow.then_some(started),
            }),
        ),
    );
    (
        Arc::new(AppState::new(config, registry, Metrics::new())),
        receiver,
        ReleaseSlowJob(gate),
        rows,
    )
}

#[tokio::test(flavor = "current_thread")]
async fn cross_request_batches_preserve_distinct_answers_extensions_and_auth() {
    let (batched, _, _, rows) = batch_state(
        ServerConfig {
            auth_token: Some("secret".into()),
            max_batch_tokens: Some(4096),
            batch_max_requests: Some(4),
            batch_wait_ms: 100,
            ..Default::default()
        },
        false,
    );
    let independent = state(Some("secret"));
    let mut requests = Vec::new();
    let mut expected = Vec::new();
    for index in 0..4 {
        let mut request = choice_request();
        request["state"] = json!(format!("distinct state {index}"));
        let response = send(
            independent.clone(),
            Method::POST,
            "/v1/systemone",
            Some(request.clone()),
            Some("secret"),
            index % 2 == 0,
        )
        .await;
        expected.push(response);
        requests.push(request);
    }
    let mut clients = Vec::new();
    for (index, request) in requests.into_iter().enumerate() {
        let state = batched.clone();
        clients.push(tokio::spawn(async move {
            send(
                state,
                Method::POST,
                "/v1/systemone",
                Some(request),
                Some("secret"),
                index % 2 == 0,
            )
            .await
        }));
    }
    for (client, expected) in clients.into_iter().zip(expected) {
        assert_eq!(client.await.unwrap(), expected);
    }
    assert_eq!(*rows.lock().unwrap(), vec![4]);
    assert_eq!(batched.metrics.cross_request_batch_count.get(), 1);
    assert_eq!(batched.metrics.batch_count.get(), 1);
    assert_eq!(
        batched.metrics.tokens_prefilled.get(),
        independent.metrics.tokens_prefilled.get()
    );
    assert_eq!(batched.metrics.queue_depth.get(), 0);
    assert_eq!(batched.metrics.prepared_waiting.get(), 0);
    assert_eq!(
        send(
            batched.clone(),
            Method::POST,
            "/v1/systemone",
            Some(choice_request()),
            None,
            false
        )
        .await
        .0,
        StatusCode::UNAUTHORIZED
    );
    // A singleton eventually executes, without falsely reporting a mixed batch.
    assert_eq!(
        send(
            batched.clone(),
            Method::POST,
            "/v1/systemone",
            Some(choice_request()),
            Some("secret"),
            false
        )
        .await
        .0,
        StatusCode::OK
    );
    assert_eq!(batched.metrics.cross_request_batch_count.get(), 1);
    assert_eq!(*rows.lock().unwrap(), vec![4, 1]);
}

#[tokio::test(flavor = "current_thread")]
async fn mixed_length_http_collation_keeps_wire_usage_and_counts_padding() {
    let (batched, _, _, rows) = batch_state(
        ServerConfig {
            auth_token: Some("secret".into()),
            max_batch_tokens: Some(4096),
            max_batch_padding_percent: 100,
            batch_max_requests: Some(4),
            batch_wait_ms: 100,
            ..Default::default()
        },
        false,
    );
    let independent = state(Some("secret"));
    let mut clients = Vec::new();
    let mut expected = Vec::new();
    for index in 0..4 {
        let mut request = choice_request();
        request["state"] = json!("distinct state ".repeat(1 + 7 * index));
        expected.push(
            send(
                independent.clone(),
                Method::POST,
                "/v1/systemone",
                Some(request.clone()),
                Some("secret"),
                index % 2 == 0,
            )
            .await,
        );
        let state = batched.clone();
        clients.push(tokio::spawn(async move {
            send(
                state,
                Method::POST,
                "/v1/systemone",
                Some(request),
                Some("secret"),
                index % 2 == 0,
            )
            .await
        }));
    }
    for (client, expected) in clients.into_iter().zip(expected) {
        assert_eq!(client.await.unwrap(), expected);
    }
    assert_eq!(*rows.lock().unwrap(), vec![4]);
    assert_eq!(batched.metrics.padded_batch_count.get(), 1);
    assert_eq!(batched.metrics.cross_request_batch_count.get(), 1);
    assert!(batched.metrics.padded_tokens.get() > 0);
    assert_eq!(
        batched.metrics.tokens_prefilled.get(),
        independent.metrics.tokens_prefilled.get() + batched.metrics.padded_tokens.get()
    );
    let (_, metrics) = send_raw(batched.clone(), Method::GET, "/metrics", Some("secret")).await;
    assert!(metrics.contains("huncho_padded_batch_count 1"));
    assert!(metrics.contains("huncho_padded_tokens"));
    assert_eq!(
        send(
            batched.clone(),
            Method::POST,
            "/v1/systemone",
            Some(choice_request()),
            None,
            false
        )
        .await
        .0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(batched.metrics.padded_batch_count.get(), 1);
    assert_eq!(batched.metrics.queue_depth.get(), 0);
    assert_eq!(batched.metrics.prepared_waiting.get(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn cross_request_cancellation_retains_running_capacity_and_discards_queued_work() {
    let (state, started, release, rows) = batch_state(
        ServerConfig {
            max_queued_per_model: 2,
            max_batch_tokens: Some(4096),
            batch_max_requests: Some(2),
            batch_wait_ms: 100,
            ..Default::default()
        },
        true,
    );
    let submit = |state: Arc<AppState>| {
        tokio::spawn(async move {
            send(
                state,
                Method::POST,
                "/v1/systemone",
                Some(choice_request()),
                None,
                false,
            )
            .await
        })
    };
    let first = submit(state.clone());
    let second = submit(state.clone());
    tokio::time::timeout(std::time::Duration::from_secs(1), started)
        .await
        .unwrap()
        .unwrap();
    first.abort();
    let _ = first.await;
    let third = submit(state.clone());
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while state.metrics.prepared_waiting.get() != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(state.metrics.queue_depth.get(), 3);
    assert_eq!(
        send(
            state.clone(),
            Method::POST,
            "/v1/systemone",
            Some(choice_request()),
            None,
            false
        )
        .await
        .0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        send(state.clone(), Method::GET, "/health", None, None, false)
            .await
            .0,
        StatusCode::OK
    );
    third.abort();
    let _ = third.await;
    drop(release);
    assert_eq!(second.await.unwrap().0, StatusCode::OK);
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while state.metrics.queue_depth.get() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(*rows.lock().unwrap(), vec![2]);
    assert_eq!(state.metrics.cross_request_batch_count.get(), 1);
    assert_eq!(state.metrics.requests_waiting.get(), 0);
    assert_eq!(state.metrics.prepared_waiting.get(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn preparation_overlaps_execution_and_bounds_ready_requests() {
    let (state, started, release) = slow_state_with_config(ServerConfig {
        max_queued_per_model: 2,
        max_prepared_per_model: 1,
        ..Default::default()
    });
    let submit = || {
        tokio::spawn(send(
            state.clone(),
            Method::POST,
            "/v1/systemone",
            Some(choice_request()),
            None,
            false,
        ))
    };
    let active = submit();
    started.await.unwrap();
    let ready = submit();
    wait_for(|| state.metrics.prepared_waiting.get() == 1).await;
    assert_eq!(state.metrics.questions_prepared.get(), 2);
    assert_eq!(state.metrics.tokens_prefilled.get(), 0);
    let pending = submit();
    wait_for(|| state.metrics.queue_depth.get() == 3).await;
    // The ready packet owns the sole preparation slot until execution or
    // cancellation. Another admitted request cannot retain prepared prompts.
    assert_eq!(state.metrics.questions_prepared.get(), 2);
    let (status, _) = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        send(state.clone(), Method::GET, "/health", None, None, false),
    )
    .await
    .unwrap();
    assert_eq!(status, StatusCode::OK);
    let (status, _) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    ready.abort();
    assert!(ready.await.unwrap_err().is_cancelled());
    wait_for(|| {
        state.metrics.questions_prepared.get() == 3 && state.metrics.prepared_waiting.get() == 1
    })
    .await;
    assert_eq!(state.metrics.prepared_waiting.get(), 1);
    assert_eq!(state.metrics.queue_depth.get(), 2);
    drop(release);
    let (status, original) = active.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    let (status, response) = pending.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response, original);
    assert_eq!(
        state.metrics.tokens_prefilled.get(),
        original["usage"]["input_tokens"].as_u64().unwrap() * 2
    );
    assert_eq!(state.metrics.queue_depth.get(), 0);
    assert_eq!(state.metrics.prepared_waiting.get(), 0);
    assert_eq!(state.metrics.requests_preparing.get(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn canceled_blocking_preparation_retains_admission_and_submits_no_forward() {
    struct SlowTokenizer {
        inner: SimpleTokenizer,
        gate: SlowGate,
        started: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    }
    impl Tokenizer for SlowTokenizer {
        fn encode(&self, text: &str, special: bool) -> huncho_core::Result<Vec<u32>> {
            if let Some(started) = self.started.lock().unwrap().take() {
                let _ = started.send(());
                let released = self.gate.0.lock().unwrap();
                let _ = self
                    .gate
                    .1
                    .wait_timeout_while(released, std::time::Duration::from_secs(2), |released| {
                        !*released
                    })
                    .unwrap();
            }
            self.inner.encode(text, special)
        }
        fn decode(&self, ids: &[u32]) -> huncho_core::Result<String> {
            self.inner.decode(ids)
        }
        fn id_for(&self, token: &str) -> Option<u32> {
            self.inner.id_for(token)
        }
        fn name(&self) -> &str {
            self.inner.name()
        }
        fn mask_token_id(&self) -> Option<u32> {
            self.inner.mask_token_id()
        }
        fn cls_token_id(&self) -> Option<u32> {
            self.inner.cls_token_id()
        }
        fn sep_token_id(&self) -> Option<u32> {
            self.inner.sep_token_id()
        }
    }
    let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let release = ReleaseSlowJob(gate.clone());
    let (started, receiver) = tokio::sync::oneshot::channel();
    let mut registry = ModelRegistry::new();
    registry.insert(
        "mock-laya",
        engine_with_parts(
            "mock-laya",
            Box::new(MockBackend::with_vocab(4096)),
            Box::new(SlowTokenizer {
                inner: SimpleTokenizer::new(32768),
                gate,
                started: std::sync::Mutex::new(Some(started)),
            }),
        ),
    );
    let state = Arc::new(AppState::new(
        ServerConfig {
            max_queued_per_model: 0,
            max_prepared_per_model: 1,
            ..Default::default()
        },
        registry,
        Metrics::new(),
    ));
    let active = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    ));
    receiver.await.unwrap();
    active.abort();
    assert!(active.await.unwrap_err().is_cancelled());
    assert_eq!(state.metrics.queue_depth.get(), 1);
    assert_eq!(state.metrics.requests_preparing.get(), 1);
    let (status, _) = send(state.clone(), Method::GET, "/health", None, None, false).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    drop(release);
    wait_for(|| state.metrics.queue_depth.get() == 0).await;
    assert_eq!(state.metrics.questions_prepared.get(), 1);
    assert_eq!(state.metrics.tokens_prefilled.get(), 0);
    assert_eq!(state.metrics.prepared_waiting.get(), 0);
    let (status, _) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(state.metrics.tokens_prefilled.get() > 0);
    assert_eq!(state.metrics.requests_preparing.get(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn slow_inference_keeps_health_registry_and_overload_responsive_after_cancellation() {
    let (state, started, release) = slow_state(0);
    let active = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    ));
    tokio::time::timeout(std::time::Duration::from_secs(2), started)
        .await
        .unwrap()
        .unwrap();
    assert!(
        state.registry.try_write().is_ok(),
        "inference must release the model registry"
    );
    let (status, _) = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        send(state.clone(), Method::GET, "/health", None, None, false),
    )
    .await
    .unwrap();
    assert_eq!(status, StatusCode::OK);
    let (status, body) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "queue_full");
    active.abort();
    assert!(active.await.unwrap_err().is_cancelled());
    // The running blocking job owns its permits even after the HTTP future dies.
    assert_eq!(state.metrics.queue_depth.get(), 1);
    let (status, _) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    drop(release);
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while state.metrics.queue_depth.get() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let (status, _) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(state.metrics.queue_depth.get(), 0);
    assert_eq!(state.metrics.requests_waiting.get(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn queued_request_cancellation_releases_admission_without_running_inference() {
    let (state, started, release) = slow_state(1);
    let active = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    ));
    tokio::time::timeout(std::time::Duration::from_secs(2), started)
        .await
        .unwrap()
        .unwrap();
    let waiting = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    ));
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while state.metrics.requests_waiting.get() != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let (status, _) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    waiting.abort();
    assert!(waiting.await.unwrap_err().is_cancelled());
    assert_eq!(state.metrics.requests_waiting.get(), 0);
    assert_eq!(state.metrics.queue_depth.get(), 1);
    let replacement = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    ));
    drop(release);
    assert_eq!(active.await.unwrap().0, StatusCode::OK);
    assert_eq!(replacement.await.unwrap().0, StatusCode::OK);
    assert_eq!(state.metrics.queue_depth.get(), 0);
    assert_eq!(
        state
            .metrics
            .queue_wait
            .with_label_values(&["mock-laya"])
            .get_sample_count(),
        2
    );
    assert_eq!(
        state
            .metrics
            .evaluation_latency
            .with_label_values(&["mock-laya"])
            .get_sample_count(),
        2
    );
}

async fn wait_for(condition: impl Fn() -> bool) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !condition() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn inflight_sharing_preserves_auth_extensions_admission_and_exact_answers() {
    let (state, started, release) = slow_state_with_config(ServerConfig {
        max_queued_per_model: 2,
        max_prepared_per_model: 1,
        coalesce_bytes: 1024 * 1024,
        auth_token: Some("secret".into()),
        ..Default::default()
    });
    let first = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("secret"),
        false,
    ));
    started.await.unwrap();
    let follower = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("secret"),
        false,
    ));
    wait_for(|| state.metrics.requests_coalesced.get() == 1).await;
    let extended = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("secret"),
        true,
    ));
    wait_for(|| state.metrics.requests_waiting.get() == 1).await;
    assert_eq!(state.metrics.queue_depth.get(), 3);
    assert_eq!(state.metrics.coalesced_waiting.get(), 1);
    let (status, _) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, body) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("secret"),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "queue_full");
    drop(release);
    let (status, original) = first.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    let (status, shared) = follower.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(original, shared);
    assert!(shared.get("extensions").is_none());
    let (status, extended) = extended.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    assert!(extended["extensions"]["raw_logits"].is_object());
    let tokens = original["usage"]["input_tokens"].as_u64().unwrap();
    assert_eq!(state.metrics.tokens_prefilled.get(), tokens * 2);
    assert_eq!(state.metrics.questions_prepared.get(), 2);
    assert_eq!(state.metrics.requests_coalesced.get(), 1);
    assert_eq!(state.metrics.result_cache_hits.get(), 0);
    // Coalescing alone retains no completed answers.
    let (status, repeated) = send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        Some("secret"),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(original, repeated);
    assert_eq!(state.metrics.tokens_prefilled.get(), tokens * 3);
    assert_eq!(state.metrics.queue_depth.get(), 0);
    assert_eq!(state.metrics.coalesced_waiting.get(), 0);
    assert_eq!(state.metrics.questions_prepared.get(), 3);
    assert_eq!(state.metrics.prepared_waiting.get(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn canceling_the_original_caller_does_not_cancel_a_shared_result() {
    let (state, started, release) = slow_state_with_config(ServerConfig {
        max_queued_per_model: 1,
        coalesce_bytes: 1024 * 1024,
        ..Default::default()
    });
    let first = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    ));
    started.await.unwrap();
    let follower = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    ));
    wait_for(|| state.metrics.requests_coalesced.get() == 1).await;
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    assert_eq!(state.metrics.queue_depth.get(), 2);
    drop(release);
    let (status, shared) = follower.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        state.metrics.tokens_prefilled.get(),
        shared["usage"]["input_tokens"].as_u64().unwrap()
    );
    assert_eq!(state.metrics.queue_depth.get(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn abandoning_all_shared_queued_callers_releases_capacity_without_inference() {
    let (state, started, release) = slow_state_with_config(ServerConfig {
        max_queued_per_model: 2,
        coalesce_bytes: 1024 * 1024,
        ..Default::default()
    });
    let active = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    ));
    started.await.unwrap();
    let mut different = choice_request();
    different["state"] = json!("a different exact request");
    let queued = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(different.clone()),
        None,
        false,
    ));
    wait_for(|| state.metrics.requests_waiting.get() == 1).await;
    let follower = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(different),
        None,
        false,
    ));
    wait_for(|| state.metrics.requests_coalesced.get() == 1).await;
    queued.abort();
    assert!(queued.await.unwrap_err().is_cancelled());
    assert_eq!(state.metrics.queue_depth.get(), 3);
    follower.abort();
    assert!(follower.await.unwrap_err().is_cancelled());
    wait_for(|| state.metrics.queue_depth.get() == 1 && state.metrics.requests_waiting.get() == 0)
        .await;
    drop(release);
    let (status, response) = active.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        state.metrics.tokens_prefilled.get(),
        response["usage"]["input_tokens"].as_u64().unwrap()
    );
    assert_eq!(state.metrics.queue_depth.get(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn coalesced_failures_keep_the_error_contract_and_are_retried() {
    struct FailingBackend {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        gate: SlowGate,
        started: Option<tokio::sync::oneshot::Sender<()>>,
    }
    impl Backend for FailingBackend {
        fn id(&self) -> BackendId {
            BackendId::Onnx
        }
        fn capabilities(&self) -> huncho_core::backend::Capabilities {
            MockBackend::with_vocab(512).capabilities()
        }
        fn fork(
            &mut self,
            _: huncho_core::backend::CacheHandle,
        ) -> huncho_core::error::Result<huncho_core::backend::CacheHandle> {
            Err(huncho_core::error::Error::Unsupported("no cache".into()))
        }
        fn forward(
            &mut self,
            _: huncho_core::backend::ForwardInput,
        ) -> huncho_core::error::Result<huncho_core::backend::ForwardOutput> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if let Some(started) = self.started.take() {
                let _ = started.send(());
                let released = self.gate.0.lock().unwrap();
                let _ = self
                    .gate
                    .1
                    .wait_timeout_while(released, std::time::Duration::from_secs(2), |released| {
                        !*released
                    })
                    .unwrap();
            }
            Err(huncho_core::error::Error::Backend(
                "observed failure".into(),
            ))
        }
    }
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let release = ReleaseSlowJob(gate.clone());
    let (started, receiver) = tokio::sync::oneshot::channel();
    let mut registry = ModelRegistry::new();
    registry.insert(
        "mock-laya",
        engine_with_backend(
            "mock-laya",
            Box::new(FailingBackend {
                calls: calls.clone(),
                gate,
                started: Some(started),
            }),
        ),
    );
    let state = Arc::new(AppState::new(
        ServerConfig {
            coalesce_bytes: 1024 * 1024,
            ..Default::default()
        },
        registry,
        Metrics::new(),
    ));
    let first = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    ));
    receiver.await.unwrap();
    let follower = tokio::spawn(send(
        state.clone(),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        false,
    ));
    wait_for(|| state.metrics.requests_coalesced.get() == 1).await;
    drop(release);
    let original = first.await.unwrap();
    assert_eq!(original.0, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(original.1["error"]["code"], "backend_error");
    assert_eq!(original, follower.await.unwrap());
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        original,
        send(
            state.clone(),
            Method::POST,
            "/v1/systemone",
            Some(choice_request()),
            None,
            false
        )
        .await
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(state.metrics.queue_depth.get(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn replica_jobs_overlap_without_releasing_running_capacity_on_http_cancellation() {
    use huncho_core::backend::{CacheHandle, Capabilities, ForwardInput, ForwardOutput};
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct ReplicaBackend {
        inner: MockBackend,
        gate: SlowGate,
        started: tokio::sync::mpsc::UnboundedSender<usize>,
        next: Arc<AtomicUsize>,
        index: usize,
    }
    impl Backend for ReplicaBackend {
        fn id(&self) -> BackendId {
            self.inner.id()
        }
        fn capabilities(&self) -> Capabilities {
            self.inner.capabilities()
        }
        fn replica(&self) -> huncho_core::Result<Box<dyn Backend>> {
            Ok(Box::new(Self {
                inner: MockBackend::with_vocab(4096),
                gate: self.gate.clone(),
                started: self.started.clone(),
                next: self.next.clone(),
                index: self.next.fetch_add(1, Ordering::SeqCst),
            }))
        }
        fn forward(&mut self, input: ForwardInput) -> huncho_core::Result<ForwardOutput> {
            let _ = self.started.send(self.index);
            let released = self.gate.0.lock().unwrap();
            let (released, timeout) = self
                .gate
                .1
                .wait_timeout_while(released, std::time::Duration::from_secs(3), |released| {
                    !*released
                })
                .unwrap();
            assert!(
                !timeout.timed_out(),
                "replica did not execute concurrently or test did not release it"
            );
            drop(released);
            self.inner.forward(input)
        }
        fn fork(&mut self, handle: CacheHandle) -> huncho_core::Result<CacheHandle> {
            self.inner.fork(handle)
        }
    }
    let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let release = ReleaseSlowJob(gate.clone());
    let (started, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut registry = ModelRegistry::new();
    registry.insert(
        "mock-laya",
        engine_with_backend(
            "mock-laya",
            Box::new(ReplicaBackend {
                inner: MockBackend::with_vocab(4096),
                gate,
                started,
                index: 0,
                next: Arc::new(AtomicUsize::new(1)),
            }),
        ),
    );
    registry.set_replicas(2).unwrap();
    assert!(registry.set_replicas(0).is_err());
    assert!(registry.set_replicas(9).is_err());
    assert_eq!(
        registry.get("mock-laya").unwrap().replica_engines().len(),
        2
    );
    let config = ServerConfig {
        max_queued_per_model: 1,
        max_prepared_per_model: 2,
        ..Default::default()
    };
    let state = Arc::new(AppState::new(config, registry, Metrics::new()));
    let (status, listing) = send(state.clone(), Method::GET, "/v1/models", None, None, false).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listing["models"][0]["replicas"], 2);
    let request = |state: Arc<AppState>| {
        tokio::spawn(send(
            state,
            Method::POST,
            "/v1/systemone",
            Some(choice_request()),
            None,
            true,
        ))
    };
    let first = request(state.clone());
    let second = request(state.clone());
    let a = tokio::time::timeout(std::time::Duration::from_secs(1), started_rx.recv())
        .await
        .unwrap()
        .unwrap();
    let b = tokio::time::timeout(std::time::Duration::from_secs(1), started_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_ne!(a, b);
    let queued = request(state.clone());
    for _ in 0..100 {
        if state.metrics.requests_waiting.get() == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    assert_eq!(state.metrics.requests_waiting.get(), 1);
    assert_eq!(
        send(
            state.clone(),
            Method::POST,
            "/v1/systemone",
            Some(choice_request()),
            None,
            false
        )
        .await
        .0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    queued.abort();
    let _ = queued.await;
    first.abort();
    let _ = first.await;
    assert_eq!(state.metrics.queue_depth.get(), 2);
    // Both running native jobs still occupy their contexts after one HTTP abort.
    let waiting = request(state.clone());
    for _ in 0..100 {
        if state.metrics.requests_waiting.get() == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    assert_eq!(state.metrics.requests_waiting.get(), 1);
    assert_eq!(
        send(
            state.clone(),
            Method::POST,
            "/v1/systemone",
            Some(choice_request()),
            None,
            false
        )
        .await
        .0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    drop(release);
    let (status, result) = second.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(waiting.await.unwrap().0, StatusCode::OK);
    let (_, expected) = send(
        crate::state(None),
        Method::POST,
        "/v1/systemone",
        Some(choice_request()),
        None,
        true,
    )
    .await;
    assert_eq!(result["answers"], expected["answers"]);
    assert_eq!(result["extensions"], expected["extensions"]);
    assert_eq!(
        result["usage"]["input_tokens"],
        expected["usage"]["input_tokens"]
    );
    for _ in 0..100 {
        if state.metrics.queue_depth.get() == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    assert_eq!(state.metrics.queue_depth.get(), 0);
    assert_eq!(state.metrics.requests_waiting.get(), 0);
    assert_eq!(
        send(
            state.clone(),
            Method::POST,
            "/v1/systemone",
            Some(choice_request()),
            None,
            true
        )
        .await
        .0,
        StatusCode::OK
    );
}

#[test]
fn unsupported_replica_construction_does_not_replace_any_registered_model() {
    let mut registry = ModelRegistry::new();
    registry.insert("mock-laya", mock_engine("mock-laya"));
    registry.insert(
        "unsupported",
        engine_with_backend("unsupported", Box::new(huncho_backend::NullBackend::new())),
    );
    assert!(registry.set_replicas(2).is_err());
    assert!(registry
        .models()
        .values()
        .all(|model| model.replica_engines().len() == 1));
}
