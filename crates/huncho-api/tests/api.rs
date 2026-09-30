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

    let backend: Box<dyn Backend> = Box::new(
        MockBackend::with_vocab(4096)
            .with_backend(BackendId::Onnx)
            .with_dtype("fp32"),
    );
    let tokenizer: Box<dyn Tokenizer> = Box::new(SimpleTokenizer::new(32768));
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
async fn health_reports_ok_and_model_count() {
    let (status, body) = send(state(None), Method::GET, "/health", None, None, false).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["models"], 1);
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
    assert!(body["answers"]["department"]["confidence"].as_f64().unwrap() >= 0.0);
    assert!(body["answers"]["department"]["confidence"].as_f64().unwrap() <= 1.0);
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
    let (status, body) = send(state(None), Method::POST, "/v1/systemone", Some(req), None, false).await;
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
    let (status, body) = send(state(None), Method::POST, "/v1/systemone", Some(req), None, false).await;
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
    let (_, body) = send(state(None), Method::POST, "/v1/systemone", Some(choice_request()), None, false).await;
    assert!(body.get("extensions").is_none());

    // Opt in via header.
    let (_, body) = send(state(None), Method::POST, "/v1/systemone", Some(choice_request()), None, true).await;
    let ext = body["extensions"].as_object().unwrap();
    assert_eq!(ext["backend"], "onnx");
    assert_eq!(ext["dtype"], "fp32");
    assert_eq!(ext["calibration_status"], "fit");
    assert!(ext["prompt_contract_hash"].as_str().unwrap().contains("test-hash"));
    let raw = ext["raw_logits"].as_object().unwrap();
    assert!(raw["department"].as_array().unwrap().len() >= 2);
}
