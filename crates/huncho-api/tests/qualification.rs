//! Synthetic boundary checks; these targets do not qualify released models.
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use huncho_api::{AppState, Metrics, ModelDescription, ModelRegistry, ServerConfig};
use huncho_core::{
    backend::{Backend, Capabilities, ForwardInput, ForwardOutput},
    conformance::{GoldenCase, GoldenSuite},
    contract::SystemOneRequest,
    engine::{Engine, EvalOptions},
    manifest::{BackendId, Family, ModelManifest},
    tensor::Tensor,
    tokenizer::SimpleTokenizer,
};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
    time::Duration,
};
use tower::ServiceExt;

#[derive(Default)]
struct Pause {
    armed: AtomicBool,
    started: AtomicBool,
    released: Mutex<bool>,
    wake: Condvar,
}
impl Pause {
    fn wait_if_armed(&self) {
        if self.armed.load(Ordering::Acquire) {
            let mut released = self.released.lock().unwrap();
            self.started.store(true, Ordering::Release);
            while !*released {
                released = self.wake.wait(released).unwrap();
            }
        }
    }
    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.wake.notify_all();
    }
}
struct ReleaseOnDrop(Arc<Pause>);
impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}
struct NativeFixture(Arc<AtomicUsize>, Option<Arc<Pause>>);
impl Backend for NativeFixture {
    fn fork(
        &mut self,
        _: huncho_core::backend::CacheHandle,
    ) -> huncho_core::Result<huncho_core::backend::CacheHandle> {
        Err(huncho_core::Error::Unsupported(
            "synthetic gate fixture has no prefix state".into(),
        ))
    }

    fn id(&self) -> BackendId {
        BackendId::Onnx
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.id(),
            dtype: "fp32".into(),
            max_context: 8192,
            families: vec![Family::F1],
            extra: BTreeMap::from([
                ("device".into(), "CPU".into()),
                (
                    "native_execution".into(),
                    "synthetic-boundary-fixture-v1".into(),
                ),
            ]),
            ..Default::default()
        }
    }
    fn replica(&self) -> huncho_core::Result<Box<dyn Backend>> {
        Ok(Box::new(Self(self.0.clone(), self.1.clone())))
    }
    fn forward(&mut self, input: ForwardInput) -> huncho_core::Result<ForwardOutput> {
        self.0.fetch_add(1, Ordering::Relaxed);
        if let Some(pause) = &self.1 {
            pause.wait_if_armed();
        }
        Ok(ForwardOutput::Features {
            values: Tensor::new(
                vec![input.positions.len(), 1],
                vec![0.; input.positions.len()],
            )?,
            positions: input.positions,
        })
    }
}
fn fixture() -> (Engine, Arc<AtomicUsize>, GoldenSuite) {
    fixture_with_pause(None)
}
fn fixture_with_pause(pause: Option<Arc<Pause>>) -> (Engine, Arc<AtomicUsize>, GoldenSuite) {
    let mut manifest = ModelManifest::load("../../examples/mock-model/huncho-model.json").unwrap();
    manifest.name = "native-fixture".into();
    manifest.prompt_contract.template = "f1-v1".into();
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = Engine::new(
        manifest,
        Box::new(NativeFixture(calls.clone(), pause)),
        Box::new(SimpleTokenizer::new(32768)),
        Default::default(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap()
    .with_result_cache(4096);
    let request:SystemOneRequest=serde_json::from_value(serde_json::json!({
        "model":"native-fixture","state":"fixture","questions":{"q":{"type":"choice","instructions":"choose","criteria":{"left":"L","right":"R"}}}
    })).unwrap();
    let suite = GoldenSuite {
        schema_version: "1.0".into(),
        family: "F1".into(),
        hash: None,
        cases: vec![GoldenCase {
            id: "synthetic".into(),
            request,
            expected: BTreeMap::from([(
                "q".into(),
                BTreeMap::from([("left".into(), 0.5), ("right".into(), 0.5)]),
            )]),
            targets: BTreeMap::from([("q".into(), "left".into())]),
        }],
    };
    (engine, calls, suite)
}
async fn request(state: Arc<AppState>, suite: &GoldenSuite) -> StatusCode {
    huncho_api::router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&suite.cases[0].request).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap()
        .status()
}
fn state(engine: Engine, config: ServerConfig) -> Arc<AppState> {
    let mut registry = ModelRegistry::new();
    registry.insert("native-fixture", engine);
    Arc::new(AppState::new(config, registry, Metrics::new()))
}
#[tokio::test]
async fn unqualified_native_context_is_refused_before_forward_and_exact_result_reuse() {
    let (engine, calls, suite) = fixture();
    engine
        .eval(
            &suite.cases[0].request,
            &EvalOptions {
                reference_readout: true,
                ..Default::default()
            },
        )
        .unwrap();
    let before = calls.load(Ordering::Relaxed);
    assert_eq!(
        request(state(engine, ServerConfig::default()), &suite).await,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
}
#[tokio::test]
async fn matching_live_qualification_serves_and_changed_options_refuse() {
    for candidate_readout in [false, true] {
        let (engine, calls, suite) = fixture();
        engine
            .qualify_for_serving(
                &suite,
                &EvalOptions {
                    reference_readout: true,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let before = calls.load(Ordering::Relaxed);
        let code = request(
            state(
                engine,
                ServerConfig {
                    candidate_readout,
                    ..Default::default()
                },
            ),
            &suite,
        )
        .await;
        assert_eq!(
            code,
            if candidate_readout {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::OK
            }
        );
        assert_eq!(
            calls.load(Ordering::Relaxed) - before,
            usize::from(!candidate_readout)
        );
    }
}
#[tokio::test]
async fn every_actual_replica_needs_its_own_fresh_complete_suite() {
    let (engine, calls, suite) = fixture();
    let mut registry = ModelRegistry::new();
    registry.insert("native-fixture", engine);
    registry.set_replicas(2).unwrap();
    let state = Arc::new(AppState::new(
        ServerConfig::default(),
        registry,
        Metrics::new(),
    ));
    let model = state.registry.read().await.get("native-fixture").unwrap();
    let options = model.serving_options(&state.config, false);
    model.replica_engines()[0]
        .qualify_for_serving(&suite, &options, None)
        .unwrap();
    let before = calls.load(Ordering::Relaxed);
    assert_eq!(
        request(state.clone(), &suite).await,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
    model.replica_engines()[1]
        .qualify_for_serving(&suite, &options, None)
        .unwrap();
    assert_eq!(request(state, &suite).await, StatusCode::OK);
}
#[tokio::test]
async fn native_lazy_factory_cannot_bypass_live_qualification() {
    let (engine, calls, suite) = fixture();
    let engine = Arc::new(engine);
    let mut registry = ModelRegistry::new();
    registry.enable_lazy(1, Duration::from_secs(60)).unwrap();
    registry
        .register_lazy(
            ModelDescription {
                name: "native-fixture".into(),
                family: Family::F1,
                backend: BackendId::Onnx,
                dtype: "fp32".into(),
                max_context: 8192,
                replicas: 1,
                residency: "cold".into(),
            },
            move || Ok(vec![engine.clone()]),
        )
        .unwrap();
    let state = Arc::new(AppState::new(
        ServerConfig::default(),
        registry,
        Metrics::new(),
    ));
    assert_eq!(
        request(state.clone(), &suite).await,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let model = state.registry.read().await.get("native-fixture").unwrap();
    model.replica_engines()[0]
        .qualify_for_serving(&suite, &model.serving_options(&state.config, false), None)
        .unwrap();
    assert_eq!(request(state, &suite).await, StatusCode::OK);
}
#[tokio::test]
async fn public_server_rejects_unqualified_eager_models_before_opening_listener() {
    let (engine, calls, _) = fixture();
    let state = state(
        engine,
        ServerConfig {
            bind: "127.0.0.1:0".into(),
            ..Default::default()
        },
    );
    let result = tokio::time::timeout(Duration::from_secs(1), huncho_api::serve(state))
        .await
        .unwrap();
    assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}
#[tokio::test]
async fn revocation_during_forward_prevents_publishing_the_response() {
    let pause = Arc::new(Pause::default());
    let _release = ReleaseOnDrop(pause.clone());
    let (engine, _, suite) = fixture_with_pause(Some(pause.clone()));
    let state = state(engine, ServerConfig::default());
    let model = state.registry.read().await.get("native-fixture").unwrap();
    let options = model.serving_options(&state.config, false);
    model.replica_engines()[0]
        .qualify_for_serving(&suite, &options, None)
        .unwrap();
    pause.armed.store(true, Ordering::Release);
    let owned_suite = suite.clone();
    let response = tokio::spawn(async move { request(state, &owned_suite).await });
    tokio::time::timeout(Duration::from_secs(3), async {
        while !pause.started.load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let mut incomplete = suite;
    incomplete.cases[0].targets.clear();
    assert!(model.replica_engines()[0]
        .qualify_for_serving(&incomplete, &options, None)
        .is_err());
    pause.release();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(3), response)
            .await
            .unwrap()
            .unwrap(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}
