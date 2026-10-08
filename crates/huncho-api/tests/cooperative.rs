//! Scheduler fixtures prove fairness, bounded cancellation and work accounting.
//! They are not model calibration evidence; native arithmetic tests live in backend.
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use huncho_api::{AppState, Metrics, ModelRegistry, ServerConfig};
use huncho_core::{
    backend::{
        Backend, BatchLimits, CacheHandle, CachedPrefill, Capabilities, ForkBatchWork,
        ForwardInput, ForwardOutput, PrefillWork,
    },
    engine::Engine,
    manifest::{BackendId, Family, ModelManifest},
    tensor::Tensor,
    tokenizer::SimpleTokenizer,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
    time::Duration,
};
use tower::ServiceExt;

struct Control {
    released: Mutex<bool>,
    gate: Condvar,
    started: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    trace: Mutex<Vec<(u64, usize)>>,
    live: AtomicUsize,
    next: AtomicU64,
    fail: AtomicBool,
    forwards: AtomicUsize,
    fail_batch: AtomicBool,
}
impl Control {
    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.gate.notify_all();
    }
}
struct Release(Arc<Control>);
impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}
struct Stepped {
    control: Arc<Control>,
    caches: BTreeMap<u64, (usize, usize)>,
    batching: bool,
}
impl Backend for Stepped {
    fn id(&self) -> BackendId {
        BackendId::Candle
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.id(),
            dtype: "fp32".into(),
            max_context: 512,
            supports_fork: true,
            families: vec![Family::F2],
            extra: BTreeMap::from([("device".into(), "CPU".into())]),
            ..Default::default()
        }
    }
    fn supports_resumable_prefill(&self) -> bool {
        true
    }
    fn begin_resumable_prefill(
        &mut self,
        tokens: &[u32],
        _: usize,
    ) -> huncho_core::Result<CachedPrefill> {
        let id = self.control.next.fetch_add(1, Ordering::SeqCst);
        self.caches.insert(id, (0, tokens.len()));
        self.control.live.fetch_add(1, Ordering::SeqCst);
        Ok(CachedPrefill {
            handle: CacheHandle { id },
            hit: false,
        })
    }
    fn advance_resumable_prefill(
        &mut self,
        handle: CacheHandle,
        work: &mut PrefillWork,
    ) -> huncho_core::Result<bool> {
        let (offset, total) = self.caches.get_mut(&handle.id).unwrap();
        work.forward_calls += 1;
        work.processed_tokens += 1;
        work.chunked_prefills += u64::from(*offset == 1);
        self.control
            .trace
            .lock()
            .unwrap()
            .push((handle.id, *offset));
        let sender = self.control.started.lock().unwrap().take();
        if let Some(sender) = sender {
            let _ = sender.send(());
            let guard = self.control.released.lock().unwrap();
            let (guard, timeout) = self
                .control
                .gate
                .wait_timeout_while(guard, Duration::from_secs(3), |released| !*released)
                .unwrap();
            assert!(
                *guard && !timeout.timed_out(),
                "test must release the blocked native call"
            );
        }
        if *offset == 1 && self.control.fail.load(Ordering::SeqCst) {
            return Err(huncho_core::Error::Backend(
                "injected second chunk failure".into(),
            ));
        }
        *offset += 1;
        Ok(*offset == *total)
    }
    fn fork(&mut self, handle: CacheHandle) -> huncho_core::Result<CacheHandle> {
        let cache = self.caches[&handle.id];
        assert_eq!(cache.0, cache.1);
        let id = self.control.next.fetch_add(1, Ordering::SeqCst);
        self.caches.insert(id, cache);
        self.control.live.fetch_add(1, Ordering::SeqCst);
        Ok(CacheHandle { id })
    }
    fn release_cache(&mut self, handle: CacheHandle) -> huncho_core::Result<()> {
        assert!(self.caches.remove(&handle.id).is_some());
        self.control.live.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }
    fn forward(&mut self, input: ForwardInput) -> huncho_core::Result<ForwardOutput> {
        self.control.forwards.fetch_add(1, Ordering::SeqCst);
        if let Some(handle) = input.fork_from {
            assert!(self.caches.contains_key(&handle.id));
        }
        output(input)
    }
    fn supports_batch(&self) -> bool {
        self.batching
    }
    fn supports_fork_batch(&self) -> bool {
        self.batching
    }
    fn supports_padded_fork_batch(&self) -> bool {
        self.batching
    }
    fn fork_batch_limits(&self) -> BatchLimits {
        BatchLimits {
            max_rows: if self.batching {
                (64 - self.caches.len()).min(63)
            } else {
                0
            },
            ..Default::default()
        }
    }
    fn forward_fork_batch(
        &mut self,
        parent: CacheHandle,
        inputs: Vec<ForwardInput>,
        work: &mut ForkBatchWork,
    ) -> huncho_core::Result<Vec<ForwardOutput>> {
        assert!(self.batching && inputs.len() <= self.fork_batch_limits().max_rows);
        assert_eq!(self.caches[&parent.id].0, self.caches[&parent.id].1);
        self.control.forwards.fetch_add(1, Ordering::SeqCst);
        work.forward_calls += 1;
        work.batch_calls += u64::from(inputs.len() > 1);
        let physical = inputs.len() * inputs.iter().map(|i| i.tokens.len()).max().unwrap();
        let padding = physical - inputs.iter().map(|i| i.tokens.len()).sum::<usize>();
        work.processed_tokens += physical as u64;
        work.padded_tokens += padding as u64;
        work.padded_batch_calls += u64::from(padding > 0);
        if self.control.fail_batch.load(Ordering::SeqCst) {
            return Err(huncho_core::Error::Backend("injected group failure".into()));
        }
        inputs
            .into_iter()
            .map(|input| {
                let child = self.fork(parent)?;
                work.cache_forks += 1;
                let result = output(input);
                self.release_cache(child)?;
                result
            })
            .collect()
    }
    fn forward_padded_fork_batch(
        &mut self,
        parent: CacheHandle,
        inputs: Vec<ForwardInput>,
        work: &mut ForkBatchWork,
    ) -> huncho_core::Result<Vec<ForwardOutput>> {
        self.forward_fork_batch(parent, inputs, work)
    }
}
fn output(input: ForwardInput) -> huncho_core::Result<ForwardOutput> {
    Ok(ForwardOutput::Logits {
        values: Tensor::new(
            vec![input.positions.len(), 4],
            input
                .positions
                .iter()
                .flat_map(|_| [0., 0.5, 1., 1.5])
                .collect(),
        )?,
        positions: input.positions,
    })
}

fn setup() -> (
    Arc<AppState>,
    Arc<Control>,
    tokio::sync::oneshot::Receiver<()>,
    Release,
) {
    setup_with_coalescing(0)
}
fn setup_with_coalescing(
    coalesce_bytes: usize,
) -> (
    Arc<AppState>,
    Arc<Control>,
    tokio::sync::oneshot::Receiver<()>,
    Release,
) {
    setup_profile(coalesce_bytes, false)
}
fn setup_profile(
    coalesce_bytes: usize,
    batching: bool,
) -> (
    Arc<AppState>,
    Arc<Control>,
    tokio::sync::oneshot::Receiver<()>,
    Release,
) {
    let (started, receiver) = tokio::sync::oneshot::channel();
    let control = Arc::new(Control {
        released: Mutex::new(false),
        gate: Condvar::new(),
        started: Mutex::new(Some(started)),
        trace: Mutex::new(Vec::new()),
        live: AtomicUsize::new(0),
        next: AtomicU64::new(1),
        fail: AtomicBool::new(false),
        forwards: AtomicUsize::new(0),
        fail_batch: AtomicBool::new(false),
    });
    let mut manifest: Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/mock-model/huncho-model.json"
        ))
        .unwrap(),
    )
    .unwrap();
    manifest["name"] = json!("stepped");
    manifest["family"] = json!("F2");
    manifest["head"]["kind"] = json!("pointer");
    manifest["prompt_contract"]["template"] = json!("kev-v1");
    manifest["backbone"]["max_context"] = json!(512);
    let manifest: ModelManifest = serde_json::from_value(manifest).unwrap();
    let engine = Engine::new(
        manifest,
        Box::new(Stepped {
            control: control.clone(),
            caches: BTreeMap::new(),
            batching,
        }),
        Box::new(SimpleTokenizer::new(512)),
        Default::default(),
        BackendId::Candle,
        "fp32",
    )
    .unwrap();
    let mut registry = ModelRegistry::new();
    registry.insert("stepped", engine);
    let state = Arc::new(AppState::new(
        ServerConfig {
            cooperative_prefill: true,
            prefix_cache: true,
            max_queued_per_model: 1,
            max_prepared_per_model: 2,
            coalesce_bytes,
            max_batch_tokens: batching.then_some(512),
            max_batch_padding_percent: if batching { 100 } else { 0 },
            ..Default::default()
        },
        registry,
        Metrics::new(),
    ));
    (state, control.clone(), receiver, Release(control))
}
fn body(name: &str) -> Value {
    json!({"model":"stepped","state":format!("{name} one two three four five six seven"),"questions":{"a":{"type":"noul","instructions":"Proceed?"},"b":{"type":"choice","instructions":"Pick","criteria":{"first":null,"second":null}}}})
}
async fn send(state: Arc<AppState>, body: Value) -> (StatusCode, Value) {
    let request = Request::post("/v1/systemone")
        .header("content-type", "application/json")
        .header("X-Huncho-Extensions", "true")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = huncho_api::router()
        .with_state(state)
        .oneshot(request)
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}
async fn until(condition: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn short_requests_run_before_long_prefills_finish_and_keep_admission_bounded() {
    let (state, control, started, _release) = setup();
    let mut long_body = body("long");
    long_body["state"] = json!((0..40)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" "));
    let long = tokio::spawn(send(state.clone(), long_body));
    started.await.unwrap();
    let short = tokio::spawn(send(state.clone(), body("short")));
    until(|| {
        state.metrics.questions_prepared.get() == 4 && state.metrics.requests_waiting.get() == 1
    })
    .await;
    assert_eq!(
        send(state.clone(), body("overflow")).await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    let health = huncho_api::router()
        .with_state(state.clone())
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    control.release();
    let (status_a, a) = long.await.unwrap();
    let (status_b, b) = short.await.unwrap();
    assert_eq!((status_a, status_b), (StatusCode::OK, StatusCode::OK));
    assert_eq!(a["answers"], b["answers"]);
    let trace = control.trace.lock().unwrap();
    assert!(trace.len() > 4);
    assert_eq!(trace[0].0, trace[2].0);
    assert_ne!(
        trace[0].0, trace[1].0,
        "queued request must run before the first prefix finishes"
    );
    assert!(state.metrics.prefill_interleaves.get() > 0);
    assert_eq!(control.live.load(Ordering::SeqCst), 0);
    assert_eq!(control.forwards.load(Ordering::SeqCst), 4);
    assert_eq!(state.metrics.queue_depth.get(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn cancellation_retains_current_kernel_permits_then_releases_partial_prefix() {
    let (state, control, started, _release) = setup();
    let job = tokio::spawn(send(state.clone(), body("cancel")));
    started.await.unwrap();
    job.abort();
    let _ = job.await;
    assert_eq!(state.metrics.queue_depth.get(), 1);
    assert_eq!(control.live.load(Ordering::SeqCst), 1);
    control.release();
    until(|| state.metrics.queue_depth.get() == 0 && control.live.load(Ordering::SeqCst) == 0)
        .await;
    assert_eq!(control.trace.lock().unwrap().len(), 1);
    assert_eq!(control.forwards.load(Ordering::SeqCst), 0);
    assert_eq!(state.metrics.tokens_prefilled.get(), 1);
    assert_eq!(send(state.clone(), body("retry")).await.0, StatusCode::OK);
    assert_eq!(control.live.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn coalesced_follower_keeps_resumable_job_alive_after_leader_cancels() {
    let (state, control, started, _release) = setup_with_coalescing(1 << 20);
    let leader = tokio::spawn(send(state.clone(), body("identical")));
    started.await.unwrap();
    let follower = tokio::spawn(send(state.clone(), body("identical")));
    until(|| state.metrics.requests_coalesced.get() == 1).await;
    leader.abort();
    let _ = leader.await;
    control.release();
    assert_eq!(follower.await.unwrap().0, StatusCode::OK);
    until(|| state.metrics.queue_depth.get() == 0).await;
    assert_eq!(state.metrics.questions_prepared.get(), 2);
    assert_eq!(control.forwards.load(Ordering::SeqCst), 2);
    assert_eq!(control.live.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn all_coalesced_callers_cancel_and_release_partial_work_after_current_chunk() {
    let (state, control, started, _release) = setup_with_coalescing(1 << 20);
    let leader = tokio::spawn(send(state.clone(), body("identical")));
    started.await.unwrap();
    let follower = tokio::spawn(send(state.clone(), body("identical")));
    until(|| state.metrics.requests_coalesced.get() == 1).await;
    leader.abort();
    follower.abort();
    let _ = leader.await;
    let _ = follower.await;
    control.release();
    until(|| state.metrics.queue_depth.get() == 0 && control.live.load(Ordering::SeqCst) == 0)
        .await;
    assert_eq!(control.trace.lock().unwrap().len(), 1);
    assert_eq!(control.forwards.load(Ordering::SeqCst), 0);
    assert_eq!(state.metrics.tokens_prefilled.get(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn queued_cancellation_submits_no_prefix_work_and_failed_chunk_releases_all_handles() {
    let (state, control, started, _release) = setup();
    control.fail.store(true, Ordering::SeqCst);
    let job = tokio::spawn(send(state.clone(), body("failure")));
    started.await.unwrap();
    let queued = tokio::spawn(send(state.clone(), body("cancel-queued")));
    until(|| state.metrics.requests_waiting.get() == 1).await;
    queued.abort();
    let _ = queued.await;
    assert_eq!(control.trace.lock().unwrap().len(), 1);
    control.release();
    assert_eq!(job.await.unwrap().0, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(control.live.load(Ordering::SeqCst), 0);
    assert_eq!(control.forwards.load(Ordering::SeqCst), 0);
    assert_eq!(state.metrics.tokens_prefilled.get(), 2);
    assert_eq!(state.metrics.prefill_calls.get(), 2);
    assert_eq!(state.metrics.queue_depth.get(), 0);
    control.fail.store(false, Ordering::SeqCst);
    assert_eq!(send(state.clone(), body("retry")).await.0, StatusCode::OK);
}

#[tokio::test(flavor = "current_thread")]
async fn cooperative_question_groups_keep_fairness_physical_metrics_and_parent_ownership() {
    let (state, control, started, _release) = setup_profile(0, true);
    let mut long = body("long");
    long["state"] = json!((0..40)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" "));
    let long = tokio::spawn(send(state.clone(), long));
    started.await.unwrap();
    let short = tokio::spawn(send(state.clone(), body("short")));
    until(|| {
        state.metrics.questions_prepared.get() == 4 && state.metrics.requests_waiting.get() == 1
    })
    .await;
    assert_eq!(
        send(state.clone(), body("overflow")).await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    control.release();
    let (a, a_body) = long.await.unwrap();
    let (b, b_body) = short.await.unwrap();
    assert_eq!((a, b), (StatusCode::OK, StatusCode::OK));
    assert_eq!(a_body["answers"], b_body["answers"]);
    let trace = control.trace.lock().unwrap();
    assert_ne!(
        trace[0].0, trace[1].0,
        "another prefix must run before a long prefill finishes"
    );
    assert!(state.metrics.prefill_interleaves.get() > 0);
    assert_eq!(state.metrics.fork_batch_count.get(), 2);
    assert_eq!(control.forwards.load(Ordering::SeqCst), 2);
    assert_eq!(control.live.load(Ordering::SeqCst), 0);
    assert_eq!(state.metrics.queue_depth.get(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn cooperative_group_coalescing_cancellation_and_failures_release_every_parent() {
    let (state, control, started, _release) = setup_profile(1 << 20, true);
    let leader = tokio::spawn(send(state.clone(), body("identical")));
    started.await.unwrap();
    let follower = tokio::spawn(send(state.clone(), body("identical")));
    until(|| state.metrics.requests_coalesced.get() == 1).await;
    leader.abort();
    let _ = leader.await;
    control.release();
    assert_eq!(follower.await.unwrap().0, StatusCode::OK);
    until(|| state.metrics.queue_depth.get() == 0).await;
    assert_eq!(control.forwards.load(Ordering::SeqCst), 1);
    assert_eq!(control.live.load(Ordering::SeqCst), 0);
    control.fail_batch.store(true, Ordering::SeqCst);
    assert_eq!(
        send(state.clone(), body("failed group")).await.0,
        StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(control.live.load(Ordering::SeqCst), 0);
    control.fail_batch.store(false, Ordering::SeqCst);
    assert_eq!(
        send(state.clone(), body("retry group")).await.0,
        StatusCode::OK
    );
    assert_eq!(control.live.load(Ordering::SeqCst), 0);
}
