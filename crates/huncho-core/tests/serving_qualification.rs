//! Gate/ownership tests use synthetic zero features, never model calibration evidence.
use huncho_core::{
    backend::{Backend, Capabilities, ForwardInput, ForwardOutput},
    conformance::{run_suite, ConformanceThresholds, GoldenCase, GoldenSuite},
    contract::SystemOneRequest,
    engine::{Engine, EvalOptions, EvalStats},
    manifest::{BackendId, CalibrationStatus, Family, ModelManifest},
    tensor::Tensor,
    tokenizer::SimpleTokenizer,
};
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

struct Fixture {
    calls: Arc<AtomicUsize>,
    drift: Arc<AtomicBool>,
    dtype: String,
    metadata: BTreeMap<String, String>,
}
impl Backend for Fixture {
    fn fork(
        &mut self,
        _: huncho_core::backend::CacheHandle,
    ) -> huncho_core::Result<huncho_core::backend::CacheHandle> {
        Err(huncho_core::Error::Unsupported(
            "synthetic gate fixture has no prefix state".into(),
        ))
    }

    fn id(&self) -> BackendId {
        BackendId::Candle
    }
    fn capabilities(&self) -> Capabilities {
        let mut extra = self.metadata.clone();
        extra.insert("device".into(), "CPU".into());
        Capabilities {
            id: self.id(),
            dtype: self.dtype.clone(),
            max_context: 8192,
            families: vec![Family::F1],
            extra,
            ..Default::default()
        }
    }
    fn replica(&self) -> huncho_core::Result<Box<dyn Backend>> {
        Ok(Box::new(Self {
            calls: self.calls.clone(),
            drift: self.drift.clone(),
            dtype: self.dtype.clone(),
            metadata: self.metadata.clone(),
        }))
    }
    fn supports_batch(&self) -> bool {
        true
    }
    fn forward(&mut self, input: ForwardInput) -> huncho_core::Result<ForwardOutput> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let values = input
            .positions
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if self.drift.load(Ordering::Relaxed) {
                    i as f32
                } else {
                    0.
                }
            })
            .collect();
        Ok(ForwardOutput::Features {
            values: Tensor::new(vec![input.positions.len(), 1], values)?,
            positions: input.positions,
        })
    }
    fn forward_batch(
        &mut self,
        inputs: Vec<ForwardInput>,
    ) -> huncho_core::Result<Vec<ForwardOutput>> {
        inputs
            .into_iter()
            .map(|input| self.forward(input))
            .collect()
    }
}
fn engine(
    dtype: &str,
    metadata: &[(&str, &str)],
    status: CalibrationStatus,
    exact: bool,
) -> (Engine, Arc<AtomicUsize>, Arc<AtomicBool>) {
    let mut manifest = ModelManifest::load("../../examples/mock-model/huncho-model.json").unwrap();
    manifest.prompt_contract.template = "f1-v1".into();
    manifest.calibration.entries.clear();
    manifest.calibration.default.status = status;
    if exact {
        manifest.calibration.entries.insert(
            format!("candle:{dtype}"),
            manifest.calibration.default.clone(),
        );
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let drift = Arc::new(AtomicBool::new(false));
    let backend = Fixture {
        calls: calls.clone(),
        drift: drift.clone(),
        dtype: dtype.into(),
        metadata: std::iter::once((
            "native_execution".into(),
            "synthetic-gate-fixture-v1".into(),
        ))
        .chain(metadata.iter().map(|(k, v)| (k.to_string(), v.to_string())))
        .collect(),
    };
    (
        Engine::new(
            manifest,
            Box::new(backend),
            Box::new(SimpleTokenizer::new(32768)),
            Default::default(),
            BackendId::Candle,
            dtype,
        )
        .unwrap(),
        calls,
        drift,
    )
}
fn suite() -> GoldenSuite {
    let request:SystemOneRequest=serde_json::from_value(serde_json::json!({
        "model":"mock-laya","state":"fixture","questions":{"q":{"type":"choice","instructions":"choose","criteria":{"left":"L","right":"R"}}}
    })).unwrap();
    GoldenSuite {
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
    }
}
#[test]
fn diagnostic_reports_and_fit_metadata_cannot_authorize_production() {
    let (engine, calls, _) = engine("fp32", &[], CalibrationStatus::Fit, false);
    let suite = suite();
    let options = EvalOptions::default();
    assert!(engine
        .eval_for_serving(&suite.cases[0].request, &options)
        .is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let report = run_suite(&engine, &suite, &ConformanceThresholds::default()).unwrap();
    assert!(report.passed);
    let _decoded: huncho_core::conformance::ConformanceReport =
        serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
    assert!(engine
        .require_serving_qualification(&options, None)
        .is_err());
    assert!(
        engine
            .qualify_for_serving(&suite, &options, None)
            .unwrap()
            .passed
    );
    engine
        .eval_for_serving(&suite.cases[0].request, &options)
        .unwrap();
    engine
        .require_serving_qualification(
            &EvalOptions {
                extensions: true,
                ..options.clone()
            },
            None,
        )
        .unwrap();
    let mut stats = EvalStats {
        forward_calls: 99,
        ..Default::default()
    };
    assert!(engine
        .eval_for_serving_with_stats(
            &suite.cases[0].request,
            &EvalOptions {
                max_context: Some(400),
                ..options
            },
            &mut stats
        )
        .is_err());
    assert_eq!(stats.forward_calls, 0);
}
#[test]
fn fresh_replicas_and_cache_reconfiguration_do_not_inherit_proofs() {
    let (engine, _, _) = engine("fp32", &[], CalibrationStatus::Fit, false);
    let suite = suite();
    let opts = EvalOptions::default();
    assert!(
        engine
            .qualify_for_serving(&suite, &opts, None)
            .unwrap()
            .passed
    );
    let replica = engine.replica().unwrap();
    assert!(replica.require_serving_qualification(&opts, None).is_err());
    assert!(
        replica
            .qualify_for_serving(&suite, &opts, None)
            .unwrap()
            .passed
    );
    drop(engine);
    replica
        .eval_for_serving(&suite.cases[0].request, &opts)
        .unwrap();
    let replica = replica.with_result_cache(4096);
    assert!(replica.require_serving_qualification(&opts, None).is_err());
    assert!(
        replica
            .qualify_for_serving(&suite, &opts, None)
            .unwrap()
            .passed
    );
    replica
        .eval_for_serving(&suite.cases[0].request, &opts)
        .unwrap();
    let mut work = EvalStats::default();
    replica
        .eval_for_serving_with_stats(&suite.cases[0].request, &opts, &mut work)
        .unwrap();
    assert_eq!(work.result_cache_hits, 1);
    assert!(replica
        .with_prompt_cache(4096)
        .require_serving_qualification(&opts, None)
        .is_err());
}
#[test]
fn fresh_qualification_discards_cached_diagnostic_results() {
    let (engine, calls, drift) = engine("fp32", &[], CalibrationStatus::Fit, false);
    let engine = engine.with_result_cache(4096);
    let suite = suite();
    let opts = EvalOptions::default();
    drift.store(true, Ordering::Relaxed);
    let stale = engine.eval(&suite.cases[0].request, &opts).unwrap();
    drift.store(false, Ordering::Relaxed);
    assert!(
        engine
            .qualify_for_serving(&suite, &opts, None)
            .unwrap()
            .passed
    );
    let before = calls.load(Ordering::Relaxed);
    let mut work = EvalStats::default();
    let fresh = engine
        .eval_for_serving_with_stats(&suite.cases[0].request, &opts, &mut work)
        .unwrap();
    assert_eq!(work.result_cache_hits, 0);
    assert_eq!(calls.load(Ordering::Relaxed), before + 1);
    assert_ne!(
        serde_json::to_value(stale).unwrap(),
        serde_json::to_value(&fresh).unwrap()
    );
    engine
        .eval_for_serving_with_stats(&suite.cases[0].request, &opts, &mut work)
        .unwrap();
    assert_eq!(work.result_cache_hits, 1);
}
#[test]
fn delayed_diagnostic_preparation_cannot_repopulate_the_active_cache_generation() {
    let (engine, calls, drift) = engine("fp32", &[], CalibrationStatus::Fit, false);
    let engine = engine.with_result_cache(4096);
    let suite = suite();
    let opts = EvalOptions::default();
    let old = engine
        .prepare_eval_with_stats(
            suite.cases[0].request.clone(),
            opts.clone(),
            &mut EvalStats::default(),
        )
        .unwrap();
    assert!(
        engine
            .qualify_for_serving(&suite, &opts, None)
            .unwrap()
            .passed
    );
    drift.store(true, Ordering::Relaxed);
    engine
        .eval_prepared_with_stats(old, &mut EvalStats::default())
        .unwrap();
    drift.store(false, Ordering::Relaxed);
    let before = calls.load(Ordering::Relaxed);
    let mut work = EvalStats::default();
    engine
        .eval_for_serving_with_stats(&suite.cases[0].request, &opts, &mut work)
        .unwrap();
    assert_eq!(work.result_cache_hits, 0);
    assert_eq!(calls.load(Ordering::Relaxed), before + 1);
}
#[test]
fn qualification_tokens_bind_context_and_attempt_even_when_options_are_unchanged() {
    let (engine, _, _) = engine("fp32", &[], CalibrationStatus::Fit, false);
    let suite = suite();
    let opts = EvalOptions::default();
    engine.qualify_for_serving(&suite, &opts, None).unwrap();
    let old = engine.serving_qualification_token(&opts, None).unwrap();
    engine
        .validate_serving_qualification_token(&old, &opts, None)
        .unwrap();
    engine.qualify_for_serving(&suite, &opts, None).unwrap();
    assert!(engine
        .validate_serving_qualification_token(&old, &opts, None)
        .is_err());
    let new = engine.serving_qualification_token(&opts, None).unwrap();
    let replica = engine.replica().unwrap();
    replica.qualify_for_serving(&suite, &opts, None).unwrap();
    assert!(replica
        .validate_serving_qualification_token(&new, &opts, None)
        .is_err());
}
#[test]
fn rejection_errors_and_incomplete_labels_revoke_previous_authorization() {
    let (engine, calls, drift) = engine("fp32", &[], CalibrationStatus::Fit, false);
    let mut suite = suite();
    let opts = EvalOptions::default();
    engine.qualify_for_serving(&suite, &opts, None).unwrap();
    drift.store(true, Ordering::Relaxed);
    assert!(
        !engine
            .qualify_for_serving(&suite, &opts, None)
            .unwrap()
            .passed
    );
    assert!(engine.require_serving_qualification(&opts, None).is_err());
    // A loosened diagnostic threshold still cannot install a proof.
    assert!(
        run_suite(
            &engine,
            &suite,
            &ConformanceThresholds {
                max_prob_delta: 1.,
                min_argmax_agreement: 0.,
                max_ece: 1.
            }
        )
        .unwrap()
        .passed
    );
    assert!(engine.require_serving_qualification(&opts, None).is_err());
    drift.store(false, Ordering::Relaxed);
    engine.qualify_for_serving(&suite, &opts, None).unwrap();
    suite.cases[0].targets.clear();
    let before = calls.load(Ordering::Relaxed);
    assert!(engine.qualify_for_serving(&suite, &opts, None).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), before);
    assert!(engine.require_serving_qualification(&opts, None).is_err());
    suite.cases.clear();
    assert!(engine.qualify_for_serving(&suite, &opts, None).is_err());
}
#[test]
fn exact_variant_calibration_and_actual_prefix_profiles_are_required() {
    let suite = suite();
    let opts = EvalOptions::default();
    for (dtype, status, metadata, exact) in [
        ("fp32", CalibrationStatus::Pending, vec![], false),
        (
            "q8_0-fp32",
            CalibrationStatus::Fit,
            vec![("weight_quantization", "test")],
            true,
        ),
        (
            "q8_0-fp32",
            CalibrationStatus::Refit,
            vec![("weight_quantization", "test")],
            false,
        ),
        (
            "fp32",
            CalibrationStatus::Fit,
            vec![("onnx_integrated_head", "test")],
            false,
        ),
        (
            "fp32",
            CalibrationStatus::Fit,
            vec![("kv_storage", "test")],
            false,
        ),
    ] {
        let (engine, calls, _) = engine(dtype, &metadata, status, exact);
        assert!(engine.qualify_for_serving(&suite, &opts, None).is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
    let (engine, _, _) = engine(
        "q8_0-fp32",
        &[("weight_quantization", "test")],
        CalibrationStatus::Refit,
        true,
    );
    assert!(
        engine
            .qualify_for_serving(&suite, &opts, None)
            .unwrap()
            .passed
    );
}
#[test]
fn cross_request_profile_is_bound_to_actual_group_size_and_not_scalar_receipts() {
    let (engine, _, _) = engine("fp32", &[], CalibrationStatus::Fit, false);
    let mut suite = suite();
    suite.cases.push(suite.cases[0].clone());
    let opts = EvalOptions {
        max_batch_tokens: Some(4096),
        prepare_all: true,
        ..Default::default()
    };
    let report = engine.qualify_for_serving(&suite, &opts, Some(2)).unwrap();
    assert!(report.passed);
    assert!(report.work.cross_request_batches > 0);
    engine
        .require_serving_qualification(&opts, Some(2))
        .unwrap();
    assert!(engine.require_serving_qualification(&opts, None).is_err());
    assert!(engine
        .require_serving_qualification(&opts, Some(3))
        .is_err());
    assert!(engine
        .require_serving_qualification(&EvalOptions::default(), Some(2))
        .is_err());
}
#[test]
fn changed_runtime_environment_refuses_before_serving_work() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "environment_check_child",
            "--ignored",
            "--test-threads=1",
        ])
        .env("HUNCHO_DIRECT_PAGED_ATTENTION", "0")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
#[test]
#[ignore = "isolated child process changes its own environment"]
fn environment_check_child() {
    let (engine, calls, _) = engine("fp32", &[], CalibrationStatus::Fit, false);
    let suite = suite();
    let opts = EvalOptions::default();
    assert!(
        engine
            .qualify_for_serving(&suite, &opts, None)
            .unwrap()
            .passed
    );
    let before = calls.load(Ordering::Relaxed);
    std::env::set_var("HUNCHO_DIRECT_PAGED_ATTENTION", "1");
    assert!(engine
        .eval_for_serving(&suite.cases[0].request, &opts)
        .is_err());
    assert_eq!(calls.load(Ordering::Relaxed), before);
}

#[test]
fn native_constructor_identity_cannot_borrow_another_backend_or_dtype_calibration_key() {
    let (proper, calls, drift) = engine("fp32", &[], CalibrationStatus::Fit, false);
    for (actual_dtype, requested_backend) in
        [("fp16", BackendId::Candle), ("fp32", BackendId::Onnx)]
    {
        let backend = Fixture {
            calls: calls.clone(),
            drift: drift.clone(),
            dtype: actual_dtype.into(),
            metadata: BTreeMap::from([(
                "native_execution".into(),
                "synthetic-gate-fixture-v1".into(),
            )]),
        };
        assert!(Engine::new(
            proper.manifest().clone(),
            Box::new(backend),
            Box::new(SimpleTokenizer::new(32768)),
            Default::default(),
            requested_backend,
            "fp32"
        )
        .is_err());
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}
