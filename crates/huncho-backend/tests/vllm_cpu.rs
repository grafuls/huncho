//! Real optional CPU runtime. Synthetic labels never qualify a released model.
#![cfg(feature = "vllm")]
use huncho_backend::vllm::{VllmBackend, VllmOptions};
use huncho_core::backend::{Backend, CacheHandle, ForwardInput};
use huncho_core::calibration::{argmax, calibrate};
use huncho_core::conformance::{
    run_suite_with_cross_request_batches, ConformanceThresholds, GoldenCase, GoldenSuite,
};
use huncho_core::contract::SystemOneRequest;
use huncho_core::engine::{Engine, EvalOptions};
use huncho_core::head::HeadParams;
use huncho_core::manifest::{BackendId, ModelManifest};
use huncho_core::prompt::formatter_for;
use huncho_core::tokenizer::HfTokenizer;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
const ROOT: &str = "tests/fixtures/vllm_cpu";
fn options() -> VllmOptions {
    VllmOptions {
        python: PathBuf::from(
            std::env::var_os("HUNCHO_VLLM_PYTHON").expect("explicit pinned CPU Python required"),
        ),
        threads: 2,
        batch_rows: 4,
        kv_cache_bytes: 64 * 1024 * 1024,
        timeout: Duration::from_secs(180),
    }
}
fn pair(actual: &[f32], expected: &[f32], delta: f32) {
    assert_eq!(actual.len(), expected.len());
    for temperature in [0.75, 1., 2.40605] {
        let a = calibrate(actual, temperature).unwrap();
        let b = calibrate(expected, temperature).unwrap();
        assert_eq!(argmax(&a), argmax(&b));
        assert!(
            a.iter().zip(b).all(|(a, b)| (a - b).abs() <= delta),
            "{a:?}"
        );
    }
}
#[test]
fn unsupported_profiles_and_changed_artifacts_fail_before_python_startup() {
    let root = Path::new(ROOT);
    let manifest = ModelManifest::load(root.join("huncho-model.json")).unwrap();
    // It deliberately does not exist: these errors must happen before spawning.
    let options = || VllmOptions {
        python: "/does/not/exist".into(),
        threads: 2,
        batch_rows: 1,
        kv_cache_bytes: 64 * 1024 * 1024,
        timeout: Duration::from_secs(10),
    };
    for dtype in ["fp32", "fp16", "int8"] {
        assert!(VllmBackend::load(root, &manifest, dtype, options()).is_err());
    }
    let mut other = manifest.clone();
    other.prompt_contract.template = "unsupported".into();
    assert!(VllmBackend::load(root, &other, "bf16", options()).is_err());
}
#[test]
fn modified_weights_and_incompatible_configs_fail_before_worker_startup() {
    use sha2::{Digest, Sha256};
    let source = Path::new(ROOT);
    let manifest = ModelManifest::load(source.join("huncho-model.json")).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("vllm/model")).unwrap();
    for name in [
        "vllm/artifact.json",
        "vllm/model/config.json",
        "vllm/model/model.safetensors",
    ] {
        std::fs::copy(source.join(name), root.join(name)).unwrap();
    }
    let options = || VllmOptions {
        python: "/bin/false".into(),
        threads: 2,
        batch_rows: 1,
        kv_cache_bytes: 64 * 1024 * 1024,
        timeout: Duration::from_secs(1),
    };
    let path = root.join("vllm/artifact.json");
    let mut descriptor: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let original = descriptor.clone();
    descriptor["files"]["model/model.safetensors"] = serde_json::json!("0".repeat(64));
    std::fs::write(&path, serde_json::to_vec(&descriptor).unwrap()).unwrap();
    let error = VllmBackend::load(root, &manifest, "bf16", options())
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("digest mismatch"), "{error}");
    let config_path = root.join("vllm/model/config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config["linear_key_head_dim"] = serde_json::json!(32);
    let bytes = serde_json::to_vec(&config).unwrap();
    std::fs::write(&config_path, &bytes).unwrap();
    descriptor = original;
    descriptor["files"]["model/config.json"] =
        serde_json::json!(format!("{:x}", Sha256::digest(bytes)));
    std::fs::write(&path, serde_json::to_vec(&descriptor).unwrap()).unwrap();
    let error = VllmBackend::load(root, &manifest, "bf16", options())
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("unsupported CPU Qwen"), "{error}");
}

#[test]
#[ignore = "requires explicitly pinned optional CPU vLLM/Python environment"]
fn real_cpu_pooling_preserves_frozen_scores_native_batches_and_fixed_shared_gates() {
    assert_eq!(std::env::var("HUNCHO_DEVICE").as_deref(), Ok("cpu"));
    let root = Path::new(ROOT);
    let manifest = ModelManifest::load(root.join("huncho-model.json")).unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let mut backend = VllmBackend::load(root, &manifest, "bf16", options()).unwrap();
    assert_eq!(backend.capabilities().extra["device"], "CPU");
    assert_eq!(backend.capabilities().extra["vllm_head_dtype"], "fp32");
    let rows: Vec<_> = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|c| c["rows"].as_array().unwrap())
        .collect();
    let mut baselines = Vec::new();
    for row in &rows {
        let input = ForwardInput::new(
            serde_json::from_value(row["tokens"].clone()).unwrap(),
            serde_json::from_value(row["positions"].clone()).unwrap(),
        );
        let output = backend.forward(input.clone()).unwrap();
        let expected: Vec<f32> = serde_json::from_value(row["probabilities"].clone()).unwrap();
        let actual = calibrate(output.values().data(), 2.40605).unwrap();
        assert_eq!(argmax(&actual), argmax(&expected));
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() <= 1e-3));
        baselines.push((input, output));
    }
    for (input, baseline) in &baselines {
        let grouped = backend
            .forward_batch(vec![input.clone(), input.clone()])
            .unwrap();
        for output in grouped {
            pair(output.values().data(), baseline.values().data(), 1e-4);
            assert_eq!(output.positions(), baseline.positions());
        }
    }
    let mut invalid = baselines[0].0.clone();
    invalid.retain_cache = true;
    assert!(backend
        .forward_batch(vec![baselines[0].0.clone(), invalid])
        .is_err());
    assert!(backend.fork(CacheHandle { id: 1 }).is_err());
    assert!(backend.forward_batch(vec![]).is_err());
    assert!(backend
        .forward_batch(vec![baselines[0].0.clone(); 5])
        .is_err());
    assert!(backend
        .forward_batch(vec![
            baselines[0].0.clone(),
            ForwardInput::new(vec![1], vec![0])
        ])
        .is_err());
    let output = backend
        .forward(ForwardInput::new(vec![1, 2], vec![0]))
        .unwrap();
    let expected: Vec<f32> = serde_json::from_value(reference["short_logits"].clone()).unwrap();
    assert!(output
        .values()
        .data()
        .iter()
        .zip(expected)
        .all(|(a, b)| (a - b).abs() < 0.01));
    let empty = backend
        .forward(ForwardInput::new(vec![1, 2], vec![]))
        .unwrap();
    assert_eq!(empty.values().shape(), &[0, 1]);
    // Returned tensors own their storage after further calls.
    let actual = calibrate(baselines[0].1.values().data(), 2.40605).unwrap();
    let frozen: Vec<f32> = serde_json::from_value(rows[0]["probabilities"].clone()).unwrap();
    assert!(actual
        .iter()
        .zip(frozen)
        .all(|(a, b)| (a - b).abs() <= 1e-3));
    let tokenizer = HfTokenizer::from_file_unbounded(root.join("tokenizer.json")).unwrap();
    let mut cases = Vec::new();
    for (index, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let request: SystemOneRequest = serde_json::from_value(case["request"].clone()).unwrap();
        let mut expected = BTreeMap::new();
        for ((id, q), row) in request
            .questions
            .iter()
            .zip(case["rows"].as_array().unwrap())
        {
            let prompt = formatter_for(&manifest)
                .build(&request.state, q, &tokenizer)
                .unwrap();
            assert_eq!(
                prompt.tokens,
                serde_json::from_value::<Vec<u32>>(row["tokens"].clone()).unwrap()
            );
            let probs: Vec<f32> = serde_json::from_value(row["probabilities"].clone()).unwrap();
            expected.insert(
                id.clone(),
                prompt
                    .candidates
                    .iter()
                    .zip(probs)
                    .map(|(c, p)| (c.label.clone(), p))
                    .collect::<BTreeMap<_, _>>(),
            );
        }
        let targets = expected
            .iter()
            .map(|(id, p)| (id.clone(), p.keys().next().unwrap().clone()))
            .collect();
        cases.push(GoldenCase {
            id: index.to_string(),
            request,
            expected,
            targets,
        });
    }
    let mut duplicate = cases[0].clone();
    duplicate.id = "duplicate-shape".into();
    cases.push(duplicate);
    let suite = GoldenSuite {
        schema_version: "1.0".into(),
        family: "F2".into(),
        hash: None,
        cases,
    };
    let engine = Engine::new(
        manifest,
        Box::new(backend),
        Box::new(tokenizer),
        HeadParams::default(),
        BackendId::Vllm,
        "bf16",
    )
    .unwrap();
    let opts = EvalOptions {
        prepare_all: true,
        max_batch_tokens: Some(4096),
        ..Default::default()
    };
    let report = run_suite_with_cross_request_batches(
        &engine,
        &suite,
        &ConformanceThresholds::default(),
        &opts,
        3,
    )
    .unwrap();
    assert!(report.passed, "{report:?}");
    assert!(report.work.cross_request_batches > 0);
    assert_eq!(report.outcome_calibration.as_ref().unwrap().questions, 9);
    let mut incomplete = suite.clone();
    incomplete.cases[0].targets.clear();
    assert!(run_suite_with_cross_request_batches(
        &engine,
        &incomplete,
        &ConformanceThresholds::default(),
        &opts,
        3
    )
    .is_err());
}
