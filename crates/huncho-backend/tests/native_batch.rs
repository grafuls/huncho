//! Exact-length native batches isolate sequence attention and preserve readouts.
#![cfg(feature = "candle")]

use huncho_backend::{CandleBackend, Qwen3_5Backend};
use huncho_core::backend::{Backend, ForwardInput};
use huncho_core::calibration::{argmax, calibrate};
use std::path::Path;

use candle::Device;

fn assert_parity(actual: &[f32], baseline: &[f32]) {
    assert_parity_with_delta(actual, baseline, 1e-4);
}

fn assert_parity_with_delta(actual: &[f32], baseline: &[f32], max_delta: f32) {
    for temperature in [0.75, 1.0, 2.40605] {
        let a = calibrate(actual, temperature).unwrap();
        let b = calibrate(baseline, temperature).unwrap();
        assert_eq!(argmax(&a), argmax(&b));
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() <= max_delta, "{actual:?} vs {baseline:?}");
        }
    }
}

#[test]
fn modernbert_batches_preserve_rows_and_candidate_positions() {
    assert_modernbert_batch_parity(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a compatible CUDA GPU"]
fn cuda_modernbert_batches_preserve_rows_and_candidate_positions() {
    let device = huncho_backend::device::device_from_env().unwrap();
    assert!(device.is_cuda(), "GPU parity test must run on CUDA");
    assert_modernbert_batch_parity(device);
}

fn assert_modernbert_batch_parity(device: Device) {
    let mut backend = CandleBackend::load_on_device(
        "tests/fixtures/tiny_modernbert/config.json",
        "tests/fixtures/tiny_modernbert/model.safetensors",
        16,
        "fp32",
        device,
    )
    .unwrap();
    assert!(backend.supports_batch());
    let inputs = vec![
        ForwardInput::new(vec![1, 2, 3, 4, 5], vec![1, 3]),
        ForwardInput::new(vec![5, 4, 3, 2, 1], vec![0, 2, 4]),
        ForwardInput::new(vec![2, 1, 5, 4, 3], vec![3, 1, 3]),
    ];
    let baselines: Vec<_> = inputs
        .iter()
        .map(|input| backend.forward(input.clone()).unwrap())
        .collect();
    let batched = backend.forward_batch(inputs.clone()).unwrap();
    for (output, baseline) in batched.iter().zip(&baselines) {
        assert_eq!(output.positions(), baseline.positions());
        // Generic scalar heads exercise calibrated readout parity on every
        // selected row; the fixture itself is a bare encoder, not trained Laya.
        let project = |values: &huncho_core::tensor::Tensor| {
            values
                .data()
                .chunks(8)
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(i, x)| x * (i as f32 + 1.) / 8.)
                        .sum()
                })
                .collect::<Vec<f32>>()
        };
        assert_parity(&project(output.values()), &project(baseline.values()));
    }
    assert!(backend.forward_batch(vec![]).is_err());
    assert!(backend
        .forward_batch(vec![inputs[0].clone(), ForwardInput::new(vec![1], vec![0])])
        .is_err());
    let mut invalid = inputs[0].clone();
    invalid.positions = vec![5];
    assert!(backend.forward_batch(vec![invalid]).is_err());
}

#[test]
fn qwen_pointer_batches_preserve_probabilities_and_reject_cache_branches() {
    assert_pointer_batch_parity(Device::Cpu);
}

#[test]
fn buffered_cpu_recurrence_preserves_kev_reference_batches_and_prefixes() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for dtype in ["fp32", "fp16"] {
        let load =
            || Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype).unwrap();
        let mut reference = load();
        let mut buffered = load().with_cpu_delta_rule(true).unwrap();
        assert_eq!(
            buffered.capabilities().extra["delta_rule_execution"],
            "cpu-buffered-v1"
        );
        for case in golden["cases"].as_array().unwrap() {
            let rows = case["rows"].as_array().unwrap();
            let tokens: Vec<u32> = serde_json::from_value(rows[0]["tokens"].clone()).unwrap();
            let prefix = rows[0]["prefix_len"].as_u64().unwrap() as usize;
            let parent = buffered
                .prefill_cached(&tokens[..prefix], 1024 * 1024)
                .unwrap()
                .handle;
            for row in rows {
                let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                let positions: Vec<usize> =
                    serde_json::from_value(row["positions"].clone()).unwrap();
                let input = ForwardInput::new(tokens.clone(), positions.clone());
                let expected = reference.forward(input.clone()).unwrap();
                let actual = buffered.forward(input.clone()).unwrap();
                let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(actual.values().data()), bits(expected.values().data()));
                let upstream: Vec<f32> =
                    serde_json::from_value(row["probabilities"].clone()).unwrap();
                let actual_probs = calibrate(actual.values().data(), 2.40605).unwrap();
                assert_eq!(argmax(&actual_probs), argmax(&upstream));
                assert!(actual_probs
                    .iter()
                    .zip(&upstream)
                    .all(|(a, b)| (a - b).abs() <= 1e-3));
                for output in buffered.forward_batch(vec![input.clone(), input]).unwrap() {
                    assert_parity(output.values().data(), actual.values().data());
                }
                let handle = buffered.fork(parent).unwrap();
                let mut suffix = ForwardInput::new(
                    tokens[prefix..].to_vec(),
                    positions.iter().map(|p| p - prefix).collect(),
                );
                suffix.fork_from = Some(handle);
                let cached = buffered.forward(suffix).unwrap();
                assert_parity(cached.values().data(), actual.values().data());
                buffered.release_cache(handle).unwrap();
            }
            buffered.release_cache(parent).unwrap();
        }
        assert!(buffered.with_cpu_delta_rule(false).is_err());
        let mut retained = load();
        retained.prefill(&[1, 2]).unwrap();
        assert!(retained.with_cpu_delta_rule(true).is_err());
        let mut cleared = load();
        let parent = cleared.prefill_cached(&[1, 2], 1024 * 1024).unwrap().handle;
        cleared.release_cache(parent).unwrap();
        cleared.clear_prefix_cache().unwrap();
        cleared.with_cpu_delta_rule(true).unwrap();
    }
}

#[cfg(feature = "clef")]
#[test]
fn kev_cpu_cross_request_collation_qualifies_unchanged_upstream_vectors() {
    use huncho_core::conformance::{GoldenCase, GoldenSuite};
    use huncho_core::engine::{Engine, EvalOptions};
    use huncho_core::manifest::{BackendId, Family, HeadKind, ModelManifest};
    use huncho_core::prompt::formatter_for;
    use huncho_core::tokenizer::HfTokenizer;
    let root = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for dtype in ["fp32", "fp16"] {
        let mut manifest =
            ModelManifest::load("../../examples/mock-model/huncho-model.json").unwrap();
        manifest.family = Family::F2;
        manifest.head.kind = HeadKind::Pointer;
        manifest.prompt_contract.template = "kev-v1".into();
        manifest.prompt_contract.state_budget = 512;
        manifest.prompt_contract.head_budget = 512;
        manifest.backbone.max_context = 512;
        manifest.calibration.entries.clear();
        manifest.calibration.default.temperature = 2.40605;
        let tokenizer = HfTokenizer::from_file_unbounded(root.join("tokenizer.json")).unwrap();
        let mut cases = Vec::new();
        for (index, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
            let request: huncho_core::contract::SystemOneRequest =
                serde_json::from_value(case["request"].clone()).unwrap();
            let expected = request
                .questions
                .iter()
                .zip(case["rows"].as_array().unwrap())
                .map(|((id, question), row)| {
                    let prompt = formatter_for(&manifest)
                        .build(&request.state, question, &tokenizer)
                        .unwrap();
                    let probabilities: Vec<f32> =
                        serde_json::from_value(row["probabilities"].clone()).unwrap();
                    (
                        id.clone(),
                        prompt
                            .candidates
                            .into_iter()
                            .zip(probabilities)
                            .map(|(c, p)| (c.label, p))
                            .collect(),
                    )
                })
                .collect();
            cases.push(GoldenCase {
                id: index.to_string(),
                request,
                expected,
                targets: Default::default(),
            });
        }
        // Repeated requests still submit fresh model work; duplicate IDs must
        // remain private to each request and caches cannot make the gate vacuous.
        cases.extend(cases.clone());
        let engine = Engine::new(
            manifest,
            Box::new(
                Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype).unwrap(),
            ),
            Box::new(tokenizer),
            Default::default(),
            BackendId::Candle,
            dtype,
        )
        .unwrap()
        .with_result_cache(1024 * 1024)
        .with_prompt_cache(1024 * 1024);
        let suite = GoldenSuite {
            schema_version: "1.0".into(),
            family: "F2".into(),
            hash: None,
            cases,
        };
        let options = EvalOptions {
            max_batch_tokens: Some(4096),
            prepare_all: true,
            ..Default::default()
        };
        let report = huncho_core::conformance::run_suite_with_cross_request_batches(
            &engine,
            &suite,
            &Default::default(),
            &options,
            4,
        )
        .unwrap();
        assert!(report.passed, "{dtype}: {report:?}");
        assert!(report.work.cross_request_batches > 0);
        assert_eq!(report.work.result_cache_hits, 0);
        assert_eq!(report.work.prompt_cache_hits, 0);
        assert_eq!(report.work.prepared_questions, 12);
        assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
    }
}

#[test]
fn execution_profiles_preserve_fixture_readouts_and_cache_semantics() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for dtype in ["fp32", "fp16"] {
        for (chunk_rows, fp32_attention) in [(64, false), (0, true), (64, true)] {
            let load =
                || Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype).unwrap();
            let mut native = load();
            assert!(!native
                .capabilities()
                .extra
                .contains_key("projection_chunk_rows"));
            assert!(load().with_projection_chunk_rows(4097).is_err());
            let mut profiled = load()
                .with_projection_chunk_rows(chunk_rows)
                .unwrap()
                .with_fp32_attention(fp32_attention)
                .unwrap();
            let extra = profiled.capabilities().extra;
            assert_eq!(
                extra.get("projection_chunk_rows"),
                (chunk_rows > 0).then(|| chunk_rows.to_string()).as_ref()
            );
            assert_eq!(
                extra.get("attention_compute_dtype").map(String::as_str),
                fp32_attention.then_some("fp32")
            );
            for case in golden["cases"].as_array().unwrap() {
                let rows = case["rows"].as_array().unwrap();
                let tokens: Vec<u32> = serde_json::from_value(rows[0]["tokens"].clone()).unwrap();
                let prefix = rows[0]["prefix_len"].as_u64().unwrap() as usize;
                let parent = profiled.prefill(&tokens[..prefix]).unwrap();
                for row in rows {
                    let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                    let positions: Vec<usize> =
                        serde_json::from_value(row["positions"].clone()).unwrap();
                    let input = ForwardInput::new(tokens.clone(), positions.clone());
                    let baseline = native.forward(input.clone()).unwrap();
                    let independent = profiled.forward(input.clone()).unwrap();
                    // Changed arithmetic uses the unchanged external-golden gate;
                    // cache/batch parity is measured against that profile's own
                    // independent forwards with the tighter unchanged 1e-4 gate.
                    assert_parity_with_delta(
                        independent.values().data(),
                        baseline.values().data(),
                        1e-3,
                    );
                    let batch = profiled.forward_batch(vec![input.clone(), input]).unwrap();
                    for output in batch {
                        assert_eq!(output.positions(), baseline.positions());
                        assert_parity(output.values().data(), independent.values().data());
                    }
                    let fork = profiled.fork(parent).unwrap();
                    let mut suffix = ForwardInput::new(
                        tokens[prefix..].to_vec(),
                        positions.iter().map(|p| p - prefix).collect(),
                    );
                    suffix.fork_from = Some(fork);
                    let output = profiled.forward(suffix).unwrap();
                    assert_parity(output.values().data(), independent.values().data());
                    profiled.release_cache(fork).unwrap();
                }
                profiled.release_cache(parent).unwrap();
            }
            // A profile can be retained with live caches, but changing the kernel
            // underneath a retained prefix must fail instead of mixing arithmetic.
            let handle = profiled.prefill(&[1, 2]).unwrap();
            profiled = profiled
                .with_projection_chunk_rows(chunk_rows)
                .unwrap()
                .with_fp32_attention(fp32_attention)
                .unwrap();
            profiled.release_cache(handle).unwrap();
            profiled = profiled.with_projection_chunk_rows(0).unwrap();
            assert!(!profiled
                .capabilities()
                .extra
                .contains_key("projection_chunk_rows"));
            profiled.prefill(&[1, 2]).unwrap();
            assert!(profiled.with_projection_chunk_rows(64).is_err());
            let mut profiled = load().with_fp32_attention(fp32_attention).unwrap();
            profiled.prefill(&[1, 2]).unwrap();
            assert!(profiled.with_fp32_attention(!fp32_attention).is_err());
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a compatible CUDA GPU"]
fn cuda_pointer_batches_preserve_probabilities_and_reject_cache_branches() {
    let device = huncho_backend::device::device_from_env().unwrap();
    assert!(device.is_cuda(), "GPU parity test must run on CUDA");
    assert_pointer_batch_parity(device);
}

fn assert_pointer_batch_parity(device: Device) {
    let root = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for dtype in ["fp32", "fp16"] {
        let mut backend = Qwen3_5Backend::load_kev_on_device(
            root,
            root,
            &root.join("head.pt"),
            512,
            dtype,
            device.clone(),
        )
        .unwrap();
        for case in golden["cases"].as_array().unwrap() {
            for row in case["rows"].as_array().unwrap() {
                let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                let positions: Vec<usize> =
                    serde_json::from_value(row["positions"].clone()).unwrap();
                let mut different = tokens.clone();
                different[1] = (different[1] + 1) % 384;
                let mut reordered = positions.clone();
                reordered.reverse();
                let inputs = vec![
                    ForwardInput::new(tokens.clone(), positions.clone()),
                    ForwardInput::new(different, positions),
                    ForwardInput::new(tokens, reordered),
                ];
                let baseline: Vec<_> = inputs
                    .iter()
                    .map(|input| backend.forward(input.clone()).unwrap())
                    .collect();
                let batched = backend.forward_batch(inputs).unwrap();
                for (output, baseline) in batched.iter().zip(&baseline) {
                    assert_eq!(output.positions(), baseline.positions());
                    assert_parity(output.values().data(), baseline.values().data());
                }
            }
        }
        let handle = backend.prefill(&[1]).unwrap();
        let mut branch = ForwardInput::new(vec![2], vec![0]);
        branch.fork_from = Some(handle);
        assert!(backend.forward_batch(vec![branch]).is_err());
        backend.release_cache(handle).unwrap();
        assert!(backend
            .forward_batch(vec![ForwardInput::new(vec![384], vec![0])])
            .is_err());
    }
}
