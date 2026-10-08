//! Native cache/chunk parity uses pinned upstream fixtures, never regenerated goldens.
#![cfg(feature = "candle")]

use huncho_backend::{kev::KevMetadata, Qwen3_5Backend};
use huncho_core::backend::{Backend, CacheHandle, ForwardInput};
use huncho_core::calibration::calibrate;
use std::path::Path;

use candle::Device;

const FIXTURE: &str = "tests/fixtures/tiny_kev";

fn load(dtype: &str, max_context: usize) -> Qwen3_5Backend {
    let root = Path::new(FIXTURE);
    Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), max_context, dtype).unwrap()
}

fn continuation(tokens: &[u32], positions: Vec<usize>, handle: CacheHandle) -> ForwardInput {
    let mut input = ForwardInput::new(tokens.to_vec(), positions);
    input.fork_from = Some(handle);
    input
}

fn assert_probabilities(actual: &[f32], expected: &[f32], temperature: f32) {
    let actual = calibrate(actual, temperature).unwrap();
    let expected = calibrate(expected, temperature).unwrap();
    for (a, b) in actual.iter().zip(&expected) {
        assert!(
            (a - b).abs() <= 1e-4,
            "cached={actual:?}, independent={expected:?}"
        );
    }
    let argmax = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(argmax(&actual), argmax(&expected));
}

fn assert_wire_equal(actual: &serde_json::Value, expected: &serde_json::Value) {
    use serde_json::Value;
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => assert!(
            (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= 1e-4,
            "{a} vs {b}"
        ),
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len());
            for (key, value) in a {
                assert_wire_equal(value, &b[key]);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

#[test]
fn native_forks_chunks_and_short_prefixes_preserve_probabilities() {
    assert_forks_chunks_and_short_prefixes(Device::Cpu);
}

#[test]
fn persistent_prefixes_are_exact_bounded_and_isolate_active_handles_on_cpu() {
    let root = Path::new(FIXTURE);
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let row = &golden["cases"][0]["rows"][0];
    let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
    let positions: Vec<usize> = serde_json::from_value(row["positions"].clone()).unwrap();
    let prefix = row["prefix_len"].as_u64().unwrap() as usize;
    let budget = 1024 * 1024;
    for dtype in ["fp32", "fp16"] {
        let mut backend = load(dtype, 512);
        let baseline = backend
            .forward(ForwardInput::new(tokens.clone(), positions.clone()))
            .unwrap();
        for iteration in 0..3 {
            let cached = backend.prefill_cached(&tokens[..prefix], budget).unwrap();
            assert_eq!(cached.hit, iteration != 0);
            let fork = backend.fork(cached.handle).unwrap();
            backend.release_cache(cached.handle).unwrap();
            if iteration == 2 {
                backend.clear_prefix_cache().unwrap();
            }
            let output = backend
                .forward(continuation(
                    &tokens[prefix..],
                    positions.iter().map(|p| p - prefix).collect(),
                    fork,
                ))
                .unwrap();
            assert_probabilities(output.values().data(), baseline.values().data(), 2.40605);
            backend.release_cache(fork).unwrap();
        }
        let after_clear = backend.prefill_cached(&tokens[..prefix], budget).unwrap();
        assert!(!after_clear.hit);
        backend.release_cache(after_clear.handle).unwrap();
        // Reducing the charged budget evicts the previous snapshot; an
        // oversized prefix remains valid but is never retained under that budget.
        for _ in 0..2 {
            let small = backend.prefill_cached(&tokens[..prefix], 1).unwrap();
            assert!(!small.hit);
            backend.release_cache(small.handle).unwrap();
        }
        let invalid = vec![u32::MAX];
        assert!(backend.prefill_cached(&invalid, budget).is_err());
        // The count bound is independent of available bytes and handle release.
        for token in 2..=18 {
            let cached = backend.prefill_cached(&[1, token], budget).unwrap();
            assert!(!cached.hit);
            backend.release_cache(cached.handle).unwrap();
        }
        let evicted = backend.prefill_cached(&[1, 2], budget).unwrap();
        assert!(!evicted.hit);
        backend.release_cache(evicted.handle).unwrap();
        let retained = backend.prefill_cached(&[1, 18], budget).unwrap();
        assert!(retained.hit);
        backend.release_cache(retained.handle).unwrap();
        assert!(backend.with_projection_chunk_rows(64).is_err());
        let mut backend = load(dtype, 512);
        let cached = backend.prefill_cached(&[1, 2], budget).unwrap();
        backend.release_cache(cached.handle).unwrap();
        backend.clear_prefix_cache().unwrap();
        assert!(backend.with_projection_chunk_rows(64).is_ok());
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a compatible CUDA GPU"]
fn cuda_forks_chunks_and_short_prefixes_preserve_probabilities() {
    let device = huncho_backend::device::device_from_env().unwrap();
    assert!(device.is_cuda(), "GPU parity test must run on CUDA");
    assert_forks_chunks_and_short_prefixes(device);
}

fn assert_forks_chunks_and_short_prefixes(device: Device) {
    let root = Path::new(FIXTURE);
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let temperature = KevMetadata::load(&root.join("head.pt"))
        .unwrap()
        .temperature;
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
        assert!(backend.capabilities().supports_fork);
        for case in golden["cases"].as_array().unwrap() {
            let rows = case["rows"].as_array().unwrap();
            let first: Vec<u32> = serde_json::from_value(rows[0]["tokens"].clone()).unwrap();
            let shared = rows[0]["prefix_len"].as_u64().unwrap() as usize;
            // One token exercises an incomplete convolution history. The
            // native state boundary exercises whole-request fan-out.
            for prefix_len in [1, 2, shared] {
                let parent = backend.prefill(&first[..prefix_len]).unwrap();
                for row in rows {
                    let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                    let positions: Vec<usize> =
                        serde_json::from_value(row["positions"].clone()).unwrap();
                    assert_eq!(tokens[..prefix_len], first[..prefix_len]);
                    let baseline = backend
                        .forward(ForwardInput::new(tokens.clone(), positions.clone()))
                        .unwrap();
                    for chunk_size in [1, 3, usize::MAX] {
                        let branch = backend.fork(parent).unwrap();
                        let mut offset = prefix_len;
                        // Chunks before the first candidate advance all cache
                        // components even though they request no readout.
                        while offset < positions[0] {
                            let end = offset.saturating_add(chunk_size).min(positions[0]);
                            backend
                                .forward(continuation(&tokens[offset..end], vec![], branch))
                                .unwrap();
                            offset = end;
                        }
                        let actual = backend
                            .forward(continuation(
                                &tokens[offset..],
                                positions.iter().map(|p| p - offset).collect(),
                                branch,
                            ))
                            .unwrap();
                        assert_probabilities(
                            actual.values().data(),
                            baseline.values().data(),
                            temperature,
                        );
                        backend.release_cache(branch).unwrap();
                    }
                }
                backend.release_cache(parent).unwrap();
                assert!(backend.fork(parent).is_err());
            }
        }
    }
}

#[test]
fn cache_handles_are_owned_bounded_and_transactional() {
    let mut backend = load("fp32", 16);
    let mut other = load("fp32", 16);
    assert!(backend.prefill(&[]).is_err());
    assert!(backend.prefill(&[1; 17]).is_err());
    let parent = backend.prefill(&[1, 2]).unwrap();
    let branch = backend.fork(parent).unwrap();
    assert!(other.fork(parent).is_err());
    assert!(other.forward(continuation(&[3], vec![0], branch)).is_err());
    backend.release_cache(parent).unwrap();
    assert!(backend.release_cache(parent).is_err());
    assert!(backend
        .forward(continuation(&[1; 15], vec![0], branch))
        .is_err());
    assert!(backend
        .forward(continuation(&[3], vec![1], branch))
        .is_err());
    // Malformed continuation does not advance the branch offset/state.
    assert!(backend
        .forward(continuation(&[u32::MAX], vec![0], branch))
        .is_err());
    let actual = backend
        .forward(continuation(&[3, 4], vec![0, 1], branch))
        .unwrap();
    let baseline = backend
        .forward(ForwardInput::new(vec![1, 2, 3, 4], vec![2, 3]))
        .unwrap();
    assert_probabilities(actual.values().data(), baseline.values().data(), 2.40605);
    backend.release_cache(branch).unwrap();
    let parent = backend.prefill(&[1]).unwrap();
    let mut handles = vec![parent];
    for _ in 1..64 {
        handles.push(backend.fork(parent).unwrap());
    }
    assert!(backend.fork(parent).is_err());
    for handle in handles {
        backend.release_cache(handle).unwrap();
    }
    let handle = backend.prefill(&[1]).unwrap();
    backend.release_cache(handle).unwrap();
}

#[cfg(feature = "clef")]
#[test]
fn engine_fanout_preserves_wire_usage_and_releases_on_later_errors() {
    use huncho_core::contract::SystemOneRequest;
    use huncho_core::engine::{Engine, EvalOptions, EvalStats};
    use huncho_core::head::HeadParams;
    use huncho_core::manifest::{BackendId, Family, HeadKind, ModelManifest};
    use huncho_core::prompt::formatter_for;
    use huncho_core::tokenizer::HfTokenizer;

    let root = Path::new(FIXTURE);
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let options = EvalOptions {
        extensions: true,
        prefix_cache: true,
        prepare_all: true,
        ..Default::default()
    };
    for dtype in ["fp32", "fp16"] {
        let mut manifest =
            ModelManifest::load("../../examples/mock-model/huncho-model.json").unwrap();
        manifest.name = "tiny-kev".into();
        manifest.family = Family::F2;
        manifest.head.kind = HeadKind::Pointer;
        manifest.prompt_contract.template = "kev-v1".into();
        manifest.prompt_contract.state_budget = 512;
        manifest.prompt_contract.head_budget = 512;
        manifest.backbone.max_context = 512;
        manifest.calibration.entries.clear();
        manifest.calibration.default.temperature = 2.40605;
        let engine = Engine::new(
            manifest,
            Box::new(load(dtype, 512)),
            Box::new(HfTokenizer::from_file_unbounded(root.join("tokenizer.json")).unwrap()),
            HeadParams::default(),
            BackendId::Candle,
            dtype,
        )
        .unwrap()
        .with_prompt_cache(1024 * 1024);
        for case in golden["cases"].as_array().unwrap() {
            let req: SystemOneRequest = serde_json::from_value(case["request"].clone()).unwrap();
            let baseline = engine
                .eval(
                    &req,
                    &EvalOptions {
                        extensions: true,
                        ..Default::default()
                    },
                )
                .unwrap();
            let mut stats = EvalStats::default();
            let cached = engine.eval_with_stats(&req, &options, &mut stats).unwrap();
            assert_eq!(stats.prompt_cache_hits, req.questions.len() as u64);
            assert_eq!(stats.prepared_questions, req.questions.len() as u64);
            let mut replay_stats = EvalStats::default();
            let replay = engine
                .eval_with_stats(&req, &options, &mut replay_stats)
                .unwrap();
            assert_eq!(
                serde_json::to_vec(&replay).unwrap(),
                serde_json::to_vec(&cached).unwrap()
            );
            assert_eq!(replay_stats.prompt_cache_hits, req.questions.len() as u64);
            assert_eq!(replay_stats.cache_forks, req.questions.len() as u64);
            assert_eq!(cached.usage.input_tokens, baseline.usage.input_tokens);
            assert_wire_equal(
                &serde_json::to_value(&cached.answers).unwrap(),
                &serde_json::to_value(&baseline.answers).unwrap(),
            );
            let rows = case["rows"].as_array().unwrap();
            let prefix = rows[0]["prefix_len"].as_u64().unwrap();
            assert_eq!(stats.prefill_calls, 1);
            assert_eq!(stats.forward_calls, req.questions.len() as u64);
            assert_eq!(stats.cache_forks, req.questions.len() as u64);
            assert_eq!(
                stats.reused_prefix_tokens,
                prefix * req.questions.len() as u64
            );
            assert_eq!(
                stats.processed_tokens,
                baseline.usage.input_tokens - prefix * (req.questions.len() as u64 - 1)
            );
            let a = cached.extensions.unwrap().raw_logits.unwrap();
            let b = baseline.extensions.unwrap().raw_logits.unwrap();
            for (id, logits) in a {
                assert_probabilities(&logits, &b[&id], 2.40605);
            }
            let tokenizer = HfTokenizer::from_file_unbounded(root.join("tokenizer.json")).unwrap();
            let expected = req
                .questions
                .iter()
                .zip(rows)
                .map(|((id, question), row)| {
                    let prompt = formatter_for(engine.manifest())
                        .build(&req.state, question, &tokenizer)
                        .unwrap();
                    let probabilities: Vec<f32> =
                        serde_json::from_value(row["probabilities"].clone()).unwrap();
                    (
                        id.clone(),
                        prompt
                            .candidates
                            .into_iter()
                            .zip(probabilities)
                            .map(|(candidate, p)| (candidate.label, p))
                            .collect(),
                    )
                })
                .collect();
            let suite = huncho_core::conformance::GoldenSuite {
                schema_version: "1.0".into(),
                family: "F2".into(),
                hash: None,
                cases: vec![huncho_core::conformance::GoldenCase {
                    id: "pinned-upstream".into(),
                    request: req.clone(),
                    expected,
                    targets: Default::default(),
                }],
            };
            let report = huncho_core::conformance::run_suite_with_options(
                &engine,
                &suite,
                &Default::default(),
                &options,
            )
            .unwrap();
            assert!(report.passed, "{dtype}: {report:?}");
            assert!(report.work.cache_forks > 0);
            assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
            let persistent = EvalOptions {
                persistent_prefix_bytes: 1024 * 1024,
                ..options.clone()
            };
            engine.clear_prefix_cache().unwrap();
            let mut first_work = EvalStats::default();
            let first = engine
                .eval_uncached_with_stats(&req, &persistent, &mut first_work)
                .unwrap();
            assert_eq!(first_work.prefill_calls, 1);
            assert_eq!(first_work.persistent_prefix_hits, 0);
            let mut hit_work = EvalStats::default();
            let hit = engine
                .eval_uncached_with_stats(&req, &persistent, &mut hit_work)
                .unwrap();
            assert_eq!(hit_work.prefill_calls, 0);
            assert_eq!(hit_work.persistent_prefix_hits, 1);
            assert_eq!(
                first_work.processed_tokens - hit_work.processed_tokens,
                prefix
            );
            assert_wire_equal(
                &serde_json::to_value(&first).unwrap(),
                &serde_json::to_value(&hit).unwrap(),
            );
            let report = huncho_core::conformance::run_suite_with_options(
                &engine,
                &suite,
                &Default::default(),
                &persistent,
            )
            .unwrap();
            assert!(report.passed, "{dtype}: {report:?}");
            assert_eq!(report.work.prefill_calls, 1);
            assert_eq!(report.work.persistent_prefix_hits, 1);
            assert_eq!(report.work.forward_calls, 2 * req.questions.len() as u64);
            assert_eq!(report.work.result_cache_hits, 0);
            assert_eq!(report.work.prompt_cache_hits, 0);
            let small = EvalOptions {
                persistent_prefix_bytes: 1,
                ..persistent
            };
            assert!(huncho_core::conformance::run_suite_with_options(
                &engine,
                &suite,
                &Default::default(),
                &small
            )
            .is_err());
        }
        // The first question fits and executes; a later one exceeds context.
        // Repeating past the native retained-handle bound detects leaked parents.
        let mut req: SystemOneRequest =
            serde_json::from_value(golden["cases"][0]["request"].clone()).unwrap();
        let first = req.questions.shift_remove("priority").unwrap();
        let remaining = std::mem::take(&mut req.questions);
        req.questions.insert("priority".into(), first);
        req.questions.extend(remaining);
        let tk = HfTokenizer::from_file_unbounded(root.join("tokenizer.json")).unwrap();
        let prompt = formatter_for(engine.manifest())
            .build(&req.state, req.questions.values().next().unwrap(), &tk)
            .unwrap();
        let invalid = EvalOptions {
            max_context: Some(prompt.tokens.len()),
            prepare_all: false,
            ..options.clone()
        };
        for _ in 0..70 {
            let mut stats = EvalStats::default();
            let error = engine
                .eval_with_stats(&req, &invalid, &mut stats)
                .unwrap_err();
            assert!(matches!(error, huncho_core::Error::Request(_)), "{error}");
            assert_eq!(stats.prefill_calls, 1);
            assert_eq!(stats.cache_forks, 1);
        }
        engine.eval(&req, &options).unwrap();
    }
}
