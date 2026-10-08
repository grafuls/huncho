//! CPU branch collation keeps frozen references and immutable prefix ownership.
#![cfg(feature = "candle")]
use huncho_backend::Qwen3_5Backend;
use huncho_core::{
    backend::{Backend, CacheHandle, ForkBatchWork, ForwardInput},
    calibration::{argmax, calibrate},
};
use std::path::Path;

fn load(dtype: &str, pages: usize, optimized: bool, runtime: bool) -> Qwen3_5Backend {
    let root = Path::new("tests/fixtures/tiny_kev");
    let backend = if runtime {
        Qwen3_5Backend::load_kev_runtime_lora(root, root, &root.join("head.pt"), 512, dtype)
    } else {
        Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype)
    }
    .unwrap();
    let backend = if dtype == "fp32" {
        backend.with_cpu_blas_from_env().unwrap()
    } else {
        backend
    };
    backend
        .with_kv_page_tokens(pages)
        .unwrap()
        .with_prefill_chunk_tokens(if optimized { 3 } else { 0 })
        .unwrap()
        .with_cpu_delta_rule(optimized)
        .unwrap()
        .with_cpu_causal_conv(optimized)
        .unwrap()
        .with_cpu_fused_gate(optimized)
        .unwrap()
        .with_attention_query_rows(if optimized { 7 } else { 0 })
        .unwrap()
        .with_grouped_gqa(optimized)
        .unwrap()
}
fn parity(a: &[f32], b: &[f32], bound: f32) {
    for t in [0.75, 1., 2.40605] {
        let (a, b) = (calibrate(a, t).unwrap(), calibrate(b, t).unwrap());
        assert_eq!(argmax(&a), argmax(&b));
        assert!(
            a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= bound),
            "{a:?} vs {b:?}"
        );
    }
}
fn scalar(backend: &mut Qwen3_5Backend, parent: CacheHandle, mut input: ForwardInput) -> Vec<f32> {
    let branch = backend.fork(parent).unwrap();
    input.fork_from = Some(branch);
    let out = backend.forward(input).unwrap().values().data().to_vec();
    backend.release_cache(branch).unwrap();
    out
}
#[test]
fn cached_native_batches_preserve_typed_goldens_rows_pages_and_adapters() {
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read("tests/fixtures/tiny_kev/golden.json").unwrap())
            .unwrap();
    for (dtype, pages, optimized, runtime) in [
        ("fp32", 0, false, false),
        ("fp16", 0, false, false),
        ("fp32", 16, true, false),
        ("fp16", 32, true, false),
        ("fp32", 16, true, true),
    ] {
        let mut backend = load(dtype, pages, optimized, runtime);
        assert!(backend.supports_fork_batch());
        for case in golden["cases"].as_array().unwrap() {
            let rows = case["rows"].as_array().unwrap();
            let prefix_len = rows[0]["prefix_len"].as_u64().unwrap() as usize;
            let first: Vec<u32> = serde_json::from_value(rows[0]["tokens"].clone()).unwrap();
            let parent = backend
                .prefill_cached(&first[..prefix_len], 1 << 20)
                .unwrap()
                .handle;
            let suffixes: Vec<_> = rows
                .iter()
                .map(|row| {
                    let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                    let positions: Vec<usize> =
                        serde_json::from_value(row["positions"].clone()).unwrap();
                    ForwardInput::new(
                        tokens[prefix_len..].to_vec(),
                        positions.iter().map(|p| p - prefix_len).collect(),
                    )
                })
                .collect();
            let baselines: Vec<_> = suffixes
                .iter()
                .map(|i| scalar(&mut backend, parent, i.clone()))
                .collect();
            let mut padded_work = ForkBatchWork::default();
            let padded = backend
                .forward_padded_fork_batch(parent, suffixes.clone(), &mut padded_work)
                .unwrap();
            assert_eq!(
                (
                    padded_work.cache_forks,
                    padded_work.forward_calls,
                    padded_work.batch_calls,
                    padded_work.padded_batch_calls
                ),
                (3, 1, 1, 1)
            );
            let physical = 3 * suffixes.iter().map(|i| i.tokens.len()).max().unwrap() as u64;
            assert_eq!(padded_work.processed_tokens, physical);
            assert_eq!(
                padded_work.padded_tokens,
                physical - suffixes.iter().map(|i| i.tokens.len() as u64).sum::<u64>()
            );
            for ((actual, baseline), row) in padded.iter().zip(baselines).zip(rows) {
                parity(actual.values().data(), &baseline, 1e-4);
                let p = calibrate(actual.values().data(), 2.40605).unwrap();
                let frozen: Vec<f32> =
                    serde_json::from_value(row["probabilities"].clone()).unwrap();
                assert_eq!(argmax(&p), argmax(&frozen));
                assert!(p.iter().zip(frozen).all(|(a, b)| (a - b).abs() <= 1e-3));
            }
            for row in rows {
                let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                let positions: Vec<usize> =
                    serde_json::from_value(row["positions"].clone()).unwrap();
                let independent = backend
                    .forward(ForwardInput::new(tokens.clone(), positions.clone()))
                    .unwrap();
                let suffix = ForwardInput::new(
                    tokens[prefix_len..].to_vec(),
                    positions.iter().map(|p| p - prefix_len).collect(),
                );
                let expected = scalar(&mut backend, parent, suffix.clone());
                let mut changed = suffix.clone();
                changed.tokens[0] = (changed.tokens[0] + 1) % 32;
                changed.positions.reverse();
                let changed_expected = scalar(&mut backend, parent, changed.clone());
                let mut work = ForkBatchWork::default();
                let out = backend
                    .forward_fork_batch(
                        parent,
                        vec![suffix.clone(), changed, suffix.clone()],
                        &mut work,
                    )
                    .unwrap();
                assert_eq!(
                    (work.cache_forks, work.forward_calls, work.batch_calls),
                    (3, 1, 1)
                );
                assert_eq!(work.processed_tokens, 3 * suffix.tokens.len() as u64);
                parity(out[0].values().data(), &expected, 1e-4);
                parity(out[1].values().data(), &changed_expected, 1e-4);
                parity(out[2].values().data(), independent.values().data(), 1e-4);
                let upstream: Vec<f32> =
                    serde_json::from_value(row["probabilities"].clone()).unwrap();
                let actual = calibrate(out[0].values().data(), 2.40605).unwrap();
                assert_eq!(argmax(&actual), argmax(&upstream));
                assert!(actual
                    .iter()
                    .zip(&upstream)
                    .all(|(a, b)| (a - b).abs() <= 1e-3));
                let held = out[0].values().data().to_vec();
                assert_eq!(scalar(&mut backend, parent, suffix.clone()), expected);
                // Repetition catches leaked temporary handles and stale output storage.
                for _ in 0..24 {
                    backend
                        .forward_fork_batch(
                            parent,
                            vec![suffix.clone(); 3],
                            &mut Default::default(),
                        )
                        .unwrap();
                }
                assert_eq!(out[0].values().data(), held);
            }
            backend.release_cache(parent).unwrap();
        }
    }
}
#[test]
fn cached_batches_reject_invalid_partial_foreign_and_exhausted_parents() {
    let mut backend = load("fp32", 16, false, false);
    let parent = backend.prefill(&[1, 2]).unwrap();
    let input = ForwardInput::new(vec![3, 4], vec![0, 1]);
    let expected = scalar(&mut backend, parent, input.clone());
    let other = load("fp32", 0, false, false).prefill(&[1, 2]).unwrap();
    for handle in [other, CacheHandle { id: u64::MAX }] {
        let mut work = ForkBatchWork::default();
        assert!(backend
            .forward_fork_batch(handle, vec![input.clone(); 2], &mut work)
            .is_err());
        assert_eq!(work.cache_forks + work.forward_calls, 0);
    }
    let mut bads = vec![
        ForwardInput::new(vec![], vec![]),
        ForwardInput::new(vec![3], vec![1]),
        ForwardInput::new(vec![u32::MAX, 4], vec![0]),
        ForwardInput::new(vec![3, 4], vec![]),
        ForwardInput::new(vec![3; 511], vec![0]),
        input.clone().with_logit_codes(vec![0]),
    ];
    let mut retained = input.clone();
    retained.retain_cache = true;
    bads.push(retained);
    let mut branched = input.clone();
    branched.fork_from = Some(parent);
    bads.push(branched);
    for bad in bads {
        let mut work = ForkBatchWork::default();
        assert!(backend
            .forward_fork_batch(parent, vec![input.clone(), bad], &mut work)
            .is_err());
        assert_eq!(work.cache_forks + work.forward_calls, 0);
        assert_eq!(scalar(&mut backend, parent, input.clone()), expected);
    }
    assert!(backend
        .forward_fork_batch(parent, vec![], &mut Default::default())
        .is_err());
    assert!(backend
        .forward_fork_batch(parent, vec![input.clone(); 64], &mut Default::default())
        .is_err());
    let handles: Vec<_> = (0..62).map(|_| backend.fork(parent).unwrap()).collect();
    assert!(backend
        .forward_fork_batch(parent, vec![input.clone(); 2], &mut Default::default())
        .is_err());
    for handle in handles {
        backend.release_cache(handle).unwrap();
    }
    let mut chunked = load("fp32", 16, true, false);
    let partial = chunked.begin_resumable_prefill(&[1; 10], 0).unwrap().handle;
    assert!(chunked
        .forward_fork_batch(partial, vec![input.clone(); 2], &mut Default::default())
        .is_err());
    chunked.release_cache(partial).unwrap();
    let replica = backend.replica().unwrap();
    assert!(replica.supports_fork_batch());
    assert!(backend
        .forward_fork_batch(parent, vec![input; 2], &mut Default::default())
        .is_ok());
    backend.release_cache(parent).unwrap();
}

#[cfg(feature = "clef")]
#[test]
fn engine_branch_batches_preserve_original_goldens_usage_budgets_and_nonvacuous_gates() {
    use huncho_core::{
        conformance::{run_suite_with_options, GoldenCase, GoldenSuite},
        engine::{Engine, EvalOptions, EvalStats},
        manifest::{BackendId, Family, HeadKind, ModelManifest},
        tokenizer::HfTokenizer,
    };
    let root = Path::new("tests/fixtures/tiny_kev");
    let source: serde_json::Value =
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
        let engine = Engine::new(
            manifest,
            Box::new(load(dtype, 16, true, false)),
            Box::new(HfTokenizer::from_file_unbounded(root.join("tokenizer.json")).unwrap()),
            Default::default(),
            BackendId::Candle,
            dtype,
        )
        .unwrap();
        let mut cases = Vec::new();
        for (n, case) in source["cases"].as_array().unwrap().iter().enumerate() {
            let mut request: huncho_core::contract::SystemOneRequest =
                serde_json::from_value(case["request"].clone()).unwrap();
            let mut expected = std::collections::BTreeMap::new();
            for ((id, q), row) in request
                .questions
                .clone()
                .iter()
                .zip(case["rows"].as_array().unwrap())
            {
                let tokenizer =
                    HfTokenizer::from_file_unbounded(root.join("tokenizer.json")).unwrap();
                let labels: Vec<_> = huncho_core::prompt::formatter_for(engine.manifest())
                    .build(&request.state, q, &tokenizer)
                    .unwrap()
                    .candidates
                    .into_iter()
                    .map(|c| c.label)
                    .collect();
                let p: Vec<f32> = serde_json::from_value(row["probabilities"].clone()).unwrap();
                let map: std::collections::BTreeMap<_, _> = labels.into_iter().zip(p).collect();
                expected.insert(id.clone(), map.clone());
                let duplicate = format!("{id}-duplicate");
                request.questions.insert(duplicate.clone(), q.clone());
                expected.insert(duplicate, map);
            }
            cases.push(GoldenCase {
                id: n.to_string(),
                request,
                expected,
                targets: Default::default(),
            });
        }
        let suite = GoldenSuite {
            schema_version: "1.0".into(),
            family: "F2".into(),
            hash: None,
            cases,
        };
        let opts = EvalOptions {
            prefix_cache: true,
            max_batch_tokens: Some(4096),
            extensions: true,
            ..Default::default()
        };
        let report = run_suite_with_options(&engine, &suite, &Default::default(), &opts).unwrap();
        assert!(report.passed, "{dtype}: {report:?}");
        assert_eq!(report.work.fork_batch_calls, 6);
        assert_eq!(report.work.cache_forks, 12);
        assert_eq!(report.work.cross_request_batches, 0);
        assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
        let request = &suite.cases[0].request;
        let mut stats = EvalStats::default();
        let out = engine
            .eval_uncached_with_stats(request, &opts, &mut stats)
            .unwrap();
        let reference = engine
            .eval(
                request,
                &EvalOptions {
                    extensions: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.usage.input_tokens, reference.usage.input_tokens);
        assert_eq!(stats.cache_forks, 6);
        assert_eq!(stats.fork_batch_calls, 3);
        assert_eq!(stats.forward_calls, 3);
        assert!(stats.processed_tokens < out.usage.input_tokens);
        assert_eq!(
            stats.processed_tokens + stats.reused_prefix_tokens,
            out.usage.input_tokens + 15
        ); // Original shared state is computed once.
        let rectangle = EvalOptions {
            max_batch_tokens: Some(70),
            ..opts.clone()
        };
        engine
            .eval_uncached_with_stats(request, &rectangle, &mut stats)
            .unwrap();
        // Two 43-token contexts cannot fit, even though 2*28 suffix tokens do.
        assert_eq!((stats.fork_batch_calls, stats.forward_calls), (1, 5));
        let tiny = EvalOptions {
            max_batch_tokens: Some(1),
            ..opts.clone()
        };
        engine
            .eval_uncached_with_stats(request, &tiny, &mut stats)
            .unwrap();
        assert_eq!((stats.fork_batch_calls, stats.forward_calls), (0, 6));
        assert!(
            run_suite_with_options(&engine, &suite, &Default::default(), &tiny)
                .unwrap_err()
                .to_string()
                .contains("actual native batch")
        );
        let retained = EvalOptions {
            persistent_prefix_bytes: 1 << 20,
            ..opts.clone()
        };
        let report =
            run_suite_with_options(&engine, &suite, &Default::default(), &retained).unwrap();
        assert!(report.passed);
        assert!(report.work.persistent_prefix_hits > 0);
        let mixed = EvalOptions {
            max_batch_padding_percent: 25,
            ..opts.clone()
        };
        let mut original = suite.clone();
        for case in &mut original.cases {
            case.request
                .questions
                .retain(|id, _| !id.ends_with("-duplicate"));
            case.expected.retain(|id, _| !id.ends_with("-duplicate"));
        }
        let report =
            run_suite_with_options(&engine, &original, &Default::default(), &mixed).unwrap();
        assert!(report.passed, "{dtype}: {report:?}");
        assert!(report.work.fork_padded_batch_calls > 0);
        assert!(report.work.padded_tokens > 0);
        assert!(report.optimization_parity.unwrap().max_prob_delta <= 1e-4);
        for invalid in [
            EvalOptions {
                cooperative_prefill: true,
                ..opts.clone()
            },
            EvalOptions {
                max_context: Some(10),
                ..opts.clone()
            },
        ] {
            assert!(engine
                .eval_uncached_with_stats(request, &invalid, &mut stats)
                .is_err());
            assert_eq!(
                stats.cache_forks + stats.forward_calls + stats.prefill_calls,
                0
            );
        }
        let replica = engine.replica().unwrap();
        assert!(replica.supports_fork_batch());
        assert!(
            run_suite_with_options(&replica, &suite, &Default::default(), &opts)
                .unwrap()
                .passed
        );
    }
}
