//! CPU attention query blocks preserve all causal keys and trained readouts.
#![cfg(feature = "candle")]
use huncho_backend::Qwen3_5Backend;
use huncho_core::{
    backend::{Backend, ForwardInput},
    calibration::{argmax, calibrate},
};
use std::path::Path;
fn parity(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let a = calibrate(actual, 2.40605).unwrap();
    let b = calibrate(expected, 2.40605).unwrap();
    assert_eq!(argmax(&a), argmax(&b));
    assert!(
        a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= 1e-4),
        "{a:?} vs {b:?}"
    );
}
#[test]
fn grouped_gqa_profile_cannot_change_shared_or_partial_prefix_arithmetic() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let fresh = || {
        Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, "fp32")
            .unwrap()
            .with_grouped_gqa(true)
            .unwrap()
    };
    let backend = fresh();
    let replica = backend.replica().unwrap();
    assert!(backend.with_grouped_gqa(false).is_err());
    drop(replica);
    let mut backend = fresh().with_prefill_chunk_tokens(3).unwrap();
    backend
        .begin_resumable_prefill(&[1, 2, 3, 4, 5], 0)
        .unwrap();
    assert!(backend.with_grouped_gqa(false).is_err());
    let mut backend = fresh();
    let cached = backend.prefill_cached(&[1, 2, 3], 1 << 20).unwrap();
    backend.release_cache(cached.handle).unwrap();
    backend.clear_prefix_cache().unwrap();
    let backend = backend.with_grouped_gqa(false).unwrap();
    assert!(!backend.capabilities().extra.contains_key("gqa_execution"));
}
#[test]
fn query_blocks_preserve_frozen_pointer_probabilities_prefixes_and_padded_batches() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for dtype in ["fp32", "fp16"] {
        let mut reference =
            Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype).unwrap();
        for (rows, grouped) in [
            (1, false),
            (3, false),
            (7, false),
            (64, false),
            (0, true),
            (3, true),
        ] {
            let mut blocked =
                Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype)
                    .unwrap()
                    .with_attention_query_rows(rows)
                    .unwrap()
                    .with_grouped_gqa(grouped)
                    .unwrap();
            if rows > 0 {
                assert_eq!(
                    blocked.capabilities().extra["attention_execution"],
                    "cpu-query-blocks-v1"
                );
            }
            if grouped {
                assert_eq!(
                    blocked.capabilities().extra["gqa_execution"],
                    "cpu-grouped-queries-v1"
                );
            } else {
                assert!(!blocked.capabilities().extra.contains_key("gqa_execution"));
            }
            let mut replica = blocked.replica().unwrap();
            for case in golden["cases"].as_array().unwrap() {
                for row in case["rows"].as_array().unwrap() {
                    let input = ForwardInput::new(
                        serde_json::from_value(row["tokens"].clone()).unwrap(),
                        serde_json::from_value(row["positions"].clone()).unwrap(),
                    );
                    let independent = reference.forward(input.clone()).unwrap();
                    let actual = blocked.forward(input.clone()).unwrap();
                    parity(actual.values().data(), independent.values().data());
                    let p: Vec<f32> = serde_json::from_value(row["probabilities"].clone()).unwrap();
                    assert!(calibrate(actual.values().data(), 2.40605)
                        .unwrap()
                        .iter()
                        .zip(p)
                        .all(|(a, b)| (a - b).abs() <= 1e-4));
                    assert_eq!(
                        actual.values().data(),
                        replica.forward(input.clone()).unwrap().values().data()
                    );
                    if input.positions.iter().all(|&p| p >= 7) {
                        let parent = blocked.prefill(&input.tokens[..7]).unwrap();
                        let fork = blocked.fork(parent).unwrap();
                        blocked.release_cache(parent).unwrap();
                        let mut suffix = ForwardInput::new(
                            input.tokens[7..].to_vec(),
                            input.positions.iter().map(|p| p - 7).collect(),
                        );
                        suffix.fork_from = Some(fork);
                        parity(
                            blocked.forward(suffix).unwrap().values().data(),
                            independent.values().data(),
                        );
                        blocked.release_cache(fork).unwrap();
                    }
                }
            }
            let inputs: Vec<_> = [5, 13, 47]
                .into_iter()
                .enumerate()
                .map(|(row, n)| {
                    ForwardInput::new(
                        (0..n)
                            .map(|i| ((i * 19 + row * 7 + 1) % 380) as u32)
                            .collect(),
                        vec![n - 1, 1, n - 1],
                    )
                })
                .collect();
            let expected: Vec<_> = inputs
                .iter()
                .map(|i| reference.forward(i.clone()).unwrap())
                .collect();
            for (a, b) in blocked
                .forward_padded_batch(inputs.clone())
                .unwrap()
                .iter()
                .zip(expected)
            {
                parity(a.values().data(), b.values().data());
            }
            let whole = ForwardInput::new(
                (0..512).map(|i| (i % 380) as u32).collect(),
                vec![511, 300, 511],
            );
            parity(
                blocked.forward(whole.clone()).unwrap().values().data(),
                reference.forward(whole).unwrap().values().data(),
            );
            drop(replica);
            let parent = blocked.prefill(&[1, 2, 3]).unwrap();
            assert!(blocked.with_attention_query_rows(rows + 1).is_err());
            // Consuming a failed configuration drops all owned state; a fresh
            // context remains configurable only before replicas/active prefixes.
            let blocked =
                Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype).unwrap();
            assert!(blocked.with_attention_query_rows(4097).is_err());
            let _ = parent;
        }
    }
}
#[test]
fn query_blocks_preserve_selected_candidate_logits() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let temp = tempfile::tempdir().unwrap();
    std::fs::copy(root.join("config.json"), temp.path().join("config.json")).unwrap();
    let mut tensors =
        candle::safetensors::load(root.join("model.safetensors"), &candle::Device::Cpu).unwrap();
    let embedding = tensors
        .iter()
        .find(|(name, _)| name.ends_with("embed_tokens.weight"))
        .unwrap()
        .1
        .clone();
    tensors.insert("lm_head.weight".into(), embedding);
    candle::safetensors::save(&tensors, temp.path().join("model.safetensors")).unwrap();
    for dtype in ["fp32", "fp16"] {
        let mut reference = Qwen3_5Backend::load(temp.path(), Some(root), 512, dtype).unwrap();
        for (rows, grouped) in [(3, false), (0, true), (3, true)] {
            let mut blocked = Qwen3_5Backend::load(temp.path(), Some(root), 512, dtype)
                .unwrap()
                .with_attention_query_rows(rows)
                .unwrap()
                .with_grouped_gqa(grouped)
                .unwrap();
            let mut input = ForwardInput::new(vec![1, 31, 14, 63, 2, 4, 5, 11, 15], vec![8, 1, 8]);
            input.logit_codes = Some(vec![37, 36, 37]);
            let expected = reference.forward(input.clone()).unwrap();
            let actual = blocked.forward(input).unwrap();
            assert_eq!(actual.positions(), expected.positions());
            parity(actual.values().data(), expected.values().data());
        }
    }
}
#[cfg(feature = "clef")]
#[test]
fn query_blocks_preserve_complete_joint_schema_heads_and_usage() {
    use huncho_backend::ClefBackend;
    let root = Path::new("tests/fixtures/tiny_clef");
    let manifest =
        huncho_core::manifest::ModelManifest::load(root.join("huncho-model.json")).unwrap();
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for dtype in ["fp32", "fp16"] {
        let mut baseline = ClefBackend::load(root, &manifest, dtype, candle::Device::Cpu).unwrap();
        for (rows, grouped) in [(7, false), (0, true), (7, true)] {
            let mut blocked = ClefBackend::load(root, &manifest, dtype, candle::Device::Cpu)
                .unwrap()
                .with_attention_query_rows(rows)
                .unwrap()
                .with_grouped_gqa(grouped)
                .unwrap();
            for case in golden["cases"].as_array().unwrap() {
                let req = serde_json::from_value(case["request"].clone()).unwrap();
                let expected = baseline.forward_request(&req, 4096).unwrap();
                let actual = blocked.forward_request(&req, 4096).unwrap();
                assert_eq!(actual.input_tokens, expected.input_tokens);
                assert_eq!(
                    actual.logits.keys().collect::<Vec<_>>(),
                    expected.logits.keys().collect::<Vec<_>>()
                );
                for (id, logits) in actual.logits {
                    assert_eq!(
                        logits.keys().collect::<Vec<_>>(),
                        expected.logits[&id].keys().collect::<Vec<_>>()
                    );
                    parity(
                        &logits.values().copied().collect::<Vec<_>>(),
                        &expected.logits[&id].values().copied().collect::<Vec<_>>(),
                    );
                }
            }
        }
    }
}
