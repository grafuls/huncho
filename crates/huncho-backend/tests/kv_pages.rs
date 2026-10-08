//! CPU page storage must preserve frozen probabilities and exact flat-cache scores.
#![cfg(feature = "candle")]
use huncho_backend::Qwen3_5Backend;
use huncho_core::{
    backend::{Backend, CacheHandle, ForwardInput, PrefillWork},
    calibration::{argmax, calibrate},
};
use std::path::Path;

fn load(dtype: &str, pages: usize) -> Qwen3_5Backend {
    let root = Path::new("tests/fixtures/tiny_kev");
    Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype)
        .unwrap()
        .with_kv_page_tokens(pages)
        .unwrap()
}
fn suffix(tokens: &[u32], positions: Vec<usize>, handle: CacheHandle) -> ForwardInput {
    let mut input = ForwardInput::new(tokens.to_vec(), positions);
    input.fork_from = Some(handle);
    input
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn parity(a: &[f32], b: &[f32], t: f32, bound: f32) {
    let a = calibrate(a, t).unwrap();
    let b = calibrate(b, t).unwrap();
    assert_eq!(argmax(&a), argmax(&b));
    assert!(
        a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= bound),
        "{a:?} vs {b:?}"
    );
}

#[test]
fn native_pages_forks_and_chunks_match_flat_storage_and_frozen_typed_goldens() {
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read("tests/fixtures/tiny_kev/golden.json").unwrap())
            .unwrap();
    for dtype in ["fp32", "fp16"] {
        for page_tokens in [16, 32, 64, 256] {
            let mut flat = load(dtype, 0);
            let mut paged = load(dtype, page_tokens);
            assert_eq!(
                paged.capabilities().extra["kv_storage"],
                "cpu-cow-pages-materialize-v1"
            );
            for case in golden["cases"].as_array().unwrap() {
                let rows = case["rows"].as_array().unwrap();
                let first: Vec<u32> = serde_json::from_value(rows[0]["tokens"].clone()).unwrap();
                let shared = rows[0]["prefix_len"].as_u64().unwrap() as usize;
                for prefix in [1, 2, shared] {
                    let flat_parent = flat.prefill(&first[..prefix]).unwrap();
                    let page_parent = paged.prefill(&first[..prefix]).unwrap();
                    for row in rows {
                        let tokens: Vec<u32> =
                            serde_json::from_value(row["tokens"].clone()).unwrap();
                        let positions: Vec<usize> =
                            serde_json::from_value(row["positions"].clone()).unwrap();
                        let reference: Vec<f32> =
                            serde_json::from_value(row["probabilities"].clone()).unwrap();
                        let independent = flat
                            .forward(ForwardInput::new(tokens.clone(), positions.clone()))
                            .unwrap();
                        for chunk in [1, 3, page_tokens, usize::MAX] {
                            let f = flat.fork(flat_parent).unwrap();
                            let p = paged.fork(page_parent).unwrap();
                            let mut offset = prefix;
                            while offset < positions[0] {
                                let end = offset.saturating_add(chunk).min(positions[0]);
                                flat.forward(suffix(&tokens[offset..end], vec![], f))
                                    .unwrap();
                                paged
                                    .forward(suffix(&tokens[offset..end], vec![], p))
                                    .unwrap();
                                offset = end;
                            }
                            let markers: Vec<_> = positions.iter().map(|n| n - offset).collect();
                            let expected = flat
                                .forward(suffix(&tokens[offset..], markers.clone(), f))
                                .unwrap();
                            let actual = paged
                                .forward(suffix(&tokens[offset..], markers, p))
                                .unwrap();
                            assert_eq!(
                                bits(actual.values().data()),
                                bits(expected.values().data()),
                                "{dtype}/{page_tokens}/{prefix}/{chunk}"
                            );
                            for t in [0.75, 1., 2.40605] {
                                parity(
                                    actual.values().data(),
                                    independent.values().data(),
                                    t,
                                    1e-4,
                                );
                            }
                            let probs = calibrate(actual.values().data(), 2.40605).unwrap();
                            assert_eq!(argmax(&probs), argmax(&reference));
                            assert!(probs
                                .iter()
                                .zip(&reference)
                                .all(|(a, b)| (a - b).abs() <= 1e-3));
                            flat.release_cache(f).unwrap();
                            paged.release_cache(p).unwrap();
                        }
                    }
                    flat.release_cache(flat_parent).unwrap();
                    paged.release_cache(page_parent).unwrap();
                }
            }
        }
    }
}

#[test]
fn pages_are_bounded_transactional_and_compose_with_retention_and_partial_prefills() {
    for dtype in ["fp32", "fp16"] {
        let mut paged = load(dtype, 16).with_prefill_chunk_tokens(3).unwrap();
        let mut flat = load(dtype, 0).with_prefill_chunk_tokens(3).unwrap();
        let prefix: Vec<u32> = (0..35).map(|n| 1 + n % 30).collect();
        let all: Vec<u32> = prefix.iter().copied().chain([3, 4]).collect();
        let expected = flat.forward(ForwardInput::new(all, vec![35, 36])).unwrap();
        for iteration in 0..3 {
            let cached = paged.prefill_cached(&prefix, 1 << 20).unwrap();
            assert_eq!(cached.hit, iteration > 0);
            let branch = paged.fork(cached.handle).unwrap();
            paged.release_cache(cached.handle).unwrap();
            if iteration == 2 {
                paged.clear_prefix_cache().unwrap();
            }
            // Invalid continuations cannot publish a changed tail or offset.
            assert!(paged.forward(suffix(&[u32::MAX], vec![0], branch)).is_err());
            assert!(paged.forward(suffix(&[3], vec![1], branch)).is_err());
            assert!(paged.forward(suffix(&[1; 478], vec![0], branch)).is_err());
            let actual = paged.forward(suffix(&[3, 4], vec![0, 1], branch)).unwrap();
            parity(
                actual.values().data(),
                expected.values().data(),
                2.40605,
                1e-4,
            );
            paged.release_cache(branch).unwrap();
        }
        for _ in 0..2 {
            let too_small = paged.prefill_cached(&prefix, 1).unwrap();
            assert!(!too_small.hit);
            paged.release_cache(too_small.handle).unwrap();
        }
        let partial = paged.begin_resumable_prefill(&prefix, 1 << 20).unwrap();
        let mut work = PrefillWork::default();
        assert!(!paged
            .advance_resumable_prefill(partial.handle, &mut work)
            .unwrap());
        assert!(paged.fork(partial.handle).is_err());
        paged.release_cache(partial.handle).unwrap();
        assert!(paged
            .advance_resumable_prefill(partial.handle, &mut work)
            .is_err());
        let completed = paged.begin_resumable_prefill(&prefix, 1 << 20).unwrap();
        while !paged
            .advance_resumable_prefill(completed.handle, &mut work)
            .unwrap()
        {}
        let branch = paged.fork(completed.handle).unwrap();
        let actual = paged.forward(suffix(&[3, 4], vec![0, 1], branch)).unwrap();
        parity(
            actual.values().data(),
            expected.values().data(),
            2.40605,
            1e-4,
        );
        paged.release_cache(completed.handle).unwrap();
        paged.release_cache(branch).unwrap();
        assert!(paged.with_kv_page_tokens(32).is_err()); // retained snapshot
        let mut paged = load(dtype, 16);
        let parent = paged.prefill(&prefix).unwrap();
        let mut handles = vec![parent];
        for _ in 1..64 {
            handles.push(paged.fork(parent).unwrap());
        }
        assert!(paged.fork(parent).is_err());
        for h in handles {
            paged.release_cache(h).unwrap();
        }
        let replica = paged.replica().unwrap();
        assert_eq!(replica.capabilities().extra["kv_page_tokens"], "16");
        assert!(paged.with_kv_page_tokens(32).is_err()); // shared immutable model
    }
    for invalid in [1, 15, 17, 257] {
        assert!(load("fp32", 0).with_kv_page_tokens(invalid).is_err());
    }
    let mut paged = load("fp32", 16);
    let partial = paged.with_prefill_chunk_tokens(3).unwrap();
    paged = partial;
    paged.begin_resumable_prefill(&[1, 2, 3, 4], 0).unwrap();
    assert!(paged.with_kv_page_tokens(0).is_err());
    assert!(load("fp32", 16).with_kv_page_tokens(0).is_ok());
}
