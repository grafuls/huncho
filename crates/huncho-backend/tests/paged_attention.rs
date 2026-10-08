//! Direct CPU page execution keeps frozen typed scores and ownership bounds.
#![cfg(feature = "candle")]
use huncho_backend::Qwen3_5Backend;
use huncho_core::{
    backend::{Backend, CacheHandle, ForwardInput},
    calibration::{argmax, calibrate},
};
use std::path::Path;

fn load(pages: usize, optimized: bool, runtime: bool) -> Qwen3_5Backend {
    let root = Path::new("tests/fixtures/tiny_kev");
    let backend = if runtime {
        Qwen3_5Backend::load_kev_runtime_lora(root, root, &root.join("head.pt"), 512, "fp32")
    } else {
        Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, "fp32")
    }
    .unwrap();
    backend
        .with_cpu_blas_from_env()
        .unwrap()
        .with_kv_page_tokens(pages)
        .unwrap()
        .with_attention_query_rows(7)
        .unwrap()
        .with_direct_paged_attention(true)
        .unwrap()
        .with_prefill_chunk_tokens(if optimized { 3 } else { 0 })
        .unwrap()
        .with_cpu_delta_rule(optimized)
        .unwrap()
        .with_cpu_causal_conv(optimized)
        .unwrap()
        .with_cpu_fused_gate(optimized)
        .unwrap()
        .with_grouped_gqa(optimized)
        .unwrap()
}
fn parity(a: &[f32], b: &[f32]) {
    for temperature in [0.75, 1., 2.40605] {
        let a = calibrate(a, temperature).unwrap();
        let b = calibrate(b, temperature).unwrap();
        assert_eq!(argmax(&a), argmax(&b));
        assert!(
            a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= 1e-4),
            "{a:?} vs {b:?}"
        );
    }
}
fn branch(
    backend: &mut dyn Backend,
    parent: CacheHandle,
    tokens: &[u32],
    positions: Vec<usize>,
) -> Vec<f32> {
    let fork = backend.fork(parent).unwrap();
    let mut input = ForwardInput::new(tokens.to_vec(), positions);
    input.fork_from = Some(fork);
    let result = backend.forward(input).unwrap().values().data().to_vec();
    backend.release_cache(fork).unwrap();
    result
}

#[test]
fn direct_pages_keep_original_typed_goldens_across_boundaries_chunks_and_runtime_adapters() {
    let source: serde_json::Value =
        serde_json::from_slice(&std::fs::read("tests/fixtures/tiny_kev/golden.json").unwrap())
            .unwrap();
    for (pages, optimized, runtime) in [
        (16, false, false),
        (32, false, false),
        (64, false, false),
        (128, false, false),
        (256, false, false),
        (16, true, false),
        (16, true, true),
    ] {
        let mut backend = load(pages, optimized, runtime);
        assert_eq!(
            backend.capabilities().extra["paged_attention"],
            "cpu-page-qk-pv-fp32-v2"
        );
        for case in source["cases"].as_array().unwrap() {
            for row in case["rows"].as_array().unwrap() {
                let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                let positions: Vec<usize> =
                    serde_json::from_value(row["positions"].clone()).unwrap();
                let independent = backend
                    .forward(ForwardInput::new(tokens.clone(), positions.clone()))
                    .unwrap();
                for prefix in [1, 2, row["prefix_len"].as_u64().unwrap() as usize] {
                    let parent = backend.prefill(&tokens[..prefix]).unwrap();
                    for chunk in [1, 5, usize::MAX] {
                        let fork = backend.fork(parent).unwrap();
                        let mut offset = prefix;
                        while offset < positions[0] {
                            let end = offset.saturating_add(chunk).min(positions[0]);
                            let mut input = ForwardInput::new(tokens[offset..end].to_vec(), vec![]);
                            input.fork_from = Some(fork);
                            backend.forward(input).unwrap();
                            offset = end;
                        }
                        let mut input = ForwardInput::new(
                            tokens[offset..].to_vec(),
                            positions.iter().map(|p| p - offset).collect(),
                        );
                        input.fork_from = Some(fork);
                        let out = backend.forward(input).unwrap();
                        parity(out.values().data(), independent.values().data());
                        let actual = calibrate(out.values().data(), 2.40605).unwrap();
                        let frozen: Vec<f32> =
                            serde_json::from_value(row["probabilities"].clone()).unwrap();
                        assert_eq!(argmax(&actual), argmax(&frozen));
                        assert!(actual
                            .iter()
                            .zip(&frozen)
                            .all(|(a, b)| (a - b).abs() <= 1e-3));
                        backend.release_cache(fork).unwrap();
                    }
                    backend.release_cache(parent).unwrap();
                }
            }
        }
    }
}

#[test]
fn direct_pages_retain_transactional_bounds_snapshots_and_replica_isolation() {
    let mut backend = load(16, true, false);
    let cached = backend.prefill_cached(&[1; 33], 1 << 20).unwrap();
    let expected = branch(&mut backend, cached.handle, &[2, 3], vec![0, 1]);
    let fork = backend.fork(cached.handle).unwrap();
    let mut bad = ForwardInput::new(vec![u32::MAX], vec![0]);
    bad.fork_from = Some(fork);
    assert!(backend.forward(bad).is_err());
    let mut valid = ForwardInput::new(vec![2, 3], vec![0, 1]);
    valid.fork_from = Some(fork);
    parity(backend.forward(valid).unwrap().values().data(), &expected);
    backend.release_cache(fork).unwrap();
    backend.release_cache(cached.handle).unwrap();
    let hit = backend.prefill_cached(&[1; 33], 1 << 20).unwrap();
    assert!(hit.hit);
    let mut replica = backend.replica().unwrap();
    assert!(replica.fork(hit.handle).is_err());
    assert!(backend.with_direct_paged_attention(false).is_err());
    let mut backend = load(16, true, false);
    let parent = backend.prefill(&[1; 33]).unwrap();
    let inputs = vec![ForwardInput::new(vec![2, 3], vec![0, 1]); 2];
    let out = backend
        .forward_fork_batch(parent, inputs, &mut Default::default())
        .unwrap();
    parity(out[0].values().data(), &expected);
    backend.release_cache(parent).unwrap();
    // Replicas own their caches and survive dropping the primary context.
    drop(backend);
    let parent = replica.prefill(&[1; 33]).unwrap();
    parity(
        &branch(replica.as_mut(), parent, &[2, 3], vec![0, 1]),
        &expected,
    );
    replica.release_cache(parent).unwrap();
}

#[test]
fn direct_page_profile_rejects_missing_prerequisites_and_later_invalid_settings() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let raw = || Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, "fp32").unwrap();
    assert!(raw().with_direct_paged_attention(true).is_err());
    assert!(raw()
        .with_kv_page_tokens(16)
        .unwrap()
        .with_direct_paged_attention(true)
        .is_err());
    assert!(raw()
        .with_attention_query_rows(7)
        .unwrap()
        .with_direct_paged_attention(true)
        .is_err());
    assert!(
        Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, "fp16")
            .unwrap()
            .with_kv_page_tokens(16)
            .unwrap()
            .with_attention_query_rows(7)
            .unwrap()
            .with_direct_paged_attention(true)
            .is_err()
    );
    assert!(load(16, false, false).with_kv_page_tokens(0).is_err());
    assert!(load(16, false, false).with_attention_query_rows(0).is_err());
    let mut backend = load(16, true, false);
    let _partial = backend.begin_resumable_prefill(&[1; 33], 0).unwrap().handle;
    assert!(backend.with_direct_paged_attention(false).is_err());
    let backend = load(16, false, false);
    let _replica = backend.replica().unwrap();
    assert!(backend.with_direct_paged_attention(false).is_err());
    let mut backend = load(16, true, false);
    // Cleanup remains supported after cancellation.
    let partial = backend.begin_resumable_prefill(&[1; 33], 0).unwrap().handle;
    backend.release_cache(partial).unwrap();
    assert!(backend
        .with_direct_paged_attention(false)
        .unwrap()
        .with_kv_page_tokens(0)
        .is_ok());
}
