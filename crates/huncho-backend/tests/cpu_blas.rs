//! Optional real CPU BLAS tests. Runtime/library/host identities require fresh
//! labeled qualification; frozen tiny fixtures establish numerical plumbing.
#![cfg(all(feature = "clef", feature = "cpu-blas"))]
use huncho_backend::{ClefBackend, Qwen3_5Backend};
use huncho_core::{
    backend::{Backend, ForwardInput},
    calibration::{argmax, calibrate},
    manifest::ModelManifest,
};
use std::path::Path;

fn parity(a: &[f32], b: &[f32], gate: f32) {
    assert_eq!(a.len(), b.len());
    for temperature in [0.75, 1., 2.40605] {
        let a = calibrate(a, temperature).unwrap();
        let b = calibrate(b, temperature).unwrap();
        assert_eq!(argmax(&a), argmax(&b));
        assert!(
            a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= gate),
            "{a:?} vs {b:?}"
        );
    }
}
#[test]
#[ignore = "requires explicit HUNCHO_CPU_BLAS_LIBRARY pointing to LP64 pthread OpenBLAS"]
fn native_blas_kev_preserves_frozen_probabilities_prefix_batch_and_replica_identity() {
    let root = Path::new("tests/fixtures/tiny_kev");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let load =
        |dtype| Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype).unwrap();
    assert!(load("fp16").with_cpu_blas_from_env().is_err());
    let mut original = load("fp32");
    let mut blas = load("fp32")
        .with_cpu_blas_from_env()
        .unwrap()
        .with_prefill_chunk_tokens(3)
        .unwrap();
    let extra = blas.capabilities().extra;
    assert_eq!(extra["cpu_blas_execution"], "openblas-lp64-fp32-v1");
    assert_eq!(extra["cpu_blas_library_sha256"].len(), 64);
    assert_eq!(extra["cpu_blas_threads"], "1");
    let mut replica = blas.replica().unwrap();
    assert_eq!(replica.capabilities().extra, extra);
    for case in reference["cases"].as_array().unwrap() {
        for row in case["rows"].as_array().unwrap() {
            let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
            let positions: Vec<usize> = serde_json::from_value(row["positions"].clone()).unwrap();
            let expected: Vec<f32> = serde_json::from_value(row["probabilities"].clone()).unwrap();
            let input = ForwardInput::new(tokens.clone(), positions.clone());
            let base = original.forward(input.clone()).unwrap();
            let actual = blas.forward(input.clone()).unwrap();
            parity(base.values().data(), actual.values().data(), 1e-3);
            let probabilities = calibrate(actual.values().data(), 2.40605).unwrap();
            assert!(probabilities
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() <= 1e-3));
            let other = replica.forward(input.clone()).unwrap();
            assert_eq!(actual.values().data(), other.values().data());
            let a = blas.replica().unwrap();
            let b = blas.replica().unwrap();
            let concurrent = std::thread::scope(|scope| {
                let left = input.clone();
                let right = input.clone();
                let a = scope.spawn(move || {
                    let mut a = a;
                    a.forward(left).unwrap()
                });
                let b = scope.spawn(move || {
                    let mut b = b;
                    b.forward(right).unwrap()
                });
                (a.join().unwrap(), b.join().unwrap())
            });
            assert_eq!(concurrent.0.values().data(), actual.values().data());
            assert_eq!(concurrent.1.values().data(), actual.values().data());
            for batch in blas
                .forward_batch(vec![input.clone(), input.clone()])
                .unwrap()
            {
                parity(batch.values().data(), actual.values().data(), 1e-4);
            }
            let prefix = row["prefix_len"].as_u64().unwrap() as usize;
            let parent = blas.prefill(&tokens[..prefix]).unwrap();
            let branch = blas.fork(parent).unwrap();
            let mut suffix = ForwardInput::new(
                tokens[prefix..].to_vec(),
                positions.iter().map(|p| p - prefix).collect(),
            );
            suffix.fork_from = Some(branch);
            let cached = blas.forward(suffix).unwrap();
            parity(cached.values().data(), actual.values().data(), 1e-4);
            blas.release_cache(parent).unwrap();
            blas.release_cache(branch).unwrap();
        }
    }
    // Configuration cannot move under live shared model storage or prefixes.
    assert!(blas.with_cpu_blas_from_env().is_err());
    let mut retained = load("fp32");
    let parent = retained.prefill(&[1, 2]).unwrap();
    assert!(retained.with_cpu_blas_from_env().is_err());
    let _ = parent;
}

#[test]
#[ignore = "requires explicit HUNCHO_CPU_BLAS_LIBRARY pointing to LP64 pthread OpenBLAS"]
fn native_blas_clef_keeps_joint_heads_and_f3_candidate_selection() {
    let root = Path::new("tests/fixtures/tiny_clef");
    let manifest = ModelManifest::load(root.join("huncho-model.json")).unwrap();
    assert!(
        ClefBackend::load(root, &manifest, "fp16", candle::Device::Cpu)
            .unwrap()
            .with_cpu_blas_from_env()
            .is_err()
    );
    let mut original = ClefBackend::load(root, &manifest, "fp32", candle::Device::Cpu).unwrap();
    let mut blas = ClefBackend::load(root, &manifest, "fp32", candle::Device::Cpu)
        .unwrap()
        .with_cpu_blas_from_env()
        .unwrap();
    assert_eq!(
        blas.capabilities().extra["cpu_blas_execution"],
        "openblas-lp64-fp32-v1"
    );
    let mut replica = blas.replica().unwrap();
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for case in golden["cases"].as_array().unwrap() {
        let request = serde_json::from_value(case["request"].clone()).unwrap();
        let expected = original.forward_request(&request, 4096).unwrap();
        let actual = blas.forward_request(&request, 4096).unwrap();
        assert_eq!(actual.input_tokens, expected.input_tokens);
        assert_eq!(
            replica.forward_request(&request, 4096).unwrap().logits,
            actual.logits
        );
        for (id, logits) in actual.logits {
            parity(
                &logits.values().copied().collect::<Vec<_>>(),
                &expected.logits[&id].values().copied().collect::<Vec<_>>(),
                1e-3,
            );
        }
    }
    let root = Path::new("tests/fixtures/tiny_kev");
    let package = tempfile::tempdir().unwrap();
    std::fs::copy(root.join("config.json"), package.path().join("config.json")).unwrap();
    let mut tensors =
        candle::safetensors::load(root.join("model.safetensors"), &candle::Device::Cpu).unwrap();
    let embedding = tensors
        .iter()
        .find(|(name, _)| name.ends_with("embed_tokens.weight"))
        .unwrap()
        .1
        .clone();
    tensors.insert("lm_head.weight".into(), embedding);
    candle::safetensors::save(&tensors, package.path().join("model.safetensors")).unwrap();
    let mut baseline = Qwen3_5Backend::load(package.path(), Some(root), 512, "fp32").unwrap();
    let mut candidate = Qwen3_5Backend::load(package.path(), Some(root), 512, "fp32")
        .unwrap()
        .with_cpu_blas_from_env()
        .unwrap();
    let input = ForwardInput::new(vec![1, 2, 3, 4, 5], vec![4]).with_logit_codes(vec![31, 7, 383]);
    let expected = baseline.forward(input.clone()).unwrap();
    let actual = candidate.forward(input).unwrap();
    parity(actual.values().data(), expected.values().data(), 1e-3);
}
