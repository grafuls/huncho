//! Real packed CPU kernels and immutable cache parity; no release acceptance.
#![cfg(feature = "quantization")]

use huncho_backend::qwen3_5::quantized::{convert_kev, Scheme};
use huncho_backend::Qwen3_5Backend;
use huncho_core::backend::{Backend, ForwardInput};
use huncho_core::calibration::{argmax, calibrate};
use std::path::Path;

fn assert_paired(actual: &[f32], expected: &[f32]) {
    for temperature in [0.75, 1.0, 2.40605] {
        let a = calibrate(actual, temperature).unwrap();
        let b = calibrate(expected, temperature).unwrap();
        assert_eq!(argmax(&a), argmax(&b));
        assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 1e-4));
    }
}

#[test]
fn durable_quantized_artifacts_run_packed_kernels_and_preserve_native_cache_and_batch_isolation() {
    let fixture = Path::new("tests/fixtures/tiny_kev");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.join("golden.json")).unwrap()).unwrap();
    for scheme in [Scheme::Q8_0, Scheme::Q4_0] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("backbone.gguf");
        let mut file = std::fs::File::create(&path).unwrap();
        let stats = convert_kev(fixture, fixture, scheme, &mut file).unwrap();
        assert_eq!(stats.projection_count, 15);
        assert!(stats.packed_projection_bytes < stats.source_projection_bytes);
        assert!(stats.dense_backbone_bytes > 0);
        let mut backend = Qwen3_5Backend::load_quantized_kev(
            &path,
            &fixture.join("head.pt"),
            512,
            scheme.dtype(),
        )
        .unwrap()
        .with_cpu_delta_rule(true)
        .unwrap()
        .with_cpu_causal_conv(true)
        .unwrap()
        .with_cpu_fused_gate(true)
        .unwrap();
        let metadata = backend.capabilities();
        assert_eq!(metadata.dtype, scheme.dtype());
        assert_eq!(metadata.extra["weight_quantization"], scheme.profile());
        assert_eq!(metadata.extra["pointer_head_dtype"], "fp32");
        assert_eq!(metadata.extra["activation_dtype"], "fp32");
        assert!(Qwen3_5Backend::load_quantized_kev(
            &path,
            &fixture.join("head.pt"),
            512,
            if scheme == Scheme::Q8_0 {
                "q4_0-fp32"
            } else {
                "q8_0-fp32"
            }
        )
        .is_err());
        let mut changed_logits = false;
        for case in golden["cases"].as_array().unwrap() {
            for row in case["rows"].as_array().unwrap() {
                let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                let positions: Vec<usize> =
                    serde_json::from_value(row["positions"].clone()).unwrap();
                let input = ForwardInput::new(tokens.clone(), positions.clone());
                let baseline = backend.forward(input.clone()).unwrap();
                let upstream: Vec<f32> = serde_json::from_value(row["raw_logits"].clone()).unwrap();
                changed_logits |= upstream != baseline.values().data();
                let probabilities = calibrate(baseline.values().data(), 2.40605).unwrap();
                assert!(probabilities
                    .iter()
                    .all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
                assert!((probabilities.iter().sum::<f32>() - 1.0).abs() <= 1e-5);
                // Packed independent execution is the scheduling reference;
                // this does not equate it to the upstream FP32 distribution.
                let batch = backend
                    .forward_batch(vec![input.clone(), input.clone()])
                    .unwrap();
                for output in batch {
                    assert_paired(output.values().data(), baseline.values().data());
                }
                let boundary = row["prefix_len"].as_u64().unwrap() as usize;
                let cached = backend
                    .prefill_cached(&tokens[..boundary], 1 << 20)
                    .unwrap();
                backend.release_cache(cached.handle).unwrap();
                let hit = backend
                    .prefill_cached(&tokens[..boundary], 1 << 20)
                    .unwrap();
                assert!(hit.hit);
                let branch = backend.fork(hit.handle).unwrap();
                backend.release_cache(hit.handle).unwrap();
                let mut suffix = ForwardInput::new(
                    tokens[boundary..].to_vec(),
                    positions.iter().map(|p| p - boundary).collect(),
                );
                suffix.fork_from = Some(branch);
                let forked = backend.forward(suffix).unwrap();
                assert_paired(forked.values().data(), baseline.values().data());
                backend.release_cache(branch).unwrap();
            }
        }
        assert!(
            changed_logits,
            "quantization must execute altered packed weights, not a dense fallback"
        );
        backend.clear_prefix_cache().unwrap();
        std::fs::write(&path, b"GGUF").unwrap();
        assert!(Qwen3_5Backend::load_quantized_kev(
            &path,
            &fixture.join("head.pt"),
            512,
            scheme.dtype()
        )
        .is_err());
    }
}
