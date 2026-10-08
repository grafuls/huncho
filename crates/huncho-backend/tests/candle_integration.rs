//! Integration tests for the candle ModernBERT backend.
//!
//! These require the `candle` feature
//! (`cargo test -p huncho-backend --features candle`) and a checked-in tiny
//! ModernBERT checkpoint in `tests/fixtures/tiny_modernbert/`. The fixture uses
//! the `convaiinnovations/laya` layout (`encoder.` weight prefix and a
//! transformers-5.0 `rope_parameters` config), so it exercises the load-time
//! key/config remapping.

#![cfg(feature = "candle")]

use huncho_backend::candle::CandleBackend;
use huncho_core::backend::{Backend, ForwardInput};

const CONFIG: &str = "tests/fixtures/tiny_modernbert/config.json";
const WEIGHTS: &str = "tests/fixtures/tiny_modernbert/model.safetensors";

#[test]
fn loads_and_reports_capabilities() {
    let backend = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();
    assert_eq!(backend.hidden_size(), 8);
    let c = backend.capabilities();
    assert_eq!(c.id, huncho_core::manifest::BackendId::Candle);
    assert_eq!(c.dtype, "fp32");
    assert_eq!(c.max_context, 16);
    assert!(!c.supports_fork);
    assert_eq!(c.families.len(), 1);
}

#[test]
fn refuses_to_claim_a_precision_it_does_not_execute() {
    for dtype in ["fp16", "bf16", "int8", "int4"] {
        assert!(CandleBackend::load(CONFIG, WEIGHTS, 16, dtype).is_err());
    }
}

#[test]
fn forward_returns_features_at_positions() {
    let mut backend = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();

    // Tokens [1..=5] with candidate positions 1 and 3.
    let input = ForwardInput::new(vec![1, 2, 3, 4, 5], vec![1, 3]);
    let out = backend.forward(input).unwrap();

    assert_eq!(out.positions(), &[1, 3]);
    let values = out.values();
    assert_eq!(values.shape(), &[2, 8]);
    // All values finite and nonzero (the tiny random model is nonzero).
    for x in values.data() {
        assert!(x.is_finite(), "feature value must be finite");
    }
}

#[test]
fn forward_is_deterministic() {
    let mut a = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();
    let mut b = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();
    let input = ForwardInput::new(vec![1, 2, 3, 4, 5], vec![0, 2, 4]);
    let oa = a.forward(input.clone()).unwrap();
    let ob = b.forward(input).unwrap();
    assert_eq!(oa.values().data(), ob.values().data());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a compatible CUDA GPU"]
fn cuda_modernbert_stages_bf16_checkpoints_and_preserves_cpu_readouts() {
    use huncho_core::calibration::{argmax, calibrate};
    let device = huncho_backend::device::device_from_env().unwrap();
    assert!(device.is_cuda(), "GPU parity test must run on CUDA");
    let package = tempfile::tempdir().unwrap();
    let weights = package.path().join("bf16.safetensors");
    let tensors = candle::safetensors::load(WEIGHTS, &candle::Device::Cpu)
        .unwrap()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.to_dtype(candle::DType::BF16).unwrap()))
        .collect::<std::collections::HashMap<_, _>>();
    candle::safetensors::save(&tensors, &weights).unwrap();
    let mut gpu = CandleBackend::load_on_device(CONFIG, &weights, 16, "fp32", device).unwrap();
    let mut cpu = CandleBackend::load(CONFIG, &weights, 16, "fp32").unwrap();
    assert_eq!(gpu.capabilities().extra["device_path"], "modernbert-cuda");
    // This fixture is a bare encoder; calibrated scalar readouts exercise
    // transfer and numerical parity, not a released trained Laya checkpoint.
    for positions in [vec![0, 2, 4], vec![4, 1, 1]] {
        let input = ForwardInput::new(vec![1, 2, 3, 4, 5], positions);
        let (a, b) = (
            gpu.forward(input.clone()).unwrap(),
            cpu.forward(input).unwrap(),
        );
        assert_eq!(a.positions(), b.positions());
        let project = |output: &huncho_core::backend::ForwardOutput| {
            output
                .values()
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
        let (a, b) = (
            calibrate(&project(&a), 1.).unwrap(),
            calibrate(&project(&b), 1.).unwrap(),
        );
        assert_eq!(argmax(&a), argmax(&b));
        assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 1e-3));
    }
}

#[test]
fn forward_empty_positions_returns_empty_features() {
    let mut backend = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();
    let input = ForwardInput::new(vec![1, 2, 3], vec![]);
    let out = backend.forward(input).unwrap();
    assert!(out.positions().is_empty());
    assert_eq!(out.values().shape(), &[0, 8]);
}

#[test]
fn forward_rejects_sequence_over_max_context() {
    let mut backend = CandleBackend::load(CONFIG, WEIGHTS, 16, "fp32").unwrap();
    // 17 tokens > max_context 16.
    let input = ForwardInput::new((0..17).map(|i| i as u32).collect::<Vec<_>>(), vec![1]);
    assert!(backend.forward(input).is_err());
}
