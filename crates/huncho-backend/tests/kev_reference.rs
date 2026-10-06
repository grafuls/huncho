//! End-to-end backbone + LoRA + pointer scoring against upstream PyTorch.
#![cfg(feature = "candle")]

use candle::Device;
use huncho_backend::{kev::KevMetadata, Qwen3_5Backend};
use huncho_core::backend::{Backend, ForwardInput};
use huncho_core::calibration::calibrate;
use huncho_core::manifest::Family;
use std::path::Path;

const FIXTURE: &str = "tests/fixtures/tiny_kev";

#[test]
fn kev_candle_matches_upstream_probabilities() {
    assert_matches_upstream(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a compatible CUDA GPU"]
fn kev_cuda_matches_upstream_probabilities() {
    let device = huncho_backend::device::device_from_env().unwrap();
    assert!(device.is_cuda(), "GPU reference test must run on CUDA");
    assert_matches_upstream(device);
}

fn assert_matches_upstream(device: Device) {
    let root = Path::new(FIXTURE);
    let meta = KevMetadata::load(&root.join("head.pt")).unwrap();
    assert_eq!(meta.base, "fixture/qwen3.5");
    assert_eq!(
        meta.base_revision.as_deref(),
        Some("1111111111111111111111111111111111111111")
    );
    assert!((meta.temperature - 2.40605).abs() < 1e-5);
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
        assert_eq!(backend.capabilities().families, vec![Family::F2]);
        for (case_idx, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
            for (row_idx, row) in case["rows"].as_array().unwrap().iter().enumerate() {
                let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
                let positions: Vec<usize> =
                    serde_json::from_value(row["positions"].clone()).unwrap();
                let output = backend
                    .forward(ForwardInput::new(tokens, positions.clone()))
                    .unwrap();
                assert_eq!(output.values().shape(), &[positions.len(), 1]);
                let expected: Vec<f32> =
                    serde_json::from_value(row["probabilities"].clone()).unwrap();
                let raw: Vec<f32> = serde_json::from_value(row["raw_logits"].clone()).unwrap();
                if dtype == "fp32" {
                    for (a, b) in output.values().data().iter().zip(&raw) {
                        assert!(
                            (a - b).abs() < 0.0002,
                            "raw logits differ: {:?} vs {raw:?}",
                            output.values().data()
                        );
                    }
                }
                let actual = calibrate(output.values().data(), meta.temperature).unwrap();
                let tolerance = if dtype == "fp32" { 2e-5 } else { 0.002 };
                for (a, b) in actual.iter().zip(&expected) {
                    assert!((a - b).abs() < tolerance, "{dtype} case {case_idx} row {row_idx}: actual={actual:?}, expected={expected:?}");
                }
            }
        }
        // A short row previously underflowed in the causal convolution.
        let output = backend
            .forward(ForwardInput::new(vec![1, 2], vec![0]))
            .unwrap();
        assert!(output.values().data().iter().all(|x| x.is_finite()));
        if dtype == "fp32" {
            assert!(
                (output.values().data()[0] - golden["short_logits"][0].as_f64().unwrap() as f32)
                    .abs()
                    < 0.0002
            );
        }
    }
}

// Real Kev base/adapter checkpoints can contain BF16 even when inference uses
// FP16. Exercise this path on CUDA (including pre-Ampere GPUs without BF16).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a compatible CUDA GPU"]
fn kev_bf16_checkpoint_runs_as_fp16_on_cuda() {
    let device = huncho_backend::device::device_from_env().unwrap();
    assert!(device.is_cuda());
    let root = Path::new(FIXTURE);
    let tmp = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        let dest = tmp.path().join(path.file_name().unwrap());
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            let tensors = candle::safetensors::load(&path, &Device::Cpu)
                .unwrap()
                .into_iter()
                .map(|(name, t)| (name, t.to_dtype(candle::DType::BF16).unwrap()))
                .collect::<std::collections::HashMap<_, _>>();
            candle::safetensors::save(&tensors, &dest).unwrap();
        } else if path.is_file() {
            std::fs::copy(&path, &dest).unwrap();
        }
    }
    let mut cpu = Qwen3_5Backend::load_kev(
        tmp.path(),
        tmp.path(),
        &tmp.path().join("head.pt"),
        512,
        "fp16",
    )
    .unwrap();
    let mut gpu = Qwen3_5Backend::load_kev_on_device(
        tmp.path(),
        tmp.path(),
        &tmp.path().join("head.pt"),
        512,
        "fp16",
        device,
    )
    .unwrap();
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    for case in golden["cases"].as_array().unwrap() {
        for row in case["rows"].as_array().unwrap() {
            let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
            let positions: Vec<usize> = serde_json::from_value(row["positions"].clone()).unwrap();
            let expected = cpu
                .forward(ForwardInput::new(tokens.clone(), positions.clone()))
                .unwrap();
            let actual = gpu.forward(ForwardInput::new(tokens, positions)).unwrap();
            let temperature = KevMetadata::load(&root.join("head.pt"))
                .unwrap()
                .temperature;
            let expected = calibrate(expected.values().data(), temperature).unwrap();
            let actual = calibrate(actual.values().data(), temperature).unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert!((a - b).abs() < 0.002, "GPU={actual:?}, CPU={expected:?}");
            }
        }
    }
}
