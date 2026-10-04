//! End-to-end backbone + LoRA + pointer scoring against upstream PyTorch.
#![cfg(feature = "candle")]

use huncho_backend::{kev::KevMetadata, Qwen3_5Backend};
use huncho_core::backend::{Backend, ForwardInput};
use huncho_core::calibration::calibrate;
use huncho_core::manifest::Family;
use std::path::Path;

const FIXTURE: &str = "tests/fixtures/tiny_kev";

#[test]
fn kev_candle_matches_upstream_probabilities() {
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
        let mut backend =
            Qwen3_5Backend::load_kev(root, root, &root.join("head.pt"), 512, dtype).unwrap();
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
