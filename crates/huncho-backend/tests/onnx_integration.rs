//! Integration tests for the ONNX Runtime backend.
//!
//! These require the `onnx` feature (`cargo test -p huncho-backend --features onnx`)
//! and a checked-in tiny deterministic encoder fixture. The fixture is a single
//! `Gather` embedding that mimics the offline `MockBackend` hidden-state rule:
//! for a token `t`, `emb[t][t % hidden] = t + 0.5`. This lets us prove the ONNX
//! backend returns the same features the reference produces.

#![cfg(feature = "onnx")]

use huncho_backend::{
    onnx::{OnnxExecutionProvider, OnnxOptions},
    OnnxBackend,
};
use huncho_core::backend::{Backend, ForwardInput};

fn fixture() -> &'static str {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tiny_encoder.onnx"
    )
}

#[test]
fn loads_and_reports_capabilities() {
    let b = OnnxBackend::load(fixture(), 8, 512, "fp32").unwrap();
    assert_eq!(b.id().to_string(), "onnx");
    let caps = b.capabilities();
    assert_eq!(caps.max_context, 512);
    assert_eq!(caps.dtype, "fp32");
    assert!(!caps.supports_fork);
    assert!(!caps.supports_lora);
}

#[test]
fn forward_returns_hidden_states_at_positions() {
    let mut b = OnnxBackend::load(fixture(), 8, 512, "fp32").unwrap();
    let out = b
        .forward(ForwardInput::new(vec![3, 10, 5], vec![0, 1, 2]))
        .unwrap();
    let v = out.values();
    assert_eq!(v.shape(), &[3, 8]);
    // token 3 -> col 3 = 3.5 ; token 10 -> col 2 = 10.5 ; token 5 -> col 5 = 5.5
    assert_eq!(v.row(0).unwrap(), &[0.0, 0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(
        v.row(1).unwrap(),
        &[0.0, 0.0, 10.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(v.row(2).unwrap(), &[0.0, 0.0, 0.0, 0.0, 0.0, 5.5, 0.0, 0.0]);
}

#[test]
fn forward_slices_subset_of_positions() {
    let mut b = OnnxBackend::load(fixture(), 8, 512, "fp32").unwrap();
    let out = b
        .forward(ForwardInput::new(vec![3, 10, 5], vec![2, 0]))
        .unwrap();
    let v = out.values();
    assert_eq!(v.shape(), &[2, 8]);
    // position 2 = token 5 -> col 5 = 5.5 ; position 0 = token 3 -> col 3 = 3.5
    assert_eq!(v.row(0).unwrap()[5], 5.5);
    assert_eq!(v.row(1).unwrap()[3], 3.5);
}

#[test]
fn selected_rows_preserve_repeated_positions_and_exact_float_bits() {
    let mut b = OnnxBackend::load(fixture(), 8, 512, "fp32").unwrap();
    let tokens = vec![3, 10, 5];
    let full = b
        .forward(ForwardInput::new(tokens.clone(), vec![0, 1, 2]))
        .unwrap();
    let positions = vec![2, 0, 2, 1];
    let selected = b
        .forward(ForwardInput::new(tokens, positions.clone()))
        .unwrap();
    assert_eq!(selected.positions(), positions);
    for (row, &position) in positions.iter().enumerate() {
        let bits = |row: &[f32]| row.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(selected.values().row(row).unwrap()),
            bits(full.values().row(position).unwrap())
        );
    }
}

#[test]
fn rejects_invalid_positions_and_retains_graph_width_semantics() {
    let mut b = OnnxBackend::load(fixture(), 8, 512, "fp32").unwrap();
    for position in [3, usize::MAX] {
        assert!(b
            .forward(ForwardInput::new(vec![3, 10, 5], vec![position]))
            .is_err());
    }
    let mut hinted_width = OnnxBackend::load(fixture(), 16, 512, "fp32").unwrap();
    assert_eq!(
        hinted_width
            .forward(ForwardInput::new(vec![3], vec![0]))
            .unwrap()
            .values()
            .shape(),
        &[1, 8]
    );
    assert_eq!(
        hinted_width
            .forward(ForwardInput::new(vec![3], vec![]))
            .unwrap()
            .values()
            .shape(),
        &[0, 16]
    );
}

#[test]
fn prefill_with_no_positions_returns_empty() {
    let mut b = OnnxBackend::load(fixture(), 8, 512, "fp32").unwrap();
    let out = b.forward(ForwardInput::new(vec![1, 2, 3], vec![])).unwrap();
    assert_eq!(out.values().shape(), &[0, 8]);
}

#[test]
fn rejects_sequence_over_max_context() {
    let mut b = OnnxBackend::load(fixture(), 8, 4, "fp32").unwrap();
    let res = b.forward(ForwardInput::new(vec![1, 2, 3, 4, 5], vec![0]));
    assert!(res.is_err());
}

fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn compact_graph_and_bound_output_reuse_preserve_exact_rows() {
    let compact = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tiny_encoder_readout.onnx"
    );
    let mut reference = OnnxBackend::load(fixture(), 8, 512, "fp32").unwrap();
    for (path, compact_readout) in [(fixture(), false), (compact, true)] {
        let mut backend = OnnxBackend::load_with_options(
            path,
            8,
            512,
            "fp32",
            OnnxOptions {
                compact_readout,
                output_buffer_bytes: 128,
                ..Default::default()
            },
        )
        .unwrap();
        for (tokens, positions) in [
            (vec![3, 10, 5], vec![2, 0, 2]),
            (vec![5, 3, 10], vec![0, 2, 1]),
            (vec![1, 2], vec![1]),
            (vec![1, 2, 3, 4, 5], vec![0, 1, 2, 3, 4]),
            (vec![3], vec![]),
            (vec![10, 5], vec![1, 0]),
        ] {
            let expected = reference
                .forward(ForwardInput::new(tokens.clone(), positions.clone()))
                .unwrap();
            let actual = backend
                .forward(ForwardInput::new(tokens, positions))
                .unwrap();
            assert_eq!(actual.values().shape(), expected.values().shape());
            assert_eq!(bits(actual.values().data()), bits(expected.values().data()));
            assert!(backend.retained_output_bytes() <= 128);
        }
        assert_eq!(backend.output_buffer_reuses(), 1);
        assert!(backend
            .forward(ForwardInput::new(vec![1], vec![1]))
            .is_err());
        let actual = backend
            .forward(ForwardInput::new(vec![3, 10], vec![1, 0]))
            .unwrap();
        assert_eq!(actual.values().row(0).unwrap()[2], 10.5);
        assert_eq!(backend.output_buffer_reuses(), 2);
        assert_eq!(backend.retained_output_bytes(), 64);
    }
    assert!(OnnxBackend::load_with_options(
        fixture(),
        8,
        512,
        "fp32",
        OnnxOptions {
            compact_readout: true,
            output_buffer_bytes: 0,
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn disabled_or_oversized_binding_retains_no_output_storage() {
    for bytes in [0, 1] {
        let mut backend = OnnxBackend::load_with_options(
            fixture(),
            8,
            512,
            "fp32",
            OnnxOptions {
                output_buffer_bytes: bytes,
                ..Default::default()
            },
        )
        .unwrap();
        for _ in 0..2 {
            backend
                .forward(ForwardInput::new(vec![3, 10], vec![1]))
                .unwrap();
        }
        assert_eq!(backend.retained_output_bytes(), 0);
        assert_eq!(backend.output_buffer_reuses(), 0);
    }
}

#[test]
fn provider_and_thread_configuration_validate_without_gpu_probes() {
    assert_eq!(
        OnnxExecutionProvider::parse("CPU").unwrap(),
        OnnxExecutionProvider::Cpu
    );
    assert_eq!(
        OnnxExecutionProvider::parse("cuda:2").unwrap(),
        OnnxExecutionProvider::Cuda { device: 2 }
    );
    for value in [
        "cuda",
        "auto",
        "cuda:-1",
        "cuda:x",
        "cuda:2147483648",
        "metal:0",
    ] {
        assert!(OnnxExecutionProvider::parse(value).is_err());
    }
    let mut reference = OnnxBackend::load(fixture(), 8, 512, "fp32").unwrap();
    let expected = reference
        .forward(ForwardInput::new(vec![3, 10], vec![1, 0]))
        .unwrap();
    for threads in [1, 4] {
        let mut configured = OnnxBackend::load_with_options(
            fixture(),
            8,
            512,
            "fp32",
            OnnxOptions {
                intra_threads: threads,
                output_buffer_bytes: 128,
                ..Default::default()
            },
        )
        .unwrap();
        let actual = configured
            .forward(ForwardInput::new(vec![3, 10], vec![1, 0]))
            .unwrap();
        assert_eq!(bits(actual.values().data()), bits(expected.values().data()));
        assert_eq!(
            configured.capabilities().extra["onnx_intra_threads"],
            threads.to_string()
        );
    }
    assert!(OnnxBackend::load_with_options(
        fixture(),
        8,
        512,
        "fp32",
        OnnxOptions {
            intra_threads: 257,
            ..Default::default()
        }
    )
    .is_err());
    assert!(OnnxBackend::load_with_options(
        fixture(),
        8,
        512,
        "fp32",
        OnnxOptions {
            execution_provider: OnnxExecutionProvider::Cuda { device: -1 },
            ..Default::default()
        }
    )
    .is_err());
    #[cfg(not(feature = "onnx-cuda"))]
    assert!(OnnxBackend::load_with_options(
        "unused.onnx",
        8,
        512,
        "fp32",
        OnnxOptions {
            execution_provider: OnnxExecutionProvider::Cuda { device: 0 },
            ..Default::default()
        }
    )
    .err()
    .unwrap()
    .to_string()
    .contains("onnx-cuda"));
}
