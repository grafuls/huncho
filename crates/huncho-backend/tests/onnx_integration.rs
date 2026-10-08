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
    #[cfg(not(feature = "onnx-shared"))]
    assert!(OnnxBackend::load_with_options(
        "absent.onnx",
        8,
        512,
        "fp32",
        OnnxOptions {
            shared_initializers: true,
            ..Default::default()
        }
    )
    .err()
    .unwrap()
    .to_string()
    .contains("onnx-shared"));
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

#[test]
fn native_dynamic_batches_preserve_isolated_rows_and_validate_contract() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tiny_encoder_batch.onnx"
    );
    let mut reference = OnnxBackend::load(fixture(), 8, 512, "fp32").unwrap();
    let inputs = vec![
        ForwardInput::new(vec![3, 10, 5], vec![2, 0, 2]),
        ForwardInput::new(vec![5, 3, 10], vec![0, 2]),
        ForwardInput::new(vec![1, 2, 3], vec![]),
    ];
    let expected = inputs
        .iter()
        .map(|input| reference.forward(input.clone()).unwrap())
        .collect::<Vec<_>>();
    for bytes in [0, 288] {
        let mut backend = OnnxBackend::load_with_options(
            path,
            8,
            512,
            "fp32",
            OnnxOptions {
                native_batch: true,
                output_buffer_bytes: bytes,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(backend.supports_batch());
        for _ in 0..2 {
            let actual = backend.forward_batch(inputs.clone()).unwrap();
            for (a, e) in actual.iter().zip(&expected) {
                assert_eq!(a.positions(), e.positions());
                assert_eq!(a.values().shape(), e.values().shape());
                assert_eq!(bits(a.values().data()), bits(e.values().data()));
            }
        }
        assert_eq!(backend.output_buffer_reuses(), u64::from(bytes > 0));
        assert_eq!(backend.retained_output_bytes(), bytes);
        for invalid in [
            vec![],
            vec![inputs[0].clone(); 65],
            vec![inputs[0].clone(), ForwardInput::new(vec![1], vec![0])],
            vec![ForwardInput::new(vec![1], vec![1])],
        ] {
            assert!(backend.forward_batch(invalid).is_err());
        }
        let mut cached = inputs[0].clone();
        cached.fork_from = Some(huncho_core::backend::CacheHandle { id: 1 });
        assert!(backend.forward_batch(vec![cached.clone()]).is_err());
        assert!(backend.forward(cached).is_err());
        let actual = backend.forward(inputs[1].clone()).unwrap();
        assert_eq!(
            bits(actual.values().data()),
            bits(expected[1].values().data())
        );
        assert!(backend.retained_output_bytes() <= bytes);
    }
    assert!(OnnxBackend::load_with_options(
        fixture(),
        8,
        512,
        "fp32",
        OnnxOptions {
            native_batch: true,
            ..Default::default()
        }
    )
    .is_err());
    assert!(OnnxBackend::load_with_options(
        path,
        8,
        512,
        "fp32",
        OnnxOptions {
            native_batch: true,
            compact_readout: true,
            ..Default::default()
        }
    )
    .is_err());
    assert!(!reference.supports_batch());
    assert!(reference.forward_batch(inputs).is_err());
}

#[test]
fn stable_device_profiles_refuse_incompatible_options_before_loading_any_runtime() {
    for (dtype, options, reason) in [
        (
            "fp32",
            OnnxOptions {
                device_io_bytes: 4096,
                ..Default::default()
            },
            "strict CUDA fp32",
        ),
        (
            "fp32",
            OnnxOptions {
                cuda_graph: true,
                ..Default::default()
            },
            "strict CUDA fp32",
        ),
        (
            "fp16",
            OnnxOptions {
                device_io_bytes: 4096,
                execution_provider: OnnxExecutionProvider::Cuda { device: 0 },
                ..Default::default()
            },
            "strict CUDA fp32",
        ),
        (
            "fp32",
            OnnxOptions {
                cuda_graph: true,
                execution_provider: OnnxExecutionProvider::Cuda { device: 0 },
                ..Default::default()
            },
            "nonzero stable device I/O",
        ),
        (
            "fp32",
            OnnxOptions {
                device_io_bytes: (512 << 20) + 1,
                ..Default::default()
            },
            "budget must be",
        ),
        (
            "fp32",
            OnnxOptions {
                device_io_bytes: 4096,
                execution_provider: OnnxExecutionProvider::Cuda { device: 0 },
                output_buffer_bytes: 4096,
                ..Default::default()
            },
            "strict CUDA fp32",
        ),
    ] {
        let result = OnnxBackend::load_with_options("must-not-load.onnx", 8, 512, dtype, options);
        assert!(result.err().unwrap().to_string().contains(reason));
    }
}

#[cfg(feature = "onnx-cuda")]
#[test]
#[ignore = "actual GPU checks deferred; requires compatible CUDA ORT/GPU and fresh release qualification"]
fn cuda_device_buffers_and_graph_replay_preserve_scalar_readout_scores_and_owned_outputs() {
    use huncho_core::calibration::{argmax, calibrate};
    for capture in [false, true] {
        let mut cpu = OnnxBackend::load(fixture(), 8, 128, "fp32").unwrap();
        let mut gpu = OnnxBackend::load_with_options(
            fixture(),
            8,
            128,
            "fp32",
            OnnxOptions {
                device_io_bytes: 1 << 20,
                cuda_graph: capture,
                execution_provider: OnnxExecutionProvider::Cuda { device: 0 },
                ..Default::default()
            },
        )
        .unwrap();
        let mut owned = None;
        for _ in 0..2 {
            for seq in [1, 3, 16, 31, 64, 128] {
                let positions = vec![seq - 1, 0, seq / 2];
                let input =
                    ForwardInput::new((0..seq).map(|n| (n % 16) as u32).collect(), positions);
                let expected = cpu.forward(input.clone()).unwrap();
                let actual = gpu.forward(input).unwrap();
                assert_eq!(expected.values().data(), actual.values().data());
                for t in [0.75, 1., 2.40605] {
                    let scores = |values: &huncho_core::tensor::Tensor| {
                        (0..3)
                            .map(|n| values.row(n).unwrap().iter().sum::<f32>() / 8.)
                            .collect::<Vec<_>>()
                    };
                    let a = calibrate(&scores(actual.values()), t).unwrap();
                    let b = calibrate(&scores(expected.values()), t).unwrap();
                    assert_eq!(argmax(&a), argmax(&b));
                    assert!(a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= 1e-4));
                }
                if let Some((previous, values)) = &owned {
                    assert_eq!(
                        huncho_core::backend::ForwardOutput::values(previous).data(),
                        values
                    );
                }
                owned = Some((actual, expected.values().data().to_vec()));
            }
        }
    }
}
