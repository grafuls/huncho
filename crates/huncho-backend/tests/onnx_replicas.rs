//! Actual independent CPU ORT sessions over immutable shared initializers.
#![cfg(feature = "onnx-shared")]
use huncho_backend::{
    onnx::{OnnxExecutionProvider, OnnxOptions},
    OnnxBackend,
};
use huncho_core::backend::{Backend, ForwardInput};
use std::path::Path;

fn bits(out: &huncho_core::backend::ForwardOutput) -> Vec<u32> {
    out.values().data().iter().map(|v| v.to_bits()).collect()
}

#[test]
fn actual_cpu_sessions_preserve_rows_after_source_mutation_and_primary_drop() {
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"));
    for (name, compact, native_batch) in [
        ("tiny_encoder.onnx", false, false),
        ("tiny_encoder_readout.onnx", true, false),
        ("tiny_encoder_batch.onnx", false, true),
        ("tiny_encoder_external.onnx", false, false),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let graph = dir.path().join(name);
        std::fs::copy(root.join(name), &graph).unwrap();
        if name == "tiny_encoder_external.onnx" {
            std::fs::copy(
                root.join("tiny_encoder_external.weights"),
                dir.path().join("tiny_encoder_external.weights"),
            )
            .unwrap();
        }
        let mut baseline = OnnxBackend::load_with_options(
            &graph,
            8,
            512,
            "fp32",
            OnnxOptions {
                compact_readout: compact,
                native_batch,
                intra_threads: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(baseline.replica().is_err());
        let inputs = [
            ForwardInput::new(vec![3, 10, 5], vec![2, 0, 2]),
            ForwardInput::new(vec![5, 3, 10], vec![0, 2, 1]),
            ForwardInput::new(vec![1, 2], vec![1]),
        ];
        let expected: Vec<_> = inputs
            .iter()
            .map(|input| bits(&baseline.forward(input.clone()).unwrap()))
            .collect();
        let mut primary = OnnxBackend::load_with_options(
            &graph,
            8,
            512,
            "fp32",
            OnnxOptions {
                compact_readout: compact,
                native_batch,
                intra_threads: 1,
                output_buffer_bytes: 1024,
                shared_initializers: true,
                ..Default::default()
            },
        )
        .unwrap();
        let identity = primary.capabilities().extra;
        assert_eq!(identity["onnx_initializer_residency"], "immutable-cpu-v1");
        assert_eq!(identity["onnx_shared_initializer_bytes"], "512");
        for (input, expected) in inputs.iter().zip(&expected) {
            assert_eq!(bits(&primary.forward(input.clone()).unwrap()), *expected);
        }
        assert!(primary.retained_output_bytes() > 0);
        // Source files no longer participate in replicas. This also proves the
        // external-data profile cannot silently reread altered weights.
        std::fs::write(&graph, b"changed graph").unwrap();
        if name == "tiny_encoder_external.onnx" {
            std::fs::write(dir.path().join("tiny_encoder_external.weights"), [0u8; 512]).unwrap();
        }
        let contexts: Vec<_> = (0..3).map(|_| primary.replica().unwrap()).collect();
        drop(primary);
        drop(baseline);
        std::thread::scope(|scope| {
            let jobs: Vec<_> = contexts
                .into_iter()
                .map(|mut context| {
                    let inputs = &inputs;
                    let expected = &expected;
                    let identity = &identity;
                    scope.spawn(move || {
                        assert_eq!(context.capabilities().extra, *identity);
                        for _ in 0..4 {
                            for (input, expected) in inputs.iter().zip(expected) {
                                assert_eq!(
                                    bits(&context.forward(input.clone()).unwrap()),
                                    *expected
                                );
                            }
                            if native_batch {
                                let outputs = context
                                    .forward_batch(vec![inputs[0].clone(), inputs[1].clone()])
                                    .unwrap();
                                assert_eq!(bits(&outputs[0]), expected[0]);
                                assert_eq!(bits(&outputs[1]), expected[1]);
                            }
                        }
                    })
                })
                .collect();
            for job in jobs {
                job.join().unwrap();
            }
        });
    }
}

#[test]
fn shared_loading_rejects_gpu_selection_before_runtime_initialization() {
    // No runtime/device availability check is involved in rejecting this shape.
    let result = OnnxBackend::load_with_options(
        "does-not-exist.onnx",
        8,
        512,
        "fp32",
        OnnxOptions {
            shared_initializers: true,
            execution_provider: OnnxExecutionProvider::Cuda { device: 0 },
            ..Default::default()
        },
    );
    assert!(result.is_err());
}
