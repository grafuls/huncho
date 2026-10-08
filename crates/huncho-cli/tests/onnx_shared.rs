//! Optional real CPU ORT replicas and their startup calibration gates.
#![cfg(feature = "onnx-shared")]
use serde_json::Value;
use std::{path::Path, process::Command};

fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_huncho"));
    command
        .env("HUNCHO_ONNX_EP", "cpu")
        .env("HUNCHO_ONNX_THREADS", "1")
        .env("HUNCHO_ONNX_SHARED_INITIALIZERS", "1")
        .env_remove("HUNCHO_ONNX_NATIVE_BATCH")
        .env_remove("HUNCHO_ONNX_COMPACT_READOUT")
        .env_remove("HUNCHO_ONNX_OUTPUT_BUFFER_BYTES")
        .env_remove("HUNCHO_DEVICE")
        .env_remove("HUNCHO_BACKEND");
    command
}

#[test]
fn cpu_onnx_replica_bench_records_each_actual_context_and_requires_labeled_serving() {
    let fixture = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/mock-model"
    ));
    let manifest = fixture.join("huncho-model.json");
    let golden = fixture.join("golden.json");
    let output = command()
        .args([
            "bench",
            "--manifest",
            manifest.to_str().unwrap(),
            "--replicas",
            "2",
            "--concurrency",
            "2",
            "--iterations",
            "4",
            "--questions",
            "3",
            "--workload",
            "mixed",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["replicas"], 2);
    assert_eq!(
        report["execution_metadata"]["onnx_initializer_residency"],
        "immutable-cpu-v1"
    );
    for context in report["replica_work"].as_array().unwrap() {
        assert_eq!(context["requests"], 2);
        assert_eq!(context["work"]["forward_calls"], 6);
    }
    assert_eq!(report["work"]["forward_calls"], 12);
    let output = command()
        .args([
            "conform",
            "--manifest",
            manifest.to_str().unwrap(),
            "--golden",
            golden.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["max_prob_delta"], 0.0);
    assert_eq!(
        report["execution_metadata"]["onnx_initializer_residency"],
        "immutable-cpu-v1"
    );
    // The deterministic fixture has no observed outcomes. Agreement cannot
    // authorize a changed execution profile or a real model replica pool.
    let suite = format!("mock-laya={}", golden.display());
    for replicas in ["1", "2"] {
        for qualified in [false, true] {
            let mut cmd = command();
            cmd.args([
                "serve",
                "--manifest",
                manifest.to_str().unwrap(),
                "--replicas",
                replicas,
                "--bind",
                "127.0.0.1:0",
            ]);
            if qualified {
                cmd.args(["--qualification-golden", &suite]);
            }
            let output = cmd.output().unwrap();
            assert!(!output.status.success());
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(
                error.contains(if qualified {
                    "observed target labels"
                } else {
                    "--qualification-golden"
                }),
                "{error}"
            );
        }
    }
}
