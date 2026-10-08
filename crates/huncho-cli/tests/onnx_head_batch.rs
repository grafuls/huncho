//! CPU process gates for strict per-row typed raw heads; no released acceptance.
#![cfg(all(feature = "onnx", feature = "tokenizers", feature = "qualification"))]
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};
fn run(args: &[&str], native: &str, shared: bool) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env("HUNCHO_DEVICE", "cpu")
        .env("HUNCHO_ONNX_EP", "cpu")
        .env("HUNCHO_ONNX_THREADS", "1")
        .env("HUNCHO_ONNX_INTEGRATED_HEAD", "1")
        .env("HUNCHO_ONNX_NATIVE_BATCH", native)
        .env("HUNCHO_ONNX_OUTPUT_BUFFER_BYTES", "4096")
        .env(
            "HUNCHO_ONNX_SHARED_INITIALIZERS",
            if shared { "1" } else { "0" },
        )
        .env("HUNCHO_ONNX_DEVICE_IO_BYTES", "0")
        .env("HUNCHO_ONNX_CUDA_GRAPH", "0")
        .env_remove("HUNCHO_ONNX_COMPACT_READOUT")
        .env_remove("HUNCHO_LAYA_SELECTED_HEAD")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if start.elapsed() > Duration::from_secs(20) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("head batch CLI hung or unqualified serving started");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}
fn package() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let root = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/integrated_f1"
    ));
    let tmp = tempfile::tempdir().unwrap();
    for name in ["tokenizer.json", "golden.json"] {
        std::fs::copy(root.join(name), tmp.path().join(name)).unwrap();
    }
    std::fs::copy(
        root.join("batch-masked.onnx"),
        tmp.path().join("model.onnx"),
    )
    .unwrap();
    let manifest = tmp.path().join("huncho-model.json");
    // New graph artifact/profile; scalar fitted fixture entries are not a
    // release certificate. Fresh gates below retain original fixed targets.
    std::fs::copy(root.join("huncho-model.json"), &manifest).unwrap();
    let golden = tmp.path().join("golden.json");
    (tmp, manifest, golden)
}
#[test]
fn cli_native_heads_bind_identity_and_actual_padding_and_refuse_missing_labels() {
    let (tmp, manifest, golden) = package();
    let receipt = tmp.path().join("native-head-receipt.json");
    let args = [
        "conform",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "onnx",
        "--dtype",
        "fp32",
        "--golden",
        golden.to_str().unwrap(),
        "--max-batch-tokens",
        "4096",
        "--max-batch-padding-percent",
        "50",
        "--batch-max-requests",
        "4",
        "--write-qualification",
        receipt.to_str().unwrap(),
        "--json",
    ];
    let output = run(&args, "1", false);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(
        report["execution_metadata"]["onnx_integrated_head"],
        "graph-integrated-f1-batch-v1"
    );
    assert_eq!(
        report["execution_metadata"]["onnx_native_batch"],
        "raw-f1-row-markers-v1"
    );
    assert!(report["work"]["padded_batch_calls"].as_u64().unwrap() > 0);
    assert!(report["work"]["cross_request_batches"].as_u64().unwrap() > 0);
    assert!(
        report["optimization_parity"]["max_prob_delta"]
            .as_f64()
            .unwrap()
            <= 1e-4
    );
    assert_eq!(report["optimization_parity"]["argmax_agreement"], 1.0);
    let record: Value = serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
    assert_eq!(record["outcome_gates_passed"], true); // Synthetic gate plumbing only.
    let mut unlabeled: Value = serde_json::from_slice(&std::fs::read(&golden).unwrap()).unwrap();
    for case in unlabeled["cases"].as_array_mut().unwrap() {
        case["targets"] = json!({});
    }
    let path = tmp.path().join("unlabeled.json");
    std::fs::write(&path, serde_json::to_vec(&unlabeled).unwrap()).unwrap();
    let binding = format!("browser-synthetic-f1={}", path.display());
    let serve = [
        "serve",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "onnx",
        "--dtype",
        "fp32",
        "--max-batch-tokens",
        "4096",
        "--max-batch-padding-percent",
        "50",
        "--qualification-golden",
        &binding,
        "--bind",
        "127.0.0.1:0",
    ];
    let output = run(&serve, "1", false);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("observed target labels"));
    let mut scalar = args.to_vec();
    let write = scalar
        .iter()
        .position(|s| *s == "--write-qualification")
        .unwrap();
    scalar.drain(write..write + 2);
    let output = run(&scalar, "0", false);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("int64"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
#[cfg(feature = "onnx-shared")]
#[test]
fn shared_cpu_raw_head_sessions_report_isolated_native_work() {
    let (_tmp, manifest, _golden) = package();
    let output = run(
        &[
            "bench",
            "--manifest",
            manifest.to_str().unwrap(),
            "--backend",
            "onnx",
            "--dtype",
            "fp32",
            "--replicas",
            "2",
            "--concurrency",
            "2",
            "--iterations",
            "2",
            "--questions",
            "3",
            "--workload",
            "mixed",
            "--max-batch-tokens",
            "4096",
            "--max-batch-padding-percent",
            "50",
            "--json",
        ],
        "1",
        true,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["replicas"], 2);
    assert!(
        report["execution_metadata"]["onnx_shared_initializer_bytes"]
            .as_str()
            .unwrap()
            .parse::<usize>()
            .unwrap()
            > 0
    );
    assert!(report["work"]["batch_calls"].as_u64().unwrap() > 0);
    assert_eq!(report["work"]["cache_forks"], 0);
    let contexts = report["replica_work"].as_array().unwrap();
    assert_eq!(contexts.len(), 2);
    for context in contexts {
        assert_eq!(context["requests"], 1);
        assert!(context["work"]["batch_calls"].as_u64().unwrap() > 0);
    }
}
