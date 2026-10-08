//! Actual CPU ORT process coverage; synthetic outcomes never release a model.
#![cfg(all(feature = "onnx", feature = "tokenizers"))]
use serde_json::{json, Value};
use std::{
    path::Path,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};
fn run(args: &[&str], flag: &str) -> Output {
    run_device_flags(args, flag, "0", "0")
}
fn run_device_flags(args: &[&str], flag: &str, bytes: &str, graph: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env("HUNCHO_DEVICE", "cpu")
        .env("HUNCHO_ONNX_EP", "cpu")
        .env("HUNCHO_ONNX_THREADS", "1")
        .env("HUNCHO_ONNX_INTEGRATED_HEAD", flag)
        .env("HUNCHO_ONNX_OUTPUT_BUFFER_BYTES", "64")
        .env("HUNCHO_ONNX_DEVICE_IO_BYTES", bytes)
        .env("HUNCHO_ONNX_CUDA_GRAPH", graph)
        .env_remove("HUNCHO_ONNX_COMPACT_READOUT")
        .env_remove("HUNCHO_ONNX_NATIVE_BATCH")
        .env_remove("HUNCHO_ONNX_SHARED_INITIALIZERS")
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
            panic!("integrated ONNX command hung or unqualified serving started");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}
#[test]
fn native_raw_head_process_binds_identity_and_requires_exact_labeled_serving() {
    let root = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/integrated_f1"
    ));
    let manifest = root.join("huncho-model.json");
    let golden = root.join("golden.json");
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
        "--json",
    ];
    let output = run(&args, "1");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(
        report["execution_metadata"]["onnx_integrated_head"],
        "graph-integrated-f1-v1"
    );
    assert_eq!(report["outcome_calibration"]["questions"], 12);
    assert_eq!(report["work"]["forward_calls"], 12);
    assert!(report["max_prob_delta"].as_f64().unwrap() <= 1e-6);
    let output = run(&args, "on");
    assert!(
        !output.status.success()
            && String::from_utf8_lossy(&output.stderr).contains("must be 0, 1, false or true")
    );
    let tmp = tempfile::tempdir().unwrap();
    for name in ["model.onnx", "huncho-model.json", "tokenizer.json"] {
        std::fs::copy(root.join(name), tmp.path().join(name)).unwrap();
    }
    let mut unlabeled: Value = serde_json::from_slice(&std::fs::read(&golden).unwrap()).unwrap();
    for case in unlabeled["cases"].as_array_mut().unwrap() {
        case["targets"] = json!({});
    }
    let path = tmp.path().join("unlabeled.json");
    std::fs::write(&path, serde_json::to_vec(&unlabeled).unwrap()).unwrap();
    let binding = format!("browser-synthetic-f1={}", path.display());
    for provided in [false, true] {
        let mut args = vec![
            "serve",
            "--manifest",
            manifest.to_str().unwrap(),
            "--backend",
            "onnx",
            "--dtype",
            "fp32",
            "--bind",
            "127.0.0.1:0",
        ];
        if provided {
            args.extend(["--qualification-golden", &binding]);
        }
        let output = run(&args, "1");
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(if provided {
                "observed target labels"
            } else {
                "--qualification-golden"
            }),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let local_manifest = tmp.path().join("huncho-model.json");
    let mut value: Value =
        serde_json::from_slice(&std::fs::read(&local_manifest).unwrap()).unwrap();
    value["calibration"]["entries"] = json!({});
    std::fs::write(&local_manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let binding = format!("browser-synthetic-f1={}", golden.display());
    let output = run(
        &[
            "serve",
            "--manifest",
            local_manifest.to_str().unwrap(),
            "--backend",
            "onnx",
            "--dtype",
            "fp32",
            "--qualification-golden",
            &binding,
            "--bind",
            "127.0.0.1:0",
        ],
        "1",
    );
    assert!(
        !output.status.success()
            && String::from_utf8_lossy(&output.stderr)
                .contains("explicit fitted/refitted onnx:fp32")
    );
    value["prompt_contract"]["template"] = json!("F1-default");
    std::fs::write(&local_manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let output = run(
        &[
            "bench",
            "--manifest",
            local_manifest.to_str().unwrap(),
            "--backend",
            "onnx",
            "--dtype",
            "fp32",
            "--iterations",
            "1",
        ],
        "1",
    );
    assert!(
        !output.status.success()
            && String::from_utf8_lossy(&output.stderr).contains("laya-v1 prompt contract")
    );
}

#[test]
fn device_buffer_and_graph_flags_refuse_cpu_and_invalid_profiles_without_accessing_gpu() {
    let manifest = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/integrated_f1/huncho-model.json"
    ));
    let args = [
        "bench",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "onnx",
        "--dtype",
        "fp32",
        "--iterations",
        "1",
    ];
    for (bytes, graph, reason) in [
        ("4096", "0", "strict CUDA fp32"),
        ("0", "1", "strict CUDA fp32"),
        ("536870913", "0", "budget must be"),
        ("broken", "0", "HUNCHO_ONNX_DEVICE_IO_BYTES"),
        ("0", "broken", "HUNCHO_ONNX_CUDA_GRAPH"),
    ] {
        let output = run_device_flags(&args, "1", bytes, graph);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(reason),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
