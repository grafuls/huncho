//! Explicit CPU runtime process coverage; fixture labels never qualify a release.
#![cfg(all(feature = "clef", feature = "cpu-blas"))]
use serde_json::{json, Value};
use std::{
    path::Path,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};
fn run(args: &[&str], threads: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env("HUNCHO_DEVICE", "cpu")
        .env("RAYON_NUM_THREADS", "1")
        .env("CANDLE_NUM_THREADS", "1")
        .env("HUNCHO_CPU_BLAS_THREADS", threads)
        .env_remove("HUNCHO_REPLICAS")
        .env_remove("HUNCHO_CPU_FUSED_GATE")
        .env_remove("HUNCHO_CPU_DELTA_RULE")
        .env_remove("HUNCHO_CPU_CAUSAL_CONV")
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
            panic!("unqualified BLAS serving started or hung");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}
#[test]
#[ignore = "requires explicit HUNCHO_CPU_BLAS_LIBRARY pointing to LP64 pthread OpenBLAS"]
fn cli_records_actual_blas_and_refuses_unsupported_precision_and_unqualified_serving() {
    assert!(std::env::var_os("HUNCHO_CPU_BLAS_LIBRARY").is_some());
    let source = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/tiny_clef"
    ));
    let tmp = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
        }
    }
    let manifest_path = tmp.path().join("huncho-model.json");
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["calibration"] =
        json!({"default":{"temperature":1.,"confidence":"peak","status":"fit"},"entries":{}});
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let package = tmp.path().to_str().unwrap();
    let args = [
        "bench",
        "--model",
        package,
        "--backend",
        "clef",
        "--dtype",
        "fp32",
        "--iterations",
        "1",
        "--questions",
        "3",
        "--json",
    ];
    let output = run(&args, "1");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        report["execution_metadata"]["cpu_blas_execution"],
        "openblas-lp64-fp32-v1"
    );
    assert_eq!(report["execution_metadata"]["cpu_blas_threads"], "1");
    assert_eq!(
        report["execution_metadata"]["cpu_blas_library_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    let output = run(&args, "0");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("1..256"));
    let output = run(
        &[
            "bench",
            "--model",
            package,
            "--backend",
            "clef",
            "--dtype",
            "fp16",
            "--iterations",
            "1",
        ],
        "1",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("CPU FP32"));
    let output = run(
        &[
            "serve",
            "--model",
            package,
            "--backend",
            "clef",
            "--dtype",
            "fp32",
            "--bind",
            "127.0.0.1:0",
        ],
        "1",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires --qualification-golden"));
}
