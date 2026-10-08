//! Lifecycle logging must not contaminate machine-readable stdout.
use std::path::PathBuf;
use std::process::Command;

#[test]
fn bench_and_conformance_stdout_are_standalone_json_with_info_logging() {
    let package = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/mock-model/huncho-model.json");
    let golden = package.parent().unwrap().join("golden.json");
    for command in ["bench", "conform"] {
        let mut process = Command::new(env!("CARGO_BIN_EXE_huncho"));
        process
            .env("RUST_LOG", "info")
            .env_remove("HUNCHO_BACKEND")
            .env_remove("HUNCHO_DTYPE")
            .args([command, "--manifest"])
            .arg(&package)
            .args(["--backend", "mock", "--json"]);
        if command == "bench" {
            process.args(["--iterations", "1"]);
        } else {
            process.arg("--golden").arg(&golden);
        }
        let output = process.output().unwrap();
        let report: serde_json::Value =
            serde_json::from_slice(&output.stdout).unwrap_or_else(|e| {
                panic!(
                    "{command}: {e}: {}",
                    String::from_utf8_lossy(&output.stdout)
                )
            });
        assert!(report.is_object());
        assert!(String::from_utf8_lossy(&output.stderr).contains("loading"));
        if command == "bench" {
            assert!(output.status.success());
            assert_eq!(report["iterations"], 1);
        } else {
            assert!(report["passed"].is_boolean());
        }
    }
}

#[test]
fn bench_distinguishes_repeated_result_reuse_from_distinct_model_work() {
    for repeat in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_huncho"));
        command.args([
            "bench",
            "--questions",
            "3",
            "--iterations",
            "4",
            "--result-cache-bytes",
            "1048576",
            "--json",
        ]);
        if repeat {
            command.arg("--repeat-inputs");
        }
        let output = command.output().unwrap();
        assert!(output.status.success());
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            report["work"]["result_cache_hits"],
            if repeat { 4 } else { 0 }
        );
        assert_eq!(report["work"]["forward_calls"], if repeat { 0 } else { 12 });
        assert_eq!(report["result_cache_bytes"], 1048576);
    }
}

#[test]
fn replica_bench_accounts_for_every_timed_request_and_shared_result_cache() {
    for repeated in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_huncho"));
        command.args([
            "bench",
            "--questions",
            "3",
            "--workload",
            "mixed",
            "--iterations",
            "9",
            "--concurrency",
            "4",
            "--replicas",
            "2",
            "--result-cache-bytes",
            "1048576",
            "--json",
        ]);
        if repeated {
            command.arg("--repeat-inputs");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["replicas"], 2);
        assert_eq!(report["concurrency"], 4);
        let contexts = report["replica_work"].as_array().unwrap();
        assert_eq!(contexts.len(), 2);
        assert_eq!(contexts[0]["requests"], 5);
        assert_eq!(contexts[1]["requests"], 4);
        for field in ["forward_calls", "processed_tokens", "result_cache_hits"] {
            let total: u64 = contexts
                .iter()
                .map(|context| context["work"][field].as_u64().unwrap())
                .sum();
            assert_eq!(total, report["work"][field].as_u64().unwrap());
        }
        assert_eq!(
            report["work"]["forward_calls"],
            if repeated { 0 } else { 27 }
        );
        assert_eq!(
            report["work"]["result_cache_hits"],
            if repeated { 9 } else { 0 }
        );
    }
    for arguments in [
        vec!["--replicas", "2", "--concurrency", "1"],
        vec!["--replicas", "2", "--concurrency", "2", "--iterations", "1"],
        vec!["--replicas", "9"],
        vec![
            "--replicas",
            "2",
            "--concurrency",
            "2",
            "--prefix-cache",
            "--persistent-prefix-bytes",
            "1",
        ],
        vec!["--persistent-prefix-bytes", "100"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_huncho"))
            .arg("bench")
            .args(arguments)
            .output()
            .unwrap();
        assert!(!output.status.success());
    }
}

#[test]
fn calibration_json_dry_run_accepts_false_and_leaves_manifest_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest = tmp.path().join("huncho-model.json");
    let package = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/mock-model/huncho-model.json");
    std::fs::copy(package, &manifest).unwrap();
    let original = std::fs::read(&manifest).unwrap();
    let data = tmp.path().join("fit.json");
    std::fs::write(
        &data,
        r#"{"rows":[[1.0,2.0],[2.0,1.0]],"targets":[1,0],"qtypes":["choice","noul"]}"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .args(["calibrate", "--manifest"])
        .arg(&manifest)
        .args(["--backend", "onnx", "--dtype", "fp32", "--data"])
        .arg(&data)
        .args(["--save", "false", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["saved"], false);
    assert_eq!(report["fitting_rows"], 2);
    assert_eq!(report["entry"]["status"], "refit");
    assert!(report["entry"]["temperature_by_options"]["noul:2"].is_number());
    assert_eq!(std::fs::read(&manifest).unwrap(), original);
}

#[test]
fn prepared_prompt_cache_reuses_preparation_but_still_submits_forwards() {
    for repeat in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_huncho"));
        command.env("HUNCHO_PROMPT_CACHE_BYTES", "1048576").args([
            "bench",
            "--questions",
            "3",
            "--iterations",
            "4",
            "--json",
        ]);
        if repeat {
            command.arg("--repeat-inputs");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            report["work"]["prompt_cache_hits"],
            if repeat { 12 } else { 0 }
        );
        assert_eq!(report["work"]["result_cache_hits"], 0);
        assert_eq!(report["work"]["forward_calls"], 12);
        assert!(report["work"]["processed_tokens"].as_u64().unwrap() > 0);
    }
    let output = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env("HUNCHO_PROMPT_CACHE_BYTES", "invalid")
        .args(["bench", "--iterations", "1", "--json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("HUNCHO_PROMPT_CACHE_BYTES"));
}
