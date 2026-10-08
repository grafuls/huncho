//! Real CPU serving/receipts; fixture labels exercise plumbing, not releases.
#![cfg(all(feature = "clef", feature = "qualification"))]
use serde_json::{json, Value};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};
fn command(args: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_huncho"));
    c.env("HUNCHO_DEVICE", "cpu")
        .env("CANDLE_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .env("HUNCHO_ATTENTION_QUERY_ROWS", "0")
        .env("HUNCHO_GROUPED_GQA", "0")
        .env("HUNCHO_CLEF_VECTOR_HEAD", "0")
        .env("HUNCHO_CLEF_GROUPED_POOL", "0")
        .env_remove("HUNCHO_CPU_BLAS_LIBRARY")
        .env_remove("HUNCHO_CPU_BLAS_THREADS")
        .env_remove("HUNCHO_CPU_DELTA_RULE")
        .env_remove("HUNCHO_CPU_CAUSAL_CONV")
        .env_remove("HUNCHO_CPU_FUSED_GATE")
        .env_remove("HUNCHO_REPLICAS")
        .env_remove("HUNCHO_BACKEND")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    c
}
fn run(mut command: Command) -> Output {
    let mut child = command.spawn().unwrap();
    let start = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if start.elapsed() > Duration::from_secs(60) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("CPU command timed out or unqualified serving started");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}
fn package() -> (tempfile::TempDir, PathBuf, PathBuf, Value) {
    let root = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/tiny_clef"
    ));
    let tmp = tempfile::tempdir().unwrap();
    let pkg = tmp.path().join("package");
    std::fs::create_dir(&pkg).unwrap();
    for name in [
        "config.json",
        "model.safetensors",
        "joint_head_config.json",
        "joint_head.safetensors",
        "tokenizer.json",
    ] {
        std::fs::copy(root.join(name), pkg.join(name)).unwrap();
    }
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(root.join("huncho-model.json")).unwrap()).unwrap();
    // Simulate fitted status only in a temporary synthetic package. The
    // checked-in fixture stays Pending; source temperatures/goldens stay fixed.
    manifest["calibration"]["default"]["status"] = json!("fit");
    let manifest_path = pkg.join("huncho-model.json");
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let cases:Vec<_>=reference["cases"].as_array().unwrap().iter().enumerate().map(|(index,case)| {
        let mut expected=case["probabilities"].clone(); let mut targets=serde_json::Map::new();
        for (id,q) in case["request"]["questions"].as_object().unwrap() {
            if q["type"]=="noul" { let p=&expected[id]; expected[id]=json!({"yes":p["true"],"no":p["false"]}); }
            targets.insert(id.clone(),expected[id].as_object().unwrap().keys().next().unwrap().clone().into());
        }
        json!({"id":index.to_string(),"request":case["request"],"expected":expected,"targets":targets})
    }).collect();
    let golden = tmp.path().join("synthetic-labeled-golden.json");
    std::fs::write(
        &golden,
        serde_json::to_vec(&json!({"schema_version":"1.0","family":"F5","cases":cases})).unwrap(),
    )
    .unwrap();
    (tmp, manifest_path, golden, reference)
}
struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn http(bind: &str, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut socket = TcpStream::connect(bind).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(20)))
        .unwrap();
    socket
        .set_write_timeout(Some(Duration::from_secs(20)))
        .unwrap();
    write!(socket,"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
    let mut response = String::new();
    socket.read_to_string(&mut response).unwrap();
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    (
        headers.split_whitespace().nth(1).unwrap().parse().unwrap(),
        body.into(),
    )
}
fn counter(metrics: &str, name: &str) -> u64 {
    metrics
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{name} ")))
        .unwrap()
        .parse()
        .unwrap()
}
#[test]
fn real_cpu_joint_serving_collates_complete_schemas_and_binds_fresh_qualification_receipts() {
    let (tmp, manifest, golden, reference) = package();
    let receipt = tmp.path().join("qualification.json");
    let output = run(command(&[
        "conform",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "clef",
        "--dtype",
        "fp32",
        "--golden",
        golden.to_str().unwrap(),
        "--max-batch-tokens",
        "3000",
        "--batch-max-requests",
        "3",
        "--max-batch-padding-percent",
        "10",
        "--write-qualification",
        receipt.to_str().unwrap(),
        "--json",
    ]));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["work"]["forward_calls"], 1);
    assert_eq!(report["work"]["cross_request_batches"], 1);
    assert_eq!(report["work"]["padded_tokens"], 226);
    assert_eq!(report["work"]["prepared_questions"], 9);
    assert_eq!(report["outcome_calibration"]["questions"], 9);
    assert_eq!(
        report["execution_metadata"]["request_batch_execution"],
        "cpu-causal-right-pad-unpad-joint-v1"
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let bind = listener.local_addr().unwrap().to_string();
    drop(listener);
    let golden_binding = format!("tiny-clef={}", golden.display());
    let record_binding = format!("tiny-clef={}", receipt.display());
    let args = [
        "serve",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "clef",
        "--dtype",
        "fp32",
        "--bind",
        &bind,
        "--max-batch-tokens",
        "3000",
        "--batch-max-requests",
        "3",
        "--max-batch-padding-percent",
        "10",
        "--batch-wait-ms",
        "100",
        "--qualification-golden",
        &golden_binding,
        "--qualification-record",
        &record_binding,
    ];
    let mut server = Server(command(&args).spawn().unwrap());
    let start = Instant::now();
    while TcpStream::connect(&bind).is_err() {
        if let Some(status) = server.0.try_wait().unwrap() {
            let mut error = String::new();
            server
                .0
                .stderr
                .as_mut()
                .unwrap()
                .read_to_string(&mut error)
                .unwrap();
            panic!("qualified server exited {status}: {error}");
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "CPU qualification timed out"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let (_, before) = http(&bind, "GET", "/metrics", "");
    let barrier = Arc::new(Barrier::new(3));
    let handles: Vec<_> = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let bind = bind.clone();
            let barrier = barrier.clone();
            let body = serde_json::to_string(&case["request"]).unwrap();
            std::thread::spawn(move || {
                barrier.wait();
                http(&bind, "POST", "/v1/systemone", &body)
            })
        })
        .collect();
    for (index, handle) in handles.into_iter().enumerate() {
        let (status, body) = handle.join().unwrap();
        assert_eq!(status, 200, "{body}");
        let actual: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            actual["usage"]["input_tokens"],
            reference["cases"][index]["tokens"]
                .as_array()
                .unwrap()
                .len()
        );
        assert_eq!(actual["usage"]["output_tokens"], 0);
        for (id, q) in reference["cases"][index]["request"]["questions"]
            .as_object()
            .unwrap()
        {
            let expected = &reference["cases"][index]["probabilities"][id];
            if q["type"] == "noul" {
                assert!(
                    (actual["answers"][id]["noul"].as_f64().unwrap()
                        - expected["true"].as_f64().unwrap())
                    .abs()
                        <= 1e-5
                );
            } else {
                for (label, p) in expected.as_object().unwrap() {
                    assert!(
                        (actual["answers"][id]["probabilities"][label]
                            .as_f64()
                            .unwrap()
                            - p.as_f64().unwrap())
                        .abs()
                            <= 1e-5
                    );
                }
            }
        }
    }
    let (_, after) = http(&bind, "GET", "/metrics", "");
    for (name, delta) in [
        ("huncho_cross_request_batch_count", 1),
        ("huncho_padded_batch_count", 1),
        ("huncho_padded_tokens", 226),
        ("huncho_tokens_prefilled", 2481),
        ("huncho_questions_prepared", 9),
    ] {
        assert_eq!(
            counter(&after, name) - counter(&before, name),
            delta,
            "{name}: {after}"
        );
    }
    drop(server); // Close the actual CLI process before another command.
    let missing = run(command(&[
        "serve",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "clef",
        "--dtype",
        "fp32",
        "--bind",
        "127.0.0.1:0",
        "--max-batch-tokens",
        "3000",
        "--qualification-golden",
        &golden_binding,
    ]));
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr)
        .contains("F5 batching requires --batch-max-requests"));
    let mut unlabeled: Value = serde_json::from_slice(&std::fs::read(&golden).unwrap()).unwrap();
    for c in unlabeled["cases"].as_array_mut().unwrap() {
        c.as_object_mut().unwrap().remove("targets");
    }
    std::fs::write(&golden, serde_json::to_vec(&unlabeled).unwrap()).unwrap();
    let rejected = run(command(&args));
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("observed target labels"));
}

#[test]
fn real_cpu_benchmark_collates_whole_requests_on_independent_replicas() {
    let (_tmp, manifest, _golden, _reference) = package();
    let output = run(command(&[
        "bench",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "clef",
        "--dtype",
        "fp32",
        "--questions",
        "3",
        "--workload",
        "mixed",
        "--iterations",
        "6",
        "--concurrency",
        "2",
        "--replicas",
        "2",
        "--batch-max-requests",
        "3",
        "--max-batch-tokens",
        "16384",
        "--max-batch-padding-percent",
        "10",
        "--json",
    ]));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["iterations"], 6);
    assert_eq!(report["batch_max_requests"], 3);
    assert_eq!(report["replicas"], 2);
    assert_eq!(report["work"]["forward_calls"], 2);
    assert_eq!(report["work"]["cross_request_batches"], 2);
    assert_eq!(report["work"]["prepared_questions"], 18);
    for replica in report["replica_work"].as_array().unwrap() {
        assert_eq!(replica["requests"], 3);
        assert_eq!(replica["work"]["forward_calls"], 1);
        assert_eq!(replica["work"]["cross_request_batches"], 1);
    }
    assert!(report["mean_ms"].as_f64().unwrap() > 0.0);
    assert!(report["measurement"]
        .as_str()
        .unwrap()
        .contains("entire group's completion"));
    let vacuous = run(command(&[
        "bench",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "clef",
        "--dtype",
        "fp32",
        "--iterations",
        "2",
        "--batch-max-requests",
        "2",
        "--max-batch-tokens",
        "1",
        "--json",
    ]));
    assert!(!vacuous.status.success());
    assert!(String::from_utf8_lossy(&vacuous.stderr).contains("submitted no actual batch"));
}
