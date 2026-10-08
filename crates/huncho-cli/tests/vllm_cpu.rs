//! Actual optional CPU vLLM CLI gates; synthetic fixtures remain Pending on disk.
#![cfg(feature = "vllm")]
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};
fn package() -> (tempfile::TempDir, PathBuf, PathBuf, Value) {
    let root = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/vllm_cpu"
    ));
    let tmp = tempfile::tempdir().unwrap();
    fn copy(source: &Path, target: &Path) {
        std::fs::create_dir_all(target).unwrap();
        for item in std::fs::read_dir(source).unwrap() {
            let item = item.unwrap();
            let dest = target.join(item.file_name());
            if item.file_type().unwrap().is_dir() {
                copy(&item.path(), &dest);
            } else {
                std::fs::copy(item.path(), dest).unwrap();
            }
        }
    }
    copy(root, tmp.path());
    let manifest = tmp.path().join("huncho-model.json");
    let reference: Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let mut cases = Vec::new();
    for (index, c) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let mut expected = serde_json::Map::new();
        let mut targets = serde_json::Map::new();
        for ((id, q), r) in c["request"]["questions"]
            .as_object()
            .unwrap()
            .iter()
            .zip(c["rows"].as_array().unwrap())
        {
            let keys: Vec<String> = match q["type"].as_str().unwrap() {
                "choice" => q["criteria"].as_object().unwrap().keys().cloned().collect(),
                "noul" => vec!["no".into(), "yes".into()],
                "score" => (0..q["criteria"].as_array().unwrap().len())
                    .map(|i| i.to_string())
                    .collect(),
                _ => panic!(),
            };
            let probabilities: serde_json::Map<String, Value> = keys
                .iter()
                .cloned()
                .zip(r["probabilities"].as_array().unwrap().iter().cloned())
                .collect();
            targets.insert(id.clone(), json!(keys[0]));
            expected.insert(id.clone(), probabilities.into());
        }
        cases.push(json!({"id":index.to_string(),"request":c["request"],"expected":expected,"targets":targets}));
    }
    let mut duplicate = cases[0].clone();
    duplicate["id"] = json!("duplicate-shape");
    cases.push(duplicate);
    let golden = tmp.path().join("synthetic-labeled.json");
    std::fs::write(
        &golden,
        serde_json::to_vec(&json!({"schema_version":"1.0","family":"F2","cases":cases})).unwrap(),
    )
    .unwrap();
    (tmp, manifest, golden, reference)
}
fn command(args: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_huncho"));
    c.args(args)
        .env("HUNCHO_DEVICE", "cpu")
        .env("HUNCHO_VLLM_THREADS", "2")
        .env("HUNCHO_VLLM_BATCH_ROWS", "4")
        .env("HUNCHO_VLLM_KV_BYTES", "67108864")
        .env("HUNCHO_ATTENTION_QUERY_ROWS", "0")
        .env("HUNCHO_GROUPED_GQA", "0")
        .env_remove("HUNCHO_REPLICAS")
        .env_remove("HUNCHO_BACKEND")
        .env_remove("HUNCHO_CPU_BLAS_LIBRARY")
        .env_remove("HUNCHO_CPU_BLAS_THREADS")
        .env_remove("HUNCHO_LAYA_SELECTED_HEAD")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    c
}
fn run(mut command: Command) -> Output {
    // Drain both pipes while the process runs; native errors can be verbose.
    let mut child = command.spawn().unwrap();
    let mut out = child.stdout.take().unwrap();
    let mut err = child.stderr.take().unwrap();
    let stdout = std::thread::spawn(move || {
        let mut b = Vec::new();
        out.read_to_end(&mut b).unwrap();
        b
    });
    let stderr = std::thread::spawn(move || {
        let mut b = Vec::new();
        err.read_to_end(&mut b).unwrap();
        b
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() > Duration::from_secs(180) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("CPU CLI timed out");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Output {
        status,
        stdout: stdout.join().unwrap(),
        stderr: stderr.join().unwrap(),
    }
}
struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn http(bind: &str, method: &str, path: &str, body: &str) -> (u16, Value) {
    let mut socket = TcpStream::connect(bind).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(60)))
        .unwrap();
    write!(socket,"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
    let mut result = String::new();
    socket.read_to_string(&mut result).unwrap();
    let (headers, body) = result.split_once("\r\n\r\n").unwrap();
    (
        headers.split_whitespace().nth(1).unwrap().parse().unwrap(),
        serde_json::from_str(body).unwrap(),
    )
}
#[test]
fn optional_backend_is_not_silently_replaced_and_requires_explicit_cpu_interpreter() {
    let (_tmp, manifest, _golden, _reference) = package();
    let mut c = command(&[
        "bench",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "vllm",
        "--dtype",
        "bf16",
        "--iterations",
        "1",
    ]);
    c.env_remove("HUNCHO_VLLM_PYTHON");
    let result = run(c);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("explicit HUNCHO_VLLM_PYTHON"));
}
#[test]
#[ignore = "requires explicitly pinned optional CPU vLLM/Python environment"]
fn real_cpu_cli_qualifies_raw_pooling_and_refuses_unqualified_serving() {
    assert!(std::env::var_os("HUNCHO_VLLM_PYTHON").is_some());
    let (tmp, manifest, golden, reference) = package();
    let manifest_s = manifest.to_str().unwrap();
    let binding = format!("tiny-vllm-kev={}", golden.display());
    let pending = run(command(&[
        "serve",
        "--manifest",
        manifest_s,
        "--backend",
        "vllm",
        "--dtype",
        "bf16",
        "--bind",
        "127.0.0.1:0",
        "--qualification-golden",
        &binding,
    ]));
    assert!(!pending.status.success());
    assert!(String::from_utf8_lossy(&pending.stderr).contains("is pending"));
    // Pending analysis collects actual raw rows without fitting or changing
    // the package. The labels below exercise plumbing, not outcome calibration.
    let original_manifest = std::fs::read(&manifest).unwrap();
    let suite: Value = serde_json::from_slice(&std::fs::read(&golden).unwrap()).unwrap();
    let fitting = tmp.path().join("fit-inputs.jsonl");
    let mut file = std::fs::File::create(&fitting).unwrap();
    for case in suite["cases"].as_array().unwrap() {
        writeln!(
            file,
            "{}",
            json!({"id":case["id"],"request":case["request"],"targets":case["targets"]})
        )
        .unwrap();
    }
    drop(file);
    let captured = tmp.path().join("captured");
    let capture = run(command(&[
        "capture-logits",
        "--model",
        manifest_s,
        "--backend",
        "vllm",
        "--dtype",
        "bf16",
        "--data",
        fitting.to_str().unwrap(),
        "--output",
        captured.to_str().unwrap(),
    ]));
    assert!(
        capture.status.success(),
        "{}",
        String::from_utf8_lossy(&capture.stderr)
    );
    assert_eq!(std::fs::read(&manifest).unwrap(), original_manifest);
    let audit: Value = serde_json::from_slice(&capture.stdout).unwrap();
    assert_eq!(audit["qualified"], false);
    assert_eq!(audit["questions"], 9);
    assert_eq!(audit["identity"]["backend"], "vllm");
    assert_eq!(audit["identity"]["device"], "CPU");
    assert_eq!(audit["identity"]["dtype"], "bf16");
    assert_eq!(audit["work"]["forward_calls"], 9);
    assert_eq!(audit["work"]["batch_calls"], 0);
    let fit: Value =
        serde_json::from_slice(&std::fs::read(captured.join("fit.json")).unwrap()).unwrap();
    assert_eq!(fit["rows"].as_array().unwrap().len(), 9);
    assert!(fit["targets"].as_array().unwrap().iter().all(|t| t == 0));
    for row in fit["rows"].as_array().unwrap() {
        assert!(row
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v.as_f64().is_some_and(f64::is_finite)));
    }
    assert_eq!(
        fit["qtypes"],
        json!(["choice", "noul", "score", "choice", "noul", "score", "choice", "noul", "score"])
    );
    let lines = std::fs::read_to_string(captured.join("fit-logits.jsonl")).unwrap();
    assert_eq!(lines.lines().count(), 9);
    for line in lines.lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        assert_eq!(row["target"], 0);
        assert_eq!(
            row["labels"].as_array().unwrap().len(),
            row["logits"].as_array().unwrap().len()
        );
    }
    // Fitted status only in the temporary synthetic package. No temperature changes.
    let mut m: Value = serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    m["calibration"]["entries"]["vllm:bf16"]["status"] = json!("fit");
    std::fs::write(&manifest, serde_json::to_vec(&m).unwrap()).unwrap();
    let receipt = tmp.path().join("receipt.json");
    let result = run(command(&[
        "conform",
        "--manifest",
        manifest_s,
        "--backend",
        "vllm",
        "--dtype",
        "bf16",
        "--golden",
        golden.to_str().unwrap(),
        "--max-batch-tokens",
        "4096",
        "--batch-max-requests",
        "3",
        "--write-qualification",
        receipt.to_str().unwrap(),
        "--json",
    ]));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["device"], "CPU");
    assert_eq!(report["dtype"], "bf16");
    assert_eq!(report["execution_metadata"]["vllm_decode"], "disabled");
    assert_eq!(report["outcome_calibration"]["questions"], 9);
    assert!(report["work"]["cross_request_batches"].as_u64().unwrap() > 0);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let bind = listener.local_addr().unwrap().to_string();
    drop(listener);
    let record = format!("tiny-vllm-kev={}", receipt.display());
    let mut server = Server(
        command(&[
            "serve",
            "--manifest",
            manifest_s,
            "--backend",
            "vllm",
            "--dtype",
            "bf16",
            "--bind",
            &bind,
            "--max-batch-tokens",
            "4096",
            "--batch-max-requests",
            "3",
            "--qualification-golden",
            &binding,
            "--qualification-record",
            &record,
        ])
        .spawn()
        .unwrap(),
    );
    let start = Instant::now();
    while TcpStream::connect(&bind).is_err() {
        if let Some(status) = server.0.try_wait().unwrap() {
            let mut err = String::new();
            server
                .0
                .stderr
                .as_mut()
                .unwrap()
                .read_to_string(&mut err)
                .unwrap();
            panic!("CPU server exited {status}: {err}");
        }
        assert!(start.elapsed() < Duration::from_secs(180));
        std::thread::sleep(Duration::from_millis(20));
    }
    let body = serde_json::to_string(&reference["cases"][0]["request"]).unwrap();
    let (status, response) = http(&bind, "POST", "/v1/systemone", &body);
    assert_eq!(status, 200, "{response}");
    assert_eq!(response["usage"]["output_tokens"], 0);
    let tokens: usize = reference["cases"][0]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["tokens"].as_array().unwrap().len())
        .sum();
    assert_eq!(response["usage"]["input_tokens"], tokens);
    drop(server);
    let bench = run(command(&[
        "bench",
        "--manifest",
        manifest_s,
        "--backend",
        "vllm",
        "--dtype",
        "bf16",
        "--iterations",
        "4",
        "--questions",
        "3",
        "--workload",
        "mixed",
        "--batch-max-requests",
        "2",
        "--max-batch-tokens",
        "4096",
        "--json",
    ]));
    assert!(
        bench.status.success(),
        "{}",
        String::from_utf8_lossy(&bench.stderr)
    );
    let value: Value = serde_json::from_slice(&bench.stdout).unwrap();
    assert_eq!(value["iterations"], 4);
    assert!(value["work"]["cross_request_batches"].as_u64().unwrap() > 0);
    m["calibration"]["default"]["status"] = json!("fit");
    m["calibration"]["entries"]
        .as_object_mut()
        .unwrap()
        .remove("vllm:bf16");
    std::fs::write(&manifest, serde_json::to_vec(&m).unwrap()).unwrap();
    let default_only = run(command(&[
        "serve",
        "--manifest",
        manifest_s,
        "--backend",
        "vllm",
        "--dtype",
        "bf16",
        "--bind",
        "127.0.0.1:0",
        "--qualification-golden",
        &binding,
    ]));
    assert!(!default_only.status.success());
    assert!(String::from_utf8_lossy(&default_only.stderr)
        .contains("explicit fitted/refitted vllm:bf16"));
    m["calibration"]["entries"]["vllm:bf16"] =
        json!({"temperature":2.406050072164233,"confidence":"peak","status":"fit"});
    std::fs::write(&manifest, serde_json::to_vec(&m).unwrap()).unwrap();
    let missing = run(command(&[
        "serve",
        "--manifest",
        manifest_s,
        "--backend",
        "vllm",
        "--dtype",
        "bf16",
        "--bind",
        "127.0.0.1:0",
    ]));
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("requires --qualification-golden"));
    let mut suite: Value = serde_json::from_slice(&std::fs::read(&golden).unwrap()).unwrap();
    for case in suite["cases"].as_array_mut().unwrap() {
        case.as_object_mut().unwrap().remove("targets");
    }
    std::fs::write(&golden, serde_json::to_vec(&suite).unwrap()).unwrap();
    let unlabeled = run(command(&[
        "serve",
        "--manifest",
        manifest_s,
        "--backend",
        "vllm",
        "--dtype",
        "bf16",
        "--bind",
        "127.0.0.1:0",
        "--qualification-golden",
        &binding,
    ]));
    assert!(!unlabeled.status.success());
    assert!(String::from_utf8_lossy(&unlabeled.stderr).contains("observed target labels"));
}
