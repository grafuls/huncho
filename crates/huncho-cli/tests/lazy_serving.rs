//! Real CPU cold loads with fixed upstream vectors. Synthetic labels exercise
//! gates only; they do not qualify released Clef or any device/precision.
#![cfg(all(feature = "clef", feature = "qualification"))]
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn package() -> (tempfile::TempDir, PathBuf, PathBuf, Value) {
    let root = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/tiny_clef"
    ));
    let tmp = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
        }
    }
    let manifest = tmp.path().join("huncho-model.json");
    let mut m: Value = serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    m["calibration"]["default"]["status"] = json!("fit");
    std::fs::write(&manifest, serde_json::to_vec(&m).unwrap()).unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let cases: Vec<_> = reference["cases"].as_array().unwrap().iter().enumerate().map(|(index, case)| {
        let mut expected = case["probabilities"].clone();
        for (id, question) in case["request"]["questions"].as_object().unwrap() {
            if question["type"] == "noul" {
                expected[id] = json!({"yes":case["probabilities"][id]["true"], "no":case["probabilities"][id]["false"]});
            }
        }
        let labels: serde_json::Map<String, Value> = expected.as_object().unwrap().iter()
            .map(|(id, probabilities)| (id.clone(), json!(probabilities.as_object().unwrap().keys().next().unwrap()))).collect();
        json!({"id":index.to_string(), "request":case["request"], "expected":expected, "targets":labels})
    }).collect();
    let golden = tmp.path().join("labeled-gate-fixture.json");
    std::fs::write(
        &golden,
        serde_json::to_vec(&json!({"schema_version":"1.0", "family":"F5", "cases":cases})).unwrap(),
    )
    .unwrap();
    (tmp, manifest, golden, reference)
}

fn command(manifest: &Path, golden: &Path, bind: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_huncho"));
    command
        .env("HUNCHO_DEVICE", "cpu")
        .env("RAYON_NUM_THREADS", "1")
        .env("CANDLE_NUM_THREADS", "1")
        .env("HUNCHO_ATTENTION_QUERY_ROWS", "0")
        .env("HUNCHO_ONNX_EP", "cpu")
        .env("HUNCHO_PREFILL_CHUNK_TOKENS", "0")
        .env_remove("HUNCHO_CPU_DELTA_RULE")
        .env_remove("HUNCHO_CPU_CAUSAL_CONV")
        .env_remove("HUNCHO_CPU_FUSED_GATE")
        .env_remove("HUNCHO_BASE_CACHE_BYTES")
        .env_remove("HUNCHO_REPLICAS")
        .env_remove("HUNCHO_COOPERATIVE_PREFILL")
        .env_remove("HUNCHO_CPU_BLAS_LIBRARY")
        .env_remove("HUNCHO_CPU_BLAS_THREADS")
        .args([
            "serve",
            "--lazy",
            "--manifest",
            manifest.to_str().unwrap(),
            "--backend",
            "clef",
            "--dtype",
            "fp32",
            "--bind",
            bind,
            "--idle-evict-secs",
            "1",
            "--qualification-golden",
            &format!("tiny-clef={}", golden.display()),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    command
}

fn http(bind: &str, method: &str, path: &str, body: &str) -> (u16, Value) {
    let mut socket = TcpStream::connect(bind).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    write!(socket, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut response = String::new();
    socket.read_to_string(&mut response).unwrap();
    let (headers, json) = response.split_once("\r\n\r\n").unwrap();
    (
        headers.split_whitespace().nth(1).unwrap().parse().unwrap(),
        serde_json::from_str(json).unwrap(),
    )
}

fn start(mut command: Command, bind: &str) -> Server {
    let mut child = command.spawn().unwrap();
    let start = Instant::now();
    while TcpStream::connect(bind).is_err() {
        if let Some(status) = child.try_wait().unwrap() {
            let output = child.wait_with_output().unwrap();
            panic!(
                "lazy server exited {status}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        if start.elapsed() > Duration::from_secs(10) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("lazy startup hung");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    Server(child)
}

fn await_cold(bind: &str) {
    let start = Instant::now();
    loop {
        let (_, models) = http(bind, "GET", "/v1/models", "");
        if models["models"][0]["residency"] == "cold" {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "idle model retained: {models}"
        );
        std::thread::sleep(Duration::from_millis(30));
    }
}

#[test]
fn actual_cpu_cold_reloads_requalify_frozen_vectors_and_fail_closed_on_changed_inputs() {
    let (_tmp, manifest, golden, reference) = package();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let bind = listener.local_addr().unwrap().to_string();
    drop(listener);
    let _server = start(command(&manifest, &golden, &bind), &bind);
    let (_, models) = http(&bind, "GET", "/v1/models", "");
    assert_eq!(models["models"][0]["residency"], "cold");
    let request = serde_json::to_string(&reference["cases"][0]["request"]).unwrap();
    let (status, first) = http(&bind, "POST", "/v1/systemone", &request);
    assert_eq!(status, 200, "{first}");
    await_cold(&bind);
    let (status, second) = http(&bind, "POST", "/v1/systemone", &request);
    assert_eq!(status, 200, "{second}");
    assert_eq!(
        first, second,
        "cold reload must preserve the full wire response"
    );
    await_cold(&bind);
    std::fs::write(&golden, b"{}").unwrap();
    let (status, error) = http(&bind, "POST", "/v1/systemone", &request);
    assert_eq!(status, 503);
    assert!(error["error"]["message"]
        .as_str()
        .unwrap()
        .contains("pinned lazy input changed"));
    let (_, models) = http(&bind, "GET", "/v1/models", "");
    assert_eq!(models["models"][0]["residency"], "failed");
}

#[test]
fn lazy_registration_and_preload_reject_incomplete_or_drifted_observed_vectors() {
    let (_tmp, manifest, golden, _) = package();
    let mut suite: Value = serde_json::from_slice(&std::fs::read(&golden).unwrap()).unwrap();
    suite["cases"][0].as_object_mut().unwrap().remove("targets");
    std::fs::write(&golden, serde_json::to_vec(&suite).unwrap()).unwrap();
    let output = command(&manifest, &golden, "127.0.0.1:0").output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("complete observed-label"));
    let (_tmp2, manifest2, golden2, _) = package();
    let mut drifted: Value = serde_json::from_slice(&std::fs::read(&golden2).unwrap()).unwrap();
    let values = drifted["cases"][0]["expected"]["a_urgent"]
        .as_object_mut()
        .unwrap();
    values.insert("yes".into(), json!(0.9));
    values.insert("no".into(), json!(0.1));
    std::fs::write(&golden2, serde_json::to_vec(&drifted).unwrap()).unwrap();
    let output = command(&manifest2, &golden2, "127.0.0.1:0")
        .args(["--preload", "tiny-clef"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("qualification failed"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
