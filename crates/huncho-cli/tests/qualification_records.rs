//! Real CPU checkpoint receipts bind startup evidence without bypassing gates.
#![cfg(all(feature = "qualification", feature = "clef"))]

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Running(Child);
impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_huncho"));
    command
        .env("HUNCHO_DEVICE", "cpu")
        .env("RAYON_NUM_THREADS", "1")
        .env("CANDLE_NUM_THREADS", "1")
        .env_remove("HUNCHO_PROJECTION_CHUNK_ROWS")
        .env_remove("HUNCHO_ATTENTION_FP32")
        .env_remove("HUNCHO_PREFIX_CACHE")
        .env_remove("HUNCHO_MAX_BATCH_TOKENS")
        .env_remove("HUNCHO_BATCH_MAX_REQUESTS")
        .env_remove("HUNCHO_MAX_PREPARED_PER_MODEL")
        .env_remove("HUNCHO_CANDIDATE_READOUT");
    command
}

fn wait_exit(mut child: Running) -> std::process::Output {
    let start = Instant::now();
    while child.0.try_wait().unwrap().is_none() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "command unexpectedly started serving or hung"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    // Pipes remain readable after try_wait; drain before dropping the guard.
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    child
        .0
        .stdout
        .take()
        .unwrap()
        .read_to_end(&mut stdout)
        .unwrap();
    child
        .0
        .stderr
        .take()
        .unwrap()
        .read_to_end(&mut stderr)
        .unwrap();
    std::process::Output {
        status: child.0.wait().unwrap(),
        stdout,
        stderr,
    }
}

#[test]
fn real_cpu_receipts_reject_stale_inputs_options_goldens_and_outcome_substitution() {
    let tmp = tempfile::tempdir().unwrap();
    let package = tmp.path().join("package");
    std::fs::create_dir(&package).unwrap();
    let fixture = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/tiny_kev"
    ));
    for file in [
        "config.json",
        "model.safetensors",
        "adapter_config.json",
        "adapter_model.safetensors",
        "head.pt",
        "tokenizer.json",
    ] {
        std::fs::copy(fixture.join(file), package.join(file)).unwrap();
    }
    let mut manifest: Value = serde_json::from_slice(
        &std::fs::read(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/mock-model/huncho-model.json"
        )))
        .unwrap(),
    )
    .unwrap();
    manifest["name"] = json!("tiny-kev");
    manifest["family"] = json!("F2");
    manifest["head"] = json!({"kind":"pointer", "weights":"head.pt", "width":4});
    manifest["backbone"]["artifacts"] =
        json!({"candle":[{"path":"adapter_model.safetensors","dtype":"fp32"}]});
    manifest["backbone"]["tokenizer"] = json!("tokenizer.json");
    manifest["backbone"]["max_context"] = json!(512);
    manifest["prompt_contract"]["template"] = json!("kev-v1");
    manifest["prompt_contract"]["state_budget"] = json!(512);
    manifest["prompt_contract"]["head_budget"] = json!(512);
    manifest["calibration"] =
        json!({"default":{"temperature":2.40605,"confidence":"peak","status":"fit"},"entries":{}});
    let manifest_path = package.join("huncho-model.json");
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(fixture.join("golden.json")).unwrap()).unwrap();
    let tokenizer =
        huncho_core::tokenizer::HfTokenizer::from_file_unbounded(package.join("tokenizer.json"))
            .unwrap();
    let typed: huncho_core::manifest::ModelManifest =
        serde_json::from_value(manifest.clone()).unwrap();
    let mut cases = Vec::new();
    for (index, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let request: huncho_core::contract::SystemOneRequest =
            serde_json::from_value(case["request"].clone()).unwrap();
        let expected: std::collections::BTreeMap<String, std::collections::BTreeMap<String, f32>> =
            request
                .questions
                .iter()
                .zip(case["rows"].as_array().unwrap())
                .map(|((id, question), row)| {
                    let prompt = huncho_core::prompt::formatter_for(&typed)
                        .build(&request.state, question, &tokenizer)
                        .unwrap();
                    let probabilities: Vec<f32> =
                        serde_json::from_value(row["probabilities"].clone()).unwrap();
                    (
                        id.clone(),
                        prompt
                            .candidates
                            .into_iter()
                            .zip(probabilities)
                            .map(|(candidate, p)| (candidate.label, p))
                            .collect(),
                    )
                })
                .collect();
        cases.push(json!({"id":index.to_string(), "request":request, "expected":expected}));
    }
    let golden = tmp.path().join("golden.json");
    std::fs::write(
        &golden,
        serde_json::to_vec(&json!({"schema_version":"1.0","family":"F2","cases":cases})).unwrap(),
    )
    .unwrap();
    let receipt = tmp.path().join("receipt.json");
    let generate = || {
        command()
            .args([
                "conform",
                "--model",
                package.to_str().unwrap(),
                "--backend",
                "candle",
                "--dtype",
                "fp32",
                "--golden",
                golden.to_str().unwrap(),
                "--write-qualification",
                receipt.to_str().unwrap(),
                "--json",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };
    let output = wait_exit(Running(generate()));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    let bytes = std::fs::read(&receipt).unwrap();
    let record: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(record["outcome_gates_passed"], false);
    for role in [
        "base/model.safetensors",
        "adapter/weights",
        "head",
        "tokenizer",
    ] {
        assert_eq!(
            record["identity"]["artifacts"][role]["sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
    }
    let output = wait_exit(Running(generate()));
    assert!(!output.status.success());
    assert_eq!(std::fs::read(&receipt).unwrap(), bytes);

    let record_binding = format!("tiny-kev={}", receipt.display());
    let golden_binding = format!("tiny-kev={}", golden.display());
    let serve = |extra: &[&str]| {
        let mut command = command();
        command
            .args([
                "serve",
                "--model",
                package.to_str().unwrap(),
                "--backend",
                "candle",
                "--dtype",
                "fp32",
                "--qualification-record",
                &record_binding,
                "--qualification-golden",
                &golden_binding,
            ])
            .args(extra);
        command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };
    let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let address_string = address.to_string();
    let mut running = Running(serve(&["--bind", &address_string]));
    let start = Instant::now();
    loop {
        if let Ok(mut socket) = std::net::TcpStream::connect(address) {
            socket
                .set_read_timeout(Some(Duration::from_secs(1)))
                .unwrap();
            socket
                .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                .unwrap();
            let mut response = String::new();
            socket.read_to_string(&mut response).unwrap();
            assert!(response.contains("200 OK"));
            break;
        }
        assert!(
            running.0.try_wait().unwrap().is_none(),
            "matching receipt failed startup"
        );
        assert!(start.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(running);
    assert!(
        !wait_exit(Running(serve(&["--max-prepared-per-model", "1"])))
            .status
            .success()
    );
    let old_golden = std::fs::read(&golden).unwrap();
    let mut changed = old_golden.clone();
    changed.push(b' ');
    std::fs::write(&golden, &changed).unwrap();
    let output = wait_exit(Running(serve(&[])));
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("does not match"));
    std::fs::write(&golden, &old_golden).unwrap();
    let mut substituted = record.clone();
    substituted["outcome_gates_passed"] = json!(true);
    std::fs::write(&receipt, serde_json::to_vec(&substituted).unwrap()).unwrap();
    let output = wait_exit(Running(serve(&[])));
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("outcome gates"));
    std::fs::write(&receipt, &bytes).unwrap();
    // A source config mutation that leaves actual model arithmetic unchanged
    // still invalidates the artifact identity, even after fresh conformance.
    let config_path = package.join("config.json");
    let mut config = std::fs::read(&config_path).unwrap();
    config.push(b' ');
    std::fs::write(&config_path, config).unwrap();
    let output = wait_exit(Running(serve(&[])));
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("does not match"));
}
