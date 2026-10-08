//! CPU-only process coverage for interleaved qualification and serving refusal.
#![cfg(feature = "clef")]
use serde_json::{json, Value};
use std::{
    path::Path,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

fn run(args: &[&str], chunk: &str) -> Output {
    run_profile(args, chunk, "0")
}
fn run_profile(args: &[&str], chunk: &str, query_rows: &str) -> Output {
    run_attention_profile(args, chunk, query_rows, "0")
}
fn run_attention_profile(args: &[&str], chunk: &str, query_rows: &str, grouped: &str) -> Output {
    run_storage_profile(args, chunk, query_rows, grouped, "0")
}
fn run_storage_profile(
    args: &[&str],
    chunk: &str,
    query_rows: &str,
    grouped: &str,
    pages: &str,
) -> Output {
    run_adapter_profile(args, chunk, query_rows, grouped, pages, "0")
}
fn run_adapter_profile(
    args: &[&str],
    chunk: &str,
    query_rows: &str,
    grouped: &str,
    pages: &str,
    lora: &str,
) -> Output {
    run_page_kernel_profile(args, chunk, query_rows, grouped, pages, lora, "0")
}
fn run_page_kernel_profile(
    args: &[&str],
    chunk: &str,
    query_rows: &str,
    grouped: &str,
    pages: &str,
    lora: &str,
    direct: &str,
) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env("HUNCHO_DEVICE", "cpu")
        .env("HUNCHO_PREFILL_CHUNK_TOKENS", chunk)
        .env("HUNCHO_ATTENTION_QUERY_ROWS", query_rows)
        .env("HUNCHO_GROUPED_GQA", grouped)
        .env("HUNCHO_KV_PAGE_TOKENS", pages)
        .env("HUNCHO_DIRECT_PAGED_ATTENTION", direct)
        .env("HUNCHO_RUNTIME_LORA", lora)
        .env("RAYON_NUM_THREADS", "1")
        .env("CANDLE_NUM_THREADS", "1")
        .env_remove("HUNCHO_CPU_DELTA_RULE")
        .env_remove("HUNCHO_CPU_CAUSAL_CONV")
        .env_remove("HUNCHO_CPU_FUSED_GATE")
        .env_remove("HUNCHO_COOPERATIVE_PREFILL")
        .env_remove("HUNCHO_REPLICAS")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let started = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if started.elapsed() > Duration::from_secs(20) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("cooperative qualification hung or unqualified serving started");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}

fn package() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let root = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/tiny_kev"
    ));
    let tmp = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
        }
    }
    let mut manifest: Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/mock-model/huncho-model.json"
        ))
        .unwrap(),
    )
    .unwrap();
    manifest["name"] = json!("tiny-kev");
    manifest["family"] = json!("F2");
    manifest["head"] = json!({"kind":"pointer","weights":"head.pt","width":4});
    manifest["backbone"]["source"] = json!({"kind":"local","path":"."});
    manifest["backbone"]["artifacts"] =
        json!({"candle":[{"path":"model.safetensors","dtype":"fp32"}]});
    manifest["backbone"]["hidden_size"] = json!(16);
    manifest["backbone"]["max_context"] = json!(512);
    manifest["backbone"]["tokenizer"] = json!("tokenizer.json");
    manifest["prompt_contract"]["template"] = json!("kev-v1");
    manifest["prompt_contract"]["state_budget"] = json!(512);
    manifest["prompt_contract"]["head_budget"] = json!(512);
    manifest["calibration"] =
        json!({"default":{"temperature":2.40605,"confidence":"peak","status":"fit"},"entries":{}});
    let manifest_path = tmp.path().join("huncho-model.json");
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    let typed_manifest: huncho_core::manifest::ModelManifest =
        serde_json::from_value(manifest).unwrap();
    let formatter = huncho_core::prompt::formatter_for(&typed_manifest);
    let tokenizer =
        huncho_core::tokenizer::HfTokenizer::from_file_unbounded(root.join("tokenizer.json"))
            .unwrap();
    let cases: Vec<_> = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let request: huncho_core::contract::SystemOneRequest =
                serde_json::from_value(case["request"].clone()).unwrap();
            let expected: serde_json::Map<String, Value> = request
                .questions
                .iter()
                .zip(case["rows"].as_array().unwrap())
                .map(|((id, q), row)| {
                    let prompt = formatter.build(&request.state, q, &tokenizer).unwrap();
                    let probabilities = row["probabilities"].as_array().unwrap();
                    (
                        id.clone(),
                        Value::Object(
                            prompt
                                .candidates
                                .into_iter()
                                .zip(probabilities)
                                .map(|(c, p)| (c.label, p.clone()))
                                .collect(),
                        ),
                    )
                })
                .collect();
            json!({"id":index.to_string(),"request":case["request"],"expected":expected})
        })
        .collect();
    let golden_path = tmp.path().join("original-probabilities.json");
    std::fs::write(
        &golden_path,
        serde_json::to_vec(&json!({"schema_version":"1.0","family":"F2","cases":cases})).unwrap(),
    )
    .unwrap();
    (tmp, manifest_path, golden_path)
}

#[cfg(feature = "qualification")]
#[test]
fn cached_branch_cli_requires_actual_native_batches_and_fresh_labeled_gates() {
    let (tmp, manifest, original) = package();
    let mut suite: Value = serde_json::from_slice(&std::fs::read(&original).unwrap()).unwrap();
    for case in suite["cases"].as_array_mut().unwrap() {
        for (id, question) in case["request"]["questions"].as_object().unwrap().clone() {
            let duplicate = format!("{id}-duplicate");
            let probabilities = case["expected"][&id].clone();
            case["request"]["questions"][&duplicate] = question;
            case["expected"][&duplicate] = probabilities;
        }
    }
    let path = tmp.path().join("duplicate-frozen-questions.json");
    std::fs::write(&path, serde_json::to_vec(&suite).unwrap()).unwrap();
    let receipt = tmp.path().join("cached-batch-numerical.json");
    let args = [
        "conform",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "candle",
        "--dtype",
        "fp32",
        "--golden",
        path.to_str().unwrap(),
        "--prefix-cache",
        "--max-batch-tokens",
        "4096",
        "--write-qualification",
        receipt.to_str().unwrap(),
        "--json",
    ];
    let output = run_storage_profile(&args, "3", "7", "1", "16");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(
        report["execution_metadata"]["cached_branch_batch"],
        "cpu-kev-equal-suffix-v1"
    );
    assert_eq!(report["work"]["fork_batch_calls"], 6);
    assert_eq!(report["work"]["cache_forks"], 12);
    assert_eq!(report["work"]["cross_request_batches"], 0);
    assert!(
        report["optimization_parity"]["max_prob_delta"]
            .as_f64()
            .unwrap()
            <= 1e-4
    );
    let numerical: Value = serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
    assert_eq!(numerical["outcome_gates_passed"], false);
    let binding = format!("tiny-kev={}", path.display());
    let serve = [
        "serve",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "candle",
        "--dtype",
        "fp32",
        "--prefix-cache",
        "--max-batch-tokens",
        "4096",
        "--qualification-golden",
        &binding,
        "--bind",
        "127.0.0.1:0",
    ];
    let output = run_storage_profile(&serve, "3", "7", "1", "16");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("observed target labels"));
    // Fixture labels test gate plumbing only, never released-model calibration.
    for case in suite["cases"].as_array_mut().unwrap() {
        let labels: serde_json::Map<String, Value> = case["expected"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, p)| {
                (
                    id.clone(),
                    json!(p.as_object().unwrap().keys().next().unwrap()),
                )
            })
            .collect();
        case["targets"] = json!(labels);
    }
    std::fs::write(&path, serde_json::to_vec(&suite).unwrap()).unwrap();
    let labeled_receipt = tmp.path().join("cached-batch-labeled-fixture.json");
    let mut labeled = args.to_vec();
    let receipt_index = labeled
        .iter()
        .position(|s| *s == "--write-qualification")
        .unwrap()
        + 1;
    labeled[receipt_index] = labeled_receipt.to_str().unwrap();
    let output = run_storage_profile(&labeled, "3", "7", "1", "16");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: Value = serde_json::from_slice(&std::fs::read(&labeled_receipt).unwrap()).unwrap();
    assert_eq!(record["outcome_gates_passed"], true);
    let mut tiny = labeled.clone();
    // Correct the token budget by name rather than relying on argument positions.
    let budget = tiny
        .iter()
        .position(|s| *s == "--max-batch-tokens")
        .unwrap()
        + 1;
    tiny[budget] = "1";
    let write = tiny
        .iter()
        .position(|s| *s == "--write-qualification")
        .unwrap();
    tiny.drain(write..write + 2);
    let output = run_storage_profile(&tiny, "3", "7", "1", "16");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("actual native batch"));
    let mut original_labeled = suite.clone();
    for case in original_labeled["cases"].as_array_mut().unwrap() {
        for field in ["expected", "targets"] {
            case[field]
                .as_object_mut()
                .unwrap()
                .retain(|id, _| !id.ends_with("-duplicate"));
        }
        case["request"]["questions"]
            .as_object_mut()
            .unwrap()
            .retain(|id, _| !id.ends_with("-duplicate"));
    }
    let mixed_path = tmp.path().join("original-mixed-typed-fixture.json");
    std::fs::write(&mixed_path, serde_json::to_vec(&original_labeled).unwrap()).unwrap();
    let mut mixed = labeled.clone();
    mixed.extend(["--max-batch-padding-percent", "25"]);
    let gold = mixed.iter().position(|s| *s == "--golden").unwrap() + 1;
    mixed[gold] = mixed_path.to_str().unwrap();
    let write = mixed
        .iter()
        .position(|s| *s == "--write-qualification")
        .unwrap();
    mixed.drain(write..write + 2);
    let output = run_storage_profile(&mixed, "3", "7", "1", "16");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(report["work"]["fork_padded_batch_calls"].as_u64().unwrap() > 0);
    assert!(report["work"]["padded_tokens"].as_u64().unwrap() > 0);
    assert!(
        report["optimization_parity"]["max_prob_delta"]
            .as_f64()
            .unwrap()
            <= 1e-4
    );
    let mut cooperative = mixed.clone();
    cooperative.push("--cooperative-prefill");
    let output = run_storage_profile(&cooperative, "3", "7", "1", "16");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["cooperative_prefill"], true);
    for key in ["prefill_yields", "prefill_interleaves", "fork_batch_calls", "fork_padded_batch_calls"] {
        assert!(report["work"][key].as_u64().unwrap() > 0);
    }
    let mut invalid = serve.to_vec();
    invalid.push("--cooperative-prefill");
    let output = run_storage_profile(&invalid, "0", "7", "1", "16");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("cooperative prefill"));
}

#[cfg(feature = "qualification")]
#[test]
fn native_cpu_cooperative_groups_start_with_fresh_proofs_and_serve_original_typed_probabilities() {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    struct Stop(std::process::Child);
    impl Drop for Stop {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let (_tmp, manifest, golden) = package();
    let mut suite: Value = serde_json::from_slice(&std::fs::read(&golden).unwrap()).unwrap();
    for case in suite["cases"].as_array_mut().unwrap() {
        case["targets"] = Value::Object(
            case["expected"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(id, p)| {
                    (
                        id.clone(),
                        json!(p.as_object().unwrap().keys().next().unwrap()),
                    )
                })
                .collect(),
        );
    }
    std::fs::write(&golden, serde_json::to_vec(&suite).unwrap()).unwrap();
    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let binding = format!("tiny-kev={}", golden.display());
    let mut command = Command::new(env!("CARGO_BIN_EXE_huncho"));
    for (name, value) in [
        ("HUNCHO_DEVICE", "cpu"),
        ("RAYON_NUM_THREADS", "1"),
        ("CANDLE_NUM_THREADS", "1"),
        ("HUNCHO_PREFILL_CHUNK_TOKENS", "3"),
        ("HUNCHO_ATTENTION_QUERY_ROWS", "7"),
        ("HUNCHO_GROUPED_GQA", "1"),
        ("HUNCHO_KV_PAGE_TOKENS", "16"),
        ("HUNCHO_DIRECT_PAGED_ATTENTION", "1"),
        ("HUNCHO_RUNTIME_LORA", "1"),
    ] {
        command.env(name, value);
    }
    for name in [
        "HUNCHO_CPU_BLAS_LIBRARY",
        "HUNCHO_CPU_DELTA_RULE",
        "HUNCHO_CPU_CAUSAL_CONV",
        "HUNCHO_CPU_FUSED_GATE",
        "HUNCHO_REPLICAS",
        "HUNCHO_BATCH_MAX_REQUESTS",
    ] {
        command.env_remove(name);
    }
    let mut server = Stop(
        command
            .args([
                "serve",
                "--manifest",
                manifest.to_str().unwrap(),
                "--backend",
                "candle",
                "--dtype",
                "fp32",
                "--prefix-cache",
                "--cooperative-prefill",
                "--max-batch-tokens",
                "4096",
                "--max-batch-padding-percent",
                "25",
                "--qualification-golden",
                &binding,
                "--bind",
                &address.to_string(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let started = Instant::now();
    let mut stream = loop {
        if let Ok(stream) = TcpStream::connect(address) {
            break stream;
        }
        if let Some(status) = server.0.try_wait().unwrap() {
            let mut error = String::new();
            server
                .0
                .stderr
                .as_mut()
                .unwrap()
                .read_to_string(&mut error)
                .unwrap();
            panic!("native server exited {status}: {error}");
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "native startup qualification hung"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let body = serde_json::to_string(&suite["cases"][0]["request"]).unwrap();
    write!(stream, "POST /v1/systemone HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nX-Huncho-Extensions: true\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let response: Value = serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
    for (id, expected) in suite["cases"][0]["expected"].as_object().unwrap() {
        let answer = &response["answers"][id];
        for (label, p) in expected.as_object().unwrap() {
            let actual = if answer.get("probabilities").is_some() {
                answer["probabilities"][label].as_f64().unwrap()
            } else {
                let yes = answer["noul"].as_f64().unwrap();
                if label == "yes" {
                    yes
                } else {
                    1. - yes
                }
            };
            assert!((actual - p.as_f64().unwrap()).abs() <= 1e-3, "{id}/{label}");
        }
    }
    assert_eq!(response["usage"]["output_tokens"], 0);
    let mut stream = TcpStream::connect(address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    write!(
        stream,
        "GET /metrics HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut metrics = String::new();
    stream.read_to_string(&mut metrics).unwrap();
    for name in [
        "huncho_fork_batch_count",
        "huncho_fork_padded_batch_count",
        "huncho_fork_count",
    ] {
        let value: u64 = metrics
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{name} ")))
            .unwrap_or_else(|| panic!("missing {name}: {metrics}"))
            .trim()
            .parse()
            .unwrap();
        assert!(value > 0);
    }
}

#[test]
fn cooperative_conformance_requires_real_interleaving_on_unchanged_native_goldens() {
    let (tmp, manifest_path, golden_path) = package();
    let manifest = manifest_path.to_str().unwrap();
    let golden = golden_path.to_str().unwrap();
    let output = run(
        &[
            "conform",
            "--manifest",
            manifest,
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--golden",
            golden,
            "--prefix-cache",
            "--cooperative-prefill",
            "--json",
        ],
        "3",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["cooperative_prefill"], true);
    assert!(report["work"]["prefill_yields"].as_u64().unwrap() > 0);
    assert!(report["work"]["prefill_interleaves"].as_u64().unwrap() > 0);
    assert_eq!(
        report["execution_metadata"]["native_execution"],
        "candle-qwen35-v1"
    );
    let mut one: Value = serde_json::from_slice(&std::fs::read(&golden_path).unwrap()).unwrap();
    one["cases"].as_array_mut().unwrap().truncate(1);
    // Arbitrary fixture labels exercise startup gates only, never production
    // calibration. The native frozen probabilities remain unchanged.
    let labels: serde_json::Map<String, Value> = one["cases"][0]["expected"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(id, expected)| {
            (
                id.clone(),
                json!(expected.as_object().unwrap().keys().next().unwrap()),
            )
        })
        .collect();
    one["cases"][0]["targets"] = json!(labels);
    let single_path = tmp.path().join("single-labeled-fixture.json");
    std::fs::write(&single_path, serde_json::to_vec(&one).unwrap()).unwrap();
    let binding = format!("tiny-kev={}", single_path.display());
    let args = [
        "serve",
        "--manifest",
        manifest,
        "--backend",
        "candle",
        "--dtype",
        "fp32",
        "--prefix-cache",
        "--cooperative-prefill",
        "--qualification-golden",
        binding.as_str(),
        "--bind",
        "127.0.0.1:0",
    ];
    let output = run(&args, "3");
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("interleaved"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = run(&args, "0");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("configured"));
}

#[cfg(feature = "qualification")]
#[test]
fn padded_native_cli_reports_actual_work_and_refuses_unlabeled_or_vacuous_startup() {
    let (tmp, manifest_path, golden_path) = package();
    let manifest = manifest_path.to_str().unwrap();
    let golden = golden_path.to_str().unwrap();
    let receipt = tmp.path().join("padded-receipt.json");
    let args = [
        "conform",
        "--manifest",
        manifest,
        "--backend",
        "candle",
        "--dtype",
        "fp32",
        "--golden",
        golden,
        "--max-batch-tokens",
        "1024",
        "--max-batch-padding-percent",
        "25",
        "--write-qualification",
        receipt.to_str().unwrap(),
        "--json",
    ];
    let output = run(&args, "0");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["max_batch_padding_percent"], 25);
    assert!(report["work"]["padded_batch_calls"].as_u64().unwrap() > 0);
    assert!(report["work"]["padded_tokens"].as_u64().unwrap() > 0);
    assert!(
        report["optimization_parity"]["max_prob_delta"]
            .as_f64()
            .unwrap()
            <= 1e-4
    );
    let record: Value = serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
    assert_eq!(record["outcome_gates_passed"], false);
    assert_eq!(record["report"]["max_batch_padding_percent"], 25);
    let output = run(
        &[
            "bench",
            "--manifest",
            manifest,
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--max-batch-tokens",
            "1024",
            "--max-batch-padding-percent",
            "25",
            "--iterations",
            "2",
            "--questions",
            "6",
            "--workload",
            "mixed",
            "--json",
        ],
        "0",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["max_batch_padding_percent"], 25);
    assert!(report["work"]["padded_batch_calls"].as_u64().unwrap() > 0);
    let binding = format!("tiny-kev={golden}");
    let output = run(
        &[
            "serve",
            "--manifest",
            manifest,
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--max-batch-tokens",
            "1024",
            "--max-batch-padding-percent",
            "25",
            "--qualification-golden",
            &binding,
            "--bind",
            "127.0.0.1:0",
        ],
        "0",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("observed target labels"));
    let mut one: Value = serde_json::from_slice(&std::fs::read(&golden_path).unwrap()).unwrap();
    one["cases"].as_array_mut().unwrap().truncate(1);
    let first = one["cases"][0]["request"]["questions"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone();
    one["cases"][0]["request"]["questions"]
        .as_object_mut()
        .unwrap()
        .retain(|id, _| id == &first);
    one["cases"][0]["expected"]
        .as_object_mut()
        .unwrap()
        .retain(|id, _| id == &first);
    let target = one["cases"][0]["expected"][&first]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone();
    one["cases"][0]["targets"] = json!({first:target});
    let singleton = tmp.path().join("singleton-labeled-test.json");
    std::fs::write(&singleton, serde_json::to_vec(&one).unwrap()).unwrap();
    let binding = format!("tiny-kev={}", singleton.display());
    let output = run(
        &[
            "serve",
            "--manifest",
            manifest,
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--max-batch-tokens",
            "1024",
            "--max-batch-padding-percent",
            "25",
            "--qualification-golden",
            &binding,
            "--bind",
            "127.0.0.1:0",
        ],
        "0",
    );
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("actually batches"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(feature = "qualification")]
#[test]
fn cpu_query_block_profile_binds_qualification_and_preserves_unchanged_goldens() {
    let (tmp, manifest, golden) = package();
    let manifest = manifest.to_str().unwrap();
    let receipt = tmp.path().join("query-receipt.json");
    let output = run_profile(
        &[
            "conform",
            "--manifest",
            manifest,
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--golden",
            golden.to_str().unwrap(),
            "--prefix-cache",
            "--cooperative-prefill",
            "--write-qualification",
            receipt.to_str().unwrap(),
            "--json",
        ],
        "3",
        "7",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["execution_metadata"]["attention_query_rows"], "7");
    assert_eq!(
        report["execution_metadata"]["attention_execution"],
        "cpu-query-blocks-v1"
    );
    assert!(
        report["optimization_parity"]["max_prob_delta"]
            .as_f64()
            .unwrap()
            <= 1e-4
    );
    assert!(report["work"]["prefill_interleaves"].as_u64().unwrap() > 0);
    let record: Value = serde_json::from_slice(&std::fs::read(receipt).unwrap()).unwrap();
    assert_eq!(record["outcome_gates_passed"], false);
    assert!(
        std::fs::read_to_string(tmp.path().join("query-receipt.json"))
            .unwrap()
            .contains("HUNCHO_ATTENTION_QUERY_ROWS")
    );
    let binding = format!("tiny-kev={}", golden.display());
    let output = run_profile(
        &[
            "serve",
            "--manifest",
            manifest,
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--qualification-golden",
            &binding,
            "--bind",
            "127.0.0.1:0",
        ],
        "0",
        "7",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("observed target labels"));
    for invalid in ["4097", "-1", "true", "broken"] {
        let output = run_profile(
            &[
                "bench",
                "--manifest",
                manifest,
                "--backend",
                "candle",
                "--iterations",
                "1",
            ],
            "0",
            invalid,
        );
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("HUNCHO_ATTENTION_QUERY_ROWS"));
    }
}

#[cfg(feature = "qualification")]
#[test]
fn grouped_cpu_gqa_binds_fresh_receipts_and_actual_prefix_interleaving() {
    let (tmp, manifest, golden) = package();
    for rows in ["0", "7"] {
        let receipt = tmp.path().join(format!("grouped-{rows}.json"));
        let output = run_attention_profile(
            &[
                "conform",
                "--manifest",
                manifest.to_str().unwrap(),
                "--backend",
                "candle",
                "--dtype",
                "fp32",
                "--golden",
                golden.to_str().unwrap(),
                "--prefix-cache",
                "--cooperative-prefill",
                "--json",
                "--write-qualification",
                receipt.to_str().unwrap(),
            ],
            "3",
            rows,
            "1",
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["passed"], true);
        assert_eq!(
            report["execution_metadata"]["gqa_execution"],
            "cpu-grouped-queries-v1"
        );
        assert!(
            report["optimization_parity"]["max_prob_delta"]
                .as_f64()
                .unwrap()
                <= 1e-4
        );
        assert!(report["work"]["prefill_interleaves"].as_u64().unwrap() > 0);
        let record: Value = serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
        assert_eq!(record["outcome_gates_passed"], false);
        let text = std::fs::read_to_string(&receipt).unwrap();
        assert!(text.contains("HUNCHO_GROUPED_GQA") && text.contains("cpu-grouped-queries-v1"));
    }
    let binding = format!("tiny-kev={}", golden.display());
    let output = run_attention_profile(
        &[
            "serve",
            "--manifest",
            manifest.to_str().unwrap(),
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--qualification-golden",
            &binding,
            "--bind",
            "127.0.0.1:0",
        ],
        "0",
        "0",
        "1",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("observed target labels"));
    let output = run_attention_profile(
        &[
            "bench",
            "--manifest",
            manifest.to_str().unwrap(),
            "--backend",
            "candle",
            "--iterations",
            "1",
        ],
        "0",
        "0",
        "broken",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("HUNCHO_GROUPED_GQA"));
}

#[cfg(feature = "qualification")]
#[test]
fn kv_pages_bind_fresh_receipts_and_actual_prefix_work_and_refuse_vacuous_serving() {
    let (tmp, manifest, golden) = package();
    let manifest = manifest.to_str().unwrap();
    for pages in ["16", "32", "64", "256"] {
        let receipt = tmp.path().join(format!("pages-{pages}.json"));
        let output = run_storage_profile(
            &[
                "conform",
                "--manifest",
                manifest,
                "--backend",
                "candle",
                "--dtype",
                "fp32",
                "--golden",
                golden.to_str().unwrap(),
                "--prefix-cache",
                "--cooperative-prefill",
                "--json",
                "--write-qualification",
                receipt.to_str().unwrap(),
            ],
            "3",
            "7",
            "1",
            pages,
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["passed"], true);
        assert_eq!(
            report["execution_metadata"]["kv_storage"],
            "cpu-cow-pages-materialize-v1"
        );
        assert_eq!(report["execution_metadata"]["kv_page_tokens"], pages);
        assert!(report["work"]["cache_forks"].as_u64().unwrap() > 0);
        assert!(report["work"]["prefill_interleaves"].as_u64().unwrap() > 0);
        assert!(
            report["optimization_parity"]["max_prob_delta"]
                .as_f64()
                .unwrap()
                <= 1e-4
        );
        let record: Value = serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
        assert_eq!(record["outcome_gates_passed"], false);
        let text = std::fs::read_to_string(receipt).unwrap();
        assert!(
            text.contains("HUNCHO_KV_PAGE_TOKENS") && text.contains("cpu-cow-pages-materialize-v1")
        );
    }
    // Synthetic labels prove gate wiring only; original frozen probabilities stay unchanged.
    let mut labeled: Value = serde_json::from_slice(&std::fs::read(&golden).unwrap()).unwrap();
    for case in labeled["cases"].as_array_mut().unwrap() {
        let targets: serde_json::Map<String, Value> = case["expected"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, expected)| {
                (
                    id.clone(),
                    json!(expected.as_object().unwrap().keys().next().unwrap()),
                )
            })
            .collect();
        case["targets"] = json!(targets);
    }
    let labeled_path = tmp.path().join("labeled-fixture.json");
    std::fs::write(&labeled_path, serde_json::to_vec(&labeled).unwrap()).unwrap();
    let receipt = tmp.path().join("labeled-pages.json");
    let output = run_storage_profile(
        &[
            "conform",
            "--manifest",
            manifest,
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--golden",
            labeled_path.to_str().unwrap(),
            "--prefix-cache",
            "--persistent-prefix-bytes",
            "1048576",
            "--write-qualification",
            receipt.to_str().unwrap(),
            "--json",
        ],
        "3",
        "0",
        "0",
        "16",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert!(report["work"]["persistent_prefix_hits"].as_u64().unwrap() > 0);
    assert!(
        report["optimization_parity"]["max_prob_delta"]
            .as_f64()
            .unwrap()
            <= 1e-4
    );
    assert_eq!(report["optimization_parity"]["argmax_agreement"], 1.0);
    let record: Value = serde_json::from_slice(&std::fs::read(receipt).unwrap()).unwrap();
    assert_eq!(record["outcome_gates_passed"], true);
    let binding = format!("tiny-kev={}", golden.display());
    let base = [
        "serve",
        "--manifest",
        manifest,
        "--backend",
        "candle",
        "--dtype",
        "fp32",
        "--qualification-golden",
        &binding,
        "--bind",
        "127.0.0.1:0",
    ];
    let output = run_storage_profile(&base, "0", "0", "0", "16");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires --prefix-cache"));
    let mut prefix = base.to_vec();
    prefix.push("--prefix-cache");
    let output = run_storage_profile(&prefix, "0", "0", "0", "16");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("observed target labels"));
    let output = run_storage_profile(
        &[
            "serve",
            "--manifest",
            manifest,
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--prefix-cache",
            "--bind",
            "127.0.0.1:0",
        ],
        "0",
        "0",
        "0",
        "16",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires --qualification-golden"));
    for invalid in ["1", "15", "17", "257", "-1", "true", "broken"] {
        let output = run_storage_profile(
            &[
                "bench",
                "--manifest",
                manifest,
                "--backend",
                "candle",
                "--iterations",
                "1",
            ],
            "0",
            "0",
            "0",
            invalid,
        );
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("HUNCHO_KV_PAGE_TOKENS"));
    }
}

#[cfg(feature = "qualification")]
#[test]
fn runtime_lora_binds_fresh_qualification_and_keeps_frozen_probabilities_and_refusal_gates() {
    let (tmp, manifest, golden) = package();
    let manifest = manifest.to_str().unwrap();
    for optimized in [false, true] {
        let receipt = tmp.path().join(format!("runtime-lora-{optimized}.json"));
        let mut args = vec![
            "conform",
            "--manifest",
            manifest,
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--golden",
            golden.to_str().unwrap(),
            "--write-qualification",
            receipt.to_str().unwrap(),
            "--json",
        ];
        if optimized {
            args.extend(["--prefix-cache", "--cooperative-prefill"]);
        }
        let output = run_adapter_profile(
            &args,
            if optimized { "3" } else { "0" },
            if optimized { "7" } else { "0" },
            if optimized { "1" } else { "0" },
            if optimized { "16" } else { "0" },
            "1",
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["passed"], true);
        assert_eq!(
            report["execution_metadata"]["adapter_execution"],
            "cpu-fp32-runtime-lora-v1"
        );
        assert_eq!(report["execution_metadata"]["runtime_lora_targets"], "3");
        if optimized {
            assert!(report["work"]["cache_forks"].as_u64().unwrap() > 0);
            assert!(
                report["optimization_parity"]["max_prob_delta"]
                    .as_f64()
                    .unwrap()
                    <= 1e-4
            );
        }
        let record: Value = serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
        assert_eq!(record["outcome_gates_passed"], false);
        assert!(std::fs::read_to_string(receipt)
            .unwrap()
            .contains("HUNCHO_RUNTIME_LORA"));
    }
    let mut labeled: Value = serde_json::from_slice(&std::fs::read(&golden).unwrap()).unwrap();
    for case in labeled["cases"].as_array_mut().unwrap() {
        let targets: serde_json::Map<String, Value> = case["expected"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, expected)| {
                (
                    id.clone(),
                    json!(expected.as_object().unwrap().keys().next().unwrap()),
                )
            })
            .collect();
        case["targets"] = json!(targets);
    }
    let labeled_path = tmp.path().join("labeled-lora-fixture.json");
    std::fs::write(&labeled_path, serde_json::to_vec(&labeled).unwrap()).unwrap();
    let receipt = tmp.path().join("labeled-lora-receipt.json");
    let output = run_adapter_profile(
        &[
            "conform",
            "--manifest",
            manifest,
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--golden",
            labeled_path.to_str().unwrap(),
            "--write-qualification",
            receipt.to_str().unwrap(),
            "--json",
        ],
        "0",
        "0",
        "0",
        "0",
        "1",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: Value = serde_json::from_slice(&std::fs::read(receipt).unwrap()).unwrap();
    assert_eq!(record["outcome_gates_passed"], true); // synthetic plumbing only
    let binding = format!("tiny-kev={}", golden.display());
    for with_golden in [false, true] {
        let mut args = vec![
            "serve",
            "--manifest",
            manifest,
            "--backend",
            "candle",
            "--dtype",
            "fp32",
            "--bind",
            "127.0.0.1:0",
        ];
        if with_golden {
            args.extend(["--qualification-golden", &binding]);
        }
        let output = run_adapter_profile(&args, "0", "0", "0", "0", "1");
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(if with_golden {
                "observed target labels"
            } else {
                "requires --qualification-golden"
            })
        );
    }
    for (dtype, lora, expected) in [
        ("fp16", "1", "runtime LoRA requires"),
        ("fp32", "broken", "HUNCHO_RUNTIME_LORA"),
    ] {
        let output = run_adapter_profile(
            &[
                "bench",
                "--manifest",
                manifest,
                "--backend",
                "candle",
                "--dtype",
                dtype,
                "--iterations",
                "1",
            ],
            "0",
            "0",
            "0",
            "0",
            lora,
        );
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains(expected));
    }
}

#[cfg(feature = "qualification")]
#[test]
fn direct_page_profile_keeps_frozen_gates_and_requires_fresh_labeled_prefix_qualification() {
    let (tmp, manifest, golden) = package();
    let receipt = tmp.path().join("direct-pages-numerical.json");
    let args = [
        "conform",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "candle",
        "--dtype",
        "fp32",
        "--golden",
        golden.to_str().unwrap(),
        "--prefix-cache",
        "--persistent-prefix-bytes",
        "1048576",
        "--json",
        "--write-qualification",
        receipt.to_str().unwrap(),
    ];
    for (pages, chunks, grouped, runtime) in [("16", "0", "0", "0"), ("256", "3", "1", "1")] {
        let profile_receipt = tmp
            .path()
            .join(format!("direct-pages-{pages}-{chunks}.json"));
        let mut profile_args = args.to_vec();
        let output_index = profile_args
            .iter()
            .position(|x| *x == "--write-qualification")
            .unwrap()
            + 1;
        profile_args[output_index] = profile_receipt.to_str().unwrap();
        let output =
            run_page_kernel_profile(&profile_args, chunks, "7", grouped, pages, runtime, "1");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["passed"], true);
        assert_eq!(
            report["execution_metadata"]["kv_storage"],
            "cpu-cow-pages-direct-v1"
        );
        assert_eq!(
            report["execution_metadata"]["paged_attention"],
            "cpu-page-qk-pv-fp32-v1"
        );
        assert!(report["work"]["cache_forks"].as_u64().unwrap() > 0);
        assert!(report["work"]["persistent_prefix_hits"].as_u64().unwrap() > 0);
        assert!(
            report["optimization_parity"]["max_prob_delta"]
                .as_f64()
                .unwrap()
                <= 1e-4
        );
        assert_eq!(report["optimization_parity"]["argmax_agreement"], 1.0);
        let record: Value =
            serde_json::from_slice(&std::fs::read(&profile_receipt).unwrap()).unwrap();
        assert_eq!(record["outcome_gates_passed"], false);
        assert!(std::fs::read_to_string(&profile_receipt)
            .unwrap()
            .contains("HUNCHO_DIRECT_PAGED_ATTENTION"));
    }
    // Synthetic targets prove startup plumbing only, not released calibration.
    let mut suite: Value = serde_json::from_slice(&std::fs::read(&golden).unwrap()).unwrap();
    for case in suite["cases"].as_array_mut().unwrap() {
        let targets: serde_json::Map<String, Value> = case["expected"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, p)| {
                (
                    id.clone(),
                    json!(p.as_object().unwrap().keys().next().unwrap()),
                )
            })
            .collect();
        case["targets"] = json!(targets);
    }
    let labeled = tmp.path().join("direct-pages-labeled-fixture.json");
    std::fs::write(&labeled, serde_json::to_vec(&suite).unwrap()).unwrap();
    let mut labeled_args = args.to_vec();
    let i = labeled_args.iter().position(|x| *x == "--golden").unwrap() + 1;
    labeled_args[i] = labeled.to_str().unwrap();
    let output = run_page_kernel_profile(&labeled_args, "3", "7", "1", "16", "0", "1");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: Value = serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
    assert_eq!(record["outcome_gates_passed"], true);
    let binding = format!("tiny-kev={}", golden.display());
    let serving = [
        "serve",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "candle",
        "--dtype",
        "fp32",
        "--qualification-golden",
        &binding,
        "--bind",
        "127.0.0.1:0",
    ];
    let output = run_page_kernel_profile(&serving, "0", "7", "0", "16", "0", "1");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires --prefix-cache"));
    let mut prefix = serving.to_vec();
    prefix.push("--prefix-cache");
    let output = run_page_kernel_profile(&prefix, "0", "7", "0", "16", "0", "1");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("observed target labels"));
    // Missing prerequisites and precision/backend changes fail before device selection.
    for (pages, queries, dtype, direct) in [
        ("0", "7", "fp32", "1"),
        ("16", "0", "fp32", "1"),
        ("16", "7", "fp16", "1"),
        ("16", "7", "fp32", "broken"),
    ] {
        let output = run_page_kernel_profile(
            &[
                "bench",
                "--manifest",
                manifest.to_str().unwrap(),
                "--backend",
                "candle",
                "--dtype",
                dtype,
                "--iterations",
                "1",
            ],
            "0",
            queries,
            "0",
            pages,
            "0",
            direct,
        );
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("direct paged attention")
                || error.contains("HUNCHO_DIRECT_PAGED_ATTENTION"),
            "{error}"
        );
    }
}
