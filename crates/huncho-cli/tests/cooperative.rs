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
    let mut child = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env("HUNCHO_DEVICE", "cpu")
        .env("HUNCHO_PREFILL_CHUNK_TOKENS", chunk)
        .env("HUNCHO_ATTENTION_QUERY_ROWS", query_rows)
        .env("HUNCHO_GROUPED_GQA", grouped)
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
