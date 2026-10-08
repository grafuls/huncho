//! CPU-only process coverage for interleaved qualification and serving refusal.
#![cfg(feature = "clef")]
use serde_json::{json, Value};
use std::{
    path::Path,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

fn run(args: &[&str], chunk: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env("HUNCHO_DEVICE", "cpu")
        .env("HUNCHO_PREFILL_CHUNK_TOKENS", chunk)
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

#[test]
fn cooperative_conformance_requires_real_interleaving_on_unchanged_native_goldens() {
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
