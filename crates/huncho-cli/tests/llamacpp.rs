//! CPU runtime selection, artifact identity and fresh calibration guards.
#![cfg(all(feature = "llamacpp", feature = "tokenizers"))]
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::Path,
    process::{Command, Output},
};
fn command(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_huncho"))
        .args(args)
        .env("HUNCHO_DEVICE", "cpu")
        .env("HUNCHO_LLAMA_THREADS", "2")
        .env("RAYON_NUM_THREADS", "1")
        .env("CANDLE_NUM_THREADS", "1")
        .env_remove("HUNCHO_BACKEND")
        .env_remove("HUNCHO_DTYPE")
        .env_remove("HUNCHO_REPLICAS")
        .env_remove("HUNCHO_PREFIX_CACHE")
        .env_remove("HUNCHO_MAX_BATCH_TOKENS")
        .env_remove("HUNCHO_BATCH_MAX_REQUESTS")
        .env_remove("HUNCHO_MAX_PREPARED_PER_MODEL")
        .env_remove("HUNCHO_RESULT_CACHE_BYTES")
        .output()
        .unwrap()
}
fn fails(output: Output, expected: &str) {
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains(expected), "{error}");
}
#[test]
fn real_cpu_runtime_selects_exact_dtype_and_keeps_numerical_and_labeled_acceptance_distinct() {
    let temp = tempfile::tempdir().unwrap();
    let pkg = temp.path().join("pkg");
    std::fs::create_dir(&pkg).unwrap();
    let fixture = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/tiny_kev"
    ));
    for name in ["head.pt", "tokenizer.json"] {
        std::fs::copy(fixture.join(name), pkg.join(name)).unwrap();
    }
    std::fs::copy(
        fixture.join("../llamacpp/kev-f32.gguf"),
        pkg.join("backbone.gguf"),
    )
    .unwrap();
    let mut m: Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/mock-model/huncho-model.json"
        ))
        .unwrap(),
    )
    .unwrap();
    m["name"] = json!("tiny-kev");
    m["family"] = json!("F2");
    m["head"] = json!({"kind":"pointer","weights":"head.pt","width":4});
    m["backbone"]["hidden_size"] = json!(16);
    m["backbone"]["max_context"] = json!(512);
    m["backbone"]["tokenizer"] = json!("tokenizer.json");
    m["backbone"]["artifacts"] = json!({"llamacpp":[{"path":"backbone.gguf","dtype":"gguf-f32"}]});
    m["prompt_contract"]["template"] = json!("kev-v1");
    m["prompt_contract"]["state_budget"] = json!(512);
    m["prompt_contract"]["head_budget"] = json!(512);
    m["calibration"] =
        json!({"default":{"temperature":2.40605,"confidence":"peak","status":"fit"},"entries":{}});
    let path = pkg.join("huncho-model.json");
    std::fs::write(&path, serde_json::to_vec(&m).unwrap()).unwrap();
    let before = std::fs::read(&path).unwrap();
    let typed: huncho_core::manifest::ModelManifest = serde_json::from_value(m.clone()).unwrap();
    let tokenizer =
        huncho_core::tokenizer::HfTokenizer::from_file_unbounded(pkg.join("tokenizer.json"))
            .unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(fixture.join("golden.json")).unwrap()).unwrap();
    let mut cases = Vec::new();
    for (index, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let request: huncho_core::contract::SystemOneRequest =
            serde_json::from_value(case["request"].clone()).unwrap();
        let expected: BTreeMap<String, BTreeMap<String, f32>> = request
            .questions
            .iter()
            .zip(case["rows"].as_array().unwrap())
            .map(|((id, q), row)| {
                let prompt = huncho_core::prompt::formatter_for(&typed)
                    .build(&request.state, q, &tokenizer)
                    .unwrap();
                let p: Vec<f32> = serde_json::from_value(row["probabilities"].clone()).unwrap();
                (
                    id.clone(),
                    prompt
                        .candidates
                        .into_iter()
                        .zip(p)
                        .map(|(c, p)| (c.label, p))
                        .collect(),
                )
            })
            .collect();
        cases.push(json!({"id":index.to_string(),"request":request,"expected":expected}));
    }
    let golden = temp.path().join("golden.json");
    std::fs::write(
        &golden,
        serde_json::to_vec(&json!({"schema_version":"1.0","family":"F2","cases":cases})).unwrap(),
    )
    .unwrap();
    let receipt = temp.path().join("receipt.json");
    let conform = command(&[
        "conform",
        "--model",
        pkg.to_str().unwrap(),
        "--golden",
        golden.to_str().unwrap(),
        "--write-qualification",
        receipt.to_str().unwrap(),
        "--json",
    ]);
    assert!(
        conform.status.success(),
        "{}",
        String::from_utf8_lossy(&conform.stderr)
    );
    let report: Value = serde_json::from_slice(&conform.stdout).unwrap();
    assert_eq!(report["backend"], "llamacpp");
    assert_eq!(report["dtype"], "gguf-f32");
    assert_eq!(report["passed"], true);
    assert!(report["max_prob_delta"].as_f64().unwrap() < 1e-6);
    let prefix = command(&[
        "conform",
        "--model",
        pkg.to_str().unwrap(),
        "--golden",
        golden.to_str().unwrap(),
        "--prefix-cache",
        "--persistent-prefix-bytes",
        "1048576",
        "--json",
    ]);
    assert!(
        prefix.status.success(),
        "{}",
        String::from_utf8_lossy(&prefix.stderr)
    );
    let prefix: Value = serde_json::from_slice(&prefix.stdout).unwrap();
    assert_eq!(prefix["passed"], true);
    assert_eq!(
        prefix["execution_metadata"]["llamacpp_prefix_state"],
        "full-hybrid-sequence-snapshot-v1"
    );
    assert!(
        prefix["optimization_parity"]["max_prob_delta"]
            .as_f64()
            .unwrap()
            <= 1e-4
    );
    assert_eq!(prefix["optimization_parity"]["argmax_agreement"], 1.);
    assert!(prefix["work"]["cache_forks"].as_u64().unwrap() > 0);
    assert!(prefix["work"]["persistent_prefix_hits"].as_u64().unwrap() > 0);
    assert!(prefix["outcome_calibration"].is_null());
    let audit: Value = serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
    assert_eq!(audit["outcome_gates_passed"], false);
    assert!(std::fs::read_to_string(&receipt)
        .unwrap()
        .contains("backbone/gguf"));
    fails(
        command(&["serve", "--model", pkg.to_str().unwrap()]),
        "requires --qualification-golden",
    );
    let binding = format!("tiny-kev={}", golden.display());
    fails(
        command(&[
            "serve",
            "--model",
            pkg.to_str().unwrap(),
            "--qualification-golden",
            &binding,
        ]),
        "observed target labels",
    );
    let bench = command(&[
        "bench",
        "--model",
        pkg.to_str().unwrap(),
        "--iterations",
        "3",
        "--concurrency",
        "2",
        "--replicas",
        "2",
        "--questions",
        "3",
        "--workload",
        "mixed",
        "--json",
    ]);
    assert!(
        bench.status.success(),
        "{}",
        String::from_utf8_lossy(&bench.stderr)
    );
    let report: Value = serde_json::from_slice(&bench.stdout).unwrap();
    assert_eq!(report["replica_work"].as_array().unwrap().len(), 2);
    assert_eq!(report["work"]["forward_calls"], 9);
    assert_eq!(
        report["execution_metadata"]["llamacpp_execution"],
        "cpu-qwen35-masked-prefill-v1"
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let invalid = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .args([
            "bench",
            "--model",
            pkg.to_str().unwrap(),
            "--iterations",
            "1",
        ])
        .env("HUNCHO_DEVICE", "cuda")
        .output()
        .unwrap();
    fails(invalid, "llamacpp currently requires HUNCHO_DEVICE=cpu");
    m["family"] = json!("F3");
    m["head"]["kind"] = json!("candidate-logit");
    m["f3"] = json!({"candidate_codes":["A","B"],"candidate_token_ids":[36,37],"system_prompt":"","prompt_code_sha256":"fixture","max_input_tokens":512});
    std::fs::write(&path, serde_json::to_vec(&m).unwrap()).unwrap();
    let bench = command(&[
        "bench",
        "--model",
        pkg.to_str().unwrap(),
        "--iterations",
        "1",
        "--json",
    ]);
    assert!(
        bench.status.success(),
        "{}",
        String::from_utf8_lossy(&bench.stderr)
    );
    let report: Value = serde_json::from_slice(&bench.stdout).unwrap();
    assert_eq!(report["dtype"], "gguf-f32");
}
