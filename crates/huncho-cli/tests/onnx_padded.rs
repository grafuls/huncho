//! Actual CPU masked ONNX processes; synthetic mean-head fixtures never release a model.
#![cfg(all(feature = "onnx-shared", feature = "tokenizers"))]
use huncho_core::{
    conformance::{GoldenCase, GoldenSuite},
    contract::{Answer, SystemOneRequest},
    engine::Engine,
    manifest::{BackendId, ModelManifest},
    tokenizer::HfTokenizer,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::Path,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

fn command(args: &[&str]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env("HUNCHO_DEVICE", "cpu")
        .env("HUNCHO_ONNX_EP", "cpu")
        .env("HUNCHO_ONNX_THREADS", "1")
        .env("HUNCHO_ONNX_NATIVE_BATCH", "1")
        .env("HUNCHO_ONNX_SHARED_INITIALIZERS", "1")
        .env("HUNCHO_ONNX_OUTPUT_BUFFER_BYTES", "8192")
        .env("HUNCHO_ONNX_INTEGRATED_HEAD", "0")
        .env("HUNCHO_ONNX_COMPACT_READOUT", "0")
        .env("RAYON_NUM_THREADS", "1")
        .env("CANDLE_NUM_THREADS", "1")
        .env("HUNCHO_LAYA_SELECTED_HEAD", "0")
        .env("HUNCHO_ATTENTION_QUERY_ROWS", "0")
        .env("HUNCHO_GROUPED_GQA", "0")
        .env_remove("HUNCHO_CPU_BLAS_LIBRARY")
        .env_remove("HUNCHO_CPU_BLAS_THREADS")
        .env_remove("HUNCHO_CPU_FUSED_GATE")
        .env_remove("HUNCHO_REPLICAS")
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
            panic!("CPU F1 command hung or unqualified serving started");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}
fn report(args: &[&str]) -> Value {
    let output = command(args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn probabilities(answer: &Answer) -> BTreeMap<String, f32> {
    match answer {
        Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => {
            probabilities.clone()
        }
        Answer::Noul { noul } => BTreeMap::from([("no".into(), 1.0 - noul), ("yes".into(), *noul)]),
    }
}
#[test]
fn masked_onnx_processes_bind_real_rectangles_replicas_and_labeled_startup() {
    let root = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures"
    ));
    let tmp = tempfile::tempdir().unwrap();
    for name in [
        "tiny_encoder_masked.onnx",
        "tiny_encoder_masked-tokenizer.json",
    ] {
        std::fs::copy(root.join(name), tmp.path().join(name)).unwrap();
    }
    let mut manifest: Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/mock-model/huncho-model.json"
        ))
        .unwrap(),
    )
    .unwrap();
    manifest["name"] = json!("synthetic-onnx-mask-process");
    manifest["backbone"]["source"] = json!({"kind":"local","path":"."});
    manifest["backbone"]["artifacts"] =
        json!({"onnx":[{"path":"tiny_encoder_masked.onnx","dtype":"fp32"}]});
    manifest["backbone"]["hidden_size"] = json!(8);
    manifest["backbone"]["tokenizer"] = json!("tiny_encoder_masked-tokenizer.json");
    manifest["backbone"]["max_context"] = json!(128);
    manifest["head"]["weights"] = json!("");
    manifest["prompt_contract"]["max_len"] = json!(128);
    manifest["prompt_contract"]["head_max_len"] = json!(64);
    manifest.as_object_mut().unwrap().remove("reference");
    let manifest_path = tmp.path().join("huncho-model.json");
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let typed: ModelManifest = serde_json::from_value(manifest).unwrap();
    let backend = huncho_backend::OnnxBackend::load_with_options(
        root.join("tiny_encoder_masked.onnx"),
        8,
        128,
        "fp32",
        huncho_backend::onnx::OnnxOptions {
            native_batch: true,
            intra_threads: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let engine = Engine::new(
        typed,
        Box::new(backend),
        Box::new(HfTokenizer::from_file(root.join("tiny_encoder_masked-tokenizer.json")).unwrap()),
        Default::default(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap();
    let request: Value = json!({"model":"synthetic-onnx-mask-process","state":"refund","questions":{
        "team":{"type":"choice","instructions":"Team?","criteria":{"shipping":null,"billing":"Charges","returns":"Refunds"}},
        "urgent":{"type":"noul","instructions":"Urgent?"},
        "priority":{"type":"score","instructions":"Priority?","criteria":["low","medium","high","very high"]}}});
    let mut cases = Vec::new();
    for (index, state) in ["refund", "a customer needs a refund"]
        .into_iter()
        .enumerate()
    {
        let mut value = request.clone();
        value["state"] = json!(state);
        let request: SystemOneRequest = serde_json::from_value(value).unwrap();
        let response = engine.eval(&request, &Default::default()).unwrap();
        cases.push(GoldenCase {
            id: index.to_string(),
            request,
            expected: response
                .answers
                .iter()
                .map(|(id, a)| (id.clone(), probabilities(a)))
                .collect(),
            targets: Default::default(),
        });
    }
    let suite = GoldenSuite {
        schema_version: "1.0".into(),
        family: "F1".into(),
        hash: None,
        cases,
    };
    let golden_path = tmp.path().join("unlabeled.json");
    std::fs::write(&golden_path, serde_json::to_vec(&suite).unwrap()).unwrap();
    let manifest = manifest_path.to_str().unwrap();
    let golden = golden_path.to_str().unwrap();
    let result = report(&[
        "conform",
        "--manifest",
        manifest,
        "--backend",
        "onnx",
        "--dtype",
        "fp32",
        "--golden",
        golden,
        "--max-batch-tokens",
        "384",
        "--max-batch-padding-percent",
        "25",
        "--json",
    ]);
    assert_eq!(result["passed"], true);
    assert_eq!(
        result["execution_metadata"]["padded_batch_execution"],
        "onnx-cpu-right-mask-v1"
    );
    assert!(result["work"]["padded_batch_calls"].as_u64().unwrap() > 0);
    assert!(result["work"]["padded_tokens"].as_u64().unwrap() > 0);
    assert!(
        result["optimization_parity"]["max_prob_delta"]
            .as_f64()
            .unwrap()
            <= 1e-4
    );
    assert!(result.get("outcome_calibration").is_none());
    let result = report(&[
        "bench",
        "--manifest",
        manifest,
        "--backend",
        "onnx",
        "--dtype",
        "fp32",
        "--max-batch-tokens",
        "384",
        "--max-batch-padding-percent",
        "100",
        "--replicas",
        "2",
        "--concurrency",
        "2",
        "--iterations",
        "4",
        "--questions",
        "3",
        "--workload",
        "mixed",
        "--json",
    ]);
    assert_eq!(result["replicas"], 2);
    assert!(result["work"]["padded_batch_calls"].as_u64().unwrap() > 0);
    let replicas = result["replica_work"].as_array().unwrap();
    assert_eq!(replicas.len(), 2);
    assert!(replicas
        .iter()
        .all(|r| r["requests"] == 2 && r["work"]["batch_calls"].as_u64().unwrap() > 0));
    let binding = format!("synthetic-onnx-mask-process={golden}");
    for supplied in [false, true] {
        let mut args = vec![
            "serve",
            "--manifest",
            manifest,
            "--backend",
            "onnx",
            "--dtype",
            "fp32",
            "--bind",
            "127.0.0.1:0",
            "--max-batch-tokens",
            "384",
            "--max-batch-padding-percent",
            "25",
        ];
        if supplied {
            args.extend(["--qualification-golden", &binding]);
        }
        let output = command(&args);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(if supplied {
                "observed target labels"
            } else {
                "--qualification-golden"
            }),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
