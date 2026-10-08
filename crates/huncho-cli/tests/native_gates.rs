//! Fitted source metadata alone cannot qualify a different native execution.
#![cfg(feature = "clef")]
use serde_json::{json, Value};
use std::{
    path::Path,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};

fn run(args: &[&str]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env("HUNCHO_DEVICE", "cpu")
        .env("RAYON_NUM_THREADS", "1")
        .env("CANDLE_NUM_THREADS", "1")
        .env_remove("HUNCHO_CPU_DELTA_RULE")
        .env_remove("HUNCHO_CPU_CAUSAL_CONV")
        .env_remove("HUNCHO_CPU_FUSED_GATE")
        .env_remove("HUNCHO_BASE_CACHE_BYTES")
        .env_remove("HUNCHO_BACKEND")
        .env_remove("HUNCHO_REPLICAS")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let started = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if started.elapsed() > Duration::from_secs(15) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("unqualified runtime unexpectedly started serving or hung");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}

#[test]
fn unoptimized_fitted_cpu_kev_and_clef_require_complete_labeled_startup_suites() {
    let root = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures"
    ));
    for (name, backend) in [("tiny_kev", "candle"), ("tiny_clef", "clef")] {
        let tmp = tempfile::tempdir().unwrap();
        let package = tmp.path();
        for entry in std::fs::read_dir(root.join(name)).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                std::fs::copy(entry.path(), package.join(entry.file_name())).unwrap();
            }
        }
        let mut manifest: Value = if name == "tiny_clef" {
            serde_json::from_slice(&std::fs::read(package.join("huncho-model.json")).unwrap())
                .unwrap()
        } else {
            let mut m: Value = serde_json::from_slice(
                &std::fs::read(Path::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../examples/mock-model/huncho-model.json"
                )))
                .unwrap(),
            )
            .unwrap();
            m["family"] = json!("F2");
            m["head"]["kind"] = json!("pointer");
            m["head"]["weights"] = json!("head.pt");
            m["backbone"]["source"] = json!({"kind":"local","path":"."});
            m["backbone"]["artifacts"] =
                json!({"candle":[{"path":"model.safetensors","dtype":"fp32"}]});
            m["backbone"]["hidden_size"] = json!(16);
            m["backbone"]["max_context"] = json!(512);
            m["backbone"]["tokenizer"] = json!("tokenizer.json");
            m["prompt_contract"]["template"] = json!("kev-v1");
            m
        };
        manifest["calibration"] = json!({"default":{"temperature":2.40605,"confidence":"peak","status":"fit"},"entries":{}});
        manifest["name"] = json!("native-fit");
        manifest.as_object_mut().unwrap().remove("reference");
        std::fs::write(
            package.join("huncho-model.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let args = [
            "serve",
            "--model",
            package.to_str().unwrap(),
            "--backend",
            backend,
            "--dtype",
            "fp32",
            "--bind",
            "127.0.0.1:0",
        ];
        let output = run(&args);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("requires --qualification-golden"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let golden = package.join("unlabeled.json");
        // Valid numerical fixture; intentionally no observed labels. No native
        // prediction is copied into it, and no calibration acceptance is inferred.
        std::fs::write(&golden,serde_json::to_vec(&json!({"schema_version":"1.0","family":manifest["family"],"cases":[{"id":"unlabeled", "request":{"model":"native-fit","state":{"x":1},"questions":{"q":{"type":"noul","instructions":"Proceed?"}}}, "expected":{"q":{"yes":0.5,"no":0.5}}}]})).unwrap()).unwrap();
        let binding = format!("native-fit={}", golden.display());
        let mut qualified = args.to_vec();
        qualified.extend(["--qualification-golden", binding.as_str()]);
        let output = run(&qualified);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("observed target labels"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
