#![cfg(feature = "candle")]
use serde_json::{json, Value};
use std::{path::Path, process::Command};

#[test]
fn selected_head_requires_explicit_valid_flag_and_loaded_trained_f1_head() {
    let fixture = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/tiny_modernbert"
    ));
    let package = tempfile::tempdir().unwrap();
    for name in ["config.json", "model.safetensors"] {
        std::fs::copy(fixture.join(name), package.path().join(name)).unwrap();
    }
    let mut manifest: Value = serde_json::from_slice(
        &std::fs::read(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/mock-model/huncho-model.json"
        )))
        .unwrap(),
    )
    .unwrap();
    manifest["backbone"]["source"] = json!({"kind":"local", "path":"."});
    manifest["backbone"]["artifacts"] =
        json!({"candle":[{"path":"model.safetensors", "dtype":"fp32"}]});
    manifest["backbone"]["hidden_size"] = json!(8);
    manifest["backbone"]["max_context"] = json!(128);
    manifest["head"]["weights"] = json!("");
    manifest.as_object_mut().unwrap().remove("reference");
    let path = package.path().join("huncho-model.json");
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let run = |flag: &str| {
        Command::new(env!("CARGO_BIN_EXE_huncho"))
            .env("HUNCHO_DEVICE", "cpu")
            .env("HUNCHO_LAYA_SELECTED_HEAD", flag)
            .env("RAYON_NUM_THREADS", "1")
            .env("CANDLE_NUM_THREADS", "1")
            .env_remove("HUNCHO_CPU_BLAS_LIBRARY")
            .env_remove("HUNCHO_CPU_BLAS_THREADS")
            .args([
                "bench",
                "--manifest",
                path.to_str().unwrap(),
                "--backend",
                "candle",
                "--dtype",
                "fp32",
                "--iterations",
                "1",
                "--questions",
                "1",
                "--json",
            ])
            .output()
            .unwrap()
    };
    let output = run("0");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(report["execution_metadata"]
        .get("laya_head_execution")
        .is_none());
    for (flag, message) in [
        ("1", "requires a trained Laya head on CPU"),
        ("on", "must be 0, 1, false or true"),
    ] {
        let output = run(flag);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(message),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    manifest["family"] = json!("F2");
    manifest["head"]["kind"] = json!("pointer");
    manifest["prompt_contract"]["template"] = json!("kev-v1");
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let output = run("1");
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("native F1 Laya package"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
