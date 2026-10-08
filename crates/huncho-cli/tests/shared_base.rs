//! Source residency never bypasses startup calibration requirements.
#![cfg(all(feature = "shared-base", feature = "clef"))]
use std::{
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[test]
fn cpu_cli_shares_relocated_base_bytes_but_rejects_pending_models_and_invalid_budgets() {
    let root = tempfile::tempdir().unwrap();
    let source = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/tiny_clef"
    ));
    let mut manifests = Vec::new();
    for name in ["first", "second"] {
        let package = root.path().join(name);
        std::fs::create_dir(&package).unwrap();
        for file in [
            "config.json",
            "model.safetensors",
            "joint_head_config.json",
            "joint_head.safetensors",
            "tokenizer.json",
        ] {
            std::fs::copy(source.join(file), package.join(file)).unwrap();
        }
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(source.join("huncho-model.json")).unwrap())
                .unwrap();
        manifest["name"] = serde_json::json!(name);
        let path = package.join("huncho-model.json");
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        manifests.push(path);
    }
    for budget in ["8388608", "1", "invalid"] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_huncho"));
        command
            .args(["serve", "--bind", "127.0.0.1:0", "--dtype", "fp32"])
            .env("HUNCHO_DEVICE", "cpu")
            .env("HUNCHO_BASE_CACHE_BYTES", budget)
            .env("RUST_LOG", "info")
            .env("RAYON_NUM_THREADS", "1")
            .env("CANDLE_NUM_THREADS", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        for path in &manifests {
            command.arg("--manifest").arg(path);
        }
        let mut child = command.spawn().unwrap();
        let start = Instant::now();
        while child.try_wait().unwrap().is_none() {
            if start.elapsed() > Duration::from_secs(10) {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("pending shared-base models must not start serving");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        assert!(!output.status.success());
        let log = String::from_utf8(output.stderr).unwrap();
        if budget == "invalid" {
            assert!(log.contains("HUNCHO_BASE_CACHE_BYTES must be"), "{log}");
        } else {
            assert!(log.contains("is pending"), "{log}");
            assert_eq!(
                log.matches("shared CPU base cache hit").count(),
                usize::from(budget == "8388608")
            );
        }
    }
}
