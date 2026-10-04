//! Exercise the native resolver, CLI, backbone and head with no Python in PATH.
#![cfg(feature = "clef")]
use serde_json::{json, Value};
use std::{fs, path::Path, process::Command};

#[test]
fn native_clef_resolves_benchmarks_and_conforms_without_python() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let fixture = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/tiny_clef"
    ));
    for name in [
        "config.json",
        "joint_head_config.json",
        "model.safetensors",
        "joint_head.safetensors",
        "tokenizer.json",
    ] {
        fs::copy(fixture.join(name), root.join(name)).unwrap();
    }
    let run = |args: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_huncho"))
            .current_dir(root)
            .env("PATH", root)
            .env("HUNCHO_CLEF_DEVICE", "cpu")
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    };
    let bench = run(&[
        "bench",
        "--backend",
        "clef",
        "--model",
        ".",
        "--dtype",
        "fp32",
        "--questions",
        "3",
        "--iterations",
        "1",
    ]);
    assert!(bench.contains("backend=clef"));
    assert!(root.join("huncho-model.json").is_file());
    run(&[
        "bench",
        "--backend",
        "clef",
        "--manifest",
        "huncho-model.json",
        "--dtype",
        "fp16",
        "--questions",
        "1",
        "--iterations",
        "1",
    ]);
    let reference: Value =
        serde_json::from_slice(&fs::read(fixture.join("golden.json")).unwrap()).unwrap();
    let cases: Vec<_> = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut expected = c["probabilities"].clone();
            for (id, q) in c["request"]["questions"].as_object().unwrap() {
                if q["type"] == "noul" {
                    let p = &expected[id];
                    expected[id] = json!({"yes":p["true"], "no":p["false"]});
                }
            }
            json!({"id":format!("reference-{i}"), "request":c["request"], "expected":expected})
        })
        .collect();
    fs::write(
        root.join("golden.json"),
        serde_json::to_vec(&json!({"schema_version":"1.0", "family":"F5", "cases":cases})).unwrap(),
    )
    .unwrap();
    let conform = run(&[
        "conform",
        "--backend",
        "clef",
        "--model",
        ".",
        "--dtype",
        "fp32",
        "--golden",
        "golden.json",
        "--json",
    ]);
    assert!(conform.contains("\"passed\": true"));
}
