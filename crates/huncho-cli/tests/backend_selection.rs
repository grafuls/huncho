//! Check user-facing defaults and explicit overrides without Hub downloads.
use std::{
    path::PathBuf,
    process::{Command, Output},
};

fn fixture_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/mock-model/huncho-model.json")
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env_remove("HUNCHO_BACKEND")
        .env_remove("HUNCHO_DTYPE")
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn explicit_mock_remains_available_for_a_package() {
    let manifest = fixture_manifest();
    let output = run(&[
        "bench",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "mock",
        "--iterations",
        "1",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(not(feature = "onnx"))]
#[test]
fn model_defaults_fail_with_build_guidance_instead_of_using_mock() {
    let manifest = fixture_manifest();
    let golden = manifest.parent().unwrap().join("golden.json");
    for command in ["serve", "bench", "conform"] {
        for model_arg in ["--manifest", "--model"] {
            let mut args = vec![command, model_arg, manifest.to_str().unwrap()];
            if command == "conform" {
                args.extend(["--golden", golden.to_str().unwrap()]);
            }
            let output = run(&args);
            assert!(!output.status.success());
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(
                error.contains("--features onnx"),
                "{command} {model_arg}: {error}"
            );
        }
    }
}

#[cfg(feature = "onnx")]
#[test]
fn auto_onnx_package_conforms_without_a_backend_flag() {
    let manifest = fixture_manifest();
    let golden = manifest.parent().unwrap().join("golden.json");
    for model_arg in ["--manifest", "--model"] {
        let output = run(&[
            "conform",
            model_arg,
            manifest.to_str().unwrap(),
            "--golden",
            golden.to_str().unwrap(),
            "--json",
        ]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("\"passed\": true"));
    }
}

#[test]
fn explicit_backend_is_not_replaced_by_auto_selection() {
    let manifest = fixture_manifest();
    let output = run(&[
        "bench",
        "--manifest",
        manifest.to_str().unwrap(),
        "--backend",
        "clef",
        "--dtype",
        "fp32",
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .to_lowercase()
        .contains("clef"));
}
