use std::process::Command;

#[test]
fn cuda_probe_cannot_report_success_for_cpu() {
    let output = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env("HUNCHO_CLEF_DEVICE", "cpu")
        .arg("__check-cuda")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no usable Clef CUDA device"));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cpu_build_rejects_cuda_probe_in_auto_mode() {
    let output = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .env_remove("HUNCHO_CLEF_DEVICE")
        .arg("__check-cuda")
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn packaging_probe_is_not_in_public_help() {
    let output = Command::new(env!("CARGO_BIN_EXE_huncho"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(!String::from_utf8_lossy(&output.stdout).contains("__check-cuda"));
}
