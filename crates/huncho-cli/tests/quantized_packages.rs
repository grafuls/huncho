//! Conversion never changes source files or grants serving acceptance.
#![cfg(all(feature = "quantization", feature = "clef"))]

use serde_json::{json, Value};
use std::path::Path;
use std::process::{Command, Output};

fn command(args: &[&str], device: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_huncho"))
        .args(args)
        .env("HUNCHO_DEVICE", device)
        .env("RAYON_NUM_THREADS", "1")
        .env("CANDLE_NUM_THREADS", "1")
        .env_remove("HUNCHO_CPU_DELTA_RULE")
        .env_remove("HUNCHO_CPU_CAUSAL_CONV")
        .env_remove("HUNCHO_PREFILL_CHUNK_TOKENS")
        .env_remove("HUNCHO_PROJECTION_CHUNK_ROWS")
        .env_remove("HUNCHO_ATTENTION_FP32")
        .env_remove("HUNCHO_PREFIX_CACHE")
        .env_remove("HUNCHO_PERSISTENT_PREFIX_BYTES")
        .env_remove("HUNCHO_MAX_BATCH_TOKENS")
        .env_remove("HUNCHO_BATCH_MAX_REQUESTS")
        .env_remove("HUNCHO_MAX_PREPARED_PER_MODEL")
        .output()
        .unwrap()
}

fn failed(output: Output, expected: &str) {
    assert!(!output.status.success());
    let log = String::from_utf8_lossy(&output.stderr);
    assert!(log.contains(expected), "{log}");
}

#[test]
fn cpu_conversion_is_durable_no_overwrite_and_requires_exact_refit_and_heldout_gate() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let fixture = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../huncho-backend/tests/fixtures/tiny_kev"
    ));
    let files = [
        "config.json",
        "model.safetensors",
        "adapter_config.json",
        "adapter_model.safetensors",
        "head.pt",
        "tokenizer.json",
    ];
    for name in files {
        std::fs::copy(fixture.join(name), source.join(name)).unwrap();
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
    manifest["head"] = json!({"kind":"pointer", "weights":"head.pt", "width":4});
    manifest["backbone"]["artifacts"] =
        json!({"candle":[{"path":"adapter_model.safetensors","dtype":"fp32"}]});
    manifest["backbone"]["tokenizer"] = json!("tokenizer.json");
    manifest["backbone"]["max_context"] = json!(512);
    manifest["prompt_contract"]["template"] = json!("kev-v1");
    manifest["prompt_contract"]["state_budget"] = json!(512);
    manifest["prompt_contract"]["head_budget"] = json!(512);
    manifest["calibration"] =
        json!({"default":{"temperature":2.40605,"confidence":"peak","status":"fit"},"entries":{}});
    let source_manifest = source.join("huncho-model.json");
    std::fs::write(&source_manifest, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let before = std::fs::read(&source_manifest).unwrap();
    for dtype in ["q8_0-fp32", "q4_0-fp32"] {
        let output = temp.path().join(dtype);
        let source_text = source.to_str().unwrap();
        let output_text = output.to_str().unwrap();
        let convert = command(
            &[
                "quantize",
                "--model",
                source_text,
                "--output",
                output_text,
                "--dtype",
                dtype,
            ],
            "cpu",
        );
        assert!(
            convert.status.success(),
            "{}",
            String::from_utf8_lossy(&convert.stderr)
        );
        let provenance: Value = serde_json::from_slice(&convert.stdout).unwrap();
        assert_eq!(provenance["qualified"], false);
        assert!(provenance["source_files"]
            .get("base/model.safetensors")
            .is_some());
        assert!(
            provenance["statistics"]["packed_projection_bytes"]
                .as_u64()
                .unwrap()
                < provenance["statistics"]["source_projection_bytes"]
                    .as_u64()
                    .unwrap()
        );
        assert_eq!(std::fs::read(&source_manifest).unwrap(), before);
        for file in files {
            assert_eq!(
                std::fs::read(source.join(file)).unwrap(),
                std::fs::read(fixture.join(file)).unwrap()
            );
        }
        let path = output.join("huncho-model.json");
        let mut converted: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let original_converted = std::fs::read(&path).unwrap();
        assert_eq!(converted["calibration"]["default"]["status"], "pending");
        assert_eq!(converted["calibration"]["default"]["temperature"], 1.0);
        assert!(converted.get("reference").is_none());
        assert!(!output.join("adapter_model.safetensors").exists());
        assert!(!output.join("model.safetensors").exists());
        let serve = [
            "serve",
            "--model",
            output_text,
            "--backend",
            "candle",
            "--dtype",
            dtype,
        ];
        let reference: Value =
            serde_json::from_slice(&std::fs::read(fixture.join("golden.json")).unwrap()).unwrap();
        let fitting = temp.path().join(format!("{dtype}-fitting.jsonl"));
        let record = json!({"id":"fitting-only", "request":reference["cases"][0]["request"],
            "targets":{"team":"returns", "urgent":"yes", "priority":"2"}});
        std::fs::write(&fitting, format!("{}\n", record)).unwrap();
        let captured = temp.path().join(format!("{dtype}-captured"));
        let collected = command(
            &[
                "capture-logits",
                "--model",
                output_text,
                "--dtype",
                dtype,
                "--data",
                fitting.to_str().unwrap(),
                "--output",
                captured.to_str().unwrap(),
            ],
            "cpu",
        );
        assert!(
            collected.status.success(),
            "{}",
            String::from_utf8_lossy(&collected.stderr)
        );
        let audit: Value = serde_json::from_slice(&collected.stdout).unwrap();
        assert_eq!(audit["qualified"], false);
        assert_eq!(audit["work"]["forward_calls"], 3);
        assert_eq!(audit["work"]["prefill_calls"], 0);
        assert_eq!(audit["work"]["batch_calls"], 0);
        assert!(!captured.join("golden.json").exists());
        let fit: Value =
            serde_json::from_slice(&std::fs::read(captured.join("fit.json")).unwrap()).unwrap();
        assert_eq!(fit["targets"], json!([1, 1, 2]));
        assert_eq!(fit["qtypes"], json!(["choice", "noul", "score"]));
        let dry_run = command(
            &[
                "calibrate",
                "--manifest",
                path.to_str().unwrap(),
                "--backend",
                "candle",
                "--dtype",
                dtype,
                "--data",
                captured.join("fit.json").to_str().unwrap(),
                "--save",
                "false",
                "--json",
            ],
            "cpu",
        );
        assert!(
            dry_run.status.success(),
            "{}",
            String::from_utf8_lossy(&dry_run.stderr)
        );
        assert_eq!(std::fs::read(&path).unwrap(), original_converted);
        failed(command(&serve, "cpu"), "requires fitted calibration");
        failed(
            command(&serve, "cuda"),
            "packed Kev requires HUNCHO_DEVICE=cpu",
        );
        failed(
            command(
                &[
                    "quantize",
                    "--model",
                    source_text,
                    "--output",
                    output_text,
                    "--dtype",
                    dtype,
                ],
                "cpu",
            ),
            "File exists",
        );
        assert_eq!(std::fs::read(&path).unwrap(), original_converted);
        let key = format!("candle:{dtype}");
        converted["calibration"]["default"]["status"] = json!("refit");
        converted["calibration"]["entries"] = json!({});
        std::fs::write(&path, serde_json::to_vec(&converted).unwrap()).unwrap();
        failed(command(&serve, "cpu"), "explicit backend:dtype refit");
        converted["calibration"]["entries"][&key] =
            json!({"temperature":2.40605, "confidence":"peak", "status":"fit"});
        std::fs::write(&path, serde_json::to_vec(&converted).unwrap()).unwrap();
        failed(command(&serve, "cpu"), "explicit backend:dtype refit");
        // This simulates only refit metadata to exercise the remaining gate;
        // no fitting rows or fixture labels establish released calibration.
        converted["calibration"]["entries"][&key]["status"] = json!("refit");
        std::fs::write(&path, serde_json::to_vec(&converted).unwrap()).unwrap();
        failed(command(&serve, "cpu"), "requires --qualification-golden");
    }
}
