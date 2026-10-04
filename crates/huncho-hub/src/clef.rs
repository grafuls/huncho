//! Resolve native Clef releases without importing remote code or using Python.
use crate::{http, HubError, ModelRef, ResolveOptions, ResolvedPackage, Result};
use huncho_core::{
    manifest::ModelManifest,
    prompt::clef::{CONTRACT, TEMPLATE},
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::Path};

pub(crate) fn resolve(model: &str, dtype: &str, opts: &ResolveOptions) -> Result<ResolvedPackage> {
    if !matches!(dtype, "fp32" | "fp16" | "bf16") {
        return Err(HubError::Package(format!(
            "unsupported Clef dtype `{dtype}`"
        )));
    }
    let (root, source, name) = match ModelRef::parse(model, opts.revision.clone()) {
        ModelRef::Local(path) if path.is_file() => return existing(&path),
        ModelRef::Local(path) => {
            let path = path.canonicalize()?;
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let source = json!({"kind":"local", "path":path});
            (path, source, name)
        }
        ModelRef::HuggingFace { repo, revision } => {
            // Resolve the revision with one small file. All following downloads
            // use that exact snapshot's commit, including an index's shards.
            let config = http::download_file(&repo, "config.json", revision, opts)?;
            let root = config
                .parent()
                .ok_or_else(|| HubError::Package("Clef config has no parent".into()))?
                .to_path_buf();
            let commit = root
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let pinned = Some(commit.clone());
            http::download_file(&repo, "joint_head_config.json", pinned.clone(), opts)?;
            // Validate architecture before downloading a large checkpoint.
            dimensions(&root)?;
            for file in ["joint_head.safetensors", "tokenizer.json"] {
                http::download_file(&repo, file, pinned.clone(), opts)?;
            }
            match http::download_file(&repo, "model.safetensors.index.json", pinned.clone(), opts) {
                Ok(index) => {
                    for shard in shard_names(&read_json(&index)?)? {
                        http::download_file(&repo, &shard, pinned.clone(), opts)?;
                    }
                }
                Err(HubError::NotFound { .. }) => {
                    http::download_file(&repo, "model.safetensors", pinned, opts)?;
                }
                Err(e) => {
                    // An offline single-file snapshot has no cached index.
                    if opts.local_files_only && root.join("model.safetensors").is_file() {
                    } else {
                        return Err(e);
                    }
                }
            }
            let source = json!({"kind":"hf", "repo":repo, "revision":commit});
            let name = repo.rsplit('/').next().unwrap_or(&repo).to_string();
            (root, source, name)
        }
    };
    let manifest_path = root.join("huncho-model.json");
    if manifest_path.is_file() {
        return existing(&manifest_path);
    }
    let (hidden, context) = dimensions(&root)?;
    for file in ["joint_head.safetensors", "tokenizer.json"] {
        require_file(&root.join(file))?;
    }
    let index = root.join("model.safetensors.index.json");
    if index.is_file() {
        for shard in shard_names(&read_json(&index)?)? {
            require_file(&root.join(shard))?;
        }
    } else {
        require_file(&root.join("model.safetensors"))?;
    }
    let value = json!({
        "schema_version":"1.0", "name":name, "family":"F5",
        "backbone": {"source":source, "hidden_size":hidden, "max_context":context,
            "tokenizer":"tokenizer.json", "artifacts":{"clef":[
                {"path":"config.json","dtype":"fp32"}, {"path":"config.json","dtype":"fp16"}, {"path":"config.json","dtype":"bf16"}
            ]}},
        "head":{"kind":"joint-schema", "weights":"joint_head.safetensors", "width":1},
        "prompt_contract":{"template":TEMPLATE, "contract_hash":CONTRACT,
            "state_budget":context, "head_budget":context, "max_options":255, "max_len":context, "head_max_len":context},
        "calibration":{"default":{"temperature":1.0,"confidence":"max-probability","status":"pending"}}
    });
    let manifest: ModelManifest = serde_json::from_value(value)?;
    manifest
        .validate()
        .map_err(|e| HubError::Package(e.to_string()))?;
    // Write completely before exposing the manifest. A concurrent resolver must
    // not replace a file containing the user's fitted calibration.
    let temporary = root.join(format!(".huncho-clef-{}.json", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    use std::io::Write;
    let result = (|| -> Result<()> {
        file.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        match std::fs::hard_link(&temporary, &manifest_path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(e) => Err(e.into()),
        }
    })();
    let _ = std::fs::remove_file(&temporary);
    result?;
    existing(&manifest_path)
}

fn existing(path: &Path) -> Result<ResolvedPackage> {
    let m = ModelManifest::load(path).map_err(|e| HubError::Package(e.to_string()))?;
    if m.prompt_contract.template != TEMPLATE || m.prompt_contract.contract_hash != CONTRACT {
        return Err(HubError::Package("Clef requires a clef-native-v1 manifest; regenerate older Python-runner packages from the raw release".into()));
    }
    Ok(ResolvedPackage {
        manifest_path: path.to_path_buf(),
    })
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn require_file(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Err(HubError::Package(format!(
            "incomplete Clef release: missing {}",
            path.display()
        )));
    }
    Ok(())
}
fn dimensions(root: &Path) -> Result<(usize, usize)> {
    let config = read_json(&root.join("config.json"))?;
    let text = config.get("text_config").unwrap_or(&config);
    if text["model_type"] != "qwen3_5_text" && text["model_type"] != "qwen3_5" {
        return Err(HubError::Package(
            "native Clef requires a Qwen3.5 text backbone".into(),
        ));
    }
    let head = read_json(&root.join("joint_head_config.json"))?;
    let hidden = text["hidden_size"].as_u64().unwrap_or(0) as usize;
    if hidden == 0 || head["hidden_size"].as_u64() != Some(hidden as u64) {
        return Err(HubError::Package(
            "Clef backbone/head hidden-size mismatch".into(),
        ));
    }
    let context = text["max_position_embeddings"]
        .as_u64()
        .unwrap_or(16384)
        .min(16384) as usize;
    if context == 0 {
        return Err(HubError::Package(
            "Clef context limit must be positive".into(),
        ));
    }
    Ok((hidden, context))
}
fn shard_names(index: &Value) -> Result<BTreeSet<String>> {
    let map = index["weight_map"]
        .as_object()
        .filter(|m| !m.is_empty())
        .ok_or_else(|| HubError::Package("Clef weight index has no weight_map".into()))?;
    map.values()
        .map(|v| {
            let name = v
                .as_str()
                .ok_or_else(|| HubError::Package("invalid Clef shard name".into()))?;
            if !name.ends_with(".safetensors")
                || Path::new(name).components().count() != 1
                || name.contains(['/', '\\'])
                || !name.starts_with("model")
            {
                return Err(HubError::Package(format!(
                    "invalid Clef shard name `{name}`"
                )));
            }
            Ok(name.into())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_hub_release_pins_shards_and_rejects_missing_weights() {
        let cache = tempfile::tempdir().unwrap();
        let commit = "3333333333333333333333333333333333333333";
        let snapshot = cache
            .path()
            .join("models--fixture--clef/snapshots")
            .join(commit);
        std::fs::create_dir_all(&snapshot).unwrap();
        let fixture = Path::new("../huncho-backend/tests/fixtures/tiny_clef");
        for name in [
            "config.json",
            "joint_head_config.json",
            "tokenizer.json",
            "joint_head.safetensors",
        ] {
            std::fs::copy(fixture.join(name), snapshot.join(name)).unwrap();
        }
        let shard = "model-00001-of-00001.safetensors";
        std::fs::copy(fixture.join("model.safetensors"), snapshot.join(shard)).unwrap();
        std::fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map":{"x":shard}})).unwrap(),
        )
        .unwrap();
        let opts = ResolveOptions {
            revision: Some(commit.into()),
            cache_dir: Some(cache.path().to_path_buf()),
            local_files_only: true,
            show_progress: false,
            ..Default::default()
        };
        let package = crate::resolve_auto(
            "fixture/clef",
            Some("fp16"),
            &[huncho_core::manifest::BackendId::Clef],
            &opts,
        )
        .unwrap();
        let m = ModelManifest::load(package.manifest_path).unwrap();
        assert!(
            matches!(m.backbone.source, huncho_core::manifest::BackboneSource::Hf { revision, .. } if revision == commit)
        );
        assert!(!snapshot.join("joint_schema_model.py").exists());
        crate::resolve_auto(
            "fixture/clef",
            None,
            &[huncho_core::manifest::BackendId::Clef],
            &opts,
        )
        .unwrap();
        std::fs::remove_file(snapshot.join(shard)).unwrap();
        assert!(crate::resolve_auto(
            "fixture/clef",
            None,
            &[huncho_core::manifest::BackendId::Clef],
            &opts
        )
        .is_err());
    }

    #[test]
    fn shards_must_be_real_local_weight_files() {
        assert!(shard_names(&json!({"weight_map":{"a":"../bad.safetensors"}})).is_err());
        assert!(shard_names(&json!({"weight_map":{}})).is_err());
        assert_eq!(
            shard_names(
                &json!({"weight_map":{"a":"model-01.safetensors","b":"model-01.safetensors"}})
            )
            .unwrap()
            .len(),
            1
        );
    }
    #[test]
    fn local_release_preserves_calibration_and_needs_no_python() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            r#"{"model_type":"qwen3_5_text","hidden_size":16,"max_position_embeddings":1024}"#,
        )
        .unwrap();
        std::fs::write(
            root.path().join("joint_head_config.json"),
            r#"{"hidden_size":16}"#,
        )
        .unwrap();
        for file in [
            "tokenizer.json",
            "joint_head.safetensors",
            "model.safetensors",
        ] {
            std::fs::write(root.path().join(file), []).unwrap();
        }
        let package = resolve(
            root.path().to_str().unwrap(),
            "fp32",
            &ResolveOptions::default(),
        )
        .unwrap();
        let mut m = ModelManifest::load(&package.manifest_path).unwrap();
        assert_eq!(m.prompt_contract.template, TEMPLATE);
        m.calibration.default.temperature = 2.5;
        std::fs::write(&package.manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();
        resolve(
            root.path().to_str().unwrap(),
            "fp32",
            &ResolveOptions::default(),
        )
        .unwrap();
        assert_eq!(
            ModelManifest::load(&package.manifest_path)
                .unwrap()
                .calibration
                .default
                .temperature,
            2.5
        );
    }
}
