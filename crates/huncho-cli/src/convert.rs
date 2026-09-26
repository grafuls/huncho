//! `huncho convert` — produce backend artifacts plus a manifest from an HF repo
//! (CONV-01).

use std::collections::BTreeMap;
use std::path::Path;

use clap::Args;
use huncho_core::manifest::{
    self, ArtifactRef, Backbone, BackboneSource, BackendId, CalibrationConfig, CalibrationEntry,
    CalibrationStatus, ConfidenceDef, Family, HeadConfig, ModelManifest, PromptContract,
};

#[derive(Args)]
pub struct ConvertArgs {
    /// Hugging Face repo id.
    #[arg(long)]
    pub hf_repo: String,

    /// Pinned revision to convert.
    #[arg(long)]
    pub revision: String,

    /// Model family (F1-F4).
    #[arg(long, default_value = "F1")]
    pub family: String,

    /// Target backend (onnx|gguf|mlx|candle).
    #[arg(long, default_value = "onnx")]
    pub backend: String,

    /// Artifact dtype (fp32|fp16|int8|q4).
    #[arg(long, default_value = "fp32")]
    pub dtype: String,

    /// Output directory.
    #[arg(long, default_value = "out")]
    pub out: String,

    /// Backbone hidden size.
    #[arg(long)]
    pub hidden_size: Option<usize>,

    /// Backbone max context.
    #[arg(long)]
    pub max_context: Option<usize>,

    /// Path to a tokenizer.json to bundle.
    #[arg(long)]
    pub tokenizer: Option<String>,

    /// A shell command to run that actually produces the artifact. Environment
    /// vars `HUNCHO_ARTIFACT`, `HUNCHO_REPO`, `HUNCHO_REVISION`, `HUNCHO_OUT` are set.
    #[arg(long)]
    pub runner: Option<String>,

    /// (candle) A local directory containing a raw HF checkpoint to assemble
    /// into a servable package. Copies `config.json` (or `encoder/config.json`),
    /// the safetensors weights (or shards), and the tokenizer into `--out`.
    #[arg(long)]
    pub source: Option<String>,

    /// Package name; defaults to the repo basename.
    #[arg(long)]
    pub name: Option<String>,
}

/// Artifact filename for a given backend (CONV-01).
pub fn artifact_name_for(backend: BackendId) -> String {
    match backend {
        BackendId::Onnx => "model.onnx".to_string(),
        BackendId::LlamaCpp => "model.gguf".to_string(),
        BackendId::Mlx => "model.safetensors".to_string(),
        BackendId::Vllm => "model.safetensors".to_string(),
        BackendId::Candle => "model.safetensors".to_string(),
    }
}

/// Build the model manifest for a converted package (CONV-01). Pure: performs
/// no filesystem writes, but validates the resulting manifest.
pub fn build_manifest(args: &ConvertArgs) -> anyhow::Result<ModelManifest> {
    let family = Family::parse(&args.family)?;
    let backend = BackendId::parse(&args.backend)?;
    let artifact_name = artifact_name_for(backend);

    let name = args
        .name
        .clone()
        .unwrap_or_else(|| args.hf_repo.rsplit('/').next().unwrap_or("model").to_string());

    let head_kind = manifest::family_kind(family);
    let template = format!("{}-v1", family.to_string().to_lowercase());
    let contract_hash = fnv1a(&template, &args.hf_repo);

    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        backend,
        vec![ArtifactRef {
            path: artifact_name,
            dtype: args.dtype.clone(),
            quantization: match args.dtype.as_str() {
                "int8" => Some("int8".into()),
                "q4" | "4bit" | "q4_0" => Some("q4".into()),
                _ => None,
            },
        }],
    );

    let manifest = ModelManifest {
        schema_version: manifest::MANIFEST_SCHEMA_VERSION.into(),
        name,
        family,
        backbone: Backbone {
            source: BackboneSource::Hf {
                repo: args.hf_repo.clone(),
                revision: args.revision.clone(),
            },
            artifacts,
            hidden_size: args.hidden_size.unwrap_or(1024),
            max_context: args.max_context.unwrap_or(4096),
            tokenizer: args.tokenizer.clone(),
        },
        adapter: None,
        head: HeadConfig {
            kind: head_kind,
            weights: "head.safetensors".into(),
            width: 1,
            pointer_offset: None,
        },
        prompt_contract: PromptContract {
            template: template.clone(),
            option_marker_tokens: vec!["<option:0>".into()],
            state_budget: 3072,
            head_budget: 1024,
            max_options: 255,
            contract_hash,
        },
        calibration: CalibrationConfig {
            default: CalibrationEntry {
                temperature: 1.0,
                per_type_temperatures: None,
                confidence: ConfidenceDef::Peak,
                status: CalibrationStatus::Pending,
            },
            entries: BTreeMap::new(),
            eval_set_hash: None,
        },
        reference: None,
        capabilities: Default::default(),
    };
    manifest.validate()?;
    Ok(manifest)
}

/// Run the artifact converter via the configured `--runner`, if any. Sets the
/// `HUNCHO_ARTIFACT`, `HUNCHO_REPO`, `HUNCHO_REVISION`, and `HUNCHO_OUT` env
/// vars for the subprocess (CONV-01).
pub fn run_runner(runner: &str, backend: BackendId, hf_repo: &str, revision: &str, out: &str) -> anyhow::Result<()> {
    let artifact_name = artifact_name_for(backend);
    tracing::info!("running artifact converter: {runner}");
    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg(runner)
        .env("HUNCHO_ARTIFACT", artifact_name)
        .env("HUNCHO_REPO", hf_repo)
        .env("HUNCHO_REVISION", revision)
        .env("HUNCHO_OUT", out)
        .status()?;
    if !status.success() {
        anyhow::bail!("artifact converter exited with {status}");
    }
    Ok(())
}

/// Assemble a servable candle package from a local raw-HF checkpoint directory.
///
/// Handles `convaiinnovations/laya`'s layout: the config may live at
/// `<source>/config.json` or `<source>/encoder/config.json` (copied to
/// `<out>/config.json`), the weights at `<source>/model.safetensors` (or sharded
/// via a `<artifact>.index.json`), and a tokenizer either at `<source>/<tokenizer>`
/// or `<source>/tokenizer/<tokenizer>`.
pub fn assemble_candle_package(
    source: &Path,
    out: &Path,
    tokenizer: Option<&str>,
) -> anyhow::Result<()> {
    let artifact = artifact_name_for(BackendId::Candle); // "model.safetensors"
    std::fs::create_dir_all(out)?;

    // Config: standard root or Laya's `encoder/` subdir.
    let config_src = first_existing(&[
        source.join("config.json"),
        source.join("encoder/config.json"),
    ])
    .ok_or_else(|| {
        anyhow::anyhow!(
            "no config.json (or encoder/config.json) found in {}: run a `hf download` of the models repo first",
            source.display()
        )
    })?;
    std::fs::copy(&config_src, out.join("config.json"))?;

    // Weights: single file or sharded.
    let weights = source.join(&artifact);
    if weights.is_file() {
        std::fs::copy(&weights, out.join(&artifact))?;
    } else {
        let index_name = format!("{artifact}.index.json");
        let index_path = source.join(&index_name);
        if !index_path.is_file() {
            anyhow::bail!(
                "no {artifact} (or {index_name} shards) found in {}",
                source.display()
            );
        }
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let shards = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .map(|m| {
                let mut s: Vec<String> =
                    m.values().filter_map(|v| v.as_str()).map(String::from).collect();
                s.sort();
                s.dedup();
                s
            })
            .unwrap_or_default();
        if shards.is_empty() {
            anyhow::bail!("{} has no `weight_map` shards", index_path.display());
        }
        for shard in &shards {
            let src = source.join(shard);
            if !src.is_file() {
                anyhow::bail!("missing shard `{shard}` in {}", source.display());
            }
            std::fs::copy(&src, out.join(shard))?;
        }
        std::fs::copy(&index_path, out.join(&index_name))?;
    }

    // Tokenizer (declared by the manifest), if any.
    if let Some(tok) = tokenizer {
        bundle_tokenizer(source, out, tok)?;
    }
    Ok(())
}

/// Copy a tokenizer (declared by the manifest) into the package.
pub fn bundle_tokenizer(source: &Path, out: &Path, tokenizer: &str) -> anyhow::Result<()> {
    let src = first_existing(&[
        source.join(tokenizer),
        source.join("tokenizer").join(tokenizer),
    ])
    .ok_or_else(|| {
        anyhow::anyhow!("tokenizer `{tokenizer}` not found in {}", source.display())
    })?;
    std::fs::copy(&src, out.join(tokenizer))?;
    Ok(())
}

fn first_existing(paths: &[std::path::PathBuf]) -> Option<std::path::PathBuf> {
    paths.iter().find(|p| p.is_file()).cloned()
}

pub fn run(args: ConvertArgs) -> anyhow::Result<()> {
    let family = Family::parse(&args.family)?;
    let backend = BackendId::parse(&args.backend)?;
    let manifest = build_manifest(&args)?;

    let out_dir = Path::new(&args.out);
    std::fs::create_dir_all(out_dir)?;
    let manifest_path = out_dir.join("huncho-model.json");
    let json = serde_json::to_vec_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;
    tracing::info!("wrote manifest {}", manifest_path.display());

    if let Some(runner) = &args.runner {
        run_runner(runner, backend, &args.hf_repo, &args.revision, &args.out)?;
    } else {
        tracing::warn!(
            "no --runner given; place the converted artifact at {}/{}",
            out_dir.display(),
            artifact_name_for(backend)
        );
    }

    // For candle, a local raw-HF checkpoint can be assembled into the package
    // directly (no external converter needed).
    if let Some(source) = &args.source {
        if backend == BackendId::Candle {
            assemble_candle_package(
                Path::new(source),
                out_dir,
                manifest.backbone.tokenizer.as_deref(),
            )?;
        } else {
            anyhow::bail!("--source is only supported for the `candle` backend");
        }
    }

    // Write a small README describing next steps (calibration, conformance).
    let readme = format!(
        "# {}\n\nConverted model package (family {:?}, backend {:?}, dtype {})\n\n\
         Next steps:\n  1. `huncho calibrate` to fit temperatures (CONV-02)\n\
         2. `huncho conform` to verify probability fidelity (CONF-02)\n\
         3. `huncho serve --manifest huncho-model.json` to serve it\n",
        manifest.name, family, backend, args.dtype
    );
    std::fs::write(out_dir.join("README.md"), readme)?;

    Ok(())
}

fn fnv1a(parts: &str, extra: &str) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for b in parts.as_bytes().iter().chain(extra.as_bytes()) {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use huncho_core::manifest::{HeadKind, MANIFEST_SCHEMA_VERSION};
    use std::fs;

    fn args() -> ConvertArgs {
        ConvertArgs {
            hf_repo: "convaiinnovations/laya".into(),
            revision: "main".into(),
            family: "F1".into(),
            backend: "onnx".into(),
            dtype: "fp32".into(),
            out: "out".into(),
            hidden_size: None,
            max_context: None,
            tokenizer: None,
            runner: None,
            source: None,
            name: None,
        }
    }

    #[test]
    fn build_manifest_produces_valid_f1_manifest() {
        let m = build_manifest(&args()).unwrap();
        m.validate().unwrap();
        assert_eq!(m.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(m.name, "laya");
        assert_eq!(m.family, Family::F1);
        assert_eq!(m.backbone.hidden_size, 1024); // default
        assert_eq!(m.backbone.max_context, 4096); // default
        assert_eq!(m.head.kind, HeadKind::OptionMarker);
        assert_eq!(m.prompt_contract.template, "f1-v1");
        assert!(m.prompt_contract.contract_hash.len() == 16);
        let onnx = m.backbone.artifacts.get(&BackendId::Onnx).unwrap();
        assert_eq!(onnx[0].path, "model.onnx");
        assert_eq!(onnx[0].dtype, "fp32");
        assert!(onnx[0].quantization.is_none());
    }

    #[test]
    fn build_manifest_honors_overrides_and_quantization() {
        let mut a = args();
        a.hidden_size = Some(1024);
        a.max_context = Some(8192);
        a.dtype = "q4".into();
        a.backend = "llamacpp".into();
        a.name = Some("laya-q4".into());
        let m = build_manifest(&a).unwrap();
        assert_eq!(m.name, "laya-q4");
        assert_eq!(m.family, Family::F1);
        assert_eq!(m.backbone.hidden_size, 1024);
        assert_eq!(m.backbone.max_context, 8192);
        let gguf = m.backbone.artifacts.get(&BackendId::LlamaCpp).unwrap();
        assert_eq!(gguf[0].path, "model.gguf");
        assert_eq!(gguf[0].quantization.as_deref(), Some("q4"));
        assert!(m.backbone.artifacts.get(&BackendId::Onnx).is_none());
    }

    #[test]
    fn run_with_runner_copies_artifact_and_writes_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("pkg");
        let src = tmp.path().join("source.onnx");
        fs::write(&src, b"fake-onnx-bytes").unwrap();

        let mut a = args();
        a.out = out.to_string_lossy().to_string();
        a.runner = Some(format!(
            "cp {} \"$HUNCHO_OUT/$HUNCHO_ARTIFACT\"",
            src.display()
        ));
        run(a).unwrap();

        let mf = ModelManifest::load(&out.join("huncho-model.json")).unwrap();
        mf.validate().unwrap();
        assert_eq!(mf.name, "laya");
        let artifact = out.join("model.onnx");
        assert_eq!(fs::read(&artifact).unwrap(), b"fake-onnx-bytes");
        assert!(out.join("README.md").exists());
    }

    #[test]
    fn run_without_runner_writes_manifest_but_no_artifact() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("pkg");
        let mut a = args();
        a.out = out.to_string_lossy().to_string();
        run(a).unwrap();
        assert!(out.join("huncho-model.json").exists());
        assert!(out.join("README.md").exists());
        assert!(!out.join("model.onnx").exists());
    }

    #[test]
    fn assemble_standard_layout_copies_config_and_weights() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let out = tmp.path().join("out");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("config.json"), b"{\"hidden_size\": 8}").unwrap();
        fs::write(src.join("model.safetensors"), b"weights").unwrap();

        assemble_candle_package(&src, &out, None).unwrap();
        assert_eq!(fs::read(out.join("config.json")).unwrap(), b"{\"hidden_size\": 8}");
        assert_eq!(fs::read(out.join("model.safetensors")).unwrap(), b"weights");
    }

    #[test]
    fn assemble_laya_layout_remaps_encoder_config_and_tokenizer() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let out = tmp.path().join("out");
        fs::create_dir_all(src.join("encoder")).unwrap();
        fs::create_dir_all(src.join("tokenizer")).unwrap();
        fs::write(src.join("encoder/config.json"), b"{\"model_type\":\"modernbert\"}").unwrap();
        fs::write(src.join("model.safetensors"), b"weights").unwrap();
        fs::write(src.join("tokenizer/tokenizer.json"), b"{} ").unwrap();

        assemble_candle_package(&src, &out, Some("tokenizer.json")).unwrap();
        assert_eq!(
            fs::read(out.join("config.json")).unwrap(),
            b"{\"model_type\":\"modernbert\"}"
        );
        assert_eq!(fs::read(out.join("model.safetensors")).unwrap(), b"weights");
        assert_eq!(fs::read(out.join("tokenizer.json")).unwrap(), b"{} ");
    }

    #[test]
    fn assemble_sharded_weights_via_index() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let out = tmp.path().join("out");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("config.json"), b"{}").unwrap();
        fs::write(src.join("model.safetensors.index.json"),
            br#"{"weight_map": {"layer.0": "model-00001-of-00002.safetensors", "layer.1": "model-00002-of-00002.safetensors"}}"#).unwrap();
        fs::write(src.join("model-00001-of-00002.safetensors"), b"shard1").unwrap();
        fs::write(src.join("model-00002-of-00002.safetensors"), b"shard2").unwrap();

        assemble_candle_package(&src, &out, None).unwrap();
        assert_eq!(fs::read(out.join("model-00001-of-00002.safetensors")).unwrap(), b"shard1");
        assert_eq!(fs::read(out.join("model-00002-of-00002.safetensors")).unwrap(), b"shard2");
        assert!(out.join("model.safetensors.index.json").exists());
    }

    #[test]
    fn assemble_missing_config_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let out = tmp.path().join("out");
        fs::create_dir_all(&src).unwrap();
        assert!(assemble_candle_package(&src, &out, None).is_err());
    }

    #[test]
    fn run_with_source_assembles_candle_package() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let out = tmp.path().join("pkg");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("config.json"), b"{\"hidden_size\": 8}").unwrap();
        fs::write(src.join("model.safetensors"), b"weights").unwrap();

        let mut a = args();
        a.backend = "candle".into();
        a.out = out.to_string_lossy().to_string();
        a.source = Some(src.to_string_lossy().to_string());
        run(a).unwrap();

        assert!(out.join("huncho-model.json").exists());
        assert_eq!(fs::read(out.join("model.safetensors")).unwrap(), b"weights");
        assert_eq!(fs::read(out.join("config.json")).unwrap(), b"{\"hidden_size\": 8}");
        let mf = ModelManifest::load(&out.join("huncho-model.json")).unwrap();
        let candle = mf.backbone.artifacts.get(&BackendId::Candle).unwrap();
        assert_eq!(candle[0].path, "model.safetensors");
    }
}
