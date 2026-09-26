//! `s1 convert` — produce backend artifacts plus a manifest from an HF repo
//! (CONV-01).

use std::collections::BTreeMap;
use std::path::Path;

use clap::Args;
use s1_core::manifest::{
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

    /// Target backend (onnx|gguf|mlx).
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
    /// vars `S1_ARTIFACT`, `S1_REPO`, `S1_REVISION`, `S1_OUT` are set.
    #[arg(long)]
    pub runner: Option<String>,

    /// Package name; defaults to the repo basename.
    #[arg(long)]
    pub name: Option<String>,
}

pub fn run(args: ConvertArgs) -> anyhow::Result<()> {
    let family = Family::parse(&args.family)?;
    let backend = BackendId::parse(&args.backend)?;
    let artifact_name = match backend {
        BackendId::Onnx => "model.onnx".to_string(),
        BackendId::LlamaCpp => "model.gguf".to_string(),
        BackendId::Mlx => "model.safetensors".to_string(),
        BackendId::Vllm => "model.safetensors".to_string(),
    };

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
            path: artifact_name.clone(),
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

    let out_dir = Path::new(&args.out);
    std::fs::create_dir_all(out_dir)?;
    let manifest_path = out_dir.join("s1-model.json");
    let json = serde_json::to_vec_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;
    tracing::info!("wrote manifest {}", manifest_path.display());

    if let Some(runner) = &args.runner {
        tracing::info!("running artifact converter: {runner}");
        let status = std::process::Command::new("sh")
            .arg("-c")
            .arg(runner)
            .env("S1_ARTIFACT", artifact_name)
            .env("S1_REPO", &args.hf_repo)
            .env("S1_REVISION", &args.revision)
            .env("S1_OUT", &args.out)
            .status()?;
        if !status.success() {
            anyhow::bail!("artifact converter exited with {status}");
        }
    } else {
        tracing::warn!(
            "no --runner given; place the converted artifact at {}/{}",
            out_dir.display(),
            artifact_name
        );
    }

    // Write a small README describing next steps (calibration, conformance).
    let readme = format!(
        "# {}\n\nConverted model package (family {:?}, backend {:?}, dtype {})\n\n\
         Next steps:\n  1. `s1 calibrate` to fit temperatures (CONV-02)\n\
         2. `s1 conform` to verify probability fidelity (CONF-02)\n\
         3. `s1 serve --manifest s1-model.json` to serve it\n",
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
