//! The `huncho-model.json` model package (PRD §6.2).
//!
//! A model package is a single manifest plus artifacts that every backend can
//! load. This module defines the manifest schema and validation.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// The manifest schema version this crate understands.
pub const MANIFEST_SCHEMA_VERSION: &str = "1.0";

// ---------------------------------------------------------------------------
// Enumerated types
// ---------------------------------------------------------------------------

/// The decision-model families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Family {
    /// F1 Encoder — scores candidates at option-marker positions.
    F1,
    /// F2 Pointer — pointer head over option-boundary tokens.
    F2,
    /// F3 Candidate-logit — softmax over one-token answer codes.
    F3,
    /// F4 Slot head — fixed-width decision head.
    F4,
    /// F5 Joint schema — scores every question in one shared forward pass.
    F5,
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Family {
    pub fn parse(s: &str) -> Result<Family> {
        match s.trim().to_uppercase().as_str() {
            "F1" => Ok(Family::F1),
            "F2" => Ok(Family::F2),
            "F3" => Ok(Family::F3),
            "F4" => Ok(Family::F4),
            "F5" => Ok(Family::F5),
            other => Err(Error::Package(format!("unknown family `{other}`"))),
        }
    }
}

/// The backends an engine can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendId {
    #[default]
    Onnx,
    #[serde(rename = "llamacpp")]
    LlamaCpp,
    Mlx,
    Vllm,
    Candle,
    Clef,
}

impl BackendId {
    /// Check a runtime selected from model metadata against the serving build.
    pub fn require_available(self, available: &[BackendId]) -> Result<Self> {
        if available.contains(&self) {
            return Ok(self);
        }
        let hint = match self {
            Self::Onnx => "rebuild with `--features onnx,hf,tokenizers`",
            Self::Candle => "rebuild with `--features candle,hf,tokenizers`",
            Self::Clef => "rebuild with `--features clef`",
            _ => "use a build that supports this backend",
        };
        Err(Error::Unsupported(format!(
            "model requires backend `{self}`, which is not available in this build; {hint}"
        )))
    }

    pub fn parse(s: &str) -> Result<BackendId> {
        match s.trim().to_lowercase().as_str() {
            "onnx" => Ok(BackendId::Onnx),
            "llamacpp" | "llama.cpp" | "llama_cpp" => Ok(BackendId::LlamaCpp),
            "mlx" => Ok(BackendId::Mlx),
            "vllm" => Ok(BackendId::Vllm),
            "candle" => Ok(BackendId::Candle),
            "clef" => Ok(BackendId::Clef),
            other => Err(Error::Package(format!("unknown backend `{other}`"))),
        }
    }
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                BackendId::Onnx => "onnx",
                BackendId::LlamaCpp => "llamacpp",
                BackendId::Mlx => "mlx",
                BackendId::Vllm => "vllm",
                BackendId::Candle => "candle",
                BackendId::Clef => "clef",
            }
        )
    }
}

/// The kind of head attached to a family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HeadKind {
    /// F1 — score candidates at option-marker positions.
    OptionMarker,
    /// F2 — pointer over option-boundary tokens.
    Pointer,
    /// F3 — softmax over one-token answer codes using the LM head.
    CandidateLogit,
    /// F4 — fixed-width slot decision head.
    Slot,
    /// F5 — joint schema head over the complete request.
    JointSchema,
}

impl HeadKind {
    pub fn parse(s: &str) -> Result<HeadKind> {
        match s.trim().to_lowercase().as_str() {
            "option-marker" | "optionmarker" => Ok(HeadKind::OptionMarker),
            "pointer" => Ok(HeadKind::Pointer),
            "candidate-logit" | "candidatelogit" => Ok(HeadKind::CandidateLogit),
            "slot" | "slot-head" => Ok(HeadKind::Slot),
            "joint-schema" => Ok(HeadKind::JointSchema),
            other => Err(Error::Package(format!("unknown head kind `{other}`"))),
        }
    }
}

/// Confidence definition (PRD CORE-05).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceDef {
    /// Jev's peak-based confidence: `(p_max - 1/n) / (1 - 1/n)`.
    #[serde(rename = "peak")]
    Peak,
    /// Normalized entropy-based confidence (used by Laya).
    #[serde(rename = "entropy")]
    Entropy,
    /// Maximum option probability (Clef's reference definition).
    #[serde(rename = "max-probability")]
    MaxProbability,
    /// A named, model-specific definition.
    #[serde(rename = "custom")]
    Custom(String),
}

impl Default for ConfidenceDef {
    fn default() -> Self {
        ConfidenceDef::Peak
    }
}

impl fmt::Display for ConfidenceDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfidenceDef::Peak => write!(f, "peak"),
            ConfidenceDef::Entropy => write!(f, "entropy"),
            ConfidenceDef::MaxProbability => write!(f, "max-probability"),
            ConfidenceDef::Custom(s) => write!(f, "custom:{s}"),
        }
    }
}

/// Calibration status for a backend × dtype combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CalibrationStatus {
    /// Fitted against the reference implementation.
    #[default]
    Fit,
    /// A refit after quantization or dtype change.
    Refit,
    /// Calibration pending; serving may be downgraded or gated.
    Pending,
}

// ---------------------------------------------------------------------------
// Manifest components
// ---------------------------------------------------------------------------

/// Where a backbone comes from.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum BackboneSource {
    /// A Hugging Face repo + pinned revision.
    Hf { repo: String, revision: String },
    /// A local path to a pre-converted backbone.
    Local { path: String },
}

/// A converted backbone artifact for a specific backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRef {
    /// Relative path from the package root.
    pub path: String,
    /// The dtype this artifact provides (e.g. `fp32`, `fp16`, `int8`, `q4`).
    pub dtype: String,
    /// Optional quantization scheme id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
}

/// The backbone section of the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Backbone {
    pub source: BackboneSource,
    /// Per-backend converted artifacts, keyed by backend id.
    #[serde(default)]
    pub artifacts: BTreeMap<BackendId, Vec<ArtifactRef>>,
    pub hidden_size: usize,
    pub max_context: usize,
    /// Path (relative) to a `tokenizer.json`, if bundled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
}

/// An optional LoRA adapter (PRD §6.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adapter {
    pub repo: String,
    pub revision: String,
    pub rank: usize,
}

/// F3 candidate-logit configuration (Nimble family).
///
/// F3 (candidate-logit) models — e.g. the `Bespoke-Nimble` adapters — classify
/// a requested schema field by scoring one-token answer codes through the LM
/// head. This section carries the codebook the model was trained with, the
/// system prompt, and the hash of the reference prompt builder so prompts stay
/// aligned with the upstream `prompt_code_sha256`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F3Config {
    /// Candidate answer codes (e.g. `"A"`, `"B"`, ...) in serving order.
    pub candidate_codes: Vec<String>,
    /// Token ids corresponding to each candidate code.
    pub candidate_token_ids: Vec<u32>,
    /// The system prompt prepended to the classified-fields prompt.
    pub system_prompt: String,
    /// Hash of the reference prompt-building source (`prompt_code_sha256`).
    pub prompt_code_sha256: String,
    /// Maximum input tokens the prompt may occupy.
    pub max_input_tokens: usize,
}

/// The head section of the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadConfig {
    pub kind: HeadKind,
    /// Path (relative) to head weights. Runtime precision is backend-specific.
    pub weights: String,
    /// Output width of the head's logits.
    #[serde(default = "default_head_width")]
    pub width: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pointer_offset: Option<usize>,
}

fn default_head_width() -> usize {
    1
}

impl HeadConfig {
    /// Validate that the head kind is consistent with the family.
    pub fn validate_family(&self, family: Family) -> Result<()> {
        let expected = family_kind(family);
        if self.kind != expected {
            return Err(Error::Package(format!(
                "family {family} requires a {expected:?} head, manifest declares {:?}",
                self.kind
            )));
        }
        Ok(())
    }
}

/// The prompt contract (PRD §6.2 `prompt_contract`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptContract {
    /// Template id (e.g. `laya-v1`, `kev-block-causal`).
    pub template: String,
    /// Tokens that mark candidate positions (F1).
    #[serde(default)]
    pub option_marker_tokens: Vec<String>,
    /// Token budget for the state / shared prefix.
    pub state_budget: usize,
    /// Token budget for the per-question head.
    pub head_budget: usize,
    /// Maximum options allowed by the contract.
    pub max_options: usize,
    /// Hash of the prompt contract bytes, used to detect upstream drift.
    pub contract_hash: String,
    /// Total per-question sequence cap (Laya `max_len`; default 512).
    #[serde(default = "default_prompt_max_len")]
    pub max_len: usize,
    /// Head region budget (Laya `head_max_len`; default 192).
    #[serde(default = "default_prompt_head_max_len")]
    pub head_max_len: usize,
}

fn default_prompt_max_len() -> usize {
    512
}

fn default_prompt_head_max_len() -> usize {
    192
}

/// A single calibration entry (per backend × dtype).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationEntry {
    pub temperature: f32,
    /// Per question-type temperatures (F4 slot head, and per-type).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_type_temperatures: Option<BTreeMap<String, f32>>,
    /// Per `{type}:{bucket}` temperatures (Laya `temperature_by_options`).
    /// Buckets are `2`, `3-5`, `6-10`, `11+`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_by_options: Option<BTreeMap<String, f32>>,
    #[serde(default)]
    pub confidence: ConfidenceDef,
    #[serde(default)]
    pub status: CalibrationStatus,
}

/// The calibration section (PRD §6.2 `calibration`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// The backend-agnostic default entry.
    pub default: CalibrationEntry,
    /// Per `{backend}:{dtype}` overrides.
    #[serde(default)]
    pub entries: BTreeMap<String, CalibrationEntry>,
    /// Hash of the eval set the temperatures were fitted on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_set_hash: Option<String>,
}

impl CalibrationConfig {
    /// Resolve the calibration entry for a backend and dtype, falling back to
    /// `default` when no exact key exists.
    pub fn resolve(&self, backend: &str, dtype: &str) -> CalibrationEntry {
        let key = format!("{backend}:{dtype}");
        self.entries
            .get(&key)
            .cloned()
            .unwrap_or_else(|| self.default.clone())
    }

    /// The exact lookup key for a backend+dtype.
    pub fn key_for(backend: &str, dtype: &str) -> String {
        format!("{backend}:{dtype}")
    }
}

/// Pointer to golden conformance vectors (PRD CONF-01).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub family_impl: String,
    pub revision: String,
    /// Path (relative) to golden vectors.
    pub golden: String,
}

/// Declared capabilities of a model package (not the backend).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub supports_fork: bool,
    #[serde(default)]
    pub supports_multi_lora: bool,
}

/// The top-level manifest (`huncho-model.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub schema_version: String,
    pub name: String,
    pub family: Family,
    pub backbone: Backbone,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<Adapter>,
    /// F3 candidate-logit (Nimble) codebook/system-prompt config. Present only
    /// for [`Family::F3`] packages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f3: Option<F3Config>,
    pub head: HeadConfig,
    pub prompt_contract: PromptContract,
    pub calibration: CalibrationConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<Reference>,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
}

impl ModelManifest {
    /// Load and validate a manifest from a path.
    pub fn load(path: impl AsRef<Path>) -> Result<ModelManifest> {
        let bytes = std::fs::read(path.as_ref())?;
        let manifest: ModelManifest = serde_json::from_slice(&bytes)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate the manifest self-consistency.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(Error::Package(format!(
                "unsupported manifest schema_version `{}` (expected {MANIFEST_SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        if self.name.trim().is_empty() {
            return Err(Error::Package("manifest `name` must be non-empty".into()));
        }
        if self.backbone.hidden_size == 0 {
            return Err(Error::Package("backbone.hidden_size must be > 0".into()));
        }
        if self.backbone.max_context == 0 {
            return Err(Error::Package("backbone.max_context must be > 0".into()));
        }
        if self.prompt_contract.max_options == 0 {
            return Err(Error::Package(
                "prompt_contract.max_options must be > 0".into(),
            ));
        }
        self.head.validate_family(self.family)?;
        if self.prompt_contract.contract_hash.trim().is_empty() {
            return Err(Error::Package(
                "prompt_contract.contract_hash must be non-empty".into(),
            ));
        }
        if self.family == Family::F3 && self.f3.is_none() {
            return Err(Error::Package(
                "family F3 requires an `f3` candidate-logit configuration".into(),
            ));
        }
        if let Some(f3) = &self.f3 {
            if self.family != Family::F3 {
                return Err(Error::Package(format!(
                    "family {} cannot declare an `f3` configuration",
                    self.family
                )));
            }
            if f3.candidate_codes.is_empty() {
                return Err(Error::Package("f3.candidate_codes must be non-empty".into()));
            }
            if f3.candidate_codes.len() != f3.candidate_token_ids.len() {
                return Err(Error::Package(format!(
                    "f3 candidate_codes ({}) must match candidate_token_ids ({})",
                    f3.candidate_codes.len(),
                    f3.candidate_token_ids.len()
                )));
            }
            if f3.prompt_code_sha256.trim().is_empty() {
                return Err(Error::Package(
                    "f3.prompt_code_sha256 must be non-empty".into(),
                ));
            }
        }
        if self.calibration.default.temperature <= 0.0 {
            return Err(Error::Package(
                "calibration.default.temperature must be > 0.0".into(),
            ));
        }
        Ok(())
    }

    /// The canonical `{backend}:{dtype}` calibration key.
    pub fn calibration_key(&self, backend: &str, dtype: &str) -> String {
        CalibrationConfig::key_for(backend, dtype)
    }

    /// Look up the artifact for a backend with the requested dtype, if present.
    pub fn find_artifact(&self, backend: BackendId, dtype: &str) -> Option<&ArtifactRef> {
        self.backbone
            .artifacts
            .get(&backend)
            .and_then(|list| list.iter().find(|a| a.dtype == dtype))
    }

    /// Select a real runtime from the package, preferring native Candle over
    /// an ONNX export when both are available. Never selects a mock runtime.
    pub fn select_backend(
        &self,
        available: &[BackendId],
        dtype: Option<&str>,
    ) -> Result<BackendId> {
        let candidates: Vec<_> = [
            BackendId::Clef,
            BackendId::Candle,
            BackendId::Onnx,
            BackendId::LlamaCpp,
            BackendId::Mlx,
            BackendId::Vllm,
        ]
        .into_iter()
        .filter(|backend| {
            let compatible = match backend {
                BackendId::Clef => self.family == Family::F5,
                BackendId::Candle => {
                    matches!(self.family, Family::F1 | Family::F3)
                        || (self.family == Family::F2 && self.prompt_contract.template == "kev-v1")
                }
                BackendId::Onnx => self.family == Family::F1,
                _ => self.family != Family::F5,
            };
            if !compatible {
                return false;
            }
            self.backbone
                .artifacts
                .get(backend)
                .is_some_and(|artifacts| {
                    artifacts
                        .iter()
                        .any(|a| dtype.map_or(true, |d| a.dtype == d))
                })
        })
        .collect();
        if let Some(backend) = candidates.iter().find(|b| available.contains(b)) {
            return Ok(*backend);
        }
        if let Some(backend) = candidates.first() {
            return backend.require_available(available);
        }
        let requested = dtype
            .map(|d| format!(" for dtype `{d}`"))
            .unwrap_or_default();
        Err(Error::Package(format!(
            "model `{}` declares no compatible backend artifacts{requested}; check its huncho-model.json",
            self.name
        )))
    }

    /// Default precision for artifact selection. Clef's device-specific runtime
    /// precision is chosen by its loader; all Clef dtypes use the same files.
    pub fn default_dtype(&self, backend: BackendId) -> &str {
        let preferred = if self.family == Family::F3 {
            "fp16"
        } else {
            "fp32"
        };
        if self.find_artifact(backend, preferred).is_some() {
            return preferred;
        }
        self.backbone
            .artifacts
            .get(&backend)
            .and_then(|artifacts| artifacts.first())
            .map(|a| a.dtype.as_str())
            .unwrap_or(preferred)
    }
}

/// The head kind expected for each family.
pub fn family_kind(family: Family) -> HeadKind {
    match family {
        Family::F1 => HeadKind::OptionMarker,
        Family::F2 => HeadKind::Pointer,
        Family::F3 => HeadKind::CandidateLogit,
        Family::F4 => HeadKind::Slot,
        Family::F5 => HeadKind::JointSchema,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_manifest() -> ModelManifest {
        let json = serde_json::json!({
            "schema_version": "1.0",
            "name": "laya-test",
            "family": "F1",
            "backbone": {
                "source": { "kind": "hf", "repo": "convaiinnovations/laya", "revision": "deadbeef" },
                "artifacts": {
                    "onnx": [{ "path": "model.onnx", "dtype": "fp32" }]
                },
                "hidden_size": 1024,
                "max_context": 4096,
                "tokenizer": "tokenizer.json"
            },
            "head": { "kind": "option-marker", "weights": "head.safetensors", "width": 1 },
            "prompt_contract": {
                "template": "laya-v1",
                "option_marker_tokens": ["<option:0>"],
                "state_budget": 3072,
                "head_budget": 512,
                "max_options": 255,
                "contract_hash": "abc123"
            },
            "calibration": {
                "default": { "temperature": 1.0, "confidence": "peak" },
                "entries": { "onnx:fp16": { "temperature": 1.1, "confidence": "peak", "status": "refit" } }
            },
            "reference": { "family_impl": "laya-serve", "revision": "x", "golden": "golden.json" }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn loads_and_validates() {
        let m = minimal_manifest();
        assert!(m.validate().is_ok());
        assert_eq!(m.family, Family::F1);
        assert_eq!(m.find_artifact(BackendId::Onnx, "fp32").unwrap().path, "model.onnx");
    }

    #[test]
    fn rejects_family_head_mismatch() {
        let mut m = minimal_manifest();
        m.head.kind = HeadKind::Pointer;
        assert!(m.validate().is_err());
    }

    #[test]
    fn resolves_calibration() {
        let m = minimal_manifest();
        let e = m.calibration.resolve("onnx", "fp16");
        assert!((e.temperature - 1.1).abs() < 1e-6);
        assert_eq!(e.status, CalibrationStatus::Refit);
        let d = m.calibration.resolve("mlx", "fp16");
        assert!((d.temperature - 1.0).abs() < 1e-6);
        assert_eq!(d.status, CalibrationStatus::Fit);
    }

    #[test]
    fn automatic_backend_uses_available_artifacts_and_requested_dtype() {
        let mut m = minimal_manifest();
        m.backbone.artifacts.insert(
            BackendId::Candle,
            vec![ArtifactRef {
                path: "model.safetensors".into(),
                dtype: "fp16".into(),
                quantization: None,
            }],
        );
        let both = [BackendId::Onnx, BackendId::Candle];
        assert_eq!(m.select_backend(&both, None).unwrap(), BackendId::Candle);
        assert_eq!(m.default_dtype(BackendId::Candle), "fp16");
        assert_eq!(
            m.select_backend(&both, Some("fp32")).unwrap(),
            BackendId::Onnx
        );
        assert_eq!(
            m.select_backend(&[BackendId::Onnx], None).unwrap(),
            BackendId::Onnx
        );
        assert!(m
            .select_backend(&both, Some("q4"))
            .unwrap_err()
            .to_string()
            .contains("dtype `q4`"));
        assert!(m
            .select_backend(&[], None)
            .unwrap_err()
            .to_string()
            .contains("--features candle"));
    }

    #[test]
    fn automatic_backend_does_not_fall_back_for_missing_or_incompatible_artifacts() {
        let mut m = minimal_manifest();
        m.family = Family::F5;
        assert!(m
            .select_backend(&[BackendId::Onnx, BackendId::Candle], None)
            .is_err());
        m.backbone.artifacts.insert(
            BackendId::Clef,
            vec![ArtifactRef {
                path: "config.json".into(),
                dtype: "fp32".into(),
                quantization: None,
            }],
        );
        assert_eq!(
            m.select_backend(&[BackendId::Clef], None).unwrap(),
            BackendId::Clef
        );
        assert!(m
            .select_backend(&[BackendId::Candle], None)
            .unwrap_err()
            .to_string()
            .contains("--features clef"));
        m.backbone.artifacts.clear();
        assert!(m
            .select_backend(&[BackendId::Clef], None)
            .unwrap_err()
            .to_string()
            .contains("no compatible backend artifacts"));
    }
}
