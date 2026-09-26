//! The `s1-model.json` model package (PRD §6.2).
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

/// The four decision-model families (PRD §4).
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
}

impl BackendId {
    pub fn parse(s: &str) -> Result<BackendId> {
        match s.trim().to_lowercase().as_str() {
            "onnx" => Ok(BackendId::Onnx),
            "llamacpp" | "llama.cpp" | "llama_cpp" => Ok(BackendId::LlamaCpp),
            "mlx" => Ok(BackendId::Mlx),
            "vllm" => Ok(BackendId::Vllm),
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
}

impl HeadKind {
    pub fn parse(s: &str) -> Result<HeadKind> {
        match s.trim().to_lowercase().as_str() {
            "option-marker" | "optionmarker" => Ok(HeadKind::OptionMarker),
            "pointer" => Ok(HeadKind::Pointer),
            "candidate-logit" | "candidatelogit" => Ok(HeadKind::CandidateLogit),
            "slot" | "slot-head" => Ok(HeadKind::Slot),
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

/// The head section of the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadConfig {
    pub kind: HeadKind,
    /// Path (relative) to head weights in safetensors. Always fp32.
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
}

/// A single calibration entry (per backend × dtype).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationEntry {
    pub temperature: f32,
    /// Per question-type temperatures (F4 slot head, and per-type).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_type_temperatures: Option<BTreeMap<String, f32>>,
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

/// The top-level manifest (`s1-model.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub schema_version: String,
    pub name: String,
    pub family: Family,
    pub backbone: Backbone,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<Adapter>,
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
}

/// The head kind expected for each family.
pub fn family_kind(family: Family) -> HeadKind {
    match family {
        Family::F1 => HeadKind::OptionMarker,
        Family::F2 => HeadKind::Pointer,
        Family::F3 => HeadKind::CandidateLogit,
        Family::F4 => HeadKind::Slot,
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
}
