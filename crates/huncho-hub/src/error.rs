//! Error types for the Hub resolver.

use std::path::PathBuf;

/// Errors produced while resolving a model package reference.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    /// A model reference string could not be interpreted.
    #[error("invalid model reference `{ref_}`: {reason}")]
    InvalidRef { ref_: String, reason: String },

    /// A local package directory did not contain the required manifest.
    #[error("model package not found: expected `{0}`")]
    PackageNotFound(PathBuf),

    /// The manifest does not declare an artifact required for the selected
    /// backend/dtype.
    #[error("model package `{package}` has no artifact for backend `{backend}` dtype `{dtype}`")]
    MissingArtifact {
        package: String,
        backend: String,
        dtype: String,
    },

    /// A failure while talking to the Hugging Face Hub.
    #[error("huggingface hub error: {0}")]
    Hf(String),

    /// The manifest could not be loaded / validated.
    #[error("invalid model package: {0}")]
    Package(String),

    /// Filesystem error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl HubError {
    /// Wrap an `hf_hub` error with a human-readable operation context.
    pub fn hf(context: impl std::fmt::Display, err: hf_hub::HFError) -> HubError {
        HubError::Hf(format!("{context}: {err}"))
    }
}

/// A convenience result alias.
pub type Result<T> = std::result::Result<T, HubError>;
