//! `huncho-hub` — Hugging Face Hub resolution for Huncho model packages.
//!
//! Lets a Huncho model package be referenced by Hugging Face repository id
//! (`owner/repo`) instead of a local path. The package manifest
//! (`huncho-model.json`) and the artifacts it references are fetched from the
//! Hub into the standard HF cache, honoring `HF_TOKEN`, `HF_HOME`, and
//! `HF_HUB_CACHE` — the same resolution model vLLM uses for its HuggingFace
//! integration.
//!
//! This is a thin, feature-gated dependency: it is only compiled by the CLI
//! when the `hf` feature is enabled so the default build stays lightweight.

pub mod error;
pub mod http;
pub mod progress;
pub mod resolver;

pub use error::{HubError, Result};
pub use progress::FileDownloadProgress;
pub use resolver::{ModelRef, ResolveOptions, ResolvedPackage, resolve, resolve_manifest_path};
