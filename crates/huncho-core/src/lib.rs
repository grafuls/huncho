//! `huncho-core` — the portable engine core for System One decision models.
//!
//! Implements the Jev `/v1/systemone` wire contract, model packaging, per-family
//! prompt building, decision heads, and the calibration layer that guarantees
//! probability fidelity across backends and quantizations.

pub mod backend;
mod cache_key;
pub mod calibration;
pub mod conformance;
pub mod contract;
pub mod engine;
pub mod error;
pub mod head;
pub mod manifest;
pub mod prompt;
mod prompt_cache;
mod response_cache;
pub mod tensor;
pub mod tokenizer;

pub use error::{Error, ErrorBody, Result};
pub use manifest::{BackendId, Family, ModelManifest};
