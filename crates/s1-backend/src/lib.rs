//! `s1-backend` — concrete [`s1_core::backend::Backend`] implementations.
//!
//! * [`mock::MockBackend`] — deterministic, dependency-free backend used for
//!   offline tests, demonstrations, and the CI conformance harness.
//! * [`onnx::OnnxBackend`] — ONNX Runtime (CPU/CUDA/WebGPU) backend, behind the
//!   `onnx` feature. Targets F1.
//! * [`null::NullBackend`] — a backend that always raises an error, used as a
//!   placeholder when a model is not yet loaded.

pub mod mock;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod null;

pub use mock::MockBackend;
#[cfg(feature = "onnx")]
pub use onnx::OnnxBackend;
pub use null::NullBackend;
