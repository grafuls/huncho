//! `huncho-backend` — concrete [`huncho_core::backend::Backend`] implementations.
//!
//! * [`mock::MockBackend`] — deterministic, dependency-free backend used for
//!   offline tests, demonstrations, and the CI conformance harness.
//! * [`onnx::OnnxBackend`] — ONNX Runtime (CPU/CUDA/WebGPU) backend, behind the
//!   `onnx` feature. Targets F1.
//! * [`candle::CandleBackend`] — loads HF `safetensors` ModernBERT directly with
//!   candle (no ONNX/Python), behind the `candle` feature. The primary
//!   real-model path for F1.
//! * [`null::NullBackend`] — a backend that always raises an error, used as a
//!   placeholder when a model is not yet loaded.

pub mod mock;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "candle")]
pub mod candle;
pub mod null;

pub use mock::MockBackend;
#[cfg(feature = "onnx")]
pub use onnx::OnnxBackend;
#[cfg(feature = "candle")]
pub use candle::CandleBackend;
pub use null::NullBackend;
