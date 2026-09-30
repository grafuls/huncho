//! `huncho-backend` — concrete [`huncho_core::backend::Backend`] implementations.
//!
//! * [`mock::MockBackend`] — deterministic, dependency-free backend used for
//!   offline tests, demonstrations, and the CI conformance harness.
//! * [`onnx::OnnxBackend`] — ONNX Runtime (CPU/CUDA/WebGPU) backend, behind the
//!   `onnx` feature. Targets F1.
//! * [`candle::CandleBackend`] — loads HF `safetensors` ModernBERT directly with
//!   candle (no ONNX/Python), behind the `candle` feature. The primary
//!   real-model path for F1.
//! * [`qwen3_5::Qwen3_5Backend`] — the F3 (candidate-logit) path: a from-scratch
//!   candle port of the Qwen3.5 hybrid Gated DeltaNet / full-attention backbone
//!   with a PEFT LoRA merge, behind the `candle` feature.
//! * [`null::NullBackend`] — a backend that always raises an error, used as a
//!   placeholder when a model is not yet loaded.

pub mod mock;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "candle")]
pub mod candle;
#[cfg(feature = "candle")]
pub mod qwen3_5;
pub mod null;

pub use mock::MockBackend;
#[cfg(feature = "onnx")]
pub use onnx::OnnxBackend;
#[cfg(feature = "candle")]
pub use candle::CandleBackend;
#[cfg(feature = "candle")]
pub use qwen3_5::Qwen3_5Backend;
pub use null::NullBackend;
