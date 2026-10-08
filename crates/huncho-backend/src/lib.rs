//! `huncho-backend` — concrete [`huncho_core::backend::Backend`] implementations.
//!
//! * [`mock::MockBackend`] — deterministic, dependency-free backend used for
//!   offline tests, demonstrations, and the CI conformance harness.
//! * [`onnx::OnnxBackend`] — ONNX Runtime CPU backend, behind the
//!   `onnx` feature. Targets F1.
//! * [`candle::CandleBackend`] — loads HF `safetensors` ModernBERT directly with
//!   candle (no ONNX/Python), behind the `candle` feature. The primary
//!   real-model path for F1.
//! * [`qwen3_5::Qwen3_5Backend`] — the F3 (candidate-logit) and Kev F2 paths: a
//!   candle port of the Qwen3.5 hybrid Gated DeltaNet / full-attention backbone
//!   with a PEFT LoRA merge, behind the `candle` feature.
//! * [`kev`] — Kev checkpoint metadata and trained pointer-head loading.
//! * [`clef::ClefBackend`] — native whole-request Clef inference, behind `clef`.
//! * [`null::NullBackend`] — a backend that always raises an error, used as a
//!   placeholder when a model is not yet loaded.

#[cfg(feature = "candle")]
pub mod candle;
#[cfg(feature = "clef")]
pub mod clef;
#[cfg(feature = "candle")]
mod conv_cpu;
#[cfg(feature = "candle")]
mod cpu_profile;
#[cfg(feature = "cpu-blas")]
mod cpu_blas;
#[cfg(feature = "candle")]
mod delta_cpu;
#[cfg(feature = "candle")]
mod gate_cpu;
#[cfg(feature = "candle")]
pub mod device;
#[cfg(feature = "candle")]
pub mod kev;
pub mod mock;
pub mod null;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "onnx-shared")]
mod onnx_shared;
#[cfg(feature = "candle")]
pub mod qwen3_5;
#[cfg(feature = "shared-base")]
pub mod shared_base;
#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(feature = "candle")]
pub use candle::CandleBackend;
#[cfg(feature = "clef")]
pub use clef::ClefBackend;
pub use mock::MockBackend;
pub use null::NullBackend;
#[cfg(feature = "onnx")]
pub use onnx::OnnxBackend;
#[cfg(feature = "candle")]
pub use qwen3_5::Qwen3_5Backend;
#[cfg(feature = "llamacpp")]
pub use llamacpp::LlamaCppBackend;

/// Process-unique handles prevent an ID from one model accidentally selecting
/// another model's live prefix. Allocation never wraps and reuses an old ID.
fn next_cache_handle() -> huncho_core::error::Result<huncho_core::backend::CacheHandle> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let id = NEXT
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| huncho_core::error::Error::Backend("cache handle space exhausted".into()))?;
    Ok(huncho_core::backend::CacheHandle { id })
}

#[cfg(feature = "candle")]
fn device_label(device: &::candle::Device) -> String {
    match device.location() {
        ::candle::DeviceLocation::Cpu => "CPU".into(),
        ::candle::DeviceLocation::Cuda { gpu_id } => format!("GPU (CUDA device {gpu_id})"),
        ::candle::DeviceLocation::Metal { gpu_id } => format!("GPU (Metal device {gpu_id})"),
    }
}
