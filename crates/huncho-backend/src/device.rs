//! Shared CUDA selection for Kev and Clef. HUNCHO_DEVICE takes precedence
//! over the legacy HUNCHO_CLEF_DEVICE alias.
use candle::{DType, Device, Tensor};
use huncho_core::error::{Error, Result};

fn backend_error(e: candle::Error) -> Error {
    Error::Backend(format!("CUDA: {e}"))
}

/// Try CUDA first in auto mode; an unavailable device falls back to CPU.
/// The RPM launcher also handles missing shared libraries before this runs.
pub fn device_from_env() -> Result<Device> {
    let name = std::env::var("HUNCHO_DEVICE")
        .or_else(|_| std::env::var("HUNCHO_CLEF_DEVICE"))
        .unwrap_or_else(|_| "auto".into());
    select_device(&name, cuda_device)
}

fn select_device(name: &str, cuda: impl FnOnce(usize) -> Result<Device>) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "auto" => match cuda(0) {
            Ok(device) => Ok(device),
            Err(e) => {
                log::debug!("CUDA unavailable; using CPU: {e}");
                Ok(Device::Cpu)
            }
        },
        _ if name == "cuda" || name.starts_with("cuda:") => {
            let ordinal = name
                .strip_prefix("cuda:")
                .unwrap_or("0")
                .parse::<usize>()
                .map_err(|_| Error::Package(format!("invalid device `{name}`")))?;
            cuda(ordinal).map_err(|e| Error::Unsupported(format!("CUDA device unavailable ({e}); check the NVIDIA driver/runtime and GPU compatibility, or use HUNCHO_DEVICE=cpu")))
        }
        _ => Err(Error::Package(format!(
            "invalid device selection `{name}`; use auto, cpu, cuda, or cuda:N"
        ))),
    }
}

fn cuda_device(ordinal: usize) -> Result<Device> {
    let device = Device::new_cuda(ordinal).map_err(backend_error)?;
    // Creating a context alone does not prove the GPU/driver can run the PTX
    // shipped in the package. Exercise a Candle kernel and cuBLAS with FP16,
    // which is also supported by Turing GPUs such as the T4. Requiring BF16
    // here would incorrectly reject those GPUs before dtype selection.
    let x = Tensor::ones((16, 16), DType::F16, &device).map_err(backend_error)?;
    let sum = (&x + &x).map_err(backend_error)?;
    sum.matmul(&x)
        .and_then(|v| v.to_dtype(DType::F32))
        .and_then(|v| v.to_vec2::<f32>())
        .map_err(backend_error)?;
    Ok(device)
}

#[cfg(feature = "clef")]
pub(crate) fn supports_bf16(device: &Device) -> Result<bool> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda) = device {
        let (major, _) = cuda
            .cuda_stream()
            .context()
            .compute_capability()
            .map_err(|e| Error::Backend(format!("reading CUDA compute capability: {e}")))?;
        return Ok(major >= 8);
    }
    let _ = device;
    Ok(false)
}

#[cfg(test)]
mod device_tests {
    use super::*;

    #[test]
    fn cpu_never_initializes_cuda() {
        assert!(select_device("cpu", |_| panic!("CUDA must not be called"))
            .unwrap()
            .is_cpu());
    }

    #[test]
    fn auto_falls_back_when_cuda_initialization_or_kernels_fail() {
        let device = select_device("auto", |ordinal| {
            assert_eq!(ordinal, 0);
            Err(Error::Backend("CUDA initialization failed".into()))
        })
        .unwrap();
        assert!(device.is_cpu());
    }

    #[test]
    fn auto_returns_successful_device() {
        // The callback substitutes a CPU device so the selection policy can be
        // tested without requiring a GPU on the test runner.
        assert!(select_device("auto", |ordinal| {
            assert_eq!(ordinal, 0);
            Ok(Device::Cpu)
        })
        .unwrap()
        .is_cpu());
    }

    #[test]
    fn explicit_cuda_keeps_ordinal_and_never_falls_back() {
        for (name, expected) in [("cuda", 0), ("cuda:2", 2)] {
            let err = select_device(name, |ordinal| {
                assert_eq!(ordinal, expected);
                Err(Error::Backend("GPU unavailable".into()))
            })
            .unwrap_err();
            assert!(err.to_string().contains("GPU unavailable"));
        }
    }

    #[test]
    fn invalid_devices_fail_before_initialization() {
        for name in ["", "gpu", "cuda:", "cuda:-1", "cuda:abc"] {
            assert!(select_device(name, |_| panic!("invalid device must be rejected")).is_err());
        }
    }
}
