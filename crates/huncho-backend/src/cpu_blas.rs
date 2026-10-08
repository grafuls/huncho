//! Explicit LP64 OpenBLAS projection profile; no library is loaded by default.
//! ABI: OpenMathLib/OpenBLAS v0.3.33 cblas.h and openblas_get_config.c.
use candle::{CpuStorage, Device, Storage, Tensor};
use candle_nn::Linear;
use huncho_core::{Error, Result};
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::{c_char, c_int, CStr},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

type Sgemm = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    f32,
    *mut f32,
    c_int,
);
type GetString = unsafe extern "C" fn() -> *const c_char;
type GetInt = unsafe extern "C" fn() -> c_int;
type SetInt = unsafe extern "C" fn(c_int);

pub(crate) struct Runtime {
    // Keep every function pointer and OpenBLAS worker alive for all projections.
    _library: Library,
    sgemm: Sgemm,
    get_threads: GetInt,
    path: PathBuf,
    sha256: String,
    config: String,
    core: String,
    threads: c_int,
}

fn text(pointer: *const c_char) -> Result<String> {
    if pointer.is_null() {
        return Err(Error::Backend("OpenBLAS returned a null identity".into()));
    }
    // The documented OpenBLAS getters return static, NUL-terminated strings.
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map_err(|e| Error::Backend(e.to_string()))
}

impl Runtime {
    fn load(path: &Path, threads: c_int) -> Result<Self> {
        let path = std::fs::canonicalize(path)?;
        if !path.is_file() {
            return Err(Error::Request(
                "OpenBLAS path must name a regular shared library".into(),
            ));
        }
        let before = Sha256::digest(std::fs::read(&path)?);
        // Explicit caller-selected native library, never a default search path.
        let library = unsafe { Library::new(&path) }.map_err(|e| Error::Backend(e.to_string()))?;
        let symbol = |name: &[u8]| {
            Error::Backend(format!(
                "OpenBLAS does not export {}",
                String::from_utf8_lossy(name)
            ))
        };
        // Check the integer ABI before obtaining or calling any BLAS kernel.
        let config = unsafe {
            *library
                .get::<GetString>(b"openblas_get_config\0")
                .map_err(|_| symbol(b"openblas_get_config"))?
        };
        let config = text(unsafe { config() })?;
        validate_config(&config)?;
        let core = unsafe {
            *library
                .get::<GetString>(b"openblas_get_corename\0")
                .map_err(|_| symbol(b"openblas_get_corename"))?
        };
        let core = text(unsafe { core() })?;
        let get_threads = unsafe {
            *library
                .get::<GetInt>(b"openblas_get_num_threads\0")
                .map_err(|_| symbol(b"openblas_get_num_threads"))?
        };
        let set_threads = unsafe {
            *library
                .get::<SetInt>(b"openblas_set_num_threads\0")
                .map_err(|_| symbol(b"openblas_set_num_threads"))?
        };
        let sgemm = unsafe {
            *library
                .get::<Sgemm>(b"cblas_sgemm\0")
                .map_err(|_| symbol(b"cblas_sgemm"))?
        };
        unsafe {
            set_threads(threads);
        }
        if unsafe { get_threads() } != threads {
            return Err(Error::Backend(
                "OpenBLAS did not accept the fixed thread budget".into(),
            ));
        }
        if Sha256::digest(std::fs::read(&path)?) != before {
            return Err(Error::Backend(
                "OpenBLAS library changed while loading".into(),
            ));
        }
        Ok(Self {
            _library: library,
            sgemm,
            get_threads,
            path,
            sha256: format!("{before:x}"),
            config,
            core,
            threads,
        })
    }

    pub(crate) fn record(&self, extra: &mut BTreeMap<String, String>) {
        extra.insert("cpu_blas_execution".into(), "openblas-lp64-fp32-v1".into());
        extra.insert("cpu_blas_library_sha256".into(), self.sha256.clone());
        extra.insert("cpu_blas_config".into(), self.config.clone());
        extra.insert("cpu_blas_core".into(), self.core.clone());
        extra.insert("cpu_blas_threads".into(), self.threads.to_string());
    }

    pub(crate) fn forward(&self, linear: &Linear, input: &Tensor) -> candle::Result<Tensor> {
        if input.dtype() != candle::DType::F32
            || !input.device().is_cpu()
            || linear.weight().dtype() != candle::DType::F32
            || !linear.weight().device().is_cpu()
        {
            candle::bail!("OpenBLAS projections require CPU FP32 input and weights");
        }
        if unsafe { (self.get_threads)() } != self.threads {
            candle::bail!("OpenBLAS global thread budget changed after qualification");
        }
        let (output_width, width) = linear.weight().dims2()?;
        if input.rank() == 0
            || input.dim(candle::D::Minus1)? != width
            || width == 0
            || output_width == 0
        {
            candle::bail!("OpenBLAS projection dimensions do not match");
        }
        let rows = input.elem_count() / width;
        let m = c_int::try_from(rows)
            .map_err(|_| candle::Error::Msg("OpenBLAS row count exceeds LP64 ABI".into()))?;
        let n = c_int::try_from(output_width)
            .map_err(|_| candle::Error::Msg("OpenBLAS output width exceeds LP64 ABI".into()))?;
        let k = c_int::try_from(width)
            .map_err(|_| candle::Error::Msg("OpenBLAS input width exceeds LP64 ABI".into()))?;
        let length = rows
            .checked_mul(output_width)
            .ok_or_else(|| candle::Error::Msg("OpenBLAS output size overflow".into()))?;
        let input = input.contiguous()?;
        let weight = linear.weight().contiguous()?;
        let (input_storage, input_layout) = input.storage_and_layout();
        let (weight_storage, weight_layout) = weight.storage_and_layout();
        let (Storage::Cpu(CpuStorage::F32(x)), Storage::Cpu(CpuStorage::F32(w))) =
            (&*input_storage, &*weight_storage)
        else {
            candle::bail!("OpenBLAS requires CPU FP32 storage");
        };
        let (xs, xe) = input_layout
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("noncontiguous OpenBLAS input".into()))?;
        let (ws, we) = weight_layout
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("noncontiguous OpenBLAS weight".into()))?;
        if xe - xs != rows * width || we - ws != output_width * width {
            candle::bail!("OpenBLAS visible storage size mismatch");
        }
        let mut values = vec![0f32; length];
        if rows > 0 {
            // Row-major X[M,K] * W[N,K]^T => Y[M,N]. Borrowed storage guards
            // and the unique initialized output remain alive for this call.
            unsafe {
                (self.sgemm)(
                    101,
                    111,
                    112,
                    m,
                    n,
                    k,
                    1.,
                    x[xs..xe].as_ptr(),
                    k,
                    w[ws..we].as_ptr(),
                    k,
                    0.,
                    values.as_mut_ptr(),
                    n,
                );
            }
        }
        let mut shape = input.dims().to_vec();
        *shape.last_mut().unwrap() = output_width;
        let output = Tensor::from_vec(values, shape, &Device::Cpu)?;
        match linear.bias() {
            Some(bias) => output.broadcast_add(bias),
            None => Ok(output),
        }
    }
}

fn validate_config(config: &str) -> Result<()> {
    let flags: Vec<_> = config.split_whitespace().collect();
    if !config.starts_with("OpenBLAS ")
        || config
            .split_whitespace()
            .any(|part| matches!(part, "USE64BITINT" | "USE_OPENMP"))
    {
        return Err(Error::Unsupported("CPU BLAS requires LP64 OpenBLAS with pthread or sequential execution; ILP64, OpenMP and other vendors are rejected".into()));
    }
    if flags.contains(&"SINGLE_THREADED") && !flags.contains(&"USE_LOCKING") {
        return Err(Error::Unsupported(
            "single-thread OpenBLAS requires USE_LOCKING for independent concurrent contexts"
                .into(),
        ));
    }
    Ok(())
}

pub(crate) fn configured() -> Result<Option<Arc<Runtime>>> {
    let Some(path) = std::env::var_os("HUNCHO_CPU_BLAS_LIBRARY") else {
        if std::env::var_os("HUNCHO_CPU_BLAS_THREADS").is_some() {
            return Err(Error::Request(
                "CPU BLAS threads require an explicit library".into(),
            ));
        }
        return Ok(None);
    };
    let path = std::fs::canonicalize(path)?;
    let threads = std::env::var("HUNCHO_CPU_BLAS_THREADS")
        .unwrap_or_else(|_| "1".into())
        .parse::<c_int>()
        .map_err(|_| Error::Request("CPU BLAS threads must be 1..256".into()))?;
    if !(1..=256).contains(&threads) {
        return Err(Error::Request("CPU BLAS threads must be 1..256".into()));
    }
    static RUNTIME: OnceLock<std::result::Result<Arc<Runtime>, String>> = OnceLock::new();
    let runtime = RUNTIME
        .get_or_init(|| {
            Runtime::load(&path, threads)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| Error::Backend(e.clone()))?;
    if runtime.path != path || runtime.threads != threads {
        return Err(Error::Unsupported(
            "configure one immutable OpenBLAS library/thread budget per process".into(),
        ));
    }
    Ok(Some(runtime.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_incompatible_integer_abis_and_vendors() {
        assert!(validate_config("OpenBLAS 0.3.33 DYNAMIC_ARCH SkylakeX MAX_THREADS=64").is_ok());
        assert!(validate_config("OpenBLAS 0.3.33 USE64BITINT DYNAMIC_ARCH").is_err());
        assert!(validate_config("other BLAS").is_err());
        assert!(validate_config("OpenBLAS 0.3.33 USE_OPENMP").is_err());
        assert!(validate_config("OpenBLAS 0.3.33 SINGLE_THREADED").is_err());
        assert!(validate_config("OpenBLAS 0.3.33 SINGLE_THREADED USE_LOCKING").is_ok());
    }
    #[test]
    #[ignore = "requires explicit HUNCHO_CPU_BLAS_LIBRARY pointing to LP64 OpenBLAS"]
    fn real_openblas_preserves_rows_bias_and_contiguous_offsets() {
        use candle::Module;
        let runtime = configured().unwrap().unwrap();
        for (batch, seq, input_width, output_width) in [(1, 1, 1, 1), (2, 5, 7, 11), (2, 3, 64, 32)]
        {
            let make = |n: usize| {
                Tensor::from_vec(
                    (0..n)
                        .map(|i| (i % 37) as f32 / 37. - 0.5)
                        .collect::<Vec<_>>(),
                    n,
                    &Device::Cpu,
                )
                .unwrap()
            };
            let weight = make(input_width * output_width)
                .reshape((output_width, input_width))
                .unwrap();
            let bias = make(output_width);
            let linear = Linear::new(weight, Some(bias));
            let input = make(batch * (seq + 2) * input_width)
                .reshape((batch, seq + 2, input_width))
                .unwrap()
                .narrow(1, 1, seq)
                .unwrap();
            let expected = linear
                .forward(&input.contiguous().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let actual = runtime
                .forward(&linear, &input)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() <= 1e-5, "{a} vs {b}");
            }
            assert!(runtime
                .forward(&linear, &input.to_dtype(candle::DType::F16).unwrap())
                .is_err());
        }
    }

    #[test]
    #[ignore = "CPU projection microbenchmark; explicit OpenBLAS library and fixed Rayon/Candle threads required"]
    fn real_openblas_timing() {
        use candle::Module;
        use std::time::Instant;
        let runtime = configured().unwrap().unwrap();
        let input = Tensor::from_vec(
            (0..128 * 1024)
                .map(|i| (i % 37) as f32 / 37. - 0.5)
                .collect::<Vec<_>>(),
            (1, 128, 1024),
            &Device::Cpu,
        )
        .unwrap();
        let linear = Linear::new(
            Tensor::from_vec(
                (0..1536 * 1024)
                    .map(|i| (i % 53) as f32 / 53. - 0.5)
                    .collect::<Vec<_>>(),
                (1536, 1024),
                &Device::Cpu,
            )
            .unwrap(),
            None,
        );
        let expected = linear
            .forward(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = runtime
            .forward(&linear, &input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let delta = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(delta <= 1e-3);
        for _ in 0..3 {
            std::hint::black_box(linear.forward(&input).unwrap());
            std::hint::black_box(runtime.forward(&linear, &input).unwrap());
        }
        let mut pairs = Vec::new();
        for index in 0..12 {
            let time = |blas| {
                let started = Instant::now();
                for _ in 0..4 {
                    std::hint::black_box(if blas {
                        runtime.forward(&linear, &input).unwrap()
                    } else {
                        linear.forward(&input).unwrap()
                    });
                }
                started.elapsed().as_secs_f64() / 4.
            };
            let (candle, blas) = if index % 2 == 0 {
                let candle = time(false);
                (candle, time(true))
            } else {
                let blas = time(true);
                (time(false), blas)
            };
            pairs.push(serde_json::json!({"candle_seconds":candle,"openblas_seconds":blas,"candle_over_openblas":candle/blas}));
        }
        let mut metadata = BTreeMap::new();
        runtime.record(&mut metadata);
        println!(
            "CPU_BLAS_TIMING={}",
            serde_json::json!({"shape":{"input":[1,128,1024],"weight":[1536,1024]},"max_raw_delta":delta,"metadata":metadata,"pairs":pairs,"qualified":false,"limits":["Kernel microbenchmark only, not released-model throughput or calibration acceptance.","Library hash/kernel/thread profile is mandatory for any future labeled gate."]})
        );
    }
}
