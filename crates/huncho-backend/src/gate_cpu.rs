//! CPU SiLU/multiply fusion. Retain the original typed scalar operations,
//! including FP16 intermediate rounding; allocate no separate SiLU tensor.
use candle::{
    op::{BinaryOpT, Mul, Silu, UnaryOpT},
    CpuStorage, Device, Result, Storage, Tensor,
};

pub(crate) fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.dims() != up.dims()
        || gate.dtype() != up.dtype()
        || !gate.device().is_cpu()
        || !up.device().is_cpu()
    {
        candle::bail!("fused CPU gate requires matching shapes/dtypes on CPU");
    }
    let (gs, gl) = gate.storage_and_layout();
    let (us, ul) = up.storage_and_layout();
    match (&*gs, &*us) {
        (Storage::Cpu(CpuStorage::F32(g)), Storage::Cpu(CpuStorage::F32(u))) => {
            let values = match (gl.contiguous_offsets(), ul.contiguous_offsets()) {
                (Some((gs, ge)), Some((us, ue))) => g[gs..ge]
                    .iter()
                    .zip(&u[us..ue])
                    .map(|(&g, &u)| Mul::f32(Silu::f32(g), u))
                    .collect(),
                _ => {
                    candle::cpu_backend::binary_map(gl, ul, g, u, |g, u| Mul::f32(Silu::f32(g), u))
                }
            };
            Tensor::from_vec(values, gate.shape(), &Device::Cpu)
        }
        (Storage::Cpu(CpuStorage::F16(g)), Storage::Cpu(CpuStorage::F16(u))) => {
            let values = match (gl.contiguous_offsets(), ul.contiguous_offsets()) {
                (Some((gs, ge)), Some((us, ue))) => g[gs..ge]
                    .iter()
                    .zip(&u[us..ue])
                    .map(|(&g, &u)| Mul::f16(Silu::f16(g), u))
                    .collect(),
                _ => {
                    candle::cpu_backend::binary_map(gl, ul, g, u, |g, u| Mul::f16(Silu::f16(g), u))
                }
            };
            Tensor::from_vec(values, gate.shape(), &Device::Cpu)
        }
        _ => candle::bail!("fused CPU gate supports FP32/FP16 storage only"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;
    fn bits(t: Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }
    #[test]
    fn fusion_preserves_typed_rounding_and_strided_views() {
        for dtype in [DType::F32, DType::F16] {
            for width in [1, 7, 128] {
                let make = |scale: f32| {
                    Tensor::from_vec(
                        (0..2 * 5 * width)
                            .map(|i| ((i % 37) as f32 - 18.0) * scale)
                            .collect::<Vec<_>>(),
                        (2, 5, width),
                        &Device::Cpu,
                    )
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap()
                    .narrow(1, 1, 3)
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
                };
                let g = make(0.8);
                let u = make(-0.3);
                assert_eq!(
                    bits(silu_mul(&g, &u).unwrap()),
                    bits(g.silu().unwrap().broadcast_mul(&u).unwrap())
                );
                let g = g.contiguous().unwrap();
                let u = u.contiguous().unwrap();
                assert_eq!(
                    bits(silu_mul(&g, &u).unwrap()),
                    bits(g.silu().unwrap().broadcast_mul(&u).unwrap())
                );
                assert_eq!(
                    bits(silu_mul(&g, &g).unwrap()),
                    bits(g.silu().unwrap().broadcast_mul(&g).unwrap())
                );
            }
        }
    }
    #[test]
    fn unsupported_shapes_and_dtypes_fail_without_broadcasting() {
        let g = Tensor::zeros((2, 3), candle::DType::F32, &Device::Cpu).unwrap();
        let u = Tensor::zeros((1, 3), candle::DType::F32, &Device::Cpu).unwrap();
        assert!(silu_mul(&g, &u).is_err());
        assert!(silu_mul(&g, &g.to_dtype(candle::DType::F16).unwrap()).is_err());
        let b = g.to_dtype(candle::DType::BF16).unwrap();
        assert!(silu_mul(&b, &b).is_err());
    }

    #[test]
    #[ignore = "CPU gate microbenchmark; excludes projections and released models"]
    fn gate_cpu_timing() {
        for dtype in [DType::F32, DType::F16] {
            let g = Tensor::from_vec(
                (0..256 * 4096)
                    .map(|i| ((i % 37) as f32 - 18.) * 0.3)
                    .collect::<Vec<_>>(),
                (256, 4096),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let u = g.affine(-0.7, 0.3).unwrap();
            let run = |fused| {
                let start = std::time::Instant::now();
                let out = if fused {
                    silu_mul(&g, &u)
                } else {
                    g.silu().and_then(|g| g.broadcast_mul(&u))
                }
                .unwrap();
                let ms = start.elapsed().as_secs_f64() * 1000.;
                (ms, bits(out))
            };
            assert_eq!(run(false).1, run(true).1);
            let mut separate_ms = Vec::new();
            let mut fused_ms = Vec::new();
            for i in 0..12 {
                if i % 2 == 0 {
                    separate_ms.push(run(false).0);
                    fused_ms.push(run(true).0);
                } else {
                    fused_ms.push(run(true).0);
                    separate_ms.push(run(false).0);
                }
            }
            println!(
                "{}",
                serde_json::json!({"scope": "CPU SiLU/multiply only; excludes projections/tokenization/heads", "shape": [256, 4096], "dtype": format!("{dtype:?}"), "separate_ms": separate_ms, "fused_ms": fused_ms, "output_float_bits_equal": true, "qualified_release": false})
            );
        }
    }
}
