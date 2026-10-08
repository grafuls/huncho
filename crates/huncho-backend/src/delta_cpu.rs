//! CPU Gated DeltaNet recurrence with reusable scalar buffers.
//!
//! Keep FP32 multiplication/addition and ascending key reduction order. There
//! is no fast-math, FMA substitution, parallel prefix transform or reduced
//! precision state. This remains a separately qualified execution profile.
use candle::{CpuStorage, DType, Device, Layout, Result, Storage, Tensor};

struct View<'a> {
    data: &'a [f32],
    layout: &'a Layout,
}
impl<'a> View<'a> {
    fn new(storage: &'a Storage, layout: &'a Layout) -> Result<Self> {
        match storage {
            Storage::Cpu(CpuStorage::F32(data)) => Ok(Self { data, layout }),
            _ => candle::bail!("buffered delta rule requires CPU FP32 storage"),
        }
    }
    fn offset(&self, indexes: &[usize]) -> usize {
        self.layout.start_offset()
            + indexes
                .iter()
                .zip(self.layout.stride())
                .map(|(i, s)| i * s)
                .sum::<usize>()
    }
}

pub(crate) fn recurrent(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (batch, heads, seq, key_width) = query.dims4()?;
    let (vb, vh, vs, value_width) = value.dims4()?;
    if batch == 0
        || heads == 0
        || seq == 0
        || key_width == 0
        || value_width == 0
        || key.dims() != query.dims()
        || (vb, vh, vs) != (batch, heads, seq)
        || g.dims() != [batch, heads, seq]
        || beta.dims() != g.dims()
        || [query, key, value, g, beta]
            .iter()
            .any(|t| !t.device().is_cpu())
    {
        candle::bail!("invalid CPU delta-rule input dimensions or device");
    }
    if let Some(state) = initial_state {
        if state.dims() != [batch, heads, key_width, value_width] || !state.device().is_cpu() {
            candle::bail!("invalid CPU delta-rule initial state");
        }
    }
    // Cast each complete input once, rather than allocating casts per token.
    // Decay uses Candle's selected FP32 exp implementation.
    let query = query.to_dtype(DType::F32)?;
    let key = key.to_dtype(DType::F32)?;
    let value = value.to_dtype(DType::F32)?;
    let decay = g.to_dtype(DType::F32)?.exp()?;
    let beta = beta.to_dtype(DType::F32)?;
    let (qs, ql) = query.storage_and_layout();
    let (ks, kl) = key.storage_and_layout();
    let (vs, vl) = value.storage_and_layout();
    let (gs, gl) = decay.storage_and_layout();
    let (bs, bl) = beta.storage_and_layout();
    let q = View::new(&qs, ql)?;
    let k = View::new(&ks, kl)?;
    let v = View::new(&vs, vl)?;
    let g = View::new(&gs, gl)?;
    let beta = View::new(&bs, bl)?;
    let state_shape = (batch, heads, key_width, value_width);
    let elements = |dims: &[usize]| {
        dims.iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| candle::Error::Msg("CPU delta-rule allocation size overflow".into()))
    };
    let matrix_size = elements(&[key_width, value_width])?;
    let state_size = elements(&[batch, heads, key_width, value_width])?;
    let out_size = elements(&[batch, heads, seq, value_width])?;
    let mut state = if let Some(initial) = initial_state {
        initial
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
    } else {
        vec![0_f32; state_size]
    };
    let mut out = vec![0_f32; out_size];
    let mut delta = vec![0_f32; value_width];
    for b in 0..batch {
        for h in 0..heads {
            let state_start = (b * heads + h) * matrix_size;
            let state = &mut state[state_start..state_start + matrix_size];
            for t in 0..seq {
                let q_start = q.offset(&[b, h, t, 0]);
                let k_start = k.offset(&[b, h, t, 0]);
                let v_start = v.offset(&[b, h, t, 0]);
                let decay = g.data[g.offset(&[b, h, t])];
                let beta = beta.data[beta.offset(&[b, h, t])];
                delta.fill(0.);
                // Fuse decay and memory projection, retaining scalar reduction
                // order for every value column. No product is contracted.
                for i in 0..key_width {
                    let ki = k.data[k_start + i * k.layout.stride()[3]];
                    for (j, memory) in delta.iter_mut().enumerate() {
                        let cell = &mut state[i * value_width + j];
                        *cell *= decay;
                        *memory += *cell * ki;
                    }
                }
                for (j, memory) in delta.iter_mut().enumerate() {
                    *memory = (v.data[v_start + j * v.layout.stride()[3]] - *memory) * beta;
                }
                let out_start = ((b * heads + h) * seq + t) * value_width;
                let out = &mut out[out_start..out_start + value_width];
                // The state update and output projection share a matrix pass.
                for i in 0..key_width {
                    let ki = k.data[k_start + i * k.layout.stride()[3]];
                    let qi = q.data[q_start + i * q.layout.stride()[3]];
                    for (j, output) in out.iter_mut().enumerate() {
                        let cell = &mut state[i * value_width + j];
                        *cell += ki * delta[j];
                        *output += *cell * qi;
                    }
                }
            }
        }
    }
    Ok((
        Tensor::from_vec(out, (batch, heads, seq, value_width), &Device::Cpu)?,
        Tensor::from_vec(state, state_shape, &Device::Cpu)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_5::recurrent_gated_delta;

    fn inputs(
        b: usize,
        heads: usize,
        seq: usize,
        k: usize,
        v: usize,
        dtype: DType,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let make4 = |width: usize, scale: f32| {
            let data = (0..b * (seq + 2) * heads * width)
                .map(|i| ((i % 37) as f32 - 18.) * scale)
                .collect::<Vec<_>>();
            Tensor::from_vec(data, (b, seq + 2, heads, width), &Device::Cpu)
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .narrow(2, 1, seq)
                .unwrap()
        };
        let make3 = |scale: f32, shift: f32| {
            let data = (0..b * (seq + 2) * heads)
                .map(|i| (i % 11) as f32 * scale + shift)
                .collect::<Vec<_>>();
            Tensor::from_vec(data, (b, seq + 2, heads), &Device::Cpu)
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .narrow(2, 1, seq)
                .unwrap()
        };
        let state = Tensor::from_vec(
            (0..b * heads * k * v)
                .map(|i| (i % 13) as f32 * 0.001)
                .collect::<Vec<_>>(),
            (b, heads, k, v),
            &Device::Cpu,
        )
        .unwrap();
        (
            make4(k, 0.003),
            make4(k, 0.004),
            make4(v, 0.01).to_dtype(dtype).unwrap(),
            make3(-0.02, -0.1),
            make3(0.01, 0.3).to_dtype(dtype).unwrap(),
            state,
        )
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    #[test]
    fn buffered_recurrence_matches_tensor_order_and_preserves_initial_storage() {
        for dtype in [DType::F32, DType::F16] {
            for (b, heads, seq, k, v) in [(1, 1, 1, 4, 3), (2, 3, 17, 7, 5), (1, 2, 129, 16, 8)] {
                let (q, k, v, g, beta, state) = inputs(b, heads, seq, k, v, dtype);
                let before = bits(&state);
                for initial in [None, Some(&state)] {
                    let expected = recurrent_gated_delta(&q, &k, &v, &g, &beta, initial).unwrap();
                    let actual = recurrent(&q, &k, &v, &g, &beta, initial).unwrap();
                    assert_eq!(bits(&actual.0), bits(&expected.0));
                    assert_eq!(bits(&actual.1), bits(&expected.1));
                }
                assert_eq!(bits(&state), before);
                if seq > 1 {
                    let first = recurrent(
                        &q.narrow(2, 0, 1).unwrap(),
                        &k.narrow(2, 0, 1).unwrap(),
                        &v.narrow(2, 0, 1).unwrap(),
                        &g.narrow(2, 0, 1).unwrap(),
                        &beta.narrow(2, 0, 1).unwrap(),
                        Some(&state),
                    )
                    .unwrap();
                    let tail = recurrent(
                        &q.narrow(2, 1, seq - 1).unwrap(),
                        &k.narrow(2, 1, seq - 1).unwrap(),
                        &v.narrow(2, 1, seq - 1).unwrap(),
                        &g.narrow(2, 1, seq - 1).unwrap(),
                        &beta.narrow(2, 1, seq - 1).unwrap(),
                        Some(&first.1),
                    )
                    .unwrap();
                    let whole = recurrent(&q, &k, &v, &g, &beta, Some(&state)).unwrap();
                    assert_eq!(bits(&tail.1), bits(&whole.1));
                    assert_eq!(
                        bits(&Tensor::cat(&[first.0, tail.0], 2).unwrap()),
                        bits(&whole.0)
                    );
                }
            }
        }
        let (q, k, v, g, beta, state) = inputs(1, 1, 3, 4, 3, DType::F32);
        assert!(recurrent(&q, &k, &v, &g.narrow(2, 0, 2).unwrap(), &beta, None).is_err());
        assert!(recurrent(&q, &k, &v, &g, &beta, Some(&state.narrow(2, 0, 2).unwrap())).is_err());
    }

    #[test]
    #[ignore = "CPU recurrence microbenchmark; no released-model speed claim"]
    fn recurrence_cpu_timing() {
        let (q, k, v, g, beta, state) = inputs(1, 8, 256, 32, 32, DType::F32);
        let run = |buffered| {
            let start = std::time::Instant::now();
            let result = if buffered {
                recurrent(&q, &k, &v, &g, &beta, Some(&state))
            } else {
                recurrent_gated_delta(&q, &k, &v, &g, &beta, Some(&state))
            }
            .unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            (elapsed, bits(&result.0), bits(&result.1))
        };
        let expected = run(false);
        let actual = run(true);
        assert_eq!(expected.1, actual.1);
        assert_eq!(expected.2, actual.2);
        let mut tensor_ms = Vec::new();
        let mut buffered_ms = Vec::new();
        for i in 0..6 {
            if i % 2 == 0 {
                tensor_ms.push(run(false).0);
                buffered_ms.push(run(true).0);
            } else {
                buffered_ms.push(run(true).0);
                tensor_ms.push(run(false).0);
            }
        }
        println!(
            "{}",
            serde_json::json!({"scope": "CPU recurrence only; FP32; nonzero state",
            "shape": [1,8,256,32,32], "tensor_ms": tensor_ms, "buffered_ms": buffered_ms,
            "output_and_state_float_bits_equal": true, "qualified_release": false})
        );
    }
}
