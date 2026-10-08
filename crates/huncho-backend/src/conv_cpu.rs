//! Buffered causal depthwise convolution preserving the tensor loop's FP32
//! kernel-tap accumulation order, without per-tap contribution/padding tensors.
use candle::{CpuStorage, DType, Device, Result, Storage, Tensor};

pub(crate) fn causal(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (batch, channels, sequence) = input.dims3()?;
    let (wc, one, kernel) = weights.dims3()?;
    if batch == 0
        || channels == 0
        || sequence == 0
        || wc != channels
        || one != 1
        || kernel == 0
        || !input.device().is_cpu()
        || !weights.device().is_cpu()
        || weights.dtype() != DType::F32
    {
        candle::bail!("buffered causal convolution requires positive CPU dimensions and FP32 depthwise weights")
    }
    let count = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(sequence))
        .ok_or_else(|| candle::Error::Msg("causal convolution output size overflow".into()))?;
    let input = input.to_dtype(DType::F32)?;
    let (storage, layout) = input.storage_and_layout();
    let (ws, wl) = weights.storage_and_layout();
    let Storage::Cpu(CpuStorage::F32(values)) = &*storage else {
        candle::bail!("invalid CPU convolution input storage")
    };
    let Storage::Cpu(CpuStorage::F32(weights)) = &*ws else {
        candle::bail!("invalid CPU convolution weight storage")
    };
    let mut output = vec![0.0f32; count];
    for b in 0..batch {
        for c in 0..channels {
            let base = layout.start_offset() + b * layout.stride()[0] + c * layout.stride()[1];
            let row = &mut output[(b * channels + c) * sequence..(b * channels + c + 1) * sequence];
            for tap in 0..kernel {
                let shift = kernel - 1 - tap;
                if shift >= sequence {
                    continue;
                }
                let weight = weights[wl.start_offset() + c * wl.stride()[0] + tap * wl.stride()[2]];
                for (t, value) in row.iter_mut().enumerate().skip(shift) {
                    *value += weight * values[base + (t - shift) * layout.stride()[2]];
                }
            }
        }
    }
    Tensor::from_vec(output, (batch, channels, sequence), &Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reference(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (b, c, seq) = input.dims3()?;
        let kernel = weights.dim(2)?;
        let input = input.to_dtype(DType::F32)?;
        let mut output = Tensor::zeros((b, c, seq), DType::F32, &Device::Cpu)?;
        for tap in 0..kernel {
            let shift = kernel - 1 - tap;
            if shift >= seq {
                continue;
            }
            let weight = weights.narrow(2, tap, 1)?.squeeze(1)?.unsqueeze(0)?;
            let contribution = weight.broadcast_mul(&input)?;
            if shift == 0 {
                output = output.broadcast_add(&contribution)?;
            } else {
                let pad = Tensor::zeros((b, c, shift), DType::F32, &Device::Cpu)?;
                output = output.broadcast_add(&Tensor::cat(
                    &[pad, contribution.narrow(2, 0, seq - shift)?],
                    2,
                )?)?;
            }
        }
        Ok(output)
    }
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
    fn convolution_matches_original_tap_order_across_dtypes_short_sequences_and_strides() {
        for dtype in [DType::F32, DType::F16] {
            for sequence in [1, 2, 3, 4, 9] {
                let input = Tensor::from_vec(
                    (0..2 * (sequence + 2) * 3)
                        .map(|i| (i as f32 * 0.127).sin())
                        .collect(),
                    (2, sequence + 2, 3),
                    &Device::Cpu,
                )
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .narrow(1, 1, sequence)
                .unwrap()
                .transpose(1, 2)
                .unwrap();
                let weights = Tensor::from_vec(
                    (0..3 * 5).map(|i| (i as f32 * 0.361).cos()).collect(),
                    (3, 1, 5),
                    &Device::Cpu,
                )
                .unwrap()
                .narrow(2, 1, 4)
                .unwrap();
                let expected = reference(&input, &weights).unwrap();
                let actual = causal(&input, &weights).unwrap();
                assert_eq!(bits(actual.clone()), bits(expected.clone()));
                assert_eq!(
                    bits(actual.to_dtype(dtype).unwrap()),
                    bits(expected.to_dtype(dtype).unwrap())
                );
            }
        }
        let input = Tensor::zeros((1, 2, 3), DType::F32, &Device::Cpu).unwrap();
        for shape in [(1, 1, 4), (2, 2, 4), (2, 1, 0)] {
            assert!(causal(
                &input,
                &Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
            )
            .is_err());
        }
        assert!(causal(
            &Tensor::zeros((1, 2, 0), DType::F32, &Device::Cpu).unwrap(),
            &Tensor::zeros((2, 1, 4), DType::F32, &Device::Cpu).unwrap()
        )
        .is_err());
    }

    #[test]
    #[ignore = "CPU convolution microbenchmark; no released-model speed claim"]
    fn convolution_cpu_timing() {
        use std::hint::black_box;
        use std::time::Instant;
        let (channels, seq) = (128, 512);
        let input = Tensor::from_vec(
            (0..channels * seq)
                .map(|i| (i as f32 * 0.127).sin())
                .collect(),
            (1, seq, channels),
            &Device::Cpu,
        )
        .unwrap()
        .transpose(1, 2)
        .unwrap();
        let weights = Tensor::from_vec(
            (0..channels * 4)
                .map(|i| (i as f32 * 0.361).cos())
                .collect(),
            (channels, 1, 4),
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            bits(causal(&input, &weights).unwrap()),
            bits(reference(&input, &weights).unwrap())
        );
        let (mut original_ms, mut buffered_ms) = (Vec::new(), Vec::new());
        for run in 0..6 {
            for buffered in if run % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                for _ in 0..20 {
                    black_box(
                        if buffered {
                            causal(&input, &weights)
                        } else {
                            reference(&input, &weights)
                        }
                        .unwrap(),
                    );
                }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / 20.0;
                if buffered {
                    buffered_ms.push(ms);
                } else {
                    original_ms.push(ms);
                }
            }
        }
        let original = original_ms.iter().sum::<f64>() / original_ms.len() as f64;
        let buffered = buffered_ms.iter().sum::<f64>() / buffered_ms.len() as f64;
        println!(
            "{}",
            serde_json::json!({"shape":[1,channels,seq], "kernel":4, "input_strides":input.stride(), "calls_per_sample":20,
            "original_ms":original_ms,"buffered_ms":buffered_ms,"original_mean_ms":original,"buffered_mean_ms":buffered,"speedup":original/buffered,"float_bits_equal":true,"qualified_release":false})
        );
    }
}
