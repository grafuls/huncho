//! CPU query blocking keeps every causal key/value and bounds score workspace.
use candle::{Device, Result, Tensor};

pub(super) fn query_blocks(q: &Tensor, k: &Tensor, v: &Tensor, rows: usize) -> Result<Tensor> {
    let (_, _, queries, width) = q.dims4()?;
    let keys = k.dim(2)?;
    if rows == 0 || rows > 4096 || queries == 0 || keys < queries || !q.device().is_cpu() {
        candle::bail!("query blocks require nonempty CPU attention and 1..4096 rows")
    }
    let offset = keys - queries;
    let scale = 1.0 / (width as f64).sqrt();
    let kt = k.transpose(2, 3)?;
    let mut output = Vec::new();
    for start in (0..queries).step_by(rows) {
        let count = rows.min(queries - start);
        let query = q.narrow(2, start, count)?.contiguous()?;
        let scores = query.matmul(&kt)?.affine(scale, 0.0)?;
        let mask = query_mask(start, count, keys, offset)?.to_dtype(scores.dtype())?;
        let scores = scores.broadcast_add(&mask)?;
        let probabilities = candle_nn::ops::softmax(&scores, 3)?;
        output.push(probabilities.matmul(v)?);
    }
    Tensor::cat(&output, 2)
}

fn query_mask(start: usize, rows: usize, keys: usize, offset: usize) -> Result<Tensor> {
    let size = rows
        .checked_mul(keys)
        .ok_or_else(|| candle::Error::Msg("attention block mask overflow".into()))?;
    let mut mask = vec![0.0f32; size];
    for row in 0..rows {
        let first_future = (offset + start + row + 1).min(keys);
        mask[row * keys + first_future..(row + 1) * keys].fill(f32::NEG_INFINITY);
    }
    Tensor::from_vec(mask, (rows, keys), &Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;
    #[test]
    fn blocks_preserve_full_causal_key_reductions_and_absolute_prefix_masks() {
        for dtype in [DType::F32, DType::F16] {
            for (queries, offset) in [(1, 0), (11, 0), (11, 7)] {
                let keys = queries + offset;
                let tensor = |n: usize, phase: f32| {
                    Tensor::from_vec(
                        (0..n)
                            .map(|i| ((i as f32 + phase) * 0.23).sin() * 0.4)
                            .collect::<Vec<_>>(),
                        (2, 4, n / 64, 8),
                        &Device::Cpu,
                    )
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap()
                };
                let q = tensor(2 * 4 * queries * 8, 2.0);
                let k = tensor(2 * 4 * keys * 8, 3.0);
                let v = tensor(2 * 4 * keys * 8, 5.0);
                let scores = q
                    .matmul(&k.transpose(2, 3).unwrap())
                    .unwrap()
                    .affine(1.0 / 8f64.sqrt(), 0.0)
                    .unwrap();
                let mask = super::super::causal_mask_at(queries, offset)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let expected = candle_nn::ops::softmax(&scores.broadcast_add(&mask).unwrap(), 3)
                    .unwrap()
                    .matmul(&v)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for rows in [1, 3, 8, 32] {
                    let actual = query_blocks(&q, &k, &v, rows)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    assert!(actual.iter().zip(&expected).all(
                        |(a, b)| (a - b).abs() <= if dtype == DType::F32 { 1e-6 } else { 5e-4 }
                    ));
                    for start in (0..queries).step_by(rows) {
                        let count = rows.min(queries - start);
                        let block = query_mask(start, count, keys, offset)
                            .unwrap()
                            .to_vec2::<f32>()
                            .unwrap();
                        let full = super::super::causal_mask_at(queries, offset)
                            .unwrap()
                            .to_vec2::<f32>()
                            .unwrap();
                        assert_eq!(&block, &full[start..start + count]);
                        assert_eq!(block.len() * keys, count * keys);
                    }
                }
            }
        }
    }
}
