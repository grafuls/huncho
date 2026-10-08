//! CPU query blocking keeps every causal key/value and bounds score workspace.
use candle::{Device, Result, Tensor};

// Flatten only the query-head/query-row axes within each K/V head. Each row
// still reduces over all keys in the original order; K/V storage is untouched.
pub(super) fn grouped_queries(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    rows: usize,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (batch, heads, queries, width) = q.dims4()?;
    let (kb, kv_heads, keys, kw) = k.dims4()?;
    let (vb, vh, vk, vw) = v.dims4()?;
    if !q.device().is_cpu()
        || batch == 0
        || kv_heads == 0
        || heads == 0
        || queries == 0
        || width == 0
        || heads % kv_heads != 0
        || kb != batch
        || vb != batch
        || vh != kv_heads
        || keys < queries
        || vk != keys
        || kw != width
        || vw != width
        || rows > 4096
    {
        candle::bail!(
            "grouped GQA requires compatible nonempty CPU query/K/V heads and 0..4096 block rows"
        )
    }
    let groups = heads / kv_heads;
    let block_rows = if rows == 0 { queries } else { rows };
    // Projection/transposition can leave token rows strided across K/V heads.
    // Candle's CPU matmul requires these dense, but never expanded to `heads`.
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let kt = k.transpose(2, 3)?;
    let mut outputs = Vec::new();
    for start in (0..queries).step_by(block_rows) {
        let count = block_rows.min(queries - start);
        let query = q.narrow(2, start, count)?.contiguous()?.reshape((
            batch,
            kv_heads,
            groups * count,
            width,
        ))?;
        let scores = query
            .matmul(&kt)?
            .affine(1.0 / (width as f64).sqrt(), 0.0)?;
        let block_mask = if rows == 0 {
            mask.ok_or_else(|| candle::Error::Msg("missing full grouped GQA mask".into()))?
                .clone()
        } else {
            query_mask(start, count, keys, keys - queries)?
        };
        let scores = scores
            .reshape((batch, kv_heads, groups, count, keys))?
            .broadcast_add(&block_mask.to_dtype(scores.dtype())?)?;
        let probabilities = candle_nn::ops::softmax(&scores, 4)?.reshape((
            batch,
            kv_heads,
            groups * count,
            keys,
        ))?;
        outputs.push(
            probabilities
                .matmul(&v)?
                .reshape((batch, heads, count, width))?,
        );
    }
    Tensor::cat(&outputs, 2)
}

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

pub(super) fn query_mask(start: usize, rows: usize, keys: usize, offset: usize) -> Result<Tensor> {
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
    fn grouped_queries_preserve_head_mapping_and_full_key_reductions() {
        for dtype in [DType::F32, DType::F16] {
            for groups in [1, 2, 4] {
                for (queries, offset) in [(1, 0), (11, 0), (11, 7)] {
                    let keys = queries + offset;
                    let tensor = |heads: usize, tokens: usize, phase: f32| {
                        Tensor::from_vec(
                            (0..2 * heads * tokens * 8)
                                .map(|i| ((i as f32 + phase) * 0.37).sin() * 0.4)
                                .collect::<Vec<_>>(),
                            (2, tokens, heads, 8),
                            &Device::Cpu,
                        )
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap()
                    };
                    let q = tensor(2 * groups, queries, 1.0);
                    let k = tensor(2, keys, 3.0);
                    let v = tensor(2, keys, 5.0);
                    let mask = super::super::causal_mask_at(queries, offset)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap();
                    let expanded_k = super::super::repeat_interleave_head(&k, groups, 1)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let expanded_v = super::super::repeat_interleave_head(&v, groups, 1)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    // The reference matmul needs dense query rows too. Inputs
                    // to the grouped helper deliberately retain native strides.
                    let scores = q
                        .contiguous()
                        .unwrap()
                        .matmul(&expanded_k.transpose(2, 3).unwrap())
                        .unwrap()
                        .affine(1.0 / 8f64.sqrt(), 0.0)
                        .unwrap()
                        .broadcast_add(&mask)
                        .unwrap();
                    let expected = candle_nn::ops::softmax(&scores, 3)
                        .unwrap()
                        .matmul(&expanded_v)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    for rows in [0, 1, 3, 8, 32] {
                        let actual = grouped_queries(&q, &k, &v, rows, Some(&mask))
                            .unwrap()
                            .to_dtype(DType::F32)
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap();
                        assert!(actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| (a - b).abs()
                                <= if dtype == DType::F32 { 1e-6 } else { 5e-4 }));
                    }
                    assert_eq!(k.elem_count() * groups, expanded_k.elem_count());
                    assert_eq!(v.elem_count() * groups, expanded_v.elem_count());
                }
            }
        }
    }
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
