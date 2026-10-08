//! Immutable CPU KV pages, with an optional direct FP32 query-block path.
use candle::{DType, Result, Tensor};
use std::sync::Arc;

#[cfg(test)]
thread_local! {
    static PATH_COUNTS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}
#[cfg(test)]
pub(super) fn reset_path_counts() {
    PATH_COUNTS.with(|counts| counts.set((0, 0)));
}
#[cfg(test)]
pub(super) fn path_counts() -> (usize, usize) {
    PATH_COUNTS.with(|counts| counts.get())
}

struct Page {
    key: Tensor,
    value: Tensor,
    rows: usize,
}

#[derive(Clone)]
pub(super) struct PagedKv {
    page_tokens: usize,
    pages: Vec<Arc<Page>>,
    tokens: usize,
}

impl PagedKv {
    pub(super) fn new(page_tokens: usize) -> Self {
        debug_assert!((16..=256).contains(&page_tokens) && page_tokens.is_power_of_two());
        Self {
            page_tokens,
            pages: Vec::new(),
            tokens: 0,
        }
    }

    fn validate(&self, key: &Tensor, value: &Tensor) -> Result<usize> {
        let (b, h, rows, w) = key.dims4()?;
        if !key.device().is_cpu()
            || !value.device().is_cpu()
            || b != 1
            || h == 0
            || rows == 0
            || w == 0
            || key.dims() != value.dims()
            || key.dtype() != value.dtype()
        {
            candle::bail!("KV pages require matching nonempty single-row CPU keys/values")
        }
        if let Some(page) = self.pages.first() {
            let dims = page.key.dims4()?;
            if (dims.0, dims.1, dims.3) != (b, h, w) || page.key.dtype() != key.dtype() {
                candle::bail!("KV page continuation changed shape or dtype")
            }
        }
        Ok(rows)
    }

    pub(super) fn materialize_with(
        &self,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        #[cfg(test)]
        PATH_COUNTS.with(|counts| {
            let (flat, direct) = counts.get();
            counts.set((flat + 1, direct));
        });
        self.validate(key, value)?;
        if self.pages.is_empty() {
            return Ok((key.clone(), value.clone()));
        }
        let mut keys: Vec<_> = self.pages.iter().map(|page| &page.key).collect();
        let mut values: Vec<_> = self.pages.iter().map(|page| &page.value).collect();
        keys.push(key);
        values.push(value);
        Ok((Tensor::cat(&keys, 2)?, Tensor::cat(&values, 2)?))
    }

    /// A private contiguous snapshot for a native branch batch. Page payloads
    /// remain immutable; no batched suffix state is stored in these pages.
    pub(super) fn materialize(&self) -> Result<(Tensor, Tensor)> {
        #[cfg(test)]
        PATH_COUNTS.with(|counts| {
            let (flat, direct) = counts.get();
            counts.set((flat + 1, direct));
        });
        if self.pages.is_empty() {
            candle::bail!("cannot materialize an empty KV prefix")
        }
        let keys: Vec<_> = self.pages.iter().map(|page| &page.key).collect();
        let values: Vec<_> = self.pages.iter().map(|page| &page.value).collect();
        Ok((Tensor::cat(&keys, 2)?, Tensor::cat(&values, 2)?))
    }

    /// Read immutable pages directly, without concatenating or expanding K/V.
    /// Softmax still includes every causal key in its original order. QK/PV
    /// matmul shapes and PV summation change, so this is a separate arithmetic
    /// profile, not a bitwise storage optimization. The caller bounds queries.
    pub(super) fn attention(&self, q: &Tensor, query_rows: usize) -> Result<Tensor> {
        #[cfg(test)]
        PATH_COUNTS.with(|counts| {
            let (flat, direct) = counts.get();
            counts.set((flat, direct + 1));
        });
        let (batch, heads, queries, width) = q.dims4()?;
        let first = self.pages.first().ok_or_else(|| {
            candle::Error::Msg("direct page attention requires nonempty KV pages".into())
        })?;
        let (kb, kv_heads, _, kw) = first.key.dims4()?;
        if !q.device().is_cpu()
            || q.dtype() != DType::F32
            || first.key.dtype() != DType::F32
            || batch != 1
            || kb != batch
            || heads == 0
            || kv_heads == 0
            || heads % kv_heads != 0
            || queries == 0
            || queries > self.tokens
            || width == 0
            || width != kw
            || !(1..=4096).contains(&query_rows)
        {
            candle::bail!("direct page attention requires compatible single-row CPU FP32 queries and 1..4096 block rows")
        }
        let groups = heads / kv_heads;
        let offset = self.tokens - queries;
        let scale = 1.0 / (width as f64).sqrt();
        let mut outputs = Vec::new();
        for start in (0..queries).step_by(query_rows) {
            let count = query_rows.min(queries - start);
            let query = q.narrow(2, start, count)?.contiguous()?.reshape((
                batch,
                kv_heads,
                groups * count,
                width,
            ))?;
            let scores = self
                .pages
                .iter()
                .map(|page| query.matmul(&page.key.transpose(2, 3)?))
                .collect::<Result<Vec<_>>>()?;
            let scores = Tensor::cat(&scores, 3)?.affine(scale, 0.0)?.reshape((
                batch,
                kv_heads,
                groups,
                count,
                self.tokens,
            ))?;
            let mask = super::attention::query_mask(start, count, self.tokens, offset)?;
            let probabilities = candle_nn::ops::softmax(&scores.broadcast_add(&mask)?, 4)?
                .reshape((batch, kv_heads, groups * count, self.tokens))?;
            let mut key_start = 0;
            let mut combined: Option<Tensor> = None;
            for page in &self.pages {
                let contribution = probabilities
                    .narrow(3, key_start, page.rows)?
                    .contiguous()?
                    .matmul(&page.value)?;
                combined = Some(match combined {
                    Some(previous) => previous.add(&contribution)?,
                    None => contribution,
                });
                key_start += page.rows;
            }
            outputs.push(combined.unwrap().reshape((batch, heads, count, width))?);
        }
        Tensor::cat(&outputs, 2)
    }

    /// Read one shared immutable prefix for private native suffix rows. Only
    /// queries/probabilities are regrouped across batch rows; prefix K/V pages
    /// are never concatenated, expanded or repeated. Each row still has one
    /// ordered softmax over the complete causal prefix and its own suffix.
    pub(super) fn attention_suffix_batch(
        &self,
        q: &Tensor,
        key: &Tensor,
        value: &Tensor,
        query_rows: usize,
    ) -> Result<Tensor> {
        let (batch, heads, queries, width) = q.dims4()?;
        let (kb, kv_heads, suffix, kw) = key.dims4()?;
        let first = self.pages.first().ok_or_else(|| {
            candle::Error::Msg("shared page attention requires a nonempty prefix".into())
        })?;
        let (pb, ph, _, pw) = first.key.dims4()?;
        if !q.device().is_cpu()
            || !key.device().is_cpu()
            || !value.device().is_cpu()
            || q.dtype() != DType::F32
            || key.dtype() != DType::F32
            || value.dtype() != DType::F32
            || first.key.dtype() != DType::F32
            || !(1..=63).contains(&batch)
            || kb != batch
            || pb != 1
            || kv_heads == 0
            || ph != kv_heads
            || heads == 0
            || heads % kv_heads != 0
            || queries == 0
            || suffix != queries
            || width == 0
            || kw != width
            || pw != width
            || key.dims() != value.dims()
            || !(1..=4096).contains(&query_rows)
        {
            candle::bail!("shared page attention requires compatible CPU FP32 prefix and 1..63 private suffix rows with 1..4096 query block rows")
        }
        let keys = self.tokens.checked_add(suffix).ok_or_else(|| {
            candle::Error::Msg("shared page attention token count overflow".into())
        })?;
        #[cfg(test)]
        PATH_COUNTS.with(|counts| {
            let (flat, direct) = counts.get();
            counts.set((flat, direct + 1));
        });
        let groups = heads / kv_heads;
        let key = key.contiguous()?;
        let value = value.contiguous()?;
        let kt = key.transpose(2, 3)?;
        let mut outputs = Vec::new();
        for start in (0..queries).step_by(query_rows) {
            let count = query_rows.min(queries - start);
            let query = q.narrow(2, start, count)?.contiguous()?.reshape((
                batch,
                kv_heads,
                groups * count,
                width,
            ))?;
            // Move batch rows into each KV head's query axis. Candle matmul
            // can then read the single prefix page without broadcasting K/V.
            let shared_query = query.transpose(0, 1)?.contiguous()?.reshape((
                1,
                kv_heads,
                batch * groups * count,
                width,
            ))?;
            let mut scores = self
                .pages
                .iter()
                .map(|page| {
                    shared_query
                        .matmul(&page.key.transpose(2, 3)?)?
                        .reshape((kv_heads, batch, groups * count, page.rows))?
                        .transpose(0, 1)
                })
                .collect::<Result<Vec<_>>>()?;
            scores.push(query.matmul(&kt)?);
            let scores = Tensor::cat(&scores, 3)?.affine(1.0 / (width as f64).sqrt(), 0.0)?;
            let mask = super::attention::query_mask(start, count, keys, self.tokens)?;
            let probabilities = candle_nn::ops::softmax(
                &scores
                    .reshape((batch, kv_heads, groups, count, keys))?
                    .broadcast_add(&mask)?,
                4,
            )?
            .reshape((batch, kv_heads, groups * count, keys))?;
            let mut key_start = 0;
            let mut combined: Option<Tensor> = None;
            for page in &self.pages {
                let contribution = probabilities
                    .narrow(3, key_start, page.rows)?
                    .transpose(0, 1)?
                    .contiguous()?
                    .reshape((1, kv_heads, batch * groups * count, page.rows))?
                    .matmul(&page.value)?
                    .reshape((kv_heads, batch, groups * count, width))?
                    .transpose(0, 1)?;
                combined = Some(match combined {
                    Some(previous) => previous.add(&contribution)?,
                    None => contribution,
                });
                key_start += page.rows;
            }
            let suffix_values = probabilities
                .narrow(3, key_start, suffix)?
                .contiguous()?
                .matmul(&value)?;
            outputs.push(
                combined
                    .unwrap()
                    .add(&suffix_values)?
                    .contiguous()?
                    .reshape((batch, heads, count, width))?,
            );
        }
        Tensor::cat(&outputs, 2)
    }

    fn owned_page(key: Tensor, value: Tensor, rows: usize) -> Result<Arc<Page>> {
        // Narrow views must not keep an entire projection/suffix allocation.
        Ok(Arc::new(Page {
            key: key.force_contiguous()?.detach(),
            value: value.force_contiguous()?.detach(),
            rows,
        }))
    }

    pub(super) fn append(&self, key: &Tensor, value: &Tensor) -> Result<Self> {
        let rows = self.validate(key, value)?;
        let tokens = self
            .tokens
            .checked_add(rows)
            .ok_or_else(|| candle::Error::Msg("KV page token count overflow".into()))?;
        let mut pages = self.pages.clone();
        let mut offset = 0;
        if let Some(tail) = pages
            .last()
            .filter(|page| page.rows < self.page_tokens)
            .cloned()
        {
            let fill = rows.min(self.page_tokens - tail.rows);
            let k = Tensor::cat(&[&tail.key, &key.narrow(2, 0, fill)?], 2)?;
            let v = Tensor::cat(&[&tail.value, &value.narrow(2, 0, fill)?], 2)?;
            pages.pop();
            pages.push(Self::owned_page(k, v, tail.rows + fill)?);
            offset = fill;
        }
        while offset < rows {
            let count = (rows - offset).min(self.page_tokens);
            pages.push(Self::owned_page(
                key.narrow(2, offset, count)?,
                value.narrow(2, offset, count)?,
                count,
            )?);
            offset += count;
        }
        Ok(Self {
            page_tokens: self.page_tokens,
            pages,
            tokens,
        })
    }

    pub(super) fn retention_bytes(&self) -> Option<usize> {
        let mut bytes = self
            .pages
            .capacity()
            .checked_mul(std::mem::size_of::<Arc<Page>>())?;
        for page in &self.pages {
            // Charge shared payloads conservatively per snapshot; include page
            // wrappers/layout metadata rather than claiming zero-cost sharing.
            bytes = bytes.checked_add(512)?;
            for tensor in [&page.key, &page.value] {
                bytes = bytes.checked_add(
                    tensor
                        .elem_count()
                        .checked_mul(tensor.dtype().size_in_bytes())?,
                )?;
            }
        }
        Some(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    fn tensors(start: usize, rows: usize, dtype: DType) -> (Tensor, Tensor) {
        let data: Vec<_> = (start..start + rows * 8).map(|n| n as f32 / 8.).collect();
        let k = Tensor::from_vec(data, (1, 2, rows, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let v = k.affine(2., -3.).unwrap();
        (k, v)
    }
    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|n| n.to_bits())
            .collect()
    }
    #[test]
    fn shared_prefix_suffix_rows_preserve_causality_head_mapping_and_row_isolation() {
        for batch in [1, 2, 5] {
            for groups in [1, 2, 4] {
                for page_tokens in [16, 32, 64, 128, 256] {
                    for (prefix, suffix) in [(3, 1), (17, 19), (263, 3)] {
                        let tensor = |b: usize, h: usize, rows: usize, phase: f32| {
                            Tensor::from_vec(
                                (0..b * h * rows * 8)
                                    .map(|i| ((i as f32 + phase) * 0.37).sin() * 0.7)
                                    .collect::<Vec<_>>(),
                                (b, rows, h, 8),
                                &Device::Cpu,
                            )
                            .unwrap()
                            .transpose(1, 2)
                            .unwrap()
                        };
                        let q = tensor(batch, 2 * groups, suffix, 1.);
                        let pk = tensor(1, 2, prefix, 3.);
                        let pv = tensor(1, 2, prefix, 5.);
                        let sk = tensor(batch, 2, suffix, 7.);
                        let sv = tensor(batch, 2, suffix, 9.);
                        let pages = PagedKv::new(page_tokens).append(&pk, &pv).unwrap();
                        let k = Tensor::cat(&[&Tensor::cat(&vec![&pk; batch], 0).unwrap(), &sk], 2)
                            .unwrap();
                        let v = Tensor::cat(&[&Tensor::cat(&vec![&pv; batch], 0).unwrap(), &sv], 2)
                            .unwrap();
                        let expected =
                            super::super::attention::grouped_queries(&q, &k, &v, 7, None)
                                .unwrap()
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap();
                        for rows in [1, 7, 64] {
                            let actual = pages
                                .attention_suffix_batch(&q, &sk, &sv, rows)
                                .unwrap()
                                .flatten_all()
                                .unwrap()
                                .to_vec1::<f32>()
                                .unwrap();
                            assert!(actual
                                .iter()
                                .zip(&expected)
                                .all(|(a, b)| (a - b).abs() <= 1e-6));
                        }
                        let actual = pages.attention_suffix_batch(&q, &sk, &sv, 7).unwrap();
                        // A changed final suffix value in row zero affects only
                        // that row's final query, never another row or prefix.
                        let mut changed = sv
                            .contiguous()
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap();
                        changed[(suffix - 1) * 8] += 100.;
                        let changed =
                            Tensor::from_vec(changed, (batch, 2, suffix, 8), &Device::Cpu).unwrap();
                        let changed = pages.attention_suffix_batch(&q, &sk, &changed, 7).unwrap();
                        if suffix > 1 {
                            assert_eq!(
                                bits(&actual.narrow(2, 0, suffix - 1).unwrap()),
                                bits(&changed.narrow(2, 0, suffix - 1).unwrap())
                            );
                        }
                        if batch > 1 {
                            assert_eq!(
                                bits(&actual.narrow(0, 1, batch - 1).unwrap()),
                                bits(&changed.narrow(0, 1, batch - 1).unwrap())
                            );
                        }
                        assert_ne!(bits(&actual), bits(&changed));
                        let stored = pages.materialize().unwrap();
                        assert_eq!(bits(&stored.0), bits(&pk));
                        assert_eq!(bits(&stored.1), bits(&pv));
                        assert!(pages.attention_suffix_batch(&q, &sk, &sv, 0).is_err());
                        assert!(pages.attention_suffix_batch(&q, &sk, &sv, 4097).is_err());
                        assert!(pages
                            .attention_suffix_batch(&q.to_dtype(DType::F16).unwrap(), &sk, &sv, 7)
                            .is_err());
                        assert!(pages
                            .attention_suffix_batch(
                                &q,
                                &sk,
                                &sv.narrow(2, 0, suffix - 1).unwrap(),
                                7
                            )
                            .is_err());
                    }
                }
            }
        }
    }
    #[test]
    fn direct_page_attention_preserves_causal_head_mapping_and_full_softmax() {
        for groups in [1, 2, 4] {
            for page_tokens in [16, 32, 64, 128, 256] {
                for (queries, offset) in [(1, 0), (19, 0), (19, 17), (3, 263)] {
                    let keys = queries + offset;
                    let tensor = |heads: usize, rows: usize, phase: f32| {
                        Tensor::from_vec(
                            (0..heads * rows * 8)
                                .map(|i| ((i as f32 + phase) * 0.37).sin() * 0.7)
                                .collect::<Vec<_>>(),
                            (1, rows, heads, 8),
                            &Device::Cpu,
                        )
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap()
                    };
                    let q = tensor(groups * 2, queries, 1.0);
                    let k = tensor(2, keys, 3.0);
                    let v = tensor(2, keys, 5.0);
                    let pages = PagedKv::new(page_tokens).append(&k, &v).unwrap();
                    let expanded_k = super::super::repeat_interleave_head(&k, groups, 1)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let expanded_v = super::super::repeat_interleave_head(&v, groups, 1)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let expected = super::super::attention::query_blocks(
                        &q.contiguous().unwrap(),
                        &expanded_k,
                        &expanded_v,
                        7,
                    )
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                    for rows in [1, 7, 64] {
                        let actual = pages
                            .attention(&q, rows)
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap();
                        assert!(actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| (a - b).abs() < 1e-6));
                    }
                    assert!(pages.attention(&q, 0).is_err());
                    assert!(pages.attention(&q, 4097).is_err());
                    assert!(pages
                        .attention(&q.to_dtype(DType::F16).unwrap(), 7)
                        .is_err());
                    // Values beyond each query's absolute position cannot enter
                    // its distribution. Poisoning just the final future token
                    // changes only the final query in a multi-query suffix.
                    if queries > 1 {
                        let vk = v.narrow(2, 0, keys - 1).unwrap();
                        let tail = v.narrow(2, keys - 1, 1).unwrap().affine(1., 100.).unwrap();
                        let changed = Tensor::cat(&[&vk, &tail], 2).unwrap();
                        let changed = PagedKv::new(page_tokens)
                            .append(&k, &changed)
                            .unwrap()
                            .attention(&q, 7)
                            .unwrap();
                        let original = pages.attention(&q, 7).unwrap();
                        assert_eq!(
                            bits(&changed.narrow(2, 0, queries - 1).unwrap()),
                            bits(&original.narrow(2, 0, queries - 1).unwrap()),
                        );
                    }
                }
            }
        }
        let empty = PagedKv::new(16);
        assert!(empty.attention(&tensors(0, 1, DType::F32).0, 7).is_err());
    }
    #[test]
    fn shared_page_groups_enforce_native_row_capacity_and_nonempty_prefixes() {
        let (key, value) = tensors(0, 17, DType::F32);
        let pages = PagedKv::new(16).append(&key, &value).unwrap();
        for batch in [63, 64] {
            let q = Tensor::zeros((batch, 4, 2, 4), DType::F32, &Device::Cpu).unwrap();
            let k = Tensor::zeros((batch, 2, 2, 4), DType::F32, &Device::Cpu).unwrap();
            assert_eq!(
                pages.attention_suffix_batch(&q, &k, &k, 1).is_ok(),
                batch == 63
            );
            assert!(PagedKv::new(16)
                .attention_suffix_batch(&q, &k, &k, 1)
                .is_err());
        }
    }
    #[test]
    fn page_order_and_copy_on_write_tails_preserve_exact_storage_values() {
        for dtype in [DType::F32, DType::F16] {
            for page_tokens in [16, 32, 64, 128, 256] {
                let (k, v) = tensors(0, page_tokens + 3, dtype);
                let parent = PagedKv::new(page_tokens).append(&k, &v).unwrap();
                let clone = parent.clone();
                assert!(Arc::ptr_eq(&parent.pages[0], &clone.pages[0]));
                assert!(Arc::ptr_eq(&parent.pages[1], &clone.pages[1]));
                let (suffix_k, suffix_v) = tensors(9000, page_tokens * 2 + 1, dtype);
                let branch = clone.append(&suffix_k, &suffix_v).unwrap();
                for page in &branch.pages {
                    for tensor in [&page.key, &page.value] {
                        let (storage, layout) = tensor.storage_and_layout();
                        let allocated = match &*storage {
                            candle::Storage::Cpu(candle::CpuStorage::F32(data)) => data.len(),
                            candle::Storage::Cpu(candle::CpuStorage::F16(data)) => data.len(),
                            _ => panic!("expected CPU FP32/FP16 page"),
                        };
                        assert_eq!(allocated, tensor.elem_count());
                        assert_eq!(layout.start_offset(), 0);
                        assert!(layout.contiguous_offsets().is_some());
                    }
                }
                assert!(Arc::ptr_eq(&parent.pages[0], &branch.pages[0]));
                assert!(!Arc::ptr_eq(&parent.pages[1], &branch.pages[1]));
                assert_eq!(parent.tokens, page_tokens + 3);
                let (last_k, last_v) = tensors(100, 1, dtype);
                let (actual_k, actual_v) = branch.materialize_with(&last_k, &last_v).unwrap();
                let expected_k = Tensor::cat(&[&k, &suffix_k, &last_k], 2).unwrap();
                let expected_v = Tensor::cat(&[&v, &suffix_v, &last_v], 2).unwrap();
                assert_eq!(bits(&actual_k), bits(&expected_k));
                assert_eq!(bits(&actual_v), bits(&expected_v));
                let original = parent.materialize_with(&last_k, &last_v).unwrap();
                assert_eq!(
                    bits(&original.0),
                    bits(&Tensor::cat(&[&k, &last_k], 2).unwrap())
                );
                assert!(
                    parent.retention_bytes().unwrap()
                        >= (page_tokens + 3) * 16 * dtype.size_in_bytes()
                );
                let bad = tensors(1, 2, DType::F64);
                assert!(parent.append(&bad.0, &bad.1).is_err());
                assert_eq!(parent.tokens, page_tokens + 3);
            }
        }
    }
}
