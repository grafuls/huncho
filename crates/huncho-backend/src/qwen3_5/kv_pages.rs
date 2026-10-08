//! Immutable CPU KV pages. Attention still materializes the original complete
//! key order; this changes persistent storage/branch copies, not reductions.
use candle::{Result, Tensor};
use std::sync::Arc;

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
    fn page_order_and_copy_on_write_tails_preserve_exact_storage_values() {
        for dtype in [DType::F32, DType::F16] {
            for page_tokens in [16, 32, 64, 256] {
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
