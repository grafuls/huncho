//! A deterministic, dependency-free [`Backend`] used for offline tests, demos,
//! and the CI conformance harness.
//!
//! The mock emits a `Features` (hidden-state) output at the requested positions.
//! The hidden vector at each position is derived purely from the token ids, so
//! results are reproducible across runs and platforms. It is *not* a
//! reference-accurate model; it exists so the whole pipeline (prompt -> head ->
//! calibration -> conformance) can be exercised without real weights.

use std::collections::BTreeMap;

use huncho_core::backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput};
use huncho_core::error::Result;
use huncho_core::manifest::{BackendId, Family};
use huncho_core::tensor::Tensor;

/// The default vocabulary width the mock emits.
pub const DEFAULT_HIDDEN: usize = 4096;

/// A deterministic mock backend.
pub struct MockBackend {
    hidden: usize,
    max_context: usize,
    id: BackendId,
    dtype: String,
    families: Vec<Family>,
    supports_fork: bool,
    /// Simple "KV" store that records prefilled token prefixes so `fork` can
    /// reconstruct branch inputs. Reproducible and cheap.
    caches: BTreeMap<u64, Vec<u32>>,
}

impl MockBackend {
    pub fn new() -> MockBackend {
        MockBackend::with_vocab(DEFAULT_HIDDEN)
    }

    pub fn with_vocab(vocab: usize) -> MockBackend {
        MockBackend {
            hidden: vocab.min(512),
            max_context: 8192,
            id: BackendId::Onnx,
            dtype: "fp32".into(),
            families: vec![Family::F1, Family::F2, Family::F3, Family::F4],
            supports_fork: true,
            caches: BTreeMap::new(),
        }
    }

    /// Configure the reported backend id (useful for calibration-key tests).
    pub fn with_backend(mut self, id: BackendId) -> MockBackend {
        self.id = id;
        self
    }

    /// Configure the reported dtype (drives calibration lookup).
    pub fn with_dtype(mut self, dtype: impl Into<String>) -> MockBackend {
        self.dtype = dtype.into();
        self
    }

    /// Deterministic per-position hidden states.
    ///
    /// The mock emits a `Features` output so the engine's feature-projection
    /// heads (F1/F2/F4) and the mean-fallback projection can all be exercised.
    /// Each position's feature vector is a sparse, deterministic activation
    /// derived from the token id at that position, so results are stable across
    /// runs and platforms.
    fn features_for(&self, tokens: &[u32], positions: &[usize]) -> Result<Tensor> {
        let n = positions.len();
        let hidden = self.hidden.max(1);
        let mut data = vec![0.0f32; n * hidden];
        for (row, &pos) in positions.iter().enumerate() {
            let tok = tokens.get(pos).copied().unwrap_or(0usize as u32) as usize;
            let col = tok % hidden;
            data[row * hidden + col] = 4.0 + 0.01 * (tok as f32);
        }
        Tensor::new(vec![n, hidden], data)
    }
}

impl Backend for MockBackend {
    fn replica(&self) -> Result<Box<dyn Backend>> {
        Ok(Box::new(Self {
            hidden: self.hidden,
            max_context: self.max_context,
            id: self.id,
            dtype: self.dtype.clone(),
            families: self.families.clone(),
            supports_fork: self.supports_fork,
            caches: BTreeMap::new(),
        }))
    }
    fn id(&self) -> BackendId {
        self.id
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.id,
            dtype: self.dtype.clone(),
            max_context: self.max_context,
            supports_fork: self.supports_fork,
            supports_lora: false,
            families: self.families.clone(),
            extra: BTreeMap::from([("device".into(), "CPU".into())]),
        }
    }

    fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
        if input.tokens.len() > self.max_context {
            return Err(huncho_core::error::Error::Backend(format!(
                "sequence length {} exceeds mock max_context {}",
                input.tokens.len(),
                self.max_context
            )));
        }
        if input.retain_cache {
            return Err(huncho_core::error::Error::Unsupported(
                "use prefill to obtain a cache handle".into(),
            ));
        }
        if let Some(src) = input.fork_from {
            let prefix = self
                .caches
                .get(&src.id)
                .ok_or_else(|| huncho_core::error::Error::Backend("cache not found".into()))?;
            if prefix.len() + input.tokens.len() > self.max_context {
                return Err(huncho_core::error::Error::Backend(
                    "cached sequence exceeds mock max_context".into(),
                ));
            }
            let positions = input
                .positions
                .iter()
                .map(|&position| {
                    position.checked_add(prefix.len()).ok_or_else(|| {
                        huncho_core::error::Error::Backend("cached position overflow".into())
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut tokens = prefix.clone();
            tokens.extend(&input.tokens);
            let values = self.features_for(&tokens, &positions)?;
            self.caches.insert(src.id, tokens);
            return Ok(ForwardOutput::Features {
                positions: input.positions,
                values,
            });
        }
        let values = self.features_for(&input.tokens, &input.positions)?;
        Ok(ForwardOutput::Features {
            positions: input.positions,
            values,
        })
    }

    fn fork(&mut self, handle: CacheHandle) -> Result<CacheHandle> {
        if !self.caches.contains_key(&handle.id) {
            return Err(huncho_core::error::Error::Backend(format!(
                "cache {} not found",
                handle.id
            )));
        }
        let child = crate::next_cache_handle()?;
        self.caches
            .insert(child.id, self.caches[&handle.id].clone());
        Ok(child)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<CacheHandle> {
        if tokens.is_empty() || tokens.len() > self.max_context {
            return Err(huncho_core::error::Error::Backend(
                "prefill must be nonempty and within max_context".into(),
            ));
        }
        let handle = crate::next_cache_handle()?;
        self.caches.insert(handle.id, tokens.to_vec());
        Ok(handle)
    }

    fn release_cache(&mut self, handle: CacheHandle) -> Result<()> {
        self.caches
            .remove(&handle.id)
            .map(|_| ())
            .ok_or_else(|| huncho_core::error::Error::Backend("cache not found".into()))
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        MockBackend::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_output() {
        let mut b = MockBackend::with_vocab(64);
        let inp = ForwardInput::new(vec![1, 2, 3], vec![1, 2]);
        let o1 = b.forward(inp.clone()).unwrap();
        let o2 = b.forward(inp).unwrap();
        assert_eq!(o1.values().data(), o2.values().data());
    }

    #[test]
    fn fork_is_isolated() {
        let mut b = MockBackend::with_vocab(64);
        let p = b.prefill(&[1, 2, 3]).unwrap();
        let f = b.fork(p).unwrap();
        assert_ne!(p.id, f.id);
    }
}
