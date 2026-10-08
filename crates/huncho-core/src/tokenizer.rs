//! Tokenization for the `huncho` engine.
//!
//! The engine talks to any tokenizer through the [`Tokenizer`] trait so that
//! prompt building stays byte-identical to the reference implementation while
//! remaining backend-agnostic.
//!
//! Two implementations ship:
//! * [`SimpleTokenizer`] — a deterministic, dependency-free tokenizer used for
//!   offline tests and the mock backend. It is *not* reference-accurate.
//! * [`HfTokenizer`] — wraps the official Hugging Face `tokenizers` crate for
//!   byte-identical tokenization (behind the `tokenizers` feature).

use crate::error::Result;

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// A tokenizer that maps text to token ids.
pub trait Tokenizer: Send + Sync {
    /// Encode text into token ids.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    /// Decode token ids back into text (best-effort).
    fn decode(&self, ids: &[u32]) -> Result<String>;
    /// The token id for a single token string, if it exists in the vocab.
    fn id_for(&self, token: &str) -> Option<u32>;
    /// A short identifier for this tokenizer (for diagnostics).
    fn name(&self) -> &str;
    /// The token id for the mask token (`[MASK]`), if the tokenizer has one.
    /// Needed by models that score candidates at `[MASK]` positions (Laya).
    fn mask_token_id(&self) -> Option<u32>;
    /// The token id for the classification/bos token (`[CLS]`), if present.
    fn cls_token_id(&self) -> Option<u32>;
    /// The token id for the separator token (`[SEP]`), if present.
    fn sep_token_id(&self) -> Option<u32>;
}

/// Instance-local memoization of exact encode calls. The underlying tokenizer
/// must remain immutable; text and the special-token flag form the whole key.
/// Successful outputs are copied on return so callers cannot mutate the cache.
pub struct CachedTokenizer {
    inner: Box<dyn Tokenizer>,
    cache: Mutex<EncodingCache>,
}

struct EncodingCache {
    entries: [HashMap<String, Vec<u32>>; 2],
    order: VecDeque<(bool, String)>,
    bytes: usize,
    budget: usize,
}

impl EncodingCache {
    // Charge both owned key copies, token capacity and a conservative fixed
    // entry allowance. A separate entry bound limits container overhead too.
    const ENTRY_OVERHEAD: usize = 256;
    const MAX_ENTRIES: usize = 1024;

    fn cost(text: &str, ids: &[u32]) -> Option<usize> {
        text.len()
            .checked_mul(2)?
            .checked_add(ids.len().checked_mul(std::mem::size_of::<u32>())?)?
            .checked_add(Self::ENTRY_OVERHEAD)
    }

    fn insert(&mut self, text: &str, special: bool, ids: &[u32]) {
        let Some(cost) = Self::cost(text, ids).filter(|&cost| cost <= self.budget) else {
            return;
        };
        if self.entries[usize::from(special)].contains_key(text) {
            return;
        }
        while self.order.len() >= Self::MAX_ENTRIES || self.bytes > self.budget - cost {
            let Some((flag, key)) = self.order.pop_front() else {
                break;
            };
            if let Some(old) = self.entries[usize::from(flag)].remove(&key) {
                self.bytes -= Self::cost(&key, &old).unwrap();
            }
        }
        self.entries[usize::from(special)].insert(text.to_owned(), ids.to_vec());
        self.order.push_back((special, text.to_owned()));
        self.bytes += cost;
    }
}

impl CachedTokenizer {
    /// Zero disables retention. Oversized encodings bypass the FIFO cache.
    /// The budget bounds retained text/token payload plus charged entry overhead;
    /// it is not an exact measurement of allocator or HashMap capacity.
    pub fn new(inner: Box<dyn Tokenizer>, max_bytes: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(EncodingCache {
                entries: Default::default(),
                order: VecDeque::new(),
                bytes: 0,
                budget: max_bytes,
            }),
        }
    }
}

impl Tokenizer for CachedTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        if let Ok(cache) = self.cache.lock() {
            if let Some(ids) = cache.entries[usize::from(add_special_tokens)].get(text) {
                return Ok(ids.clone());
            }
        }
        // Encoding stays outside the lock. Concurrent misses may do redundant
        // work, but never serialize tokenizer computation or cache errors.
        let ids = self.inner.encode(text, add_special_tokens)?;
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(text, add_special_tokens, &ids);
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids)
    }

    fn id_for(&self, token: &str) -> Option<u32> {
        self.inner.id_for(token)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn mask_token_id(&self) -> Option<u32> {
        self.inner.mask_token_id()
    }

    fn cls_token_id(&self) -> Option<u32> {
        self.inner.cls_token_id()
    }

    fn sep_token_id(&self) -> Option<u32> {
        self.inner.sep_token_id()
    }
}

/// A deterministic, hash-based tokenizer for offline testing.
///
/// It splits on whitespace and punctuation, maps each token to a stable id via
/// FNV-1a, and reserves a special-id region for synthetic markers such as
/// option markers (`<option:0>`).
pub struct SimpleTokenizer {
    vocab: usize,
}

impl SimpleTokenizer {
    /// Special tokens are mapped into the top of the vocab so they never
    /// collide with ordinary vocabulary entries.
    const SPECIAL_RESERVED: usize = 1024;
    pub const BOS: &'static str = "<s>";
    pub const EOS: &'static str = "</s>";

    pub fn new(vocab: usize) -> SimpleTokenizer {
        SimpleTokenizer {
            vocab: vocab.max(Self::SPECIAL_RESERVED + 8),
        }
    }

    /// Stable FNV-1a hash of a byte string.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    fn tokens_of(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let is_sep = |c: char| c.is_whitespace();
        for ch in text.chars() {
            if is_sep(ch) {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    fn id_for_inner(&self, token: &str) -> u32 {
        let reserved_start = (self.vocab - Self::SPECIAL_RESERVED) as u64;
        let h = Self::fnv1a(token.as_bytes());
        let id = if token.starts_with('<') && token.ends_with('>') {
            // Synthetic markers live in the reserved region.
            reserved_start + (h % Self::SPECIAL_RESERVED as u64)
        } else {
            h % (reserved_start)
        };
        id as u32
    }

    /// The id for a special token, if within reserved range.
    pub fn special_id(&self, token: &str) -> Option<u32> {
        if token.starts_with('<') && token.ends_with('>') {
            Some(self.id_for_inner(token))
        } else {
            None
        }
    }
}

impl Tokenizer for SimpleTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if add_special_tokens {
            ids.push(self.id_for_inner(Self::BOS));
        }
        for tok in Self::tokens_of(text) {
            ids.push(self.id_for_inner(&tok));
        }
        if add_special_tokens {
            ids.push(self.id_for_inner(Self::EOS));
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        // SimpleTokenizer cannot losslessly decode; return a placeholder.
        Ok(format!("<{} tokens>", ids.len()))
    }

    fn id_for(&self, token: &str) -> Option<u32> {
        Some(self.id_for_inner(token))
    }

    fn name(&self) -> &str {
        "simple"
    }

    fn mask_token_id(&self) -> Option<u32> {
        Some(self.id_for_inner("[MASK]"))
    }

    fn cls_token_id(&self) -> Option<u32> {
        Some(self.id_for_inner("[CLS]"))
    }

    fn sep_token_id(&self) -> Option<u32> {
        Some(self.id_for_inner("[SEP]"))
    }
}

/// Wrap the Hugging Face `tokenizers` crate (feature-gated).
#[cfg(feature = "tokenizers")]
pub struct HfTokenizer {
    tokenizer: tokenizers::Tokenizer,
    // Cache exact single-token encodings, preserving normalization/subword rules.
    token_ids: std::sync::RwLock<std::collections::HashMap<String, Option<u32>>>,
}

#[cfg(feature = "tokenizers")]
impl HfTokenizer {
    /// Joint-schema encoders enforce their own total budget. Disable tokenizer
    /// padding/truncation so individual schema fragments cannot be shortened.
    pub fn from_file_unbounded(path: impl AsRef<std::path::Path>) -> Result<HfTokenizer> {
        let mut result = Self::from_file(path)?;
        result
            .tokenizer
            .with_truncation(None)
            .map_err(|e| crate::error::Error::Package(e.to_string()))?;
        result.tokenizer.with_padding(None);
        Ok(result)
    }

    /// Load a `tokenizer.json` from disk.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<HfTokenizer> {
        let tokenizer = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| crate::error::Error::Package(format!("failed to load tokenizer: {e}")))?;
        Ok(HfTokenizer {
            tokenizer,
            token_ids: Default::default(),
        })
    }
}

#[cfg(feature = "tokenizers")]
impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| crate::error::Error::Backend(format!("tokenizer encode failed: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| crate::error::Error::Backend(format!("tokenizer decode failed: {e}")))
    }

    fn id_for(&self, token: &str) -> Option<u32> {
        const MAX_KEYS: usize = 512;
        const MAX_KEY_BYTES: usize = 128;
        let cacheable = token.len() <= MAX_KEY_BYTES;
        if cacheable {
            if let Ok(cache) = self.token_ids.read() {
                if let Some(&id) = cache.get(token) {
                    return id;
                }
            }
        }
        let id = self
            .tokenizer
            .encode(token, false)
            .ok()
            .and_then(|enc| (enc.get_ids().len() == 1).then(|| enc.get_ids()[0]));
        if cacheable {
            if let Ok(mut cache) = self.token_ids.write() {
                if cache.len() < MAX_KEYS {
                    cache.insert(token.to_owned(), id);
                }
            }
        }
        id
    }

    fn name(&self) -> &str {
        "hf-tokenizers"
    }

    fn mask_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("[MASK]")
    }

    fn cls_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("[CLS]")
    }

    fn sep_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("[SEP]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingTokenizer {
        inner: SimpleTokenizer,
        calls: Arc<AtomicUsize>,
    }

    impl Tokenizer for CountingTokenizer {
        fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if text == "fail" {
                return Err(crate::error::Error::Backend("encoding failed".into()));
            }
            self.inner.encode(text, special)
        }
        fn decode(&self, ids: &[u32]) -> Result<String> {
            self.inner.decode(ids)
        }
        fn id_for(&self, token: &str) -> Option<u32> {
            self.inner.id_for(token)
        }
        fn name(&self) -> &str {
            self.inner.name()
        }
        fn mask_token_id(&self) -> Option<u32> {
            self.inner.mask_token_id()
        }
        fn cls_token_id(&self) -> Option<u32> {
            self.inner.cls_token_id()
        }
        fn sep_token_id(&self) -> Option<u32> {
            self.inner.sep_token_id()
        }
    }

    fn cached(max_bytes: usize) -> (CachedTokenizer, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let tokenizer = CachedTokenizer::new(
            Box::new(CountingTokenizer {
                inner: SimpleTokenizer::new(32768),
                calls: calls.clone(),
            }),
            max_bytes,
        );
        (tokenizer, calls)
    }

    #[test]
    fn exact_encoding_cache_isolates_flags_instances_and_caller_mutation() {
        let (tokenizer, calls) = cached(4096);
        let expected = SimpleTokenizer::new(32768).encode("a b", false).unwrap();
        let mut actual = tokenizer.encode("a b", false).unwrap();
        actual[0] = u32::MAX;
        assert_eq!(tokenizer.encode("a b", false).unwrap(), expected);
        let special = tokenizer.encode("a b", true).unwrap();
        assert_ne!(special, expected);
        assert_eq!(tokenizer.encode("a b", true).unwrap(), special);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let (other, other_calls) = cached(4096);
        assert_eq!(other.encode("a b", false).unwrap(), expected);
        assert_eq!(other_calls.load(Ordering::SeqCst), 1);
        assert_eq!(tokenizer.name(), "simple");
        assert_eq!(tokenizer.decode(&expected).unwrap(), "<2 tokens>");
        assert_eq!(tokenizer.id_for("[MASK]"), tokenizer.mask_token_id());
        assert_eq!(tokenizer.id_for("[CLS]"), tokenizer.cls_token_id());
        assert_eq!(tokenizer.id_for("[SEP]"), tokenizer.sep_token_id());
    }

    #[test]
    fn encoding_cache_evicts_bypasses_large_entries_and_never_retains_errors() {
        let cost = EncodingCache::cost("a", &[1]).unwrap();
        let (tokenizer, calls) = cached(2 * cost);
        for text in ["a", "b", "a", "c", "a"] {
            tokenizer.encode(text, false).unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 4); // FIFO hits do not reorder.
        for _ in 0..2 {
            tokenizer.encode(&"large ".repeat(100), false).unwrap();
            assert!(tokenizer.encode("fail", false).is_err());
        }
        assert_eq!(calls.load(Ordering::SeqCst), 8);
        let cache = tokenizer.cache.lock().unwrap();
        assert!(cache.bytes <= cache.budget);
        assert_eq!(cache.order.len(), 2);
        drop(cache);
        let (disabled, calls) = cached(0);
        disabled.encode("a", false).unwrap();
        disabled.encode("a", false).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let (bounded, _) = cached(usize::MAX);
        for i in 0..EncodingCache::MAX_ENTRIES + 20 {
            bounded.encode(&format!("key{i}"), false).unwrap();
        }
        assert_eq!(
            bounded.cache.lock().unwrap().order.len(),
            EncodingCache::MAX_ENTRIES
        );
    }

    #[test]
    fn concurrent_encoding_cache_keeps_exact_results_and_a_bounded_budget() {
        let (tokenizer, _) = cached(4096);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let tokenizer = &tokenizer;
                scope.spawn(move || {
                    let baseline = SimpleTokenizer::new(32768);
                    for i in 0..100 {
                        let text = format!("unicode café {i}");
                        assert_eq!(
                            tokenizer.encode(&text, i % 2 == 0).unwrap(),
                            baseline.encode(&text, i % 2 == 0).unwrap()
                        );
                    }
                });
            }
        });
        let cache = tokenizer.cache.lock().unwrap();
        assert!(cache.bytes <= cache.budget);
        assert!(cache.order.len() <= EncodingCache::MAX_ENTRIES);
    }

    #[cfg(feature = "tokenizers")]
    #[test]
    fn cached_hf_encoding_preserves_exact_normalization_and_special_token_policy() {
        let file = "tests/fixtures/minimal_tokenizer.json";
        let baseline = HfTokenizer::from_file(file).unwrap();
        let cached = CachedTokenizer::new(Box::new(HfTokenizer::from_file(file).unwrap()), 4096);
        for special in [false, true] {
            for text in ["hello world", "refund", "<option:0>", "", "CAFÉ café"] {
                for _ in 0..3 {
                    assert_eq!(
                        cached.encode(text, special).unwrap(),
                        baseline.encode(text, special).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn deterministic_encode() {
        let tk = SimpleTokenizer::new(32768);
        let a = tk.encode("hello world", true).unwrap();
        let b = tk.encode("hello world", true).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 4); // BOS hello world EOS
        assert_eq!(a[0], tk.id_for(SimpleTokenizer::BOS).unwrap());
        assert_eq!(a[3], tk.id_for(SimpleTokenizer::EOS).unwrap());
    }

    #[test]
    fn markers_in_reserved_range() {
        let tk = SimpleTokenizer::new(32768);
        let m = tk.id_for("<option:0>").unwrap();
        assert!(m >= (32768 - 1024) as u32);
        // Distinct markers get distinct ids (almost surely).
        let m2 = tk.id_for("<option:1>").unwrap();
        assert_ne!(m, m2);
    }

    #[cfg(feature = "tokenizers")]
    #[test]
    fn token_id_cache_preserves_encoding_and_is_bounded() {
        let tk = HfTokenizer::from_file("tests/fixtures/minimal_tokenizer.json").unwrap();
        for token in [
            "<option:0>",
            "refund",
            "hello world",
            "not_in_the_vocab",
            "",
        ] {
            let ids = tk.encode(token, false).unwrap();
            let expected = (ids.len() == 1).then(|| ids[0]);
            for _ in 0..3 {
                assert_eq!(tk.id_for(token), expected);
            }
        }
        assert_eq!(tk.token_ids.read().unwrap().get("hello world"), Some(&None));
        for index in 0..600 {
            tk.id_for(&format!("key_{index}"));
        }
        assert_eq!(tk.token_ids.read().unwrap().len(), 512);
        let large = "long".repeat(40);
        tk.id_for(&large);
        assert!(!tk.token_ids.read().unwrap().contains_key(&large));
        assert_eq!(tk.encode("hello world", true).unwrap(), vec![1, 7, 8, 2]);
    }

    #[cfg(feature = "tokenizers")]
    #[test]
    fn token_id_cache_is_shared_safely_across_threads() {
        let tk = HfTokenizer::from_file("tests/fixtures/minimal_tokenizer.json").unwrap();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..50 {
                        assert_eq!(tk.id_for("<option:0>"), Some(18));
                        assert_eq!(tk.id_for("hello world"), None);
                    }
                });
            }
        });
        assert_eq!(tk.token_ids.read().unwrap().len(), 2);
    }
}
