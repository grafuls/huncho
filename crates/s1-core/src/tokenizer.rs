//! Tokenization for the `s1` engine.
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
        if token.starts_with('<') && token.ends_with('>') {
            Some(self.id_for_inner(token))
        } else {
            Some(self.id_for_inner(token))
        }
    }

    fn name(&self) -> &str {
        "simple"
    }
}

/// Wrap the Hugging Face `tokenizers` crate (feature-gated).
#[cfg(feature = "tokenizers")]
pub struct HfTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "tokenizers")]
impl HfTokenizer {
    /// Load a `tokenizer.json` from disk.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<HfTokenizer> {
        let tokenizer = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| crate::error::Error::Package(format!("failed to load tokenizer: {e}")))?;
        Ok(HfTokenizer { tokenizer })
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
        let enc = self.tokenizer.encode(token, false).ok()?;
        if enc.get_ids().len() == 1 {
            Some(enc.get_ids()[0])
        } else {
            None
        }
    }

    fn name(&self) -> &str {
        "hf-tokenizers"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
