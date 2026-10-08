//! Bounded reuse of exact prepared prompts, independent of backend execution.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::prompt::{BuiltPrompt, Candidate};

pub(crate) struct PromptCache {
    entries: HashMap<Arc<[u8]>, (BuiltPrompt, usize)>,
    order: VecDeque<Arc<[u8]>>,
    bytes: usize,
    budget: usize,
}

impl PromptCache {
    const MAX_ENTRIES: usize = 1024;
    const ENTRY_OVERHEAD: usize = 256;

    pub(crate) fn new(budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            budget,
        }
    }

    pub(crate) fn get(&self, key: &[u8]) -> Option<BuiltPrompt> {
        // Engine prefix/batch paths consume and modify tokens and positions.
        // Return an owned clone so neither they nor callers can alter retention.
        self.entries.get(key).map(|(prompt, _)| prompt.clone())
    }

    fn cost(key: &[u8], prompt: &BuiltPrompt) -> Option<usize> {
        let mut bytes = key
            .len()
            .checked_add(
                prompt
                    .tokens
                    .len()
                    .checked_mul(std::mem::size_of::<u32>())?,
            )?
            .checked_add(
                prompt
                    .candidates
                    .len()
                    .checked_mul(std::mem::size_of::<Candidate>())?,
            )?
            .checked_add(std::mem::size_of::<BuiltPrompt>())?
            .checked_add(2 * Self::ENTRY_OVERHEAD)?;
        for candidate in &prompt.candidates {
            bytes = bytes
                .checked_add(candidate.label.len())?
                .checked_add(candidate.description.as_ref().map_or(0, String::len))?;
        }
        Some(bytes)
    }

    pub(crate) fn insert(&mut self, key: Vec<u8>, prompt: &BuiltPrompt) {
        let Some(cost) = Self::cost(&key, prompt).filter(|&cost| cost <= self.budget) else {
            return;
        };
        if self.entries.contains_key(key.as_slice()) {
            return;
        }
        while self.order.len() >= Self::MAX_ENTRIES || self.bytes > self.budget - cost {
            let Some(old_key) = self.order.pop_front() else {
                break;
            };
            if let Some((_, old_cost)) = self.entries.remove(&old_key) {
                self.bytes -= old_cost;
            }
        }
        let key: Arc<[u8]> = key.into();
        self.entries.insert(key.clone(), (prompt.clone(), cost));
        self.order.push_back(key);
        self.bytes += cost;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::CandidateKind;

    #[test]
    fn eviction_oversize_bounds_and_owned_return() {
        let prompt = BuiltPrompt {
            tokens: vec![9, 3, 7],
            candidates: vec![Candidate {
                kind: CandidateKind::Level,
                position: 2,
                code_id: 7,
                label: "0".into(),
                description: Some("first level".into()),
                index: 0,
            }],
            prefix_len: 1,
            qtype: 1,
        };
        let cost = PromptCache::cost(b"a", &prompt).unwrap();
        let mut cache = PromptCache::new(cost);
        cache.insert(b"a".to_vec(), &prompt);
        let mut copy = cache.get(b"a").unwrap();
        copy.tokens.clear();
        copy.candidates[0].position = 0;
        copy.candidates[0].description = None;
        let retained = cache.get(b"a").unwrap();
        assert_eq!(retained.tokens, prompt.tokens);
        assert_eq!(retained.candidates[0].position, 2);
        assert_eq!(
            retained.candidates[0].description.as_deref(),
            Some("first level")
        );
        cache.insert(b"oversized".to_vec(), &prompt);
        assert!(cache.get(b"a").is_some());
        assert!(cache.get(b"oversized").is_none());
        cache.insert(b"b".to_vec(), &prompt);
        assert!(cache.get(b"a").is_none());
        assert!(cache.get(b"b").is_some());
        assert!(cache.bytes <= cache.budget);
        let mut cache = PromptCache::new(usize::MAX);
        for i in 0u32..2048 {
            cache.insert(i.to_le_bytes().to_vec(), &prompt);
        }
        assert_eq!(cache.entries.len(), PromptCache::MAX_ENTRIES);
        assert!(cache.get(&0u32.to_le_bytes()).is_none());
        assert!(cache.get(&2047u32.to_le_bytes()).is_some());
        let mut disabled = PromptCache::new(0);
        disabled.insert(b"a".to_vec(), &prompt);
        assert!(disabled.get(b"a").is_none());
    }
}
