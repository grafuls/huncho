//! Bounded, engine-local reuse of successful calibrated responses.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::contract::{Answer, SystemOneResponse};

pub(crate) struct ResponseCache {
    entries: HashMap<Arc<[u8]>, (SystemOneResponse, usize)>,
    order: VecDeque<Arc<[u8]>>,
    bytes: usize,
    budget: usize,
}

impl ResponseCache {
    const MAX_ENTRIES: usize = 1024;
    // Charge a conservative allowance per tree/hash entry, besides all owned
    // strings and float payloads. This is not an allocator/RSS measurement.
    const ENTRY_OVERHEAD: usize = 256;

    pub(crate) fn new(budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            budget,
        }
    }

    pub(crate) fn get(&self, key: &[u8]) -> Option<SystemOneResponse> {
        self.entries.get(key).map(|(response, _)| response.clone())
    }

    fn cost(key: &[u8], response: &SystemOneResponse) -> Option<usize> {
        let mut bytes = key
            .len()
            .checked_add(response.model.len())?
            .checked_add(std::mem::size_of::<SystemOneResponse>())?
            .checked_add(2 * Self::ENTRY_OVERHEAD)?;
        let mut add = |len: usize| -> Option<()> {
            bytes = bytes.checked_add(len)?;
            Some(())
        };
        for (id, answer) in &response.answers {
            add(id.len())?;
            add(std::mem::size_of::<Answer>() + Self::ENTRY_OVERHEAD)?;
            match answer {
                Answer::Choice {
                    choice,
                    probabilities,
                    ..
                } => {
                    add(choice.len())?;
                    for label in probabilities.keys() {
                        add(label.len())?;
                        add(Self::ENTRY_OVERHEAD)?;
                    }
                }
                Answer::Score {
                    probabilities,
                    legend,
                    ..
                } => {
                    for label in probabilities.keys() {
                        add(label.len())?;
                        add(Self::ENTRY_OVERHEAD)?;
                    }
                    for (label, description) in legend {
                        add(label.len())?;
                        add(description.len())?;
                        add(Self::ENTRY_OVERHEAD)?;
                    }
                }
                Answer::Noul { .. } => {}
            }
        }
        if let Some(extras) = &response.extensions {
            for value in [
                &extras.backend,
                &extras.dtype,
                &extras.calibration_status,
                &extras.confidence_definition,
                &extras.prompt_contract_hash,
            ]
            .into_iter()
            .flatten()
            {
                add(value.len())?;
            }
            if let Some(rows) = &extras.raw_logits {
                for (id, row) in rows {
                    add(id.len())?;
                    add(row.len().checked_mul(std::mem::size_of::<f32>())?)?;
                    add(Self::ENTRY_OVERHEAD)?;
                }
            }
        }
        Some(bytes)
    }

    pub(crate) fn insert(&mut self, key: Vec<u8>, response: &SystemOneResponse) {
        let Some(cost) = Self::cost(&key, response).filter(|&cost| cost <= self.budget) else {
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
        self.entries.insert(key.clone(), (response.clone(), cost));
        self.order.push_back(key);
        self.bytes += cost;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_key;
    use crate::contract::{Question, SystemOneRequest, Usage};
    use crate::engine::EvalOptions;

    #[test]
    fn key_preserves_each_optional_noul_description() {
        let absent: SystemOneRequest = serde_json::from_value(serde_json::json!({
            "model": "test", "state": "input",
            "questions": {"q": {"type": "noul", "instructions": "yes?",
                "criteria": {"yes": null, "no": null}}}
        }))
        .unwrap();
        let opts = EvalOptions::default();
        let mut keys = vec![cache_key::request(&absent, &opts).unwrap()];
        let mut prompt_keys =
            vec![cache_key::prompt(&absent.state, &absent.questions["q"]).unwrap()];
        for (yes, no) in [(true, false), (false, true), (true, true)] {
            let mut request = absent.clone();
            let Question::Noul {
                criteria: Some(criteria),
                ..
            } = &mut request.questions["q"]
            else {
                unreachable!()
            };
            criteria.yes = yes.then_some(serde_json::Value::Null);
            criteria.no = no.then_some(serde_json::Value::Null);
            assert_eq!(
                serde_json::to_vec(&request).unwrap(),
                serde_json::to_vec(&absent).unwrap()
            );
            let key = cache_key::request(&request, &opts).unwrap();
            assert!(!keys.contains(&key));
            keys.push(key);
            let key = cache_key::prompt(&request.state, &request.questions["q"]).unwrap();
            assert!(!prompt_keys.contains(&key));
            prompt_keys.push(key);
        }
    }

    #[test]
    fn budget_eviction_oversize_and_entry_limit() {
        let response = SystemOneResponse::new("test".into(), Default::default(), Usage::new(7));
        let cost = ResponseCache::cost(b"a", &response).unwrap();
        let mut cache = ResponseCache::new(cost);
        cache.insert(b"a".to_vec(), &response);
        assert!(cache.get(b"a").is_some());
        cache.insert(b"too long".to_vec(), &response);
        assert!(cache.get(b"a").is_some());
        assert!(cache.get(b"too long").is_none());
        cache.insert(b"b".to_vec(), &response);
        assert!(cache.get(b"a").is_none());
        assert!(cache.get(b"b").is_some());
        assert!(cache.bytes <= cache.budget);

        let mut cache = ResponseCache::new(usize::MAX);
        for i in 0u32..2048 {
            cache.insert(i.to_le_bytes().to_vec(), &response);
        }
        assert_eq!(cache.entries.len(), ResponseCache::MAX_ENTRIES);
        assert!(cache.get(&0u32.to_le_bytes()).is_none());
        assert!(cache.get(&2047u32.to_le_bytes()).is_some());
        let mut disabled = ResponseCache::new(0);
        disabled.insert(b"a".to_vec(), &response);
        assert!(disabled.get(b"a").is_none());
    }
}
