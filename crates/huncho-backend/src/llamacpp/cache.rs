//! Private immutable full hybrid sequence snapshots; never partial/on-device state.
use super::*;
use huncho_core::backend::{CachedPrefill, PrefillWork};
use std::collections::{BTreeSet, VecDeque};

pub(super) const MAX_HANDLES: usize = 64;
const MAX_BYTES: usize = 512 * 1024 * 1024;
const ENTRY_BYTES: usize = 256;

pub(super) struct Snapshot {
    pub(super) data: Vec<u8>,
    pub(super) tokens: usize,
}
#[derive(Default)]
pub(super) struct Prefixes {
    pub(super) handles: BTreeMap<u64, Arc<Snapshot>>,
    retained: VecDeque<(Vec<u32>, Arc<Snapshot>)>,
}
impl Prefixes {
    fn bytes(&self) -> usize {
        let mut seen = BTreeSet::new();
        let mut bytes = (self.handles.len() + self.retained.len()) * ENTRY_BYTES;
        for snapshot in self
            .handles
            .values()
            .chain(self.retained.iter().map(|(_, s)| s))
        {
            if seen.insert(Arc::as_ptr(snapshot) as usize) {
                bytes += snapshot.data.capacity();
            }
        }
        bytes
            + self
                .retained
                .iter()
                .map(|(key, _)| key.capacity() * 4)
                .sum::<usize>()
    }
    fn retained_bytes(&self) -> usize {
        self.retained
            .iter()
            .map(|(key, snapshot)| ENTRY_BYTES + key.capacity() * 4 + snapshot.data.capacity())
            .sum()
    }
    fn trim(&mut self, budget: usize) {
        while self.retained_bytes() > budget {
            self.retained.pop_front();
        }
    }
    pub(super) fn insert(&mut self, snapshot: Arc<Snapshot>) -> Result<CacheHandle> {
        if self.handles.len() >= MAX_HANDLES || self.bytes().saturating_add(ENTRY_BYTES) > MAX_BYTES
        {
            return Err(Error::Backend(
                "llama.cpp prefix handle/memory limit reached".into(),
            ));
        }
        let handle = crate::next_cache_handle()?;
        self.handles.insert(handle.id, snapshot);
        Ok(handle)
    }
    pub(super) fn clear_retained(&mut self) {
        self.retained.clear();
    }
    fn remember(&mut self, tokens: &[u32], snapshot: Arc<Snapshot>, budget: usize) {
        let cost = ENTRY_BYTES + tokens.len() * 4 + snapshot.data.capacity();
        if cost > budget {
            return;
        }
        while self.retained.len() >= 16 || self.retained_bytes() > budget - cost {
            self.retained.pop_front();
        }
        // The snapshot already belongs to a live handle and is charged once.
        if self.bytes().saturating_add(ENTRY_BYTES + tokens.len() * 4) <= MAX_BYTES {
            self.retained.push_back((tokens.to_vec(), snapshot));
        }
    }
}

impl LlamaCppBackend {
    pub(super) fn prefix(&self, handle: CacheHandle) -> Result<Arc<Snapshot>> {
        self.prefixes
            .handles
            .get(&handle.id)
            .cloned()
            .ok_or_else(|| Error::Backend("unknown or foreign llama.cpp prefix handle".into()))
    }
    pub(super) fn clear_memory(&self) -> Result<()> {
        let memory = unsafe { ffi::llama_get_memory(self.context.pointer.as_ptr()) };
        if memory.is_null() {
            return Err(Error::Backend(
                "llama.cpp omitted hybrid model memory".into(),
            ));
        }
        unsafe { ffi::llama_memory_clear(memory, true) };
        Ok(())
    }
    pub(super) fn restore(&self, snapshot: &Snapshot) -> Result<()> {
        // Full sequence state (flags NONE) contains attention KV AND recurrence.
        // No external/untrusted byte stream enters this private API.
        let read = unsafe {
            ffi::llama_state_seq_set_data(
                self.context.pointer.as_ptr(),
                snapshot.data.as_ptr(),
                snapshot.data.len(),
                0,
            )
        };
        let memory = unsafe { ffi::llama_get_memory(self.context.pointer.as_ptr()) };
        if read != snapshot.data.len()
            || memory.is_null()
            || unsafe { ffi::llama_memory_seq_pos_max(memory, 0) } != snapshot.tokens as i32 - 1
        {
            return Err(Error::Backend(
                "llama.cpp full hybrid prefix restore failed".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn snapshot(&self, tokens: usize) -> Result<Arc<Snapshot>> {
        let ctx = self.context.pointer.as_ptr();
        // The data API synchronizes the context before copying native state.
        let size = unsafe { ffi::llama_state_seq_get_size(ctx, 0) };
        if size == 0
            || size
                > MAX_BYTES
                    .saturating_sub(self.prefixes.bytes())
                    .saturating_sub(ENTRY_BYTES)
        {
            return Err(Error::Backend(
                "llama.cpp full prefix exceeds 512 MiB charged snapshot budget".into(),
            ));
        }
        let mut data = Vec::new();
        data.try_reserve_exact(size)
            .map_err(|_| Error::Backend("llama.cpp prefix allocation failed".into()))?;
        if data.capacity()
            > MAX_BYTES
                .saturating_sub(self.prefixes.bytes())
                .saturating_sub(ENTRY_BYTES)
        {
            return Err(Error::Backend(
                "llama.cpp prefix allocation exceeds charged budget".into(),
            ));
        }
        data.resize(size, 0);
        let written = unsafe { ffi::llama_state_seq_get_data(ctx, data.as_mut_ptr(), size, 0) };
        if written != size {
            return Err(Error::Backend(
                "llama.cpp full prefix snapshot failed".into(),
            ));
        }
        Ok(Arc::new(Snapshot { data, tokens }))
    }
    pub(super) fn cached_prefill(
        &mut self,
        tokens: &[u32],
        budget: usize,
        work: &mut PrefillWork,
    ) -> Result<CachedPrefill> {
        if !matches!(self.readout.as_ref(), Readout::Pointer(_)) {
            return Err(Error::Unsupported(
                "llama.cpp prefix reuse currently requires Kev F2".into(),
            ));
        }
        if tokens.is_empty()
            || tokens.len() > self.capabilities.max_context
            || tokens
                .iter()
                .any(|&t| t as usize >= self.context.model.vocab)
            || budget > MAX_BYTES
        {
            return Err(Error::Request(
                "invalid llama.cpp prefix tokens or retention budget (0..512 MiB)".into(),
            ));
        }
        if self.prefixes.handles.len() >= MAX_HANDLES {
            return Err(Error::Backend(
                "llama.cpp prefix handle limit reached".into(),
            ));
        }
        self.prefixes.trim(budget);
        if budget > 0 {
            if let Some((_, snapshot)) = self
                .prefixes
                .retained
                .iter()
                .find(|(key, _)| key.as_slice() == tokens)
            {
                let handle = self.prefixes.insert(snapshot.clone())?;
                return Ok(CachedPrefill { handle, hit: true });
            }
        }
        self.clear_memory()?;
        work.forward_calls += 1;
        work.processed_tokens += tokens.len() as u64;
        let result = (|| {
            self.decode_tokens(tokens, &[], 0, false)?;
            let snapshot = self.snapshot(tokens.len())?;
            let handle = self.prefixes.insert(snapshot.clone())?;
            self.prefixes.remember(tokens, snapshot, budget);
            Ok(CachedPrefill { handle, hit: false })
        })();
        if result.is_err() {
            let _ = self.clear_memory();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_forks_charge_shared_bytes_once_and_retention_is_fifo_bounded() {
        let mut prefixes = Prefixes::default();
        let snapshot = Arc::new(Snapshot {
            data: vec![42; 1024],
            tokens: 3,
        });
        let parent = prefixes.insert(snapshot.clone()).unwrap();
        let child = prefixes.insert(snapshot.clone()).unwrap();
        assert_eq!(prefixes.bytes(), 1024 + 2 * ENTRY_BYTES);
        prefixes.remember(&[1, 2, 3], snapshot.clone(), 1024 * 1024);
        assert_eq!(prefixes.bytes(), 1024 + 3 * ENTRY_BYTES + 12);
        prefixes.handles.remove(&parent.id);
        assert_eq!(prefixes.handles[&child.id].data, vec![42; 1024]);
        prefixes.clear_retained();
        assert_eq!(prefixes.bytes(), 1024 + ENTRY_BYTES);
        prefixes.handles.remove(&child.id);
        assert_eq!(prefixes.bytes(), 0);
        for key in 0..18 {
            prefixes.remember(
                &[key],
                Arc::new(Snapshot {
                    data: vec![key as u8; 1024],
                    tokens: 1,
                }),
                1024 * 1024,
            );
        }
        assert_eq!(prefixes.retained.len(), 16);
        assert_eq!(prefixes.retained.front().unwrap().0, [2]);
        prefixes.trim(1024 + ENTRY_BYTES + 4);
        assert_eq!(prefixes.retained.len(), 1);
        assert_eq!(prefixes.retained.front().unwrap().0, [17]);
        prefixes.trim(0);
        assert_eq!(prefixes.bytes(), 0);
    }
}
