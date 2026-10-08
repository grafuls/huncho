//! Bounded immutable CPU base tensors, before any checkpoint-specific LoRA merge.
//! Source bytes are hashed on every lookup; adapters and inference state are
//! never retained here. Eviction drops only the cache's references.
use candle::{DType, Tensor};
use huncho_core::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    io::Read,
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct BaseCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub bypasses: u64,
    pub evictions: u64,
    pub entries: usize,
    pub charged_bytes: usize,
}

struct Entry {
    key: [u8; 32],
    tensors: HashMap<String, Tensor>,
    bytes: usize,
}
#[derive(Default)]
struct Cache {
    entries: VecDeque<Entry>,
    stats: BaseCacheStats,
}

/// Share read-only base tensor storage across distinct, independently merged
/// Qwen adapters. The bound covers cache-owned tensor payloads and metadata,
/// not live models, allocator overhead or temporary loads/merges.
pub struct BaseWeightCache {
    budget: usize,
    cache: Mutex<Cache>,
}
impl BaseWeightCache {
    pub(crate) fn enabled(&self) -> bool {
        self.budget > 0
    }

    pub fn new(charged_byte_budget: usize) -> Self {
        Self {
            budget: charged_byte_budget,
            cache: Mutex::new(Cache::default()),
        }
    }

    pub fn stats(&self) -> Result<BaseCacheStats> {
        Ok(self.cache.lock().map_err(|_| poisoned())?.stats)
    }

    /// Release retained bases; loaded models keep their independently owned
    /// tensor references and are unaffected. Work counters remain cumulative.
    pub fn clear(&self) -> Result<()> {
        let mut cache = self.cache.lock().map_err(|_| poisoned())?;
        cache.entries.clear();
        cache.stats.entries = 0;
        cache.stats.charged_bytes = 0;
        Ok(())
    }

    pub(crate) fn load(
        &self,
        files: &[PathBuf],
        dtype: DType,
        include_lm_head: bool,
        load: impl FnOnce() -> Result<HashMap<String, Tensor>>,
    ) -> Result<HashMap<String, Tensor>> {
        if self.budget == 0 {
            return load();
        }
        let key = identity(files, dtype, include_lm_head)?;
        // Serialize startup materialization to avoid duplicate resident copies
        // on concurrent misses. This mutex is never held during inference.
        let mut cache = self.cache.lock().map_err(|_| poisoned())?;
        if let Some(index) = cache.entries.iter().position(|entry| entry.key == key) {
            if identity(files, dtype, include_lm_head)? != key {
                return Err(changed());
            }
            let entry = cache.entries.remove(index).unwrap();
            let tensors = entry.tensors.clone();
            cache.entries.push_back(entry);
            cache.stats.hits += 1;
            log::info!(
                "shared CPU base cache hit ({} charged bytes)",
                cache.stats.charged_bytes
            );
            return Ok(tensors);
        }
        cache.stats.misses += 1;
        let tensors = load()?;
        if identity(files, dtype, include_lm_head)? != key {
            return Err(changed());
        }
        let bytes = charge(&tensors)?;
        if tensors.is_empty() || bytes > self.budget {
            cache.stats.bypasses += 1;
            return Ok(tensors);
        }
        while cache.entries.len() >= 16 || cache.stats.charged_bytes > self.budget - bytes {
            let entry = cache
                .entries
                .pop_front()
                .ok_or_else(|| Error::Backend("invalid shared-base cache accounting".into()))?;
            cache.stats.charged_bytes -= entry.bytes;
            cache.stats.evictions += 1;
        }
        cache.entries.push_back(Entry {
            key,
            tensors: tensors.clone(),
            bytes,
        });
        cache.stats.charged_bytes += bytes;
        cache.stats.entries = cache.entries.len();
        log::info!("retained immutable CPU base ({bytes} charged bytes)");
        Ok(tensors)
    }
}

fn poisoned() -> Error {
    Error::Backend("shared-base cache mutex poisoned".into())
}
fn changed() -> Error {
    Error::Package("base weights changed during shared-cache loading".into())
}

fn charge(tensors: &HashMap<String, Tensor>) -> Result<usize> {
    tensors.iter().try_fold(
        std::mem::size_of::<Entry>() + 32,
        |total, (name, tensor)| {
            if !tensor.device().is_cpu() {
                return Err(Error::Unsupported("shared bases support CPU only".into()));
            }
            tensor
                .elem_count()
                .checked_mul(tensor.dtype().size_in_bytes())
                .and_then(|bytes| bytes.checked_add(name.len()))
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<(String, Tensor)>() + 64))
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| Error::Package("shared-base tensor charge overflow".into()))
        },
    )
}

fn identity(files: &[PathBuf], dtype: DType, include_lm_head: bool) -> Result<[u8; 32]> {
    let mut digest = Sha256::new();
    digest.update(b"huncho-qwen-unmerged-cpu-base-v1\0");
    digest.update(format!("{dtype:?}:{include_lm_head}\0").as_bytes());
    let mut paths = files.to_vec();
    paths.sort();
    digest.update((paths.len() as u64).to_le_bytes());
    let mut buffer = vec![0u8; 1024 * 1024];
    for path in paths {
        // Only basenames affect tensor loading; identical relocated checkpoint
        // bytes can share storage without reusing an adapter or tokenizer.
        let name = path
            .file_name()
            .and_then(|v| v.to_str())
            .ok_or_else(|| Error::Package("base shard needs a UTF-8 file name".into()))?;
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        let mut file = File::open(&path)?;
        let metadata = file.metadata()?;
        let mut length = 0u64;
        digest.update(metadata.len().to_le_bytes());
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            length += read as u64;
            digest.update(&buffer[..read]);
        }
        let after = file.metadata()?;
        if length != metadata.len()
            || after.len() != metadata.len()
            || after.modified()? != metadata.modified()?
        {
            return Err(changed());
        }
    }
    Ok(digest.finalize().into())
}

/// CLI configuration is fixed for the process. Explicit library caches avoid
/// environment state and can have separate budgets/lifetimes.
pub(crate) fn configured() -> Result<Option<&'static BaseWeightCache>> {
    static CACHE: OnceLock<BaseWeightCache> = OnceLock::new();
    let budget = match std::env::var("HUNCHO_BASE_CACHE_BYTES") {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            Error::Package("HUNCHO_BASE_CACHE_BYTES must be a nonnegative integer".into())
        })?,
        Err(std::env::VarError::NotPresent) => 0,
        Err(_) => {
            return Err(Error::Package(
                "invalid HUNCHO_BASE_CACHE_BYTES value".into(),
            ))
        }
    };
    if budget == 0 && CACHE.get().is_none() {
        return Ok(None);
    }
    let cache = CACHE.get_or_init(|| BaseWeightCache::new(budget));
    if cache.budget != budget {
        return Err(Error::Package(
            "shared-base cache budget is fixed after first use; restart to change it".into(),
        ));
    }
    Ok(Some(cache))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;
    fn write(path: &std::path::Path, value: f32) {
        let weights: HashMap<String, Tensor> = HashMap::from([(
            "model.norm.weight".into(),
            Tensor::new(&[value, value + 1.], &Device::Cpu).unwrap(),
        )]);
        candle::safetensors::save(&weights, path).unwrap();
    }
    fn read(files: &[PathBuf]) -> Result<HashMap<String, Tensor>> {
        candle::safetensors::load(&files[0], &Device::Cpu)
            .map_err(|e| Error::Backend(e.to_string()))
    }
    #[test]
    fn immutable_hits_content_changes_dtype_readout_eviction_and_clear() {
        let root = tempfile::tempdir().unwrap();
        let files = vec![root.path().join("model.safetensors")];
        write(&files[0], 1.);
        let map = read(&files).unwrap();
        let cache = BaseWeightCache::new(charge(&map).unwrap());
        let first = cache
            .load(&files, DType::F32, false, || read(&files))
            .unwrap();
        let second = cache
            .load(&files, DType::F32, false, || {
                panic!("hit must not load tensors")
            })
            .unwrap();
        let (a, _) = first["model.norm.weight"].storage_and_layout();
        let (b, _) = second["model.norm.weight"].storage_and_layout();
        assert!(std::ptr::eq(&*a, &*b));
        drop((a, b));
        // Equal-size rewritten contents cannot reuse old tensors.
        write(&files[0], 3.);
        let changed = cache
            .load(&files, DType::F32, false, || read(&files))
            .unwrap();
        assert_eq!(
            changed["model.norm.weight"].to_vec1::<f32>().unwrap(),
            vec![3., 4.]
        );
        assert_eq!(
            first["model.norm.weight"].to_vec1::<f32>().unwrap(),
            vec![1., 2.]
        );
        cache
            .load(&files, DType::F32, true, || read(&files))
            .unwrap();
        cache
            .load(&files, DType::F16, true, || read(&files))
            .unwrap();
        let stats = cache.stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 4);
        assert_eq!(stats.evictions, 3);
        assert_eq!(stats.entries, 1);
        assert!(stats.charged_bytes <= cache.budget);
        cache.clear().unwrap();
        assert_eq!(cache.stats().unwrap().charged_bytes, 0);
        assert_eq!(
            first["model.norm.weight"].to_vec1::<f32>().unwrap(),
            vec![1., 2.]
        );
        let small = BaseWeightCache::new(1);
        small
            .load(&files, DType::F32, false, || read(&files))
            .unwrap();
        assert_eq!(small.stats().unwrap().bypasses, 1);
        assert_eq!(small.stats().unwrap().charged_bytes, 0);
    }
    #[test]
    fn a_source_change_during_materialization_is_never_published() {
        let root = tempfile::tempdir().unwrap();
        let files = vec![root.path().join("model.safetensors")];
        write(&files[0], 1.);
        let cache = BaseWeightCache::new(1 << 20);
        let result = cache.load(&files, DType::F32, false, || {
            let loaded = read(&files)?;
            write(&files[0], 2.);
            Ok(loaded)
        });
        assert!(result.unwrap_err().to_string().contains("changed"));
        assert_eq!(cache.stats().unwrap().entries, 0);
    }

    #[test]
    fn concurrent_misses_materialize_once_and_each_caller_owns_a_tensor_reference() {
        let root = tempfile::tempdir().unwrap();
        let files = vec![root.path().join("model.safetensors")];
        write(&files[0], 1.);
        let cache = BaseWeightCache::new(1 << 20);
        let loads = std::sync::atomic::AtomicUsize::new(0);
        let maps = std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        cache
                            .load(&files, DType::F32, false, || {
                                loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                read(&files)
                            })
                            .unwrap()
                    })
                })
                .collect();
            jobs.into_iter()
                .map(|job| job.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(loads.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(cache.stats().unwrap().hits, 3);
        cache.clear().unwrap();
        for map in maps {
            assert_eq!(
                map["model.norm.weight"].to_vec1::<f32>().unwrap(),
                vec![1., 2.]
            );
        }
    }
}
