//! A [`Backend`] that always fails to forward. Used as a placeholder when a
//! model package has not yet been loaded, and in tests that exercise error
//! paths.

use s1_core::backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput};
use s1_core::error::{Error, Result};
use s1_core::manifest::BackendId;

/// A backend whose forward calls always error.
pub struct NullBackend {
    id: BackendId,
}

impl NullBackend {
    pub fn new() -> NullBackend {
        NullBackend {
            id: BackendId::Onnx,
        }
    }

    pub fn with_id(id: BackendId) -> NullBackend {
        NullBackend { id }
    }
}

impl Default for NullBackend {
    fn default() -> Self {
        NullBackend::new()
    }
}

impl Backend for NullBackend {
    fn id(&self) -> BackendId {
        self.id
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.id,
            dtype: "none".into(),
            max_context: 0,
            supports_fork: false,
            supports_lora: false,
            families: Vec::new(),
            extra: Default::default(),
        }
    }

    fn forward(&mut self, _input: ForwardInput) -> Result<ForwardOutput> {
        Err(Error::Unsupported("NullBackend has no loaded model".into()))
    }

    fn fork(&mut self, _handle: CacheHandle) -> Result<CacheHandle> {
        Err(Error::Unsupported("NullBackend cannot fork".into()))
    }
}
