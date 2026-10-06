//! Kev checkpoint metadata and trained pointer readout.
//!
//! Reads the upstream `head.pt` ZIP/pickle format as data using Candle's
//! restricted parser; no Python interpreter or pickle callables are executed.

use std::io::BufReader;
use std::path::Path;

use candle::pickle::{Object, PthTensors, Stack};
use candle::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use huncho_core::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct KevMetadata {
    pub base: String,
    pub base_revision: Option<String>,
    pub lora_rank: usize,
    pub head_dim: usize,
    pub temperature: f32,
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Package(format!("invalid Kev head.pt: {message}"))
}

impl KevMetadata {
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(invalid)?;
        let mut archive = zip::ZipArchive::new(BufReader::new(file)).map_err(invalid)?;
        let names: Vec<_> = archive
            .file_names()
            .filter(|n| n.ends_with("/data.pkl"))
            .map(str::to_owned)
            .collect();
        if names.len() != 1 {
            return Err(invalid("expected exactly one data.pkl"));
        }
        let mut stack = Stack::empty();
        stack
            .read_loop(&mut BufReader::new(
                archive.by_name(&names[0]).map_err(invalid)?,
            ))
            .map_err(invalid)?;
        let Object::Dict(fields) = stack.finalize().map_err(invalid)? else {
            return Err(invalid("expected a metadata dictionary"));
        };
        let get = |key: &str| {
            fields.iter().find_map(|(k, v)| match k {
                Object::Unicode(k) if k == key => Some(v),
                _ => None,
            })
        };
        let string = |key: &str| -> Result<Option<String>> {
            match get(key) {
                Some(Object::Unicode(v)) => Ok(Some(v.clone())),
                None | Some(Object::None) => Ok(None),
                _ => Err(invalid(format!("`{key}` must be a string"))),
            }
        };
        let integer = |key: &str, default: usize| -> Result<usize> {
            match get(key) {
                Some(Object::Int(v)) if *v >= 0 => Ok(*v as usize),
                Some(Object::Long(v)) if *v >= 0 => usize::try_from(*v).map_err(invalid),
                None => Ok(default),
                _ => Err(invalid(format!("`{key}` must be a non-negative integer"))),
            }
        };
        for flag in ["option_isolation", "special_embeddings"] {
            match get(flag) {
                None | Some(Object::Bool(false)) => (),
                Some(Object::Bool(true)) => {
                    return Err(Error::Unsupported(format!(
                        "Kev `{flag}=true` is not supported by the Candle backend"
                    )))
                }
                _ => return Err(invalid(format!("`{flag}` must be a boolean"))),
            }
        }
        if string("weights")?.as_deref().unwrap_or("lora") != "lora" {
            return Err(Error::Unsupported(
                "Kev full-weight checkpoints are not supported; expected a LoRA checkpoint".into(),
            ));
        }
        let temperature = match get("temperature") {
            Some(Object::Float(t)) => *t as f32,
            Some(Object::Int(t)) => *t as f32,
            None => 1.0,
            _ => return Err(invalid("`temperature` must be numeric")),
        };
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(invalid("temperature must be finite and positive"));
        }
        let base = string("base")?
            .filter(|s| !s.is_empty())
            .ok_or_else(|| invalid("missing base model"))?;
        let head_dim = integer("head_dim", 256)?;
        let lora_rank = integer("lora", 0)?;
        if head_dim == 0 || lora_rank == 0 {
            return Err(invalid("head_dim and LoRA rank must be positive"));
        }
        Ok(Self {
            base,
            base_revision: string("base_revision")?,
            head_dim,
            lora_rank,
            temperature,
        })
    }
}

pub(crate) struct PointerHead {
    q: Linear,
    k: Linear,
    scale: f64,
}

impl PointerHead {
    pub(crate) fn load(path: &Path, hidden_size: usize, device: &Device) -> Result<Self> {
        let meta = KevMetadata::load(path)?;
        let tensors = PthTensors::new(path, Some("head")).map_err(invalid)?;
        let tensor = |name: &str, shape: &[usize]| -> Result<Tensor> {
            let t = tensors
                .get(name)
                .map_err(invalid)?
                .ok_or_else(|| invalid(format!("missing `{name}`")))?;
            if t.dims() != shape {
                return Err(invalid(format!(
                    "`{name}` shape {:?}, expected {shape:?}",
                    t.dims()
                )));
            }
            let t = t.to_dtype(DType::F32).map_err(invalid)?;
            if !t
                .flatten_all()
                .map_err(invalid)?
                .to_vec1::<f32>()
                .map_err(invalid)?
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(invalid(format!("`{name}` contains non-finite weights")));
            }
            t.to_device(device).map_err(invalid)
        };
        let projection = |name: &str| -> Result<Linear> {
            Ok(Linear::new(
                tensor(&format!("{name}.weight"), &[meta.head_dim, hidden_size])?,
                Some(tensor(&format!("{name}.bias"), &[meta.head_dim])?),
            ))
        };
        Ok(Self {
            q: projection("q")?,
            k: projection("k")?,
            scale: 1.0 / (meta.head_dim as f64).sqrt(),
        })
    }

    /// Uncalibrated scores. Huncho applies the checkpoint temperature once,
    /// after this readout. The pointer projections always run in fp32.
    pub(crate) fn forward(&self, decide: &Tensor, options: &Tensor) -> candle::Result<Tensor> {
        let q = self.q.forward(&decide.to_dtype(DType::F32)?)?;
        let k = self.k.forward(&options.to_dtype(DType::F32)?)?;
        k.matmul(&q.t()?)?.affine(self.scale, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn metadata_file(temperature: f64, isolate: bool) -> tempfile::NamedTempFile {
        // Protocol-2 primitives in the same ZIP entry used by torch.save.
        // No Python runtime is needed to exercise malformed metadata.
        fn string(bytes: &mut Vec<u8>, value: &str) {
            bytes.push(b'X');
            bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        let mut bytes = b"\x80\x02}(".to_vec();
        string(&mut bytes, "base");
        string(&mut bytes, "fixture/qwen3.5");
        string(&mut bytes, "lora");
        bytes.extend_from_slice(b"K\x02");
        string(&mut bytes, "head_dim");
        bytes.extend_from_slice(b"K\x04");
        string(&mut bytes, "temperature");
        bytes.push(b'G');
        bytes.extend_from_slice(&temperature.to_be_bytes());
        string(&mut bytes, "option_isolation");
        bytes.push(if isolate { 0x88 } else { 0x89 });
        bytes.extend_from_slice(b"u.");
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let mut archive = zip::ZipWriter::new(&mut file);
        archive
            .start_file("head/data.pkl", zip::write::SimpleFileOptions::default())
            .unwrap();
        archive.write_all(&bytes).unwrap();
        archive.finish().unwrap();
        file
    }

    #[test]
    fn rejects_invalid_calibration_and_unsupported_encoding() {
        for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let file = metadata_file(temperature, false);
            assert!(KevMetadata::load(file.path())
                .unwrap_err()
                .to_string()
                .contains("temperature must be finite and positive"));
        }
        let file = metadata_file(1.0, true);
        assert!(KevMetadata::load(file.path())
            .unwrap_err()
            .to_string()
            .contains("option_isolation=true"));
    }

    #[test]
    fn rejects_head_backbone_dimension_mismatch() {
        let path = Path::new("tests/fixtures/tiny_kev/head.pt");
        let err = PointerHead::load(path, 8, &Device::Cpu)
            .err()
            .expect("must reject mismatched head");
        assert!(err.to_string().contains("q.weight"));
        assert!(err.to_string().contains("expected [4, 8]"));
    }
}
