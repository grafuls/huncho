//! Immutable CPU ONNX initializers for independent sessions. AddInitializer
//! shares runtime storage. External values also replace the graph's external
//! references before validation; that ORT step copies data into the graph.
//! The source graph bytes are retained verbatim; this parser never rewrites it.
use huncho_core::{Error, Result};
use ort::{
    session::builder::PrepackedWeights,
    value::{DynValue, Tensor},
};
use prost::{bytes::Bytes, Message};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path},
    sync::{Arc, Mutex},
};

pub(crate) struct SharedSource {
    pub model: Bytes,
    pub initializers: Vec<(String, Arc<DynValue>)>,
    pub external_names: BTreeSet<String>,
    pub prepacked: PrepackedWeights,
    pub creation: Mutex<()>,
    pub bytes: usize,
    pub sha256: String,
}

impl SharedSource {
    pub fn load(path: &Path) -> Result<Self> {
        let root = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .canonicalize()?;
        let model = Bytes::from(std::fs::read(path)?);
        let proto = Model::decode(model.clone())
            .map_err(|e| Error::Package(format!("invalid shared ONNX model: {e}")))?;
        if !proto.functions.is_empty()
            || !proto.training.is_empty()
            || !proto.configuration.is_empty()
        {
            return Err(unsupported(
                "local functions, training and multi-device graphs",
            ));
        }
        let graph = proto
            .graph
            .ok_or_else(|| Error::Package("ONNX model has no graph".into()))?;
        if !graph.sparse.is_empty() {
            return Err(unsupported("sparse initializers"));
        }
        for node in &graph.nodes {
            for attr in &node.attributes {
                if attr.graph.is_some()
                    || !attr.graphs.is_empty()
                    || attr.sparse.is_some()
                    || !attr.sparse_list.is_empty()
                {
                    return Err(unsupported("nested graphs and sparse attributes"));
                }
                for tensor in attr.tensor.iter().chain(&attr.tensors) {
                    if tensor.location != 0 || !tensor.external.is_empty() {
                        return Err(unsupported("external tensor attributes"));
                    }
                }
            }
        }
        let mut files = BTreeMap::new();
        let mut initializers = BTreeMap::new();
        let mut external_names = BTreeSet::new();
        let mut bytes = 0usize;
        for tensor in graph.initializers {
            if tensor.name.is_empty() || initializers.contains_key(&tensor.name) {
                return Err(Error::Package(
                    "ONNX initializer names must be nonempty and unique".into(),
                ));
            }
            if tensor.segment.is_some() {
                return Err(unsupported("segmented initializers"));
            }
            let shape = tensor
                .dims
                .iter()
                .map(|&dim| {
                    usize::try_from(dim)
                        .map_err(|_| Error::Package("invalid ONNX initializer dimension".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            let count = shape.iter().try_fold(1usize, |count, &dim| {
                count
                    .checked_mul(dim)
                    .ok_or_else(|| Error::Package("ONNX initializer size overflow".into()))
            })?;
            let raw = raw_data(&tensor, &root, &mut files)?;
            if tensor.location == 1 {
                external_names.insert(tensor.name.clone());
            }
            let (value, width) = value(&tensor, shape, count, raw.as_deref())?;
            bytes = bytes
                .checked_add(
                    count
                        .checked_mul(width)
                        .ok_or_else(|| Error::Package("ONNX initializer byte overflow".into()))?,
                )
                .ok_or_else(|| Error::Package("ONNX shared byte overflow".into()))?;
            initializers.insert(tensor.name, Arc::new(value));
        }
        if initializers.is_empty() {
            return Err(unsupported("graphs without dense initializers"));
        }
        let mut digest = Sha256::new();
        digest.update((model.len() as u64).to_le_bytes());
        digest.update(&model);
        for (name, data) in files {
            digest.update((name.len() as u64).to_le_bytes());
            digest.update(name.as_bytes());
            digest.update((data.len() as u64).to_le_bytes());
            digest.update(data);
        }
        Ok(Self {
            model,
            initializers: initializers.into_iter().collect(),
            external_names,
            prepacked: PrepackedWeights::new(),
            creation: Mutex::new(()),
            bytes,
            sha256: format!("{:x}", digest.finalize()),
        })
    }
}

fn unsupported(kind: &str) -> Error {
    Error::Unsupported(format!("shared ONNX CPU profile does not support {kind}"))
}

fn raw_data(
    tensor: &Initializer,
    root: &Path,
    files: &mut BTreeMap<String, Bytes>,
) -> Result<Option<Bytes>> {
    if tensor.location == 0 {
        if !tensor.external.is_empty() {
            return Err(Error::Package(
                "ONNX DEFAULT initializer has external metadata".into(),
            ));
        }
        return Ok(tensor.raw.clone());
    }
    if tensor.location != 1 || tensor.raw.is_some() || tensor.has_typed_data() {
        return Err(Error::Package(
            "invalid or mixed ONNX external initializer storage".into(),
        ));
    }
    let mut attrs = BTreeMap::new();
    for entry in &tensor.external {
        if attrs
            .insert(entry.key.as_str(), entry.value.as_str())
            .is_some()
        {
            return Err(Error::Package(
                "duplicate ONNX external metadata key".into(),
            ));
        }
    }
    if attrs
        .keys()
        .any(|key| !matches!(*key, "location" | "offset" | "length" | "checksum"))
    {
        return Err(unsupported("unknown external-data metadata"));
    }
    let location = *attrs
        .get("location")
        .ok_or_else(|| Error::Package("missing ONNX external location".into()))?;
    let relative = Path::new(location);
    if location.is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
    {
        return Err(Error::Package(
            "ONNX external location must remain inside the graph directory".into(),
        ));
    }
    if !files.contains_key(location) {
        let path = root.join(relative).canonicalize()?;
        if !path.starts_with(root) || !path.is_file() {
            return Err(Error::Package(
                "ONNX external file escapes graph directory".into(),
            ));
        }
        files.insert(location.to_string(), Bytes::from(std::fs::read(path)?));
    }
    let data = &files[location];
    let number = |key: &str| {
        attrs
            .get(key)
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| Error::Package(format!("invalid ONNX external {key}")))
            })
            .transpose()
    };
    let start = number("offset")?.unwrap_or(0);
    let end = match number("length")? {
        Some(length) => start
            .checked_add(length)
            .ok_or_else(|| Error::Package("ONNX external range overflow".into()))?,
        None => data.len(),
    };
    if start > end || end > data.len() {
        return Err(Error::Package("ONNX external range exceeds file".into()));
    }
    // Checksum is advisory ONNX metadata; the snapshot SHA-256 binds actual
    // full file bytes, independently of any exporter-provided checksum string.
    Ok(Some(data.slice(start..end)))
}

fn value(
    tensor: &Initializer,
    shape: Vec<usize>,
    count: usize,
    raw: Option<&[u8]>,
) -> Result<(DynValue, usize)> {
    if raw.is_some() && tensor.has_typed_data() {
        return Err(Error::Package("mixed ONNX raw/typed initializer".into()));
    }
    macro_rules! tensor_value {
        ($ty:ty, $field:ident, $width:expr) => {{
            if tensor.typed_fields() > usize::from(!tensor.$field.is_empty()) {
                return Err(Error::Package(
                    "ONNX initializer has data for another dtype".into(),
                ));
            }
            let values: Vec<$ty> = if let Some(raw) = raw {
                if count.checked_mul($width) != Some(raw.len()) {
                    return Err(Error::Package(
                        "ONNX raw initializer byte count mismatch".into(),
                    ));
                }
                raw.chunks_exact($width)
                    .map(|chunk| <$ty>::from_le_bytes(chunk.try_into().unwrap()))
                    .collect()
            } else {
                tensor.$field.clone()
            };
            if values.len() != count {
                return Err(Error::Package(
                    "ONNX typed initializer element count mismatch".into(),
                ));
            }
            (
                Tensor::from_array((shape, values))
                    .map_err(|e| Error::Backend(format!("ONNX shared initializer: {e}")))?
                    .into_dyn(),
                $width,
            )
        }};
    }
    Ok(match tensor.dtype {
        1 => tensor_value!(f32, floats, 4),
        6 => tensor_value!(i32, ints, 4),
        7 => tensor_value!(i64, longs, 8),
        11 => tensor_value!(f64, doubles, 8),
        9 => {
            if tensor.typed_fields() > usize::from(!tensor.ints.is_empty()) {
                return Err(Error::Package(
                    "ONNX bool initializer has mixed data".into(),
                ));
            }
            let values: Vec<i32> = if let Some(raw) = raw {
                if raw.len() != count {
                    return Err(Error::Package(
                        "ONNX bool initializer byte count mismatch".into(),
                    ));
                }
                raw.iter().map(|&v| i32::from(v)).collect()
            } else {
                tensor.ints.clone()
            };
            if values.len() != count || values.iter().any(|&v| !matches!(v, 0 | 1)) {
                return Err(Error::Package("invalid ONNX bool initializer".into()));
            }
            (
                Tensor::from_array((shape, values.iter().map(|&v| v != 0).collect::<Vec<_>>()))
                    .map_err(|e| Error::Backend(e.to_string()))?
                    .into_dyn(),
                1,
            )
        }
        dtype => {
            return Err(unsupported(&format!(
                "initializer dtype {dtype}; supported: FP32, FP64, INT32, INT64, BOOL"
            )))
        }
    })
}

// Read-only projection of the official ONNX v1.20.1 protobuf schema. Unknown
// fields are skipped by prost, while original bytes are always passed to ORT.
// https://github.com/onnx/onnx/blob/v1.20.1/onnx/onnx.proto
#[derive(Clone, Message)]
struct Model {
    #[prost(message, optional, tag = "7")]
    graph: Option<Graph>,
    #[prost(bytes = "bytes", repeated, tag = "20")]
    training: Vec<Bytes>,
    #[prost(bytes = "bytes", repeated, tag = "25")]
    functions: Vec<Bytes>,
    #[prost(bytes = "bytes", repeated, tag = "26")]
    configuration: Vec<Bytes>,
}
#[derive(Clone, Message)]
struct Graph {
    #[prost(message, repeated, tag = "1")]
    nodes: Vec<Node>,
    #[prost(message, repeated, tag = "5")]
    initializers: Vec<Initializer>,
    #[prost(bytes = "bytes", repeated, tag = "15")]
    sparse: Vec<Bytes>,
}
#[derive(Clone, Message)]
struct Node {
    #[prost(message, repeated, tag = "5")]
    attributes: Vec<Attribute>,
}
#[derive(Clone, Message)]
struct Attribute {
    #[prost(message, optional, tag = "5")]
    tensor: Option<Initializer>,
    #[prost(bytes = "bytes", optional, tag = "6")]
    graph: Option<Bytes>,
    #[prost(message, repeated, tag = "10")]
    tensors: Vec<Initializer>,
    #[prost(bytes = "bytes", repeated, tag = "11")]
    graphs: Vec<Bytes>,
    #[prost(bytes = "bytes", optional, tag = "22")]
    sparse: Option<Bytes>,
    #[prost(bytes = "bytes", repeated, tag = "23")]
    sparse_list: Vec<Bytes>,
}
#[derive(Clone, Message)]
struct Initializer {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    dtype: i32,
    #[prost(bytes = "bytes", optional, tag = "3")]
    segment: Option<Bytes>,
    #[prost(float, repeated, tag = "4")]
    floats: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    ints: Vec<i32>,
    #[prost(bytes = "bytes", repeated, tag = "6")]
    strings: Vec<Bytes>,
    #[prost(int64, repeated, tag = "7")]
    longs: Vec<i64>,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(bytes = "bytes", optional, tag = "9")]
    raw: Option<Bytes>,
    #[prost(double, repeated, tag = "10")]
    doubles: Vec<f64>,
    #[prost(uint64, repeated, tag = "11")]
    unsigned: Vec<u64>,
    #[prost(message, repeated, tag = "13")]
    external: Vec<Entry>,
    #[prost(int32, tag = "14")]
    location: i32,
}
impl Initializer {
    fn typed_fields(&self) -> usize {
        [
            !self.floats.is_empty(),
            !self.ints.is_empty(),
            !self.strings.is_empty(),
            !self.longs.is_empty(),
            !self.doubles.is_empty(),
            !self.unsigned.is_empty(),
        ]
        .into_iter()
        .filter(|v| *v)
        .count()
    }
    fn has_typed_data(&self) -> bool {
        self.typed_fields() > 0
    }
}
#[derive(Clone, Message)]
struct Entry {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_snapshot_rejects_ambiguous_names_dimensions_and_nested_graphs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid.onnx");
        let weight = Initializer {
            name: "weight".into(),
            dtype: 1,
            dims: vec![1],
            floats: vec![1.],
            ..Default::default()
        };
        let check = |graph: Graph| {
            std::fs::write(
                &path,
                Model {
                    graph: Some(graph),
                    ..Default::default()
                }
                .encode_to_vec(),
            )
            .unwrap();
            assert!(SharedSource::load(&path).is_err());
        };
        check(Graph {
            initializers: vec![weight.clone(), weight.clone()],
            ..Default::default()
        });
        check(Graph {
            initializers: vec![Initializer {
                name: "".into(),
                ..weight.clone()
            }],
            ..Default::default()
        });
        check(Graph {
            initializers: vec![Initializer {
                dims: vec![-1],
                ..weight.clone()
            }],
            ..Default::default()
        });
        check(Graph {
            initializers: vec![Initializer {
                dims: vec![i64::MAX, 4],
                ..weight.clone()
            }],
            ..Default::default()
        });
        check(Graph {
            initializers: vec![weight],
            nodes: vec![Node {
                attributes: vec![Attribute {
                    graph: Some(Bytes::new()),
                    ..Default::default()
                }],
            }],
            ..Default::default()
        });
    }

    #[test]
    fn external_ranges_reject_traversal_duplicates_overflow_and_mixed_storage() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("data"), [0u8, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        let make = |location: &str, offset: &str, length: &str| Initializer {
            location: 1,
            external: [
                ("location", location),
                ("offset", offset),
                ("length", length),
            ]
            .into_iter()
            .map(|(key, value)| Entry {
                key: key.into(),
                value: value.into(),
            })
            .collect(),
            ..Default::default()
        };
        let mut files = BTreeMap::new();
        assert_eq!(
            raw_data(&make("data", "2", "4"), dir.path(), &mut files)
                .unwrap()
                .unwrap()
                .as_ref(),
            &[2, 3, 4, 5]
        );
        for (loc, off, len) in [
            ("../data", "0", "4"),
            ("/data", "0", "4"),
            ("data", "4", "9"),
            ("data", "-1", "4"),
            ("data", "18446744073709551615", "8"),
        ] {
            assert!(raw_data(&make(loc, off, len), dir.path(), &mut files).is_err());
        }
        let mut duplicate = make("data", "0", "4");
        duplicate.external.push(duplicate.external[0].clone());
        assert!(raw_data(&duplicate, dir.path(), &mut files).is_err());
        let mut mixed = make("data", "0", "4");
        mixed.raw = Some(Bytes::new());
        assert!(raw_data(&mixed, dir.path(), &mut files).is_err());
        let mut missing = make("data", "0", "4");
        missing.location = 0;
        assert!(raw_data(&missing, dir.path(), &mut files).is_err());
    }

    #[test]
    fn typed_values_preserve_raw_bits_and_reject_unsupported_or_malformed_data() {
        let tensor = Initializer {
            dtype: 1,
            ..Default::default()
        };
        let bits = [0x80000000u32, 0x7fc01234, 0x3f800000];
        let raw: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (actual, _) = value(&tensor, vec![3], 3, Some(&raw)).unwrap();
        assert_eq!(
            actual
                .try_extract_tensor::<f32>()
                .unwrap()
                .1
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
        assert!(value(&tensor, vec![2], 2, Some(&raw)).is_err());
        assert!(value(
            &Initializer {
                dtype: 1,
                ints: vec![1],
                ..Default::default()
            },
            vec![1],
            1,
            None
        )
        .is_err());
        assert!(value(
            &Initializer {
                dtype: 10,
                ..Default::default()
            },
            vec![1],
            1,
            Some(&[0, 0])
        )
        .is_err());
        assert!(value(
            &Initializer {
                dtype: 9,
                ..Default::default()
            },
            vec![1],
            1,
            Some(&[2])
        )
        .is_err());
    }
}
