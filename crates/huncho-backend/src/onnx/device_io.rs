//! Stable owned I/O allocations per exact shape. No captured allocation is
//! evicted: ORT retains captured graphs for the entire session lifetime.
use huncho_core::{Error, Result};
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, RunOptions, Session},
    value::{Tensor, TensorValueType},
};
use std::{collections::BTreeMap, sync::Mutex};
const MAX_SHAPES: usize = 32;
pub(super) const MAX_BYTES: usize = 512 << 20;
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    inputs: Vec<(String, Vec<i64>)>,
    output_name: String,
    output_shape: Vec<usize>,
}
struct Cell {
    inputs: Vec<Tensor<i64>>,
    output: Tensor<f32>, // CPU readback; device output is owned by binding.
    binding: Mutex<IoBinding>,
    run_options: Option<RunOptions>,
    #[cfg(test)]
    output_address: usize,
    // ORT values created by Tensor::new do not retain their allocator. Keep
    // it after every tensor/binding in drop order, including failed runs.
    _allocator: Mutex<Allocator>,
}
pub(super) struct State {
    device: AllocationDevice,
    device_id: i32,
    budget: usize,
    capture: bool,
    bytes: usize,
    cells: BTreeMap<Key, Cell>,
    poisoned: bool,
}
fn failed(error: impl std::fmt::Display) -> Error {
    Error::Backend(format!("ONNX stable I/O: {error}"))
}
fn bytes(shape: &[usize], width: usize) -> Result<usize> {
    shape
        .iter()
        .try_fold(width, |n, &d| n.checked_mul(d))
        .filter(|n| *n > 0)
        .ok_or_else(|| failed("nonempty bounded shapes required"))
}
impl State {
    pub(super) fn cuda(device_id: i32, budget: usize, capture: bool) -> Self {
        Self {
            device: AllocationDevice::CUDA,
            device_id,
            budget,
            capture,
            bytes: 0,
            cells: BTreeMap::new(),
            poisoned: false,
        }
    }
    pub(super) fn run<'a>(
        &'a mut self,
        session: &mut Session,
        inputs: &[(String, Tensor<i64>)],
        output_name: &str,
        output_shape: Vec<usize>,
    ) -> Result<&'a Tensor<f32>> {
        if self.poisoned {
            return Err(failed(
                "previous copy/run/readback failed; reload this backend",
            ));
        }
        let key = Key {
            inputs: inputs
                .iter()
                .map(|(n, t)| (n.clone(), t.shape().to_vec()))
                .collect(),
            output_name: output_name.into(),
            output_shape,
        };
        if !self.cells.contains_key(&key) {
            if self.cells.len() >= MAX_SHAPES {
                return Err(failed(
                    "32 stable shape slots exhausted; reload or use ordinary execution",
                ));
            }
            // Charge device inputs/output, CPU readback and conservative metadata.
            // This excludes transient caller input tensors and ORT graph/workspace storage.
            let mut charge = bytes(&key.output_shape, 8)?
                .checked_add(4096)
                .ok_or_else(|| failed("byte overflow"))?;
            for (_, t) in inputs {
                charge = charge
                    .checked_add(
                        t.shape()
                            .num_elements()
                            .checked_mul(8)
                            .ok_or_else(|| failed("input byte overflow"))?,
                    )
                    .ok_or_else(|| failed("byte overflow"))?;
            }
            if charge > self.budget.saturating_sub(self.bytes) {
                return Err(failed(
                    "stable I/O byte budget exhausted; no allocation was evicted",
                ));
            }
            let allocator = Allocator::new(
                session,
                MemoryInfo::new(
                    self.device,
                    self.device_id,
                    AllocatorType::Device,
                    MemoryType::Default,
                )
                .map_err(failed)?,
            )
            .map_err(failed)?;
            let device_inputs = inputs
                .iter()
                .map(|(_, t)| Tensor::<i64>::new(&allocator, t.shape().to_vec()).map_err(failed))
                .collect::<Result<Vec<_>>>()?;
            let device_output =
                Tensor::<f32>::new(&allocator, key.output_shape.clone()).map_err(failed)?;
            #[cfg(test)]
            let output_address = device_output.data_ptr() as usize;
            let output = Tensor::<f32>::new(&Allocator::default(), key.output_shape.clone())
                .map_err(failed)?;
            let mut binding = session.create_binding().map_err(failed)?;
            for ((name, _), value) in inputs.iter().zip(&device_inputs) {
                binding.bind_input(name, value).map_err(failed)?;
            }
            binding
                .bind_output(output_name, device_output)
                .map_err(failed)?;
            let run_options = if self.capture {
                let mut options = RunOptions::new().map_err(failed)?;
                options
                    .set("gpu_graph_id", self.cells.len().to_string())
                    .map_err(failed)?;
                Some(options)
            } else {
                None
            };
            self.cells.insert(
                key.clone(),
                Cell {
                    inputs: device_inputs,
                    output,
                    binding: Mutex::new(binding),
                    run_options,
                    #[cfg(test)]
                    output_address,
                    _allocator: Mutex::new(allocator),
                },
            );
            self.bytes += charge;
        }
        let cell = self.cells.get_mut(&key).unwrap();
        let result = (|| -> Result<()> {
            for ((_, source), target) in inputs.iter().zip(&mut cell.inputs) {
                source.copy_into(target).map_err(failed)?; // synchronous, exact stable target
            }
            let binding = cell.binding.get_mut().map_err(failed)?;
            let outputs = match &cell.run_options {
                Some(options) => session.run_binding_with_options(binding, options),
                None => session.run_binding(binding),
            }
            .map_err(failed)?;
            binding.synchronize_outputs().map_err(failed)?;
            let value = outputs
                .get(output_name)
                .ok_or_else(|| failed("missing bound output"))?
                .downcast_ref::<TensorValueType<f32>>()
                .map_err(failed)?;
            value.copy_into(&mut cell.output).map_err(failed)?;
            if cell
                .output
                .try_extract_tensor::<f32>()
                .map_err(failed)?
                .1
                .iter()
                .any(|v| !v.is_finite())
            {
                return Err(failed("nonfinite output; reload this backend"));
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.poisoned = true;
            return Err(error);
        }
        Ok(&cell.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cpu(budget: usize) -> State {
        let mut state = State::cuda(0, budget, false);
        state.device = AllocationDevice::CPU;
        state
    }
    fn graph() -> Session {
        Session::builder()
            .unwrap()
            .with_no_environment_execution_providers()
            .unwrap()
            .commit_from_file(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/tiny_encoder.onnx"
            ))
            .unwrap()
    }
    fn inputs(values: Vec<i64>) -> Vec<(String, Tensor<i64>)> {
        let n = values.len();
        vec![
            (
                "input_ids".into(),
                Tensor::from_array(([1, n], values)).unwrap(),
            ),
            (
                "attention_mask".into(),
                Tensor::from_array(([1, n], vec![1i64; n])).unwrap(),
            ),
        ]
    }
    #[test]
    fn actual_cpu_io_uses_original_addresses_for_changed_data_and_keeps_owned_copies() {
        let mut session = graph();
        let mut state = cpu(1 << 20);
        let first = state
            .run(
                &mut session,
                &inputs(vec![1, 2, 3]),
                "last_hidden_state",
                vec![1, 3, 8],
            )
            .unwrap();
        let owned = first.extract_tensor().1.to_vec();
        let address = first.data_ptr() as usize;
        let cell = state.cells.values().next().unwrap();
        let input_addresses: Vec<_> = cell.inputs.iter().map(|t| t.data_ptr() as usize).collect();
        let bound_address = cell.output_address;
        let second = state
            .run(
                &mut session,
                &inputs(vec![3, 2, 1]),
                "last_hidden_state",
                vec![1, 3, 8],
            )
            .unwrap();
        assert_eq!(second.data_ptr() as usize, address);
        assert_ne!(second.extract_tensor().1, owned);
        let cell = state.cells.values().next().unwrap();
        assert_eq!(
            input_addresses,
            cell.inputs
                .iter()
                .map(|t| t.data_ptr() as usize)
                .collect::<Vec<_>>()
        );
        let binding = cell.binding.lock().unwrap();
        let outputs = session.run_binding(&binding).unwrap();
        assert_eq!(
            outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .unwrap()
                .1
                .as_ptr() as usize,
            bound_address
        );
        drop(outputs);
        drop(binding);
        let expected = session.run(inputs(vec![1, 2, 3])).unwrap();
        assert_eq!(
            expected["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .unwrap()
                .1,
            owned
        );
    }
    #[test]
    fn slots_and_bytes_are_bounded_and_failures_poison_without_eviction_or_reallocation() {
        let mut session = graph();
        let mut state = cpu(1 << 20);
        for n in 1..=MAX_SHAPES {
            state
                .run(
                    &mut session,
                    &inputs(vec![1; n]),
                    "last_hidden_state",
                    vec![1, n, 8],
                )
                .unwrap();
        }
        let charged = state.bytes;
        assert!(state
            .run(
                &mut session,
                &inputs(vec![1; 33]),
                "last_hidden_state",
                vec![1, 33, 8]
            )
            .is_err());
        assert_eq!(state.cells.len(), MAX_SHAPES);
        assert_eq!(state.bytes, charged);
        assert!(!state.poisoned);
        state
            .run(
                &mut session,
                &inputs(vec![2]),
                "last_hidden_state",
                vec![1, 1, 8],
            )
            .unwrap();
        let mut tiny = cpu(1);
        assert!(tiny
            .run(
                &mut session,
                &inputs(vec![1]),
                "last_hidden_state",
                vec![1, 1, 8]
            )
            .is_err());
        assert!(tiny.cells.is_empty());
        assert_eq!(tiny.bytes, 0);
        let mut failed = cpu(1 << 20);
        assert!(failed
            .run(
                &mut session,
                &inputs(vec![i64::MAX]),
                "last_hidden_state",
                vec![1, 1, 8]
            )
            .is_err());
        assert!(failed.poisoned);
        assert_eq!(failed.cells.len(), 1);
        assert!(failed
            .run(
                &mut session,
                &inputs(vec![1]),
                "last_hidden_state",
                vec![1, 1, 8]
            )
            .unwrap_err()
            .to_string()
            .contains("reload"));
    }
}
