//! A minimal dense tensor used by the engine to exchange data with backends.
//!
//! The engine intentionally avoids pulling in a heavy tensor library. Backends
//! convert their native tensors into this representation at the boundary, and
//! the heads/calibration math operates on `&[f32]` with explicit shapes.

use crate::error::{Error, Result};

/// The element type of a tensor. v1 stores everything as `f32` at the engine
/// boundary; backends downcast to their native dtype internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    Bf16,
}

impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::Bf16 => 2,
        }
    }
}

/// A dense, contiguous, row-major tensor of `f32` values.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
    dtype: DType,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Tensor> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(Error::Package(format!(
                "tensor shape {shape:?} expects {expected} elements, got {}",
                data.len()
            )));
        }
        Ok(Tensor {
            shape,
            data,
            dtype: DType::F32,
        })
    }

    pub fn zeros(shape: Vec<usize>) -> Tensor {
        let len = shape.iter().product();
        Tensor {
            shape,
            data: vec![0.0; len],
            dtype: DType::F32,
        }
    }

    pub fn from_dim1(data: Vec<f32>) -> Tensor {
        let len = data.len();
        Tensor {
            shape: vec![len],
            data,
            dtype: DType::F32,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Index into the flat data.
    pub fn get(&self, idx: usize) -> Option<f32> {
        self.data.get(idx).copied()
    }

    /// Return a 1-D slice for a single row `i` (row-major). This is the common
    /// case for per-position hidden state / logit vectors.
    pub fn row(&self, i: usize) -> Result<&[f32]> {
        if self.shape.len() != 2 {
            return Err(Error::Package(format!(
                "row() requires a 2-D tensor, got shape {:?}",
                self.shape
            )));
        }
        let cols = self.shape[1];
        let start = i * cols;
        let end = start + cols;
        self.data
            .get(start..end)
            .ok_or_else(|| Error::Package(format!("row {i} out of range for shape {:?}", self.shape)))
    }

    /// Reshape to target shape, validating element count.
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor> {
        let expected: usize = shape.iter().product();
        if expected != self.numel() {
            return Err(Error::Package(format!(
                "cannot reshape {:?} ({} elements) to {:?}",
                self.shape, self.numel(), shape
            )));
        }
        Ok(Tensor {
            shape,
            data: self.data.clone(),
            dtype: self.dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_access() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(t.row(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(t.row(1).unwrap(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn shape_mismatch() {
        assert!(Tensor::new(vec![2, 3], vec![1.0, 2.0]).is_err());
    }
}
