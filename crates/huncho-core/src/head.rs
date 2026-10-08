//! Decision heads (PRD §6, `Heads`).
//!
//! Heads convert backend output (features or logits) into the per-candidate
//! logits that the calibration layer turns into probabilities. The head for a
//! family is declared in the manifest; `huncho convert` extracts the reference head
//! weights into a [`HeadParams`].

use crate::backend::ForwardOutput;
use crate::error::{Error, Result};
use crate::manifest::{Family, HeadKind};
use crate::prompt::Candidate;

/// A single linear layer (`y = W x + b`), row-major `[out_dim, in_dim]`.
#[derive(Debug, Clone)]
pub struct Linear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, weight: Vec<f32>, bias: Vec<f32>) -> Result<Linear> {
        let expected = in_dim * out_dim;
        if weight.len() != expected {
            return Err(Error::Package(format!(
                "linear weight has {} elements, expected {expected} (out={out_dim}, in={in_dim})",
                weight.len()
            )));
        }
        if bias.len() != out_dim {
            return Err(Error::Package(format!(
                "linear bias has {} elements, expected {out_dim}",
                bias.len()
            )));
        }
        Ok(Linear {
            in_dim,
            out_dim,
            weight,
            bias,
        })
    }

    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        if x.len() != self.in_dim {
            return Vec::new();
        }
        let mut out = self.bias.clone();
        for (o, value) in out.iter_mut().enumerate() {
            *value = self.project_row(x, o);
        }
        out
    }

    /// Project into caller-owned storage, retaining the scalar accumulation
    /// order used by `apply` so buffer reuse does not change calibration.
    pub fn apply_into(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != self.in_dim || out.len() != self.out_dim {
            return Err(Error::Backend(
                "linear input/output dimensions do not match".into(),
            ));
        }
        for (o, value) in out.iter_mut().enumerate() {
            *value = self.project_row(x, o);
        }
        Ok(())
    }

    fn project_row(&self, x: &[f32], row: usize) -> f32 {
        let base = row * self.in_dim;
        let mut acc = 0.0f32;
        for (&weight, &value) in self.weight[base..base + self.in_dim].iter().zip(x) {
            acc += weight * value;
        }
        self.bias[row] + acc
    }
}

/// The parameters a head needs. For the candidate-logit (F3) and the mock
/// backends this is empty: they read logits directly.
#[derive(Debug, Clone, Default)]
pub struct HeadParams {
    /// A linear projection shared across candidates (F1 option-marker, F2
    /// pointer, F4 slot are all linear/readout heads).
    pub linear: Option<Linear>,
    /// Offset for the pointer head (F2).
    pub pointer_offset: Option<usize>,
}

impl HeadParams {
    /// A shared linear projection mapping `hidden -> single logit` per
    /// candidate, used by the option-marker and pointer heads.
    pub fn scalar_linear(hidden_size: usize, weight: Vec<f32>, bias: f32) -> Result<HeadParams> {
        Ok(HeadParams {
            linear: Some(Linear::new(hidden_size, 1, weight, vec![bias])?),
            pointer_offset: None,
        })
    }
}

/// Borrow an index once. Ordered backend outputs need no allocation and permit
/// logarithmic lookup; legacy unordered outputs retain first-match behavior.
struct ReadoutIndex<'a, T> {
    keys: &'a [T],
    sorted: bool,
}

impl<'a, T: Ord> ReadoutIndex<'a, T> {
    fn new(keys: &'a [T]) -> Self {
        Self {
            keys,
            sorted: keys.windows(2).all(|pair| pair[0] <= pair[1]),
        }
    }

    fn get(&self, key: &T) -> Option<usize> {
        if self.sorted {
            let index = self.keys.partition_point(|value| value < key);
            (self.keys.get(index) == Some(key)).then_some(index)
        } else {
            self.keys.iter().position(|value| value == key)
        }
    }
}

fn row_for_position(index: &ReadoutIndex<'_, usize>, position: usize) -> Result<usize> {
    index.get(&position).ok_or_else(|| {
        Error::Backend(format!(
            "requested position {position} not present in forward output (positions {:?})",
            index.keys
        ))
    })
}

/// Compute per-candidate logits from a backend output.
///
/// * If the backend returns `Logits` (F3, and the mock backend), each
///   candidate's logit is `logits[position][code_id]`.
/// * If the backend returns `Features` (F1/F2/F4), the feature vector at each
///   candidate position is projected by `params.linear` (or, if no projection
///   is supplied, summarized by its mean as a deterministic fallback).
pub fn candidate_logits(
    family: Family,
    head_kind: HeadKind,
    output: &ForwardOutput,
    candidates: &[Candidate],
    params: &HeadParams,
) -> Result<Vec<f32>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let positions = ReadoutIndex::new(output.positions());
    match output {
        ForwardOutput::Logits { values, .. } => {
            let vocab = values.shape().last().copied().unwrap_or(0);
            let mut logits = Vec::with_capacity(candidates.len());
            for c in candidates {
                let row = row_for_position(&positions, c.position)?;
                let feat = values.row(row).map_err(|e| Error::Backend(e.to_string()))?;
                let code = c.code_id as usize;
                if code >= vocab {
                    return Err(Error::Backend(format!(
                        "candidate code {code} out of range for vocab {vocab}"
                    )));
                }
                logits.push(feat[code]);
            }
            Ok(logits)
        }
        ForwardOutput::SelectedLogits { codes, values, .. } => {
            if values.shape() != [output.positions().len(), codes.len()] {
                return Err(Error::Backend(
                    "selected-logit dimensions do not match positions/codes".into(),
                ));
            }
            let codes = ReadoutIndex::new(codes);
            let mut logits = Vec::with_capacity(candidates.len());
            for candidate in candidates {
                let row = row_for_position(&positions, candidate.position)?;
                let column = codes.get(&candidate.code_id).ok_or_else(|| {
                    Error::Backend(format!(
                        "candidate code {} missing from selected logits",
                        candidate.code_id
                    ))
                })?;
                logits.push(values.row(row).map_err(|e| Error::Backend(e.to_string()))?[column]);
            }
            Ok(logits)
        }
        ForwardOutput::Features { values, .. } => {
            let dim = values.shape().last().copied().unwrap_or(0);
            let mut logits = Vec::with_capacity(candidates.len());
            for c in candidates {
                let row = row_for_position(&positions, c.position)?;
                let feat = values.row(row).map_err(|e| Error::Backend(e.to_string()))?;
                logits.push(project_feature(family, head_kind, dim, feat, params)?);
            }
            Ok(logits)
        }
    }
}

fn project_feature(
    _family: Family,
    _head_kind: HeadKind,
    dim: usize,
    feat: &[f32],
    params: &HeadParams,
) -> Result<f32> {
    if let Some(linear) = &params.linear {
        if linear.out_dim != 1 {
            return Err(Error::Package(format!(
                "expected linear head to output 1 logit, got {}",
                linear.out_dim
            )));
        }
        if feat.len() != linear.in_dim {
            return Err(Error::Package(format!(
                "linear head expects {} features, got {}",
                linear.in_dim,
                feat.len()
            )));
        }
        return Ok(linear.project_row(feat, 0));
    }
    // Deterministic fallback when no head weights are present: the mean activation.
    if dim == 0 || feat.len() != dim {
        return Err(Error::Package(format!(
            "feature length {} does not match expected hidden size {dim}",
            feat.len()
        )));
    }
    Ok(feat.iter().sum::<f32>() / dim as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::CandidateKind;
    use crate::tensor::Tensor;

    fn cands(n: usize, position: usize, code: u32) -> Vec<Candidate> {
        (0..n)
            .map(|i| Candidate {
                kind: CandidateKind::Option,
                position,
                code_id: code + i as u32,
                label: i.to_string(),
                description: None,
                index: i,
            })
            .collect()
    }

    #[test]
    fn candidate_logit_reads_logits() {
        // 2 candidates share position 3, codes 7 and 8. Vocab 16.
        let mut data = vec![0.0f32; 4 * 16];
        data[3 * 16 + 7] = 2.5;
        data[3 * 16 + 8] = -1.5;
        let values = Tensor::new(vec![4, 16], data).unwrap();
        let out = ForwardOutput::Logits {
            positions: vec![0, 1, 2, 3],
            values,
        };
        let c = cands(2, 3, 7);
        let logits = candidate_logits(
            Family::F3,
            HeadKind::CandidateLogit,
            &out,
            &c,
            &HeadParams::default(),
        )
        .unwrap();
        assert_eq!(logits, vec![2.5, -1.5]);
    }

    #[test]
    fn candidate_logit_features_with_linear() {
        let hidden = 3;
        // W = [[2,0,0]] bias=1 => logit = 2*x0 + 1
        let w = vec![2.0, 0.0, 0.0];
        let params = HeadParams::scalar_linear(hidden, w, 1.0).unwrap();
        let mut data = vec![0.0f32; 2 * hidden];
        data[0] = 0.5; // row 0
        data[3] = 1.0; // row 1
        let values = Tensor::new(vec![2, hidden], data).unwrap();
        let out = ForwardOutput::Features {
            positions: vec![0, 1],
            values,
        };
        let c = vec![
            Candidate {
                kind: CandidateKind::Option,
                position: 0,
                code_id: 0,
                label: "0".into(),
                description: None,
                index: 0,
            },
            Candidate {
                kind: CandidateKind::Option,
                position: 1,
                code_id: 1,
                label: "1".into(),
                description: None,
                index: 1,
            },
        ];
        let logits =
            candidate_logits(Family::F1, HeadKind::OptionMarker, &out, &c, &params).unwrap();
        assert_eq!(logits, vec![2.0 * 0.5 + 1.0, 2.0 * 1.0 + 1.0]);
    }

    #[test]
    fn compact_logits_preserve_candidate_order_and_reject_missing_codes() {
        let output = ForwardOutput::SelectedLogits {
            positions: vec![2, 7],
            codes: vec![3, 11, 29],
            values: Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        };
        let mut candidates = cands(3, 7, 0);
        candidates[0].code_id = 29;
        candidates[1].code_id = 3;
        candidates[2].code_id = 11;
        candidates[2].position = 2;
        let logits = candidate_logits(
            Family::F3,
            HeadKind::CandidateLogit,
            &output,
            &candidates,
            &HeadParams::default(),
        )
        .unwrap();
        assert_eq!(logits, vec![6.0, 4.0, 2.0]);
        candidates[0].code_id = 30;
        assert!(candidate_logits(
            Family::F3,
            HeadKind::CandidateLogit,
            &output,
            &candidates,
            &HeadParams::default()
        )
        .is_err());
    }

    #[test]
    fn row_lookup_preserves_first_match_for_sorted_and_unordered_outputs() {
        for keys in [vec![2, 2, 7], vec![7, 2, 2]] {
            let lookup = ReadoutIndex::new(&keys);
            assert_eq!(lookup.get(&2), keys.iter().position(|&key| key == 2));
            assert_eq!(lookup.get(&8), None);
        }
    }

    #[test]
    fn reused_linear_and_scalar_projection_preserve_accumulation_order() {
        let linear = Linear::new(4, 1, vec![1e10, 1.0, -1e10, 0.25], vec![0.5]).unwrap();
        let input = [1.0, 1.0, 1.0, 2.0];
        let mut output = [99.0];
        linear.apply_into(&input, &mut output).unwrap();
        // Summing bias first or reassociating products would produce a different result.
        assert_eq!(output, [1.0]);
        let params = HeadParams {
            linear: Some(linear.clone()),
            pointer_offset: None,
        };
        assert_eq!(
            project_feature(Family::F1, HeadKind::OptionMarker, 4, &input, &params)
                .unwrap()
                .to_bits(),
            output[0].to_bits()
        );
        assert_eq!(linear.apply(&input), output);
        assert!(linear.apply_into(&input[..3], &mut output).is_err());
    }
}
