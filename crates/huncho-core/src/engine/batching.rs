use crate::backend::{BatchLimits, ForwardInput};

/// Grouping needs dimensions only. Owned native inputs and borrowed external
/// readouts share the same budget/padding algorithm without copying payloads.
pub(super) trait BatchShape {
    fn token_len(&self) -> usize;
    fn readout_rows(&self) -> usize;
}

impl BatchShape for ForwardInput {
    fn token_len(&self) -> usize {
        self.tokens.len()
    }
    fn readout_rows(&self) -> usize {
        self.positions.len().saturating_add(1)
    }
}

/// Greedy groups in caller order (engine supplies increasing lengths). Charge
/// the full padded rectangle, and always preserve oversized singletons intact.
pub(super) fn padded_groups<T: BatchShape>(
    inputs: Vec<(usize, T)>,
    budget: usize,
    percent: usize,
    limits: BatchLimits,
) -> Vec<Vec<(usize, T)>> {
    padded_groups_with_prefix(inputs, budget, percent, limits, 0)
}

/// Charge complete KV contexts, but bound padding against only newly submitted
/// suffix positions. Prefix residency cannot hide padding cost.
pub(super) fn padded_groups_with_prefix<T: BatchShape>(
    inputs: Vec<(usize, T)>,
    budget: usize,
    percent: usize,
    limits: BatchLimits,
    prefix: usize,
) -> Vec<Vec<(usize, T)>> {
    let mut groups = Vec::new();
    let mut group: Vec<(usize, T)> = Vec::new();
    let mut longest = 0;
    let mut logical = 0usize;
    let mut readouts = 0usize;
    for input in inputs {
        let length = input.1.token_len();
        let new_longest = longest.max(length);
        let physical = new_longest.checked_mul(group.len() + 1);
        let workspace = new_longest
            .checked_add(prefix)
            .and_then(|n| n.checked_mul(group.len() + 1));
        let new_logical = logical.checked_add(length);
        let cost = input.1.readout_rows();
        let new_readouts = readouts.checked_add(cost);
        let fits = physical
            .zip(new_logical)
            .is_some_and(|(physical, logical)| {
                workspace.is_some_and(|n| n <= budget)
                    && new_readouts
                        .is_some_and(|n| limits.max_readouts.map_or(true, |max| n <= max))
                    && (physical - logical) as u128 * 100 <= physical as u128 * percent as u128
            });
        if !group.is_empty() && (group.len() == limits.max_rows || !fits) {
            groups.push(std::mem::take(&mut group));
            longest = 0;
            logical = 0;
            readouts = 0;
        }
        longest = longest.max(length);
        logical += length;
        readouts = readouts.saturating_add(cost);
        group.push(input);
    }
    if !group.is_empty() {
        groups.push(group);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn groups_bound_actual_rectangles_padding_and_rows_without_truncation() {
        let lengths = [1, 2, 3, 3, 4, 5, 7, 9, 10, 35, 80];
        for percent in [0, 1, 25, 100] {
            let inputs = lengths
                .into_iter()
                .enumerate()
                .map(|(i, n)| (i, ForwardInput::new(vec![1; n], vec![n - 1])))
                .collect();
            let groups = padded_groups(inputs, 64, percent, BatchLimits::default());
            let mut seen = Vec::new();
            for group in groups {
                let rectangle =
                    group.len() * group.iter().map(|(_, i)| i.tokens.len()).max().unwrap();
                let logical: usize = group.iter().map(|(_, i)| i.tokens.len()).sum();
                assert!(group.len() <= 64);
                if group.len() > 1 {
                    assert!(rectangle <= 64);
                    assert!((rectangle - logical) * 100 <= rectangle * percent);
                }
                for (index, input) in group {
                    assert_eq!(input.tokens.len(), lengths[index]);
                    assert_eq!(input.positions, [lengths[index] - 1]);
                    seen.push(index);
                }
            }
            assert_eq!(seen, (0..lengths.len()).collect::<Vec<_>>());
        }
        let many = (0..129)
            .map(|i| (i, ForwardInput::new(vec![1], vec![0])))
            .collect();
        assert_eq!(
            padded_groups(many, 4096, 25, BatchLimits::default())
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            [64, 64, 1]
        );
    }

    #[test]
    fn cached_padding_bounds_suffix_work_separately_from_full_workspace() {
        let input = || {
            vec![
                (0, ForwardInput::new(vec![1; 10], vec![0])),
                (1, ForwardInput::new(vec![1; 20], vec![0])),
            ]
        };
        let limits = BatchLimits::default();
        // 10 padded positions / 40 submitted suffixes exceeds 24%, even
        // though 10 / 240 complete KV slots would look cheap.
        assert_eq!(
            padded_groups_with_prefix(input(), 240, 24, limits, 100).len(),
            2
        );
        assert_eq!(
            padded_groups_with_prefix(input(), 240, 25, limits, 100).len(),
            1
        );
        assert_eq!(
            padded_groups_with_prefix(input(), 239, 25, limits, 100).len(),
            2
        );
    }

    #[test]
    fn native_readout_and_sequence_limits_split_without_dropping_candidates() {
        let inputs = (0..7)
            .map(|i| (i, ForwardInput::new(vec![1; 256], vec![3; 99])))
            .collect();
        let groups = padded_groups(
            inputs,
            8192,
            0,
            BatchLimits {
                max_rows: 4,
                max_readouts: Some(256),
            },
        );
        assert_eq!(
            groups.iter().map(Vec::len).collect::<Vec<_>>(),
            [2, 2, 2, 1]
        );
        assert_eq!(
            groups
                .into_iter()
                .flatten()
                .map(|(i, _)| i)
                .collect::<Vec<_>>(),
            (0..7).collect::<Vec<_>>()
        );
        let inputs = (0..3)
            .map(|i| (i, ForwardInput::new(vec![1; 3], vec![1; 300])))
            .collect();
        assert_eq!(
            padded_groups(
                inputs,
                8192,
                0,
                BatchLimits {
                    max_rows: 4,
                    max_readouts: Some(256)
                }
            )
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
            [1, 1, 1]
        );
    }
}
