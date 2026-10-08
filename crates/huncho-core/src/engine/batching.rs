use crate::backend::{BatchLimits, ForwardInput};

/// Greedy groups in caller order (engine supplies increasing lengths). Charge
/// the full padded rectangle, and always preserve oversized singletons intact.
pub(super) fn padded_groups(
    inputs: Vec<(usize, ForwardInput)>,
    budget: usize,
    percent: usize,
    limits: BatchLimits,
) -> Vec<Vec<(usize, ForwardInput)>> {
    let mut groups = Vec::new();
    let mut group: Vec<(usize, ForwardInput)> = Vec::new();
    let mut longest = 0;
    let mut logical = 0usize;
    let mut readouts = 0usize;
    for input in inputs {
        let length = input.1.tokens.len();
        let new_longest = longest.max(length);
        let physical = new_longest.checked_mul(group.len() + 1);
        let new_logical = logical.checked_add(length);
        let cost = input.1.positions.len().saturating_add(1);
        let new_readouts = readouts.checked_add(cost);
        let fits = physical
            .zip(new_logical)
            .is_some_and(|(physical, logical)| {
                physical <= budget
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
