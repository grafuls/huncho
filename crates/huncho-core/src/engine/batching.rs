use crate::backend::ForwardInput;

/// Greedy groups in caller order (engine supplies increasing lengths). Charge
/// the full padded rectangle, and always preserve oversized singletons intact.
pub(super) fn padded_groups(
    inputs: Vec<(usize, ForwardInput)>,
    budget: usize,
    percent: usize,
) -> Vec<Vec<(usize, ForwardInput)>> {
    let mut groups = Vec::new();
    let mut group: Vec<(usize, ForwardInput)> = Vec::new();
    let mut longest = 0;
    let mut logical = 0usize;
    for input in inputs {
        let length = input.1.tokens.len();
        let new_longest = longest.max(length);
        let physical = new_longest.checked_mul(group.len() + 1);
        let new_logical = logical.checked_add(length);
        let fits = physical
            .zip(new_logical)
            .is_some_and(|(physical, logical)| {
                physical <= budget
                    && (physical - logical) as u128 * 100 <= physical as u128 * percent as u128
            });
        if !group.is_empty() && (group.len() == 64 || !fits) {
            groups.push(std::mem::take(&mut group));
            longest = 0;
            logical = 0;
        }
        longest = longest.max(length);
        logical += length;
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
            let groups = padded_groups(inputs, 64, percent);
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
            padded_groups(many, 4096, 25)
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            [64, 64, 1]
        );
    }
}
