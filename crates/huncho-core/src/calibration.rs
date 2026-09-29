//! Calibration math: temperature scaling, softmax, and confidence.
//!
//! This is the core differentiator of `huncho` (PRD §6.3). The engine guarantees
//! that calibrated probabilities stay calibrated across backends and
//! quantizations by applying per-(backend, dtype) temperatures and reporting
//! confidence using the documented Jev definition.

use crate::error::{Error, Result};
use crate::manifest::ConfidenceDef;

/// Laya's `temp_bucket` size segment for a given option count.
/// Buckets: `2`, `3-5`, `6-10`, `11+`.
pub fn bucket_size(n_options: usize) -> &'static str {
    if n_options <= 2 {
        "2"
    } else if n_options <= 5 {
        "3-5"
    } else if n_options <= 10 {
        "6-10"
    } else {
        "11+"
    }
}

/// Numerically stable softmax over a logit vector.
///
/// Returns a probability distribution that sums to 1.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    softmax_temperature(logits, 1.0)
}

/// Softmax with temperature scaling.
///
/// `temperature <= 0` is rejected by the caller via [`softmax_temperature`];
/// this function panics only on NaN/Inf inputs.
pub fn softmax_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let inv_t = temperature.recip();
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for &l in logits {
        let e = ((l - max) * inv_t).exp() as f64;
        exp.push(e);
        sum += e;
    }
    exp.into_iter().map(|e| (e / sum) as f32).collect()
}

/// Numerically stable log-softmax.
pub fn log_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let inv_t = temperature.recip();
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    let mut out = Vec::with_capacity(logits.len());
    for &l in logits {
        let e = ((l - max) * inv_t).exp() as f64;
        sum += e;
        out.push(l);
    }
    let log_sum = sum.ln();
    out.into_iter()
        .map(|l| (l - max) * inv_t - log_sum as f32)
        .collect()
}

/// The argmax index of a vector.
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

/// Shannon entropy in bits.
pub fn entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0f64;
    for &p in probs {
        if p > 0.0 {
            h += p as f64 * (p as f64).log2();
        }
    }
    -h as f32
}

/// Jev's peak-based confidence: `(p_max - 1/n) / (1 - 1/n)`.
///
/// For a uniform distribution this is 0; for a single peak it is 1. Matches the
/// reference values in the TypeSafe docs (e.g. `[0.61, 0.35, 0.04]` -> 0.42,
/// `[0, 0.57, 0.43]` -> 0.35).
pub fn confidence_peak(probs: &[f32]) -> f32 {
    let n = probs.len();
    if n <= 1 {
        return probs.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
    }
    let p_max = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let uniform = 1.0 / n as f32;
    let denom = 1.0 - uniform;
    if denom <= 0.0 {
        return 1.0;
    }
    ((p_max - uniform) / denom).clamp(0.0, 1.0)
}

/// Normalized entropy-based confidence: `1 - H(p) / log2(n)`.
///
/// This is the Laya definition; a uniform distribution gives 0 and a single
/// peak gives 1.
pub fn confidence_entropy(probs: &[f32]) -> f32 {
    let n = probs.len();
    if n <= 1 {
        return probs.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
    }
    let h = entropy(probs);
    let max_h = (n as f32).log2();
    if max_h <= 0.0 {
        return 1.0;
    }
    (1.0 - h / max_h).clamp(0.0, 1.0)
}

/// Compute confidence for a probability vector using the given definition.
pub fn confidence(probs: &[f32], def: &ConfidenceDef) -> f32 {
    match def {
        ConfidenceDef::Peak => confidence_peak(probs),
        ConfidenceDef::Entropy => confidence_entropy(probs),
        ConfidenceDef::Custom(_) => confidence_peak(probs),
    }
}

/// Apply temperature and softmax to a logit vector, returning calibrated
/// probabilities. Validates temperature is positive.
pub fn calibrate(logits: &[f32], temperature: f32) -> Result<Vec<f32>> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(Error::Calibration(format!(
            "temperature must be positive and finite, got {temperature}"
        )));
    }
    Ok(softmax_temperature(logits, temperature))
}

/// Fit a single scalar temperature that minimizes NLL on `(logits, target)`
/// pairs via scalar-search. `target` is the one-hot-ish target index per row.
///
/// Returns `(temperature, nll)`.
pub fn fit_temperature(rows: &[Vec<f32>], targets: &[usize]) -> Result<(f32, f64)> {
    if rows.len() != targets.len() || rows.is_empty() {
        return Err(Error::Calibration(
            "fit_temperature requires matching non-empty rows/targets".into(),
        ));
    }
    // Scalar search over log-space temperature.
    let nll = |t: f64| -> f64 {
        // `t` is in log10 space; convert to an actual (positive) temperature.
        let temp = 10f64.powf(t) as f32;
        let mut total = 0.0;
        for (row, &target) in rows.iter().zip(targets) {
            let probs = log_softmax(row, temp);
            let lp = probs.get(target).copied().unwrap_or(f32::NEG_INFINITY) as f64;
            total -= lp;
        }
        total / rows.len() as f64
    };

    // Golden-section search on log10 temperature in [-2, 2].
    let mut lo = -2.0f64;
    let mut hi = 2.0f64;
    let gr = (5.0f64.sqrt() - 1.0) / 2.0;
    let mut a = hi - gr * (hi - lo);
    let mut b = lo + gr * (hi - lo);
    let mut fa = nll(a);
    let mut fb = nll(b);
    for _ in 0..40 {
        if fa < fb {
            hi = b;
            b = a;
            fb = fa;
            a = hi - gr * (hi - lo);
            fa = nll(a);
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + gr * (hi - lo);
            fb = nll(b);
        }
    }
    let temp = 10f64.powf((lo + hi) / 2.0);
    let best_nll = nll((lo + hi) / 2.0);
    Ok((temp as f32, best_nll))
}

/// Expected calibration error (ECE) with `bins` equal-width probability bins,
/// using the max-probability calibration protocol.
///
/// `preds` is a table of predicted max-probabilities, `labels` is a slice of
/// booleans indicating whether the prediction was correct.
pub fn ece(preds: &[f32], correct: &[bool], bins: usize) -> f32 {
    debug_assert_eq!(preds.len(), correct.len());
    if preds.is_empty() {
        return 0.0;
    }
    let bins = bins.max(1);
    let mut bin_acc = vec![0.0f64; bins];
    let mut bin_cnt = vec![0usize; bins];
    for (p, &c) in preds.iter().zip(correct) {
        let idx = ((*p as f64 * bins as f64) as usize).min(bins - 1);
        bin_cnt[idx] += 1;
        if c {
            bin_acc[idx] += 1.0;
        }
    }
    let mut total = 0.0f64;
    for i in 0..bins {
        if bin_cnt[i] == 0 {
            continue;
        }
        let accuracy = bin_acc[i] / bin_cnt[i] as f64;
        let confidence = (i as f64 + 0.5) / bins as f64;
        total += (bin_cnt[i] as f64 / preds.len() as f64) * (accuracy - confidence).abs();
    }
    total as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn temperature_flattens() {
        let p1 = softmax_temperature(&[2.0, 0.0], 1.0);
        let p2 = softmax_temperature(&[2.0, 0.0], 4.0);
        // Higher temperature => flatter
        assert!(p1[0] > p2[0]);
    }

    #[test]
    fn jev_confidence_values() {
        // Reference values from the TypeSafe docs.
        assert!((confidence_peak(&[0.61, 0.35, 0.04]) - 0.42).abs() < 0.01);
        assert!((confidence_peak(&[0.0, 0.57, 0.43]) - 0.35).abs() < 0.01);
        assert!((confidence_peak(&[0.0, 0.89, 0.11]) - 0.84).abs() < 0.01);
        assert!((confidence_peak(&[0.84, 0.10, 0.06]) - 0.76).abs() < 0.01);
        assert!((confidence_peak(&[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((confidence_peak(&[0.5, 0.5]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn entropy_confidence() {
        assert!((confidence_entropy(&[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((confidence_entropy(&[0.5, 0.5]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn fit_temperature_recovers_true() {
        // Draw logits from a softmax at a known temperature, then sample targets
        // from that categorical distribution. Maximum likelihood on the sampled
        // one-hot targets recovers the generating temperature.
        let true_t = 1.8f32;
        let mut rows = Vec::new();
        let mut targets = Vec::new();
        let rng = |s: usize| (s as f32 * 0.37).sin();
        let mut state = 12345u64;
        let next = |state: &mut u64| -> f64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*state >> 11) as f64 / (1u64 << 53) as f64
        };
        for i in 0..2000 {
            // A clear class structure: class 0 strongly favored, class 2 disfavored.
            let logits = vec![
                2.5 + rng(i) * 0.4,
                rng(i + 1) * 0.4,
                -2.5 + rng(i + 2) * 0.4,
            ];
            let probs = softmax_temperature(&logits, true_t);
            let r = next(&mut state);
            let mut acc = 0.0f64;
            let mut t = 0usize;
            for (j, &p) in probs.iter().enumerate() {
                acc += p as f64;
                if r <= acc {
                    t = j;
                    break;
                }
                t = j;
            }
            rows.push(logits);
            targets.push(t);
        }
        let (temp, _) = fit_temperature(&rows, &targets).unwrap();
        assert!(
            (temp - true_t).abs() / true_t < 0.3,
            "temp={temp} true={true_t}"
        );
    }

    #[test]
    fn ece_basic() {
        // Build an approximately calibrated table: in each bin the empirical
        // accuracy equals the bin midpoint confidence (up to one sample).
        let mut preds = Vec::new();
        let mut correct = Vec::new();
        for k in 0..10 {
            let center = (k as f32 + 0.5) / 10.0;
            let n_correct = (center * 10.0).round() as usize;
            for i in 0..10 {
                preds.push(center);
                correct.push(i < n_correct);
            }
        }
        let e = ece(&preds, &correct, 10);
        assert!(e < 0.1, "ece={e}");
    }
}
