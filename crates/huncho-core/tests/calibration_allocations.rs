//! Count only this thread's calibration heap calls; no timing/RSS claim.
use huncho_core::calibration::{calibrate, calibrate_owned};
#[path = "support/allocations.rs"]
mod allocations;
use allocations::measure;

fn original(logits: &[f32], temperature: f32) -> Vec<f32> {
    let inv_t = temperature.recip();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for &logit in logits {
        let e = ((logit - max) * inv_t).exp() as f64;
        exp.push(e);
        sum += e;
    }
    exp.into_iter().map(|e| (e / sum) as f32).collect()
}

#[test]
fn owned_calibration_makes_no_heap_calls_and_borrowed_calibration_uses_one_fp32_buffer() {
    for size in [2, 3, 10, 255, 4096] {
        let logits: Vec<_> = (0..size).map(|i| (i as f32 * 0.17).sin() * 4.).collect();
        let owned = logits.clone();
        let allocation = owned.as_ptr();
        let (old, old_calls) = measure(|| original(&logits, 2.40605));
        let (borrowed, borrowed_calls) = measure(|| calibrate(&logits, 2.40605).unwrap());
        let (reused, reused_calls) = measure(|| calibrate_owned(owned, 2.40605).unwrap());
        assert_eq!(old, borrowed);
        assert_eq!(old, reused);
        assert!(old_calls.0 >= 1 && old_calls.1 >= size * 8);
        assert_eq!(borrowed_calls, (1, size * 4));
        assert_eq!(reused_calls, (0, 0));
        assert_eq!(reused.as_ptr(), allocation);
        println!("candidates={size}: historical alloc/realloc calls={}, requested_bytes={}; borrowed calls={}, requested_bytes={}; owned calls={}, requested_bytes={}",
            old_calls.0, old_calls.1, borrowed_calls.0, borrowed_calls.1, reused_calls.0, reused_calls.1);
    }
}
