# Candle kernels compatibility patch

Source: `candle-kernels` 0.11.0 from crates.io, licensed MIT OR Apache-2.0.
Archive SHA-256: `67450168a281bbb195a14cc85cf164c7a12023a54a03409a3324f476346e0789`.

This copy removes the two `__hmax_nan` and `__hmin_nan` fallback definitions
from `src/compatibility.cuh`. CUDA 12.8 and 13.3 already define these FP16
intrinsics for Turing, so Candle's pre-Ampere guard causes duplicate definitions
when compiling for `sm_75`. All other kernel sources are unchanged.

The node build uses CUDA 12.8. Keep this patch until an upstream release removes
the conflicting definitions, then return to the registry dependency.
