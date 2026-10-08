# Dedicated CPU browser workers, 2026-10-08

Actual ONNX Runtime Web 1.30.0 executes the existing fixed F1 graph in a
dedicated module Worker with CPU WASM, one thread and fixed SIMD. Chrome
launches with GPU, software rasterizer and WebGL disabled. No GPU inventory,
initialization, execution or qualification is requested. Apple work is skipped.
Native Cargo dependencies/features and all original weights, independent Rust
reference logits/probabilities, labels and temperatures are unchanged.

Packaged descriptors hash the worker module as well as the SDK/core/runtime/
model/suite. Verified immutable worker/SDK Blob snapshots execute; every
worker independently loads real graph/assets and runs the complete fixed
labeled shared-core gate before returning a public instance. A page report
cannot promote another session. Worker/page execution contexts and module/
descriptor hashes appear in the fresh report.

- [Build](build.log): shared core rebuilt using official Rust 1.97.1,
  wasm32-unknown-unknown and wasm-bindgen 0.2.129.
- [Actual CPU browser checks](browser.log) and [report](summary.json): all
  previous calling-thread/packaging/qualification checks plus dedicated-worker
  graph execution for every original typed/unicode/JSON/long-state request.
  Independent native Rust answers retain their comparison limits; actual
  worker/page wire responses also compare bitwise.
- Fresh complete labeled reports, module hash/outcome refusal, invalid inputs
  with recovery, bounded eight-request snapshots, immutable report copies,
  drain-before-dispose and repeated disposal pass. Verified synthetic startup
  and runtime-crash scripts test failure cleanup; all eight pending requests
  reject and the failed worker refuses new work. Explicit termination also
  rejects all pending work. These scripts are test transport failures, not
  alternate model/reference data.

Cargo/build and actual local runtime commands run sequentially to completion.
Synthetic labels test gate plumbing only. `qualified=false` in the retained
summary means no released-model acceptance, isolated inference/UI benchmark or
peak RSS measurement is established. CPU kernels/heads/calibration are
unchanged. Message/JSON transport adds copies; every worker owns its runtime
and model session. No shared weights, worker pool, native browser batches,
prefix cache or multithread tuning is claimed. WebGPU remains deferred.
