# Native CPU browser batches, 2026-10-08

Actual ONNX Runtime Web 1.30.0 CPU WASM executes dynamic F1 row/marker heads on
the page and in dedicated workers. Fixed SIMD/one thread are unchanged. Chrome
disables GPU/software rasterizer/WebGL; no actual GPU check or Apple work runs.
Default Cargo dependencies/features remain unchanged.

Original scalar graphs, model/weights, independent native Rust reference
scores/probabilities, manifest temperatures, requests and goldens are untouched.
The new optional generator writes four new graphs from existing recorded
weights: equal, masked, nonfinite and deliberately batch-biased. The additional
equal-row suite copies an original choice request/vector/label under another
question ID; it never recomputes probabilities or changes the original suite.
All labels are synthetic boundary fixtures, not released outcome evidence.

The strict ABI uses dynamic INT64 tokens/masks `[B,S]`, row/marker coordinates
`[R,2]`, original question types `[B]`, and only raw FP32 scores `[R,1]`.
Shared-core groups stably sort original readouts by length, cap rows/markers,
bound multi-row rectangles by the immutable descriptor's token/padding budget,
and preserve oversized singletons within declared contexts. Owned scores
scatter back before unchanged typed temperature/softmax. Wire usage is logical;
work counts every actual tensor call/padded slot. No cross-request or prefix
batching is claimed.

Fresh complete labeled conformance runs optimized groups and actual separate
batch-one questions. Shared external delta <=1e-3/full argmax/ECE drift <=0.02
and paired <=1e-4/full argmax are unchanged. Core recomputes expected work and
logical usage from original immutable requests. Missing native/mixed-padding
work, wrong counters/usage, incomplete labels, scalar ABI substitution,
missing masks, nonfinite scores and invalid budgets refuse serving. Each page
or worker session reruns its own gate; packaging grants no acceptance.

- [WASM build](build.log): official Rust 1.97.1/wasm-bindgen 0.2.129.
- [Actual browser checks](browser.log) and [report](summary.json): original
  typed/JSON/unicode/long-state values; reordered mixed type row/marker scatter;
  native equal and masked heads; packaging/worker composition; all previous
  scalar/worker/failure/queue/ownership regressions. A biased graph below the
  external delta limit still fails the fixed paired gate.
- Optimized padded profile: four calls for twelve original questions, four
  native batches, three padded calls, 672 physical tokens and 27 padded slots;
  paired delta 0/full argmax. Independent qualification calls are extra and
  remain outside the optimized work counters, as in native conformance.
- Equal profile: ten calls for sixteen questions, four native batches, no
  padding; unchanged copied synthetic vectors and paired delta 0/full argmax.
- [Workspace with external-score feature](workspace.log): core raw/external/
  native conformance, typed calibration, new context/attempt proofs, HTTP
  scheduling/ownership and dependency regressions. Core tests cover physical/
  logical work, foreign/invalid/nonfinite plans, fixed paired rejection, full
  row coverage and 64-row profile bounds.
- [Native CPU ONNX CLI](onnx-cli.log) and [native CPU Kev/Clef](candle.log):
  retained native shared conformance paths still pass unchanged references.

Initial browser assertions incorrectly assumed every case needed padding and
that the long equal-row case still needed three tensor calls. The frozen long
case has equal truncated row lengths; corrected expectations retain that input
and report three padded calls/ten equal-profile calls. Initial failure logs
remain as `browser-initial.log` and `browser-equal-initial.log`.

Commands run sequentially to process exit. Synthetic fixture success grants no
released F1 export/calibration, speedup or peak RSS measurement. Batch graph
mask/backbone/trained head semantics still need real exports and independent
held-out outcomes. Cross-request collation, prefixes, other families,
multithreading, Apple and WebGPU/device qualification remain open/deferred.
