# Direct CPU Kev shared-prefix native attention, 2026-10-09

No Apple or actual GPU inventory/execution/qualification checks run. This
default-disabled profile uses native CPU FP32 Candle only. Dependencies,
original weights, reference probabilities, temperatures and external/paired
acceptance thresholds are unchanged. Synthetic labels test gate plumbing;
they do not establish released-model calibration.

Native equal/padded suffix groups regroup queries/probabilities across rows
to read one immutable prefix page. Prefix K/V never concatenate, expand to
query heads or repeat across batch rows. Each row retains one complete ordered
causal softmax, private suffix K/V and private GDN/convolution state. Parent
pages never advance, and suffix state is discarded after the trained readout.
The existing complete-context budget and live 64-handle bound remain
conservative. Independent forwards retain their explicit flat fallback.

Page/suffix matmul shapes and PV summation change rounding. Execution identity
is now `paged_attention=cpu-page-qk-pv-fp32-v2`,
`kv_storage=cpu-cow-pages-direct-v2`, and
`paged_attention_fallback=flat-independent-v1`. Fresh complete labeled and
selected native batch/padding paired gates are mandatory; old receipts do
not authorize the new profile.

- [Kernel](kernel.log): shared-prefix attention matches full contiguous
  grouped attention within 1e-6 for distinct rows/heads, causal offsets, page
  boundaries and query blocks. Poisoned future/foreign-row values cannot enter
  another query's distribution. Original pages remain bitwise unchanged;
  malformed dtype/shape/block settings refuse execution.
- [Native path](native-path.log): actual equal/padded trained Kev groups use
  zero complete-KV materializations, preserving original frozen probabilities
  within 1e-3 and independent paired probabilities within 1e-4/full argmax.
  Runtime LoRA, repeated groups, private recurrence, immutable parents and
  early malformed-input cleanup retain the existing ownership bounds.
- [Broader CPU run](backend.log): native library, candidate projection,
  Candle, chunks, whole-schema Clef, observed-label conformance and cooperative
  batching passed. The run then caught a stale scheduling test that still
  rejected newly supported cooperative cached groups. It stopped at that test.
- [Scheduling correction](fork-regression.log): tests now compare supported
  cooperative groups against original independent probabilities and still
  refuse missing prefix reuse. The first correction tried Rust equality on
  wire types without `PartialEq`; [compile log](fork-regression-initial-compile.log)
  is retained. Explicit unchanged probability/usage assertions correct the test.
- [Remaining CPU suites](backend-remaining.log) cover the rest of the native,
  page/prefix/padding/query, adapter/replica and packed-profile regressions.
- [Final native library](native-final.log): 48 tests pass; four explicit
  timing/corpus diagnostics remain ignored. All five supported page sizes and
  native 63/64-row boundary checks are covered. No GPU feature is enabled.
- [Final OpenBLAS integration](native-openblas.log): all five page sizes,
  chunks, trained frozen vectors, runtime adapters, retained snapshots,
  transactional bounds and replica isolation pass with the optional provider.
- [CPU OpenBLAS CLI](cli-openblas.log): actual cached equal/padded groups,
  fresh labeled/prefix/paired gates,
  v2 execution metadata and strict refusal of unsupported/unlabeled profiles.
  The installed provider is explicitly optional and remains part of identity.
- [Actual CPU HTTP](http-native.log): the native CLI server with runtime
  LoRA/query blocks/grouped GQA/direct pages serves original frozen typed
  probabilities only after fresh qualification, with actual cached/padded group
  counters and zero output tokens. The test cleans up its localhost process.
  Its baseline process clears both BLAS settings; the first run cleared the
  library but inherited the thread setting and correctly refused startup
  ([retained failure](http-initial-environment.log)). The corrected test also
  checks environment isolation when the parent selects an optional provider.

Commands run sequentially to process exit with `RAYON_NUM_THREADS=1`, without
the `cuda` feature. Optional OpenBLAS uses `cpu-blas`,
`HUNCHO_CPU_BLAS_LIBRARY=/usr/lib64/libopenblas.so.0` and one provider thread.
Removing prefix copies does not establish lower latency,
total peak RSS or released acceptance. Query/probability regrouping, suffix
K/V, recurrence, convolution, projections and activations still allocate; many
small page matmuls may slow a small workload. Mixed-prefix groups and device
kernels remain open. Released Kev CPU FP32 is rejected, and Q8/Q4 outcome
qualification remains pending.
