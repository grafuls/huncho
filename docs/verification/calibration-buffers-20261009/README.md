# Candidate-logit allocation reuse, 2026-10-09

Default dependencies/features, original prompts/head weights, temperatures,
goldens and external/paired thresholds are unchanged. No Apple or actual GPU
inventory/execution/qualification checks run. This storage optimization keeps
FP32 exponentials, ordered FP64 normalization and final FP32 rounding.

`calibrate_owned` validates and consumes a temporary raw-logit vector. Every
native scalar/batch/prefix/cooperative/F5 and external CPU WASM readout uses it
when extensions are disabled. Extensions keep the raw vector and use the
borrowed calibrator with one FP32 result allocation. Buffers remain local to
the request; there is no shared mutable scratch or additional locking. Invalid
logits/temperatures still refuse results; no refit or serving proof is inferred.

- [Workspace](workspace.log): complete default workspace with external-score
  APIs; historical FP64-scratch references compare original float bits across
  13 cardinalities, 16 patterns and eight valid temperatures, including ties,
  signed zero, underflow and extreme finite values. Owned pointers/capacities
  stay unchanged. Extension vectors and invalid-input error messages match.
- [Native CPU](native.log): original typed Kev FP32/FP16, ModernBERT F1 and
  complete-schema Clef F5 scalar/native batches, cached equal/padded groups,
  cooperative cohorts/pressure, private parents and fresh fixed gates pass.
- [Official CPU WASM build](browser-build.log) and [actual browser](browser.log):
  page/scalar, dedicated workers and native equal/masked row/marker graphs retain
  frozen native probabilities and unchanged complete labeled/paired gates.
  GPU/WebGL initialization remains explicitly disabled in browser tests.
- [Actual release allocations](allocations-release.log): a test-only System
  allocator counts only successful alloc/realloc calls on the measurement
  thread, excluding input creation, output disposal and report formatting.
  Two, three, ten, 255 and 4,096 candidates give these local measurements:

  | Calibration path | Additional heap calls | Requested bytes per candidate |
  |---|---:|---:|
  | Historical FP64 scratch | 2 | 12 |
  | Borrowed FP32 probability buffer | 1 | 4 |
  | Owned existing logits buffer | 0 | 0 |

- [Release arithmetic](arithmetic-release.log): the historical float-bit,
  retained extension, input-validation and existing temperature-fitting tests
  also pass in the optimized native build.
- [Actual CPU ONNX CLI](onnx-cli.log): original scalar/native raw F1 head
  probabilities, labeled startup gates, metadata and strict device-flag refusal
  pass with the CPU ORT library. The initial baseline and
  [additional Candle run](onnx-candle-initial.log) did not enable `onnx-shared`,
  so its guarded padded/shared tests did not run. The exact required feature
  exercises them in [shared/padded CPU profiles](onnx-shared-padded.log).
- [Actual CPU HTTP](http-native.log): fresh qualification of the real Kev
  runtime-LoRA/direct-page/cooperative native-group profile, original typed
  probabilities and logical zero-decode usage with physical cached/padded
  counters. The test owns and cleans up its localhost server.

These values describe this compiler/platform's calibration calls. Requested
bytes are not simultaneous peak memory, allocator overhead or whole-request
RSS. The source logits allocation already exists in the owned case. No
end-to-end latency, throughput, cost or released-model acceptance is claimed.
Every actual serving context still needs fresh complete observed-label and
selected paired gates. Released Kev FP32 remains rejected and Q8/Q4 outcome
qualification pending; synthetic labels only verify gate plumbing.
