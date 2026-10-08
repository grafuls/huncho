# CPU ONNX integrated raw-head batches, 2026-10-08

No Apple or actual GPU check ran. CPU ORT 1.28.0 executes these fixtures; ort
2.0.0-rc.13 is the pinned Rust API. The generator uses optional ONNX 1.20.1 and
NumPy 2.5.3 tooling. Default dependencies and feature selection are unchanged.
This is synthetic software evidence, not a released Laya export, trained fit,
observed-outcome acceptance or speed/RSS measurement.

The new strict batch ABI uses dynamic tokens/masks `[B,S]`, row/marker coordinates
`[R,2]`, question types `[B]`, and raw scalar scores `[R,1]`. At most 64 rows and
8,192 markers are accepted. The graph owns the head; shared Rust calibration
applies each original question's temperature once. Masked context means use
original lengths. Owned scores scatter back in input/marker order, including
duplicates, empty marker lists and mixed types. Generic/compact/scalar ABIs keep
their separate contracts. No cache retention, vocabulary codes or device path
is introduced.

Existing weights, independent Rust raw scores, probability goldens, targets,
tokenizers and temperatures are untouched. Three new graphs are generated from
those recorded weights; the generator does not regenerate references. Fixed
targets are arbitrary synthetic data for gate plumbing only. Full argmax,
external delta/ECE and paired 1e-4 thresholds remain unchanged.

- [Native tests](native.log): three new native/engine tests and four existing
  scalar/shared tests pass. Original mixed typed requests and cross-request
  padding, frozen raw values, original probabilities, types, coordinates,
  ownership/reuse, limits/invalid inputs, nonfinite outputs and isolated shared
  replicas are covered.
- [CLI tests](cli.log): six process groups pass. Actual padded/cross-request
  work and receipts bind the new ABI; missing labels cannot start serving.
  Shared sessions report actual per-context benchmark work. Old scalar,
  feature-mask, device-option preflight and shared-session regressions pass.
- [CPU backend regression](backend.log): full ONNX/shared/Candle/Clef suite.
- [Final changed-path checks](final-native.log): raw/head/mask/shared/compact
  native paths after removing unused type-vector allocation from feature rows.
- [CUDA static checks](cuda-compile.log): optional ONNX CUDA/shared tests compile
  against the CPU ORT distribution only. No GPU initialization/execution follows.
- [Default workspace](default-workspace.log): dependency/profile regressions.

Cargo and actual CLI commands run sequentially to process exit. Native tests
use `--features onnx-shared,clef`; CLI tests use
`--features onnx-shared,tokenizers,qualification`. All ORT commands use
`ORT_PREFER_DYNAMIC_LINK=1` and the CPU distribution's library directory in
`LD_LIBRARY_PATH`. The future device feature check compiles only; no ignored
device test runs.

Graphs must actually preserve trained backbone/head and mask semantics. ABI
metadata alone does not prove calibration. Every real serving context still
needs exact fitted/refitted `onnx:fp32` metadata plus fresh complete observed-label
gates. Scalar graph receipts cannot authorize the new artifact/profile.
Released Laya exports/qualification and controlled latency/RSS remain open.
