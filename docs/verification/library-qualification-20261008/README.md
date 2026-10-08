# Core and HTTP serving qualification, 2026-10-08

No Apple or actual GPU checks run. Default dependencies and numerical,
calibration, label and paired thresholds are unchanged. Synthetic zero-feature
fixtures test boundary plumbing only; they do not release a model.

The core runs fresh complete observed-label conformance for the actual engine,
options and selected compute/runtime environment. A private proof is installed
only on success; each new attempt immediately revokes the previous proof.
Native backend/dtype identity must match the engine calibration key. Pending
entries, inherited/non-refitted quantized calibration, missing exact vLLM or
integrated-head calibration and vacuous prefix profiles refuse production.

Production evaluations retain opaque context/attempt tokens before work and
validate them before publishing. A later successful attempt with identical
options cannot promote an old result. Every actual HTTP replica is checked
before admission and after inference/coalescing; eager server startup checks
before binding, while lazy factories cannot bypass request checks. Changed
options/runtime environment fail before forwards or exact-result reuse.
Presentation-only extensions normalize to the same arithmetic profile.

Fresh gates clear response caches and advance their shared generation. Delayed
diagnostic preparations can insert only into their older namespace; they cannot
restore entries usable by fresh production. Cache reconfiguration revokes
proofs and new replicas cannot inherit them. Context/token state is private and
cannot be reconstructed from serialized reports or unsigned receipts.

- [Default workspace](workspace.log): core/API/CLI units and existing scheduling,
  caches, HTTP batches/residency/cancellation regressions; ten core boundary
  tests pass plus an isolated environment child executed by its parent. Six
  HTTP boundary tests cover lazy loading, every replica, prefilled result caches,
  changed options, listener refusal and revocation during a blocked forward.
- [Earlier targeted core](core.log) and [API](api.log) checks retain intermediate
  coverage; workspace checks include the subsequent tokens/cache generations.
- [Native CPU Candle CLI](candle-cli.log): actual Kev fork, chunk, padding,
  direct pages, grouped GQA, runtime LoRA and fresh labeled startup profiles.
- [Native CPU ONNX CLI](onnx-cli.log): actual scalar/batch/masked integrated
  readouts and shared contexts, original fixed references and startup gates.
- [CPU browser WASM rebuild/runtime](browser-build.log), [browser](browser.log):
  separate async qualification and shared core compatibility.

Cargo and actual local CLI commands run sequentially to process exit. Initial
fixture compile/parse failures are retained as `core-initial.log` and
`core-fixture-errors.log`; required fork/instructions were added before success.

Raw `eval`, prepared/external APIs, bench and diagnostic conformance remain
research interfaces. Custom native runtime implementers must declare
`native_execution` when exposing production inference. The async browser SDK
retains its own fresh actual-runtime gate. These in-process proofs do not
establish independent dataset provenance or signed portable attestation. The
released CPU Kev profile remains rejected/pending; fixture success grants no
released outcome acceptance, speed/RSS measurement or GPU qualification.
