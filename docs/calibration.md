# Calibration & confidence

`calibrated probabilities stay calibrated` is the core differentiator of `huncho`.
The engine applies a per-(backend, dtype) temperature to head logits before
softmax. Temperature scaling alone does not establish empirical calibration.
Every real serving runtime must pass fresh complete labeled conformance for
its loaded execution profile, including an unoptimized package marked `fit`.
Confidence is computed using the documented definition.

## Pipeline

For each question:

1. The prompt builder produces the candidate positions and per-candidate
   code ids.
2. The head reads the hidden states / logits at those positions and produces a
   candidate logit vector.
3. The calibration layer applies the temperature and softmax:
   `p = softmax(logits / temperature)`.
4. Confidence is computed from the probability vector.

The temperature is resolved from the manifest's `calibration` section by the
key `{backend}:{dtype}`, falling back to `default`. F4 (and per-type calibration)
can override temperature per question type.

## Temperature scaling

```rust
pub fn softmax_temperature(logits: &[f32], temperature: f32) -> Vec<f32>
```

A higher temperature flattens the distribution; a lower temperature sharpens it.
`huncho calibrate` fits a scalar temperature that minimizes the negative log
likelihood on a held-out `(logits, target)` set and writes it back into the
manifest.

The core keeps the existing FP32 exponent, ordered FP64 sum/division and final
FP32 rounding. `calibrate_owned(Vec<f32>, temperature)` consumes raw logits
and normalizes in their allocation; `calibrate(&[f32], temperature)` retains
the caller's raw logits and allocates one FP32 result buffer. Both reject the
same invalid logits/temperatures. Native scalar/batch/prefix/cooperative/F5 and
browser readout paths consume temporary logits when extensions are disabled.
Extensions retain the original raw vector. No temperature, reduction order or
candidate mapping changes, and no refit is inferred from buffer reuse.
[Bitwise and allocation checks](verification/calibration-buffers-20261009/README.md)
cover the original arithmetic, unchanged gates and actual CPU inference.

## Confidence

Three definitions are supported, selectable per model via `calibration.default.confidence`:

### Clef maximum probability (`max-probability`)

Confidence is the largest calibrated option probability, including for score
questions. A uniform two-option distribution has confidence `0.5`.

### Jev peak-based confidence (`peak`)

```
confidence = (p_max - 1/n) / (1 - 1/n)
```

- Uniform distribution → `0`.
- Single peak → `1`.

### Laya entropy-based confidence (`entropy`)

```
confidence = 1 - H(p) / log2(n)
```

- Uniform distribution → `0`.
- Single peak → `1`.

The confidence definition is reported in the response `extensions` field
(`confidence_definition`) when extensions are enabled, so downstream consumers
know which formula was applied.

## Conformance (probability fidelity)

`huncho conform` runs golden vectors against any backend and reports:

- **Max probability delta** — the largest absolute difference between the
  backend's probabilities and the reference's. Default tolerance `1e-3`.
- **Argmax agreement** — the fraction of cases where the backend picks the same
  label as the reference.
- **ECE drift** — the difference between the backend's and the reference's
  expected calibration error. Default bound `0.02`.

The suite passes only if all three meet their thresholds. A refit must also
pass those unchanged gates; a failed variant remains rejected. Fit and final
evaluation inputs must be separate, and evaluation labels must cover every
question. Unlabeled numerical agreement is diagnostic evidence.

`serve --qualification-golden MODEL=PATH` is required for native Candle,
Clef, ONNX and llama.cpp execution, even without optimization flags or a
refitted entry. Every replica is checked concurrently before the listener
opens. The runtime reports `native_execution` in its capabilities/receipts;
the explicit offline mock retains its demo path. Source `fit` metadata, a
`default` fallback temperature or a retained unsigned receipt cannot replace
fresh outcome checks. Quantized variants additionally require an exact
backend:dtype `refit` entry. Diagnostic library inference, `bench` and
numerical-only `conform` remain available for analysis and do not authorize
serving.

## Library serving qualification

`Engine::qualify_for_serving(&suite, &options, cross_request_max_requests)` runs
fresh complete observed-label conformance at the fixed thresholds. A passing
report installs a private in-process proof for that actual engine, execution
options and selected compute/runtime environment. Every replica/new context
needs its own complete run. Diagnostic reports, source `fit` metadata and
deserialized receipts cannot install proofs. Pending entries are refused;
quantization requires an explicit exact backend:dtype `refit`, and vLLM and
integrated ONNX heads require explicit fitted/refitted entries.

Use `eval_for_serving` or `eval_for_serving_with_stats` for programmatic
production evaluation. They check before model/cache work and retain an opaque
qualification token to check again before returning a result. A new attempt
revokes the previous proof even if it fails; a later successful run with the
same options cannot authorize an older in-flight result. Cache configuration
also revokes the proof. Qualification clears the response cache and advances
its shared generation so delayed diagnostic plans cannot repopulate active
entries. Pure unchanged offline mock demos keep their existing path.

The HTTP server checks eager contexts before opening its listener and checks
all actual replicas before admission and after inference/coalescing. Lazy
factories cannot bypass the request boundary. Unqualified profiles return 503
with `model_unqualified`, including when an exact response is retained.
Programmatic HTTP applications use each actual handle's `serving_options` and
`serving_cross_request_max_requests` after creating `AppState`, then qualify
every `replica_engines()` context before calling `serve` or exposing `router`.
Presentation-only extensions share the same proof; changed arithmetic or
scheduling options require fresh qualification.

All built-in real runtimes declare `native_execution`. Custom production
runtimes must declare that capability and match their engine backend/dtype
identity. Raw `eval`, prepared/external APIs and conformance remain explicit
diagnostic interfaces; they do not claim serving authorization. The browser
SDK has its separate fresh actual async-runtime gate. These proofs establish
the existing numerical/outcome checks, not dataset independence/provenance or
a signed portable execution certificate. Applications remain responsible for
trusted pinned references and independent real held-out outcomes.
[Boundary verification](verification/library-qualification-20261008/README.md).

This rule closes a measured portability gap: the released Kev CPU FP32
profile fails the complete held-out probability-delta gate, despite fitted
upstream metadata and passing argmax/ECE-drift checks. Original and buffered
CPU logits match on its eight worst cases; CPU FP16 also differs on selected
cases. [Retained CPU evidence](verification/kev-cpu-heldout-20261008/summary.json)
records the failure without changing temperatures or reference vectors.

## Reference values

The seed values from the Jev docs are used as unit-test anchors in
[`calibration.rs`](../crates/huncho-core/src/calibration.rs):

- `[0.61, 0.35, 0.04]` → peak confidence `0.42`
- `[0.00, 0.57, 0.43]` → peak confidence `0.35`
- `[1.0, 0.0]` → `1.0`; `[0.5, 0.5]` → `0.0`

## CLI

```bash
huncho calibrate --manifest huncho-model.json --backend onnx --dtype fp32 \
  --data fit-data.json
```

`fit-data.json` carries `{ "rows": [[...logits]], "targets": [index] }`. The
fitted temperature is written back into `calibration.entries["onnx:fp32"]`.
