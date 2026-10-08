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
backend:dtype `refit` entry. Library inference, `bench` and numerical-only
`conform` remain available for analysis and do not authorize serving.

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
