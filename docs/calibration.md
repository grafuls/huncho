# Calibration & confidence

`calibrated probabilities stay calibrated` is the core differentiator of `huncho`.
The engine applies a per-(backend, dtype) temperature to the head logits before
softmax, so that probabilities are calibrated across backends and
quantizations. Confidence is computed using the documented definition.

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

The suite passes only if all three meet their thresholds. A backend (or
quantized variant) that fails must ship with **refitted** temperatures or be
rejected.

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
