# mock-laya (PHASE 1 tracer bullet)

A fully self-contained model package for the M0 tracer bullet: an F1 model
served behind `/v1/systemone` on **ONNX Runtime CPU**. It needs no external
weights, so it can be built, served, and conformance-checked in CI.

## Files

- `huncho-model.json` — the model manifest (F1, option-marker head, `laya-v1`
  prompt contract).
- `mock-model.onnx` — a real ONNX Runtime artifact. It reproduces the offline
  mock reference's deterministic hidden-state rule (`hidden=512`,
  `col = tok % 512`, `value = 4 + 0.01*tok`) for every token position, so the
  ONNX engine and the reference agree exactly.
- `golden.json` — the reference conformance vectors (generated once by the
  mock backend, CONF-01).

## Run it

Build with the `onnx` feature, then serve / conform against the real backend:

```bash
cargo build --release -p huncho-cli --features onnx

# Serve on ONNX Runtime
cargo run -p huncho-cli --features onnx -- \
  serve --manifest examples/mock-model/huncho-model.json
# POST /v1/systemone with model="mock-laya"

# Conformance: ONNX engine vs. reference golden
cargo run -p huncho-cli --features onnx -- \
  conform --manifest examples/mock-model/huncho-model.json \
          --golden examples/mock-model/golden.json
```

The same commands with `--backend mock` drive the in-process mock backend instead,
which is how `golden.json` was generated and how the harness runs without an
ONNX build.

## Regenerating the artifact

If you change the mock reference rule in `huncho-backend/src/mock.rs`, regenerate
`mock-model.onnx` so the ONNX and reference implementations stay in lockstep
(see `crates/huncho-backend/tests/onnx_conformance.rs` for the exact invariant).
