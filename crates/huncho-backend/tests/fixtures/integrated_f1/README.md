# Integrated F1 CPU fixture

These small graphs contain synthetic embeddings, pooled context, a typed
affine input and an MLP scalar scorer. They are not ModernBERT/Laya weights.
The tokenizer, four states and twelve question outcomes are public synthetic
test data. Fixed targets exercise conformance plumbing; they are not observed
outcomes and cannot release a model. `reference.json` retains `qualified=false`.
Raw scores and calibrated typed answers come from independent native Rust
arithmetic, rather than the ONNX graph under test.

`model.onnx` uses the strict integrated ABI; `masked.onnx` adds a mask input.
`wrong-output.onnx` and `nonfinite.onnx` test load/inference rejection.
`weights.json` records the independent scalar reference inputs.

Reproduce from the repository root with optional tooling (ONNX 1.20.1, NumPy):

```sh
python browser/tests/generate_fixture.py --out-dir crates/huncho-backend/tests/fixtures/integrated_f1
RAYON_NUM_THREADS=2 cargo run --locked --manifest-path browser/wasm/Cargo.toml \
  --example reference -- crates/huncho-backend/tests/fixtures/integrated_f1
HUNCHO_DEVICE=cpu cargo test --locked -p huncho-backend \
  --features onnx-shared,clef --test onnx_integrated -- --test-threads=1
HUNCHO_DEVICE=cpu cargo test --locked -p huncho-cli \
  --features onnx,tokenizers,qualification --test onnx_integrated -- --test-threads=1
```

Tests require a compatible CPU ONNX Runtime; an offline dynamic distribution
can be supplied with `ORT_LIB_LOCATION`, `ORT_PREFER_DYNAMIC_LINK=1` and its
library directory in `LD_LIBRARY_PATH`. All actual execution is CPU. Tests
that refuse GPU options do so before runtime initialization, without probes.
Graph generators do not export or qualify any released model.
