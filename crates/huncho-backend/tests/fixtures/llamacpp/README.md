# Native CPU GGUF fixture

`kev-f32.gguf` and `kev-f16.gguf` are a standard dense Qwen3.5 GGUF converted from
[`tiny_kev`](../tiny_kev/README.md), using Huncho's CPU FP32 LoRA merge and the
pinned upstream mapper at `26394b4e6749a41c3633db040e0987500a5f7013`.
The trained-format pointer remains in `tiny_kev/head.pt`; existing upstream
`golden.json` is the independent reference and is never regenerated here.
Weights are random test data, not a useful decision model or calibration dataset.

SHA-256 FP32: `83f9463c39501ef8924ac0b0bbe783cb0561640d09eab679f98a4fbd0a281d18`.

SHA-256 F16: `9c76a5b4eb4d9109a1bf45280b0f78f56bd57e4973549b878594d48945de6f7a`.
The fixture includes one linear-attention and one full-attention layer, unequal
key/value head counts, partial rotary and nonzero LoRA updates. It has 384 tokens
and 16 hidden features. GGUF ties its auxiliary LM projection to embeddings;
F2 decision scoring still uses the trained external pointer.

Tests execute real native CPU kernels without Python, network or a GPU. Rebuild
only when deliberately updating the artifact, from a clean pinned upstream
checkout and CPU Python environment (validated with PyTorch 2.8.0+cpu,
Transformers 5.17.0, Safetensors 0.8.0 and NumPy 2.3.4):

```bash
HUNCHO_LLAMA_TOOLS=/path/to/pinned-llama.cpp \
HUNCHO_LLAMA_PYTHON=/path/to/cpu-venv/bin/python \
cargo test -p huncho-backend --features llamacpp --test llamacpp \
  regenerate_gguf_from_native_fp32_merge -- --ignored
```

`generate.py` supplies the tiny tokenizer's exact existing vocabulary IDs and
merges because its test-only BPE fingerprint is not in upstream's registry. All
weight, norm, RoPE and value-head transformations use the official pinned class.
The production CLI calls the unmodified official converter.
