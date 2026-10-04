# Native Clef backend

`--backend clef` runs the Qwen3.5 text backbone and Clef's trained joint schema
head directly in Rust with Candle. It reads the published safetensors and
tokenizer files. Users do not need Python, PyTorch, pip, model conversion, or a
second server. All questions in a request share one backbone pass.

## Build and run

For CPU:

```bash
cargo build --release -p huncho-cli --features clef
./target/release/huncho serve --backend clef --model Cloudflare/clef
```

For NVIDIA GPUs, build with a CUDA toolkit installed:

```bash
cargo build --release -p huncho-cli --features cuda
./target/release/huncho serve --backend clef --model Cloudflare/clef
```

`clef` includes the Rust Hub client, tokenizer, and Candle. `cuda` includes
`clef` and Candle's CUDA kernels. A distributed binary can include these at
build time; its users only need the appropriate system GPU driver/libraries
and downloaded weights. This change does not publish prebuilt binaries.

The CPU build defaults to FP16 backbone weights with an FP32 joint head. The
CUDA build defaults to GPU 0 and BF16. Use `--dtype fp32`, `--dtype fp16`, or
`--dtype bf16` to override; BF16 is rejected on CPU. `HUNCHO_CLEF_DEVICE` accepts
`auto`, `cpu`, `cuda`, or `cuda:N`. A CUDA build's `auto` requires a working GPU.

The full model still needs substantial memory: 27 billion parameters at two
bytes each is approximately 54 GB just for weights, plus working memory.
CPU inference is useful for validation but will be slow for this checkpoint.

Requests use `"model": "clef"`. The resolver accepts raw releases using the
Qwen3.5 text architecture and the same joint-head layout; incompatible
architectures are rejected before loading weights.

## Downloads and local packages

The Rust resolver downloads config, tokenizer, backbone shards, and joint-head
weights from one pinned Hub commit. It honors `--revision`, `--token`,
`HF_TOKEN`, and `--cache-dir`. Repository Python code is neither downloaded nor
executed. It generates `huncho-model.json` alongside the files, and preserves
existing calibration when the package is loaded again.

An already downloaded release works without network access:

```bash
./target/release/huncho serve --backend clef --model ./models/clef
./target/release/huncho serve --backend clef --manifest ./models/clef/huncho-model.json
```

Required files: `config.json`, `tokenizer.json`, `joint_head_config.json`,
`joint_head.safetensors`, and either `model.safetensors` or
`model.safetensors.index.json` with every listed shard. `huncho convert` is not
needed. The native package uses family `F5`, head `joint-schema`, and prompt
template `clef-native-v1`.

## Contract and limits

- Text and JSON state are supported. The HTTP API rejects images and videos.
- Up to 64 questions are scored jointly, retaining caller question order.
  Choice options are sorted by ID as in Clef. Noul descriptions accept Huncho's
  `yes`/`no` and `true`/`false` aliases; output retains the Huncho contract.
- Generated packages cap the complete prompt at 16,384 tokens, or the model's
  smaller declared limit. Oversized inputs are rejected before inference;
  state is never silently truncated. Shared input tokens are counted once.
- Confidence is the maximum option probability. Temperature defaults to 1.0
  with calibration status `pending`; fit temperatures on your own evaluation
  data. Existing `clef:<dtype>` entries are honored.
- This initial port uses unfused attention and a per-token DeltaNet recurrence.
  It does not provide quantization, a KV cache, multimodal processing, or
  performance parity with optimized PyTorch/CUDA kernels.

## Validation

```bash
cargo test --workspace --features huncho-cli/clef --offline
./target/release/huncho conform --backend clef --model Cloudflare/clef \
  --golden ./clef-golden.json
```

Checked-in tiny-model fixtures were generated with Cloudflare's
[reference implementation](https://huggingface.co/Cloudflare/clef/blob/main/joint_schema_model.py),
PyTorch 2.8.0 CPU and Transformers 5.17.0. The fixtures record the reference
source SHA-256. Tests compare prompt tokens and spans exactly, FP32 raw logits
within `3e-5`, and FP16 probabilities within `0.002`. They cover all three
question types, question reordering, nested JSON, Unicode, and empty
instructions. CLI tests resolve, benchmark, and conform with Python absent
from `PATH`. Regeneration alone uses `scripts/generate_clef_fixture.py`.

The full Cloudflare checkpoint and CUDA execution have not been validated in
this environment. Run a release-matched F5 golden suite and benchmark your
workload on the target hardware before relying on accuracy or latency claims.
