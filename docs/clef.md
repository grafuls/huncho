# Native Clef backend

Huncho automatically selects the native Clef backend, which runs the Qwen3.5
text backbone and Clef's trained joint schema head directly in Rust with Candle. It reads the published safetensors and
tokenizer files. Users do not need Python, PyTorch, pip, model conversion, or a
second server. All questions in a request share one backbone pass.

## Build and run

For CPU:

```bash
cargo build --release -p huncho-cli --features clef
./target/release/huncho serve --model Cloudflare/clef
```

For NVIDIA GPUs, build with a CUDA toolkit installed:

```bash
cargo build --release -p huncho-cli --features cuda
./target/release/huncho serve --model Cloudflare/clef
```

`clef` includes the Rust Hub client, tokenizer, and Candle. `cuda` includes
`clef` and Candle's CUDA kernels. A source-built CUDA executable links NVIDIA
libraries and needs those libraries even when forcing CPU. The unified RPM
bundles separate CPU and CUDA executables behind the `huncho` command, allowing
it to run on a CPU host without NVIDIA libraries.

For the RPM install, driver/runtime requirements, and the shared systemd unit,
see [GPU setup](gpu-setup.md).

`HUNCHO_CLEF_DEVICE=auto` (the default) tries GPU 0 and falls back to CPU when
initialization or a small kernel probe fails. Use `cpu` to force CPU, or `cuda`
/ `cuda:N` to require a GPU and report errors if it is unavailable. Clef uses
BF16 on CUDA and FP16 backbone weights with an FP32 joint head on CPU. Use
`--dtype fp32`, `--dtype fp16`, or `--dtype bf16` to override; BF16 is rejected
on CPU. Explicit dtype overrides are not changed on fallback. Once a device
has been selected, model-loading and inference errors are reported, including
insufficient GPU memory.

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
./target/release/huncho serve --model ./models/clef
./target/release/huncho serve --manifest ./models/clef/huncho-model.json
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

CPU Clef supports the shared-weight `--replicas 1..8` serving pool. Backbone,
joint head, lexical embeddings and tokenizer share storage; each job evaluates
one complete schema with its own activations. It does not split fields or
enable cross-request F5 tensor batching. Configure backbone kernels before
creating replicas; later shared-backbone changes are rejected. Every context
must pass fresh complete labeled startup qualification concurrently. Fixture
replicas retain identical raw float bits for fp32/fp16 under concurrent calls,
including grouped pooling/projections and buffered CPU kernels. Released Clef
pool calibration, throughput and RSS measurements remain outstanding.

`HUNCHO_CLEF_GROUPED_POOL=1` enables CPU-only grouped option pooling (default
off). It gathers all option token embeddings once, then keeps each original
span mean. Summary scoring groups questions by their exact option count and
restores original field order; it introduces no padding or schema splitting.
It composes with `HUNCHO_CLEF_VECTOR_HEAD=1`. The compact lexical gather can
retain more temporary rows than the scalar loop, so latency and peak memory
depend on the schema. Backbone and joint-field attention still run once for
the complete request.

Metadata records `joint_pool_execution=grouped-spans-summary-v1`. Changed BMM
and reduction shapes require fresh complete labeled startup qualification,
even with a fitted source temperature. CPU fp32/fp16 fixtures exercise all
typed questions, multi-token options, singleton and repeated nonadjacent
cardinality groups, both projection modes and unchanged temperatures. This is
fixture parity; full-checkpoint calibration and throughput remain unqualified.
No GPU checks are performed for this profile.

```bash
cargo test --workspace --features huncho-cli/clef --offline
./target/release/huncho conform --model Cloudflare/clef \
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
