# CPU vLLM decision backend

The optional `vllm` feature runs a local, pinned CPU pooling worker for Kev
F2/Qwen3.5. It executes a complete prefill and the trained pointer head, returns
raw finite marker scores, and leaves typed temperature/softmax to Huncho core.
The custom model has no vocabulary head, sampler or decode loop. This is a
separate optional Python runtime; the default Rust build gains no runtime
dependency. Apple and actual GPU checks are deferred.

## Runtime and package

Build `cargo build --release -p huncho-cli --features vllm`. Select an explicit
absolute `HUNCHO_VLLM_PYTHON` executable in an isolated Linux x86_64 CPU-only
environment with distribution versions `vllm==0.31.0+cpu`,
`torch==2.13.0+cpu`, and `transformers==5.17.0`. Huncho clears inherited worker
environment options, disables accelerator/plugin discovery before imports,
requires CPU-only Torch and forces offline loading. It never substitutes
Candle, an HTTP chat endpoint or generated-token log probabilities.

Follow [vLLM's CPU installation guidance](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/cpu/)
for its native libraries and CPU instruction requirements. The wheel exercised
here was `vllm-0.31.0+cpu-cp38-abi3-manylinux_2_39_x86_64.whl`, SHA256
`d1eb6f3b6e548cfe372ecc4c85d0d2aa6986f183a2bfcfccdff93549b5463227`.
Its glibc requirement does not establish portability to older hosts. On the
tested isolated environment, the Intel OpenMP library needed explicit
`LD_PRELOAD`; that library's bytes enter the execution identity.

CPU Qwen3.5 GDN in this pinned runtime requires BF16 execution and
128-dimensional recurrent key/value heads. The trained pointer remains FP32.
Other dtypes, families, quantized weights and runtime adapters are refused.
LoRA must be merged offline into FP32 storage. CPU runtime GDN LoRA is not
implemented upstream in this version.

Convert a local pinned source package into a **new** package:

```bash
/absolute/cpu-venv/bin/python scripts/export_kev_vllm.py /source/kev-package \
  --base /source/pinned-qwen35 --out /new/vllm-package
```

The exporter requires complete Qwen3.5 backbone tensors, standard constant-rank
LoRA and complete q/k pointer projections. It verifies names/shapes using a
meta model, pins source bytes before reading weights and rechecks them before
and after export. Existing output directories and unsupported embedding/head
or adapter semantics are refused. No source file or temperature changes.
The new `vllm:bf16` calibration entry inherits the source FP32 temperature and
overrides with status **Pending**.

The exact artifact points to `vllm/artifact.json`, whose strict profile is
`kev-pointer-vllm-cpu-bf16-v1`. It pins only `model/config.json` and
`model/model.safetensors`; unpinned files, digest changes and paths outside the
package are rejected. Source provenance accompanies the export. Huncho also
binds the embedded worker source, Python executable, installed core runtime
bytes, native libraries and compiler/environment profile to qualification.
Changing any arithmetic/runtime/package profile requires fresh checks.

## Execution and calibration

Set `HUNCHO_DEVICE=cpu` (or leave it unset). Options are:

| Variable | Default | Bound/purpose |
|---|---:|---|
| `HUNCHO_VLLM_PYTHON` | required | Absolute isolated CPU interpreter |
| `HUNCHO_VLLM_THREADS` | 2 | 1..64 CPU threads |
| `HUNCHO_VLLM_BATCH_ROWS` | 1 | 1..8 independent native rows |
| `HUNCHO_VLLM_KV_BYTES` | 1073741824 | 64 MiB..16 GiB native KV budget |
| `HUNCHO_VLLM_TIMEOUT_SECS` | 180 | 1..600 seconds per worker reply |

With more than one configured row, existing per-/cross-request batching can
collate equal-length independent prompts within token/readout limits. The
runtime submits token IDs directly, with exact original marker positions and
final decision position. It requires one actual complete model forward per
submitted group and rejects silent splitting. Returned tensors own their
storage. Invalid inputs are rejected before execution; worker timeout/protocol
failure invalidates the context. IPC has a 32 MiB frame bound.

The pinned [pooling API](https://docs.vllm.ai/en/v0.31.0/models/pooling_models/)
is used with a custom raw pointer pooler and `use_activation=False`. Ordinary
classification activation would alter the contract. Prefix caching, prefix
forks, partial/chunked prefills, padding, replicas, quantization and device
execution are unsupported in this increment. Native KV allocations remain
useful for complete prefills but are not a Huncho retained-prefix capability.
The byte budget excludes weights, recurrence and runtime/workspace overhead.

Pending packages may collect actual raw fitting rows offline:

```bash
HUNCHO_DEVICE=cpu HUNCHO_VLLM_PYTHON=/absolute/cpu-venv/bin/python \
  huncho capture-logits --model /new/vllm-package --backend vllm --dtype bf16 \
  --data /independent/fit.jsonl --output /new/fitting-rows
```

Use separate observed-outcome fitting and held-out datasets. Capture neither
fits temperatures nor authorizes serving. `huncho calibrate`, followed by
`huncho conform --backend vllm --dtype bf16` and an actual complete labeled
suite, must establish the unchanged probability, argmax, ECE-drift and enabled
batch-parity gates. Serving requires an explicit fitted/refitted `vllm:bf16`
entry and fresh `--qualification-golden MODEL=PATH`, including when no
optimization flags are set. DEFAULT calibration cannot authorize this runtime.
Unsigned receipts bind local inputs but never replace fresh evaluation.

## Evidence and limits

The frozen synthetic fixture uses independent FP32 eager PyTorch seed-711
hybrid weights, three merged LoRA targets and the original Kev token encodings.
BF16 vLLM raw pooling stays below the fixed `1e-3` probability threshold with
all six argmaxes unchanged; measured maximum probability delta is about
`0.0006074` at the original temperature. Native paired batches pass the fixed
`1e-4` gate. Actual CLI tests exercise Pending fitting capture, fresh receipts,
HTTP serving, physical/logical work, grouped benchmarking and refusal of
missing, unlabeled and DEFAULT-only qualification. Package/digest/profile
negative checks run without Python. The fixture remains Pending on disk.

These synthetic labels establish gate plumbing, not released Kev calibration.
No released 4B acceptance, controlled speed/RSS result or GPU qualification
follows. Runtime startup and comprehensive identity hashing add cost; IPC and
BF16 drift may outweigh the kernel/batching benefit on small workloads.
See [retained CPU verification](verification/vllm-cpu-20261008/README.md).
