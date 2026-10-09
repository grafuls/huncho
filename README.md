# Huncho

**Huncho** is a portable serving engine for Jev-style "System One" decision
models. It takes a **state** and a set of **typed questions** and
returns **calibrated probabilities** — no text generation. It is to decision
models what llama.cpp / vLLM are to generative LLMs, with one non-negotiable
guarantee: *calibrated probabilities stay calibrated* across every backend and
quantization.

```
state + typed questions ──► huncho ──► calibrated probabilities (no decode loop)
```

The engine implements the **Jev `/v1/systemone` wire contract**, so an unmodified
TypeSafe Python SDK works against Huncho with only a base-URL change.

## Highlights

- **Five decision-head families** behind one contract: F1 (option-marker), F2
  (pointer), F3 (candidate-logit), F4 (slot head), F5 (joint schema).
- **Calibration as a first-class guarantee**: per-backend × dtype temperatures,
  softmax, and confidence (Jev peak-based, or per-model definitions).
- **Conformance gating**: an offline harness (`huncho conform`) compares any backend
  against golden vectors and gates releases on probability fidelity, argmax
  agreement, and ECE drift.
- **Efficient decision serving**: native tensor batches, shared Kev prefixes,
  bounded caches, and shared-weight CPU replicas. See [performance](#performance).
- **Offline-first**: a dependency-free deterministic `MockBackend` lets the whole
  pipeline (prompt → head → calibration → conformance) run with no weights.
- **Static, portable**: pure-Rust core with optional ONNX Runtime.
- **Hugging Face-native**: a model package can be resolved by repo id
  (`--model owner/repo`) so manifests, weights, head, and golden vectors are
  pulled straight from the Hub — the same run-time resolution model vLLM uses.

## Performance

Huncho computes decision scores in a forward pass, with no generated tokens or
decode loop. Its performance features reduce repeated preparation, model work,
and memory copies while keeping typed calibration in the shared core.

| Feature | Benefit | Supported paths / controls |
|---|---|---|
| **Native tensor batching** | Combine question rows in one model call; compatible paths support bounded right padding. Clef batches complete joint schemas. | CPU ModernBERT/Laya, Qwen/Kev, compatible ONNX graphs, and Clef. `--max-batch-tokens`, `--max-batch-padding-percent`, and optional [cross-request collation](docs/operations.md#cross-request-batches). |
| **Shared Kev prefixes** | Prefill shared state once, then fork question branches. Optional retained snapshots avoid repeated state prefill across requests. | Native CPU Kev: `--prefix-cache`, `--persistent-prefix-bytes`; [cached question batches](docs/operations.md#cpu-kev-batches-from-a-shared-prefix). |
| **Cooperative prefill scheduling** | Let queued requests run between bounded prefix chunks and question groups. | Native CPU Kev: `--cooperative-prefill` with prefix reuse and configured chunking; [configuration and limits](docs/operations.md). |
| **Bounded exact reuse** | Reuse tokenization, prepared prompts, and complete responses; coalesce identical in-flight HTTP requests. | F1–F4 tokenizer/prompt caches; engine result caches and HTTP coalescing. Disabled by default; [cache budgets and controls](docs/operations.md#serve-flags). |
| **Concurrent CPU contexts** | Serve through independently locked contexts sharing immutable weights; overlap bounded prompt preparation with inference. | Supported CPU runtimes: `--replicas`, `--max-prepared-per-model`; [replica support and limits](docs/operations.md#bounded-shared-weight-cpu-replicas). |
| **Shared bases across adapters** | Retain one immutable CPU base instead of duplicating dense weights for each supported adapter. Lazy residency bounds loaded model groups. | Optional `shared-base` build and CPU FP32 F2/F3 runtime LoRA; [adapter execution](docs/operations.md#cpu-fp32-runtime-lora) and [lazy residency](docs/operations.md#optional-cpu-lazy-residency). |
| **Lean heads and calibration** | Project only requested F3 candidates, compute selected Laya final-layer marker queries, vectorize Clef heads, and normalize owned logits in their existing buffer. | Family-specific [head profiles](docs/backends.md); shared-core [calibration buffer reuse](docs/verification/calibration-buffers-20261009/README.md) preserves tested probability bits. |
| **CPU compute and attention tuning** | Optional OpenBLAS matmuls, fused MLP gates, bounded attention query blocks, and grouped K/V without head expansion. Direct CPU FP32 Kev page attention shares prefix K/V across suffix rows. | Native Qwen/Clef profiles and optional `cpu-blas`; [kernel profiles](docs/backends.md) and [direct page attention](docs/operations.md#direct-cpu-kev-page-attention). |
| **Browser inference workers** | Move loading, tokenization, qualification, and CPU inference off the page thread; use native equal-length or masked F1 graph batches. | Separate [CPU WASM browser package](browser/README.md), with bounded dedicated workers and fresh per-session gates. |

Optional [llama.cpp](docs/llamacpp.md) and [vLLM](docs/vllm.md) CPU backends also
execute decision readouts without a decode loop. The pinned vLLM Kev path
supports equal-length native batches and local two-rank CPU tensor sharding.
Experimental CPU Q8_0/Q4_0 packages are available behind `quantization`, with
[variant refits and held-out acceptance](docs/quantization.md) required.

Most tuning controls are opt-in and depend on the family, backend, and execution
profile. Every real serving context requires fresh complete labeled conformance;
selected batching/prefix paths also require paired independent-forward checks.
Caches cannot bypass those gates. The default build keeps external runtimes
optional. Impact depends on the workload: the [implementation roadmap and
evidence](docs/optimization-roadmap.md) record current coverage, measurements,
and qualification limits separately from released-model acceptance.

## Crates

| Crate | Purpose |
|---|---|
| `huncho-core` | Wire contract, model packaging, prompt building, heads, calibration, conformance, engine. |
| `huncho-backend` | Offline mock/null backends; optional ONNX, Candle/ModernBERT, Qwen/Kev, Clef, llama.cpp, and vLLM runtimes. |
| `huncho-api` | HTTP API: `/v1/systemone`, `/health`, `/v1/models`, `/metrics`, auth. |
| `huncho-hub` | Hugging Face Hub resolution of model packages by repo id (feature `hf`). |
| `huncho-cli` | `huncho serve`, `convert`, `calibrate`, `conform`, `bench`. |

## Quick start

The [RPM package](packaging/rpm/README.md) bundles CPU and CUDA runtimes behind
one `huncho` command. Kev and Clef automatically use a compatible NVIDIA GPU when the
driver and runtime are available, otherwise CPU. Set `HUNCHO_DEVICE=cpu`
to force CPU or `cuda` / `cuda:N` to require a GPU. See [GPU setup](docs/gpu-setup.md).

Serve a built-in deterministic mock model (no weights required):

```bash
cargo run --release -p huncho-cli -- serve --mock --bind 127.0.0.1:8080
```

Query it:

```bash
curl -s http://127.0.0.1:8080/v1/models

curl -s -X POST http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The customer wants a refund because the shoes are too small.",
    "model": "mock",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team handles this?",
        "criteria": { "returns": "Refund request", "billing": "Charging issue" }
      },
      "is_refund": {
        "type": "noul",
        "instructions": "Is the customer requesting a refund?"
      }
    }
  }'
```

Registering a model package with the offline mock backend:

```bash
cargo run --release -p huncho-cli -- serve --manifest examples/mock-model/huncho-model.json --backend mock
```

Registering a model package from the Hugging Face Hub by repo id (build with
`--features onnx,hf,tokenizers` for an ONNX package):

```bash
cargo run --release -p huncho-cli --features onnx,hf,tokenizers -- serve --model my-org/laya \
  --qualification-golden 'MODEL_NAME=/path/to/pinned-labeled-golden.json'
```

`serve`, `bench`, and `conform` select a backend for each model automatically.
Clef uses its native runtime; supported safetensors packages use Candle; ONNX
packages use ONNX Runtime. `--backend` is an optional override. The required
runtime must be included in the build; a missing runtime produces a build hint.

Every real runtime, including unoptimized packages marked `fit`, requires fresh
complete labeled startup conformance before serving. Use the manifest's name
for `MODEL_NAME` and keep fitting inputs separate from the pinned evaluation
suite. A failed variant remains rejected. Numerical-only `conform` and `bench`
are available for analysis; see [calibration gates](docs/calibration.md).

The manifest and the selected backend's artifacts are fetched into the HF cache,
pinned to the resolved commit, so `serve --model` is deterministic across runs.

Serve Kev-4B natively with Candle (add the `cuda` feature for GPU support):

```bash
cargo build --release -p huncho-cli --features hf,candle,tokenizers
./target/release/huncho serve --model jaredpalmer/kev-4b --bind 127.0.0.1:8080 \
  --qualification-golden 'kev-4b=/path/to/pinned-labeled-golden.json'
```

The first load downloads the adapter, tokenizer, pointer head, and the base
checkpoint pinned by `head.pt`. Requests use `"model": "kev-4b"`. The default
is fp16 on CUDA and fp32 on CPU; `--dtype` overrides it. This path supports Qwen3.5 Kev
LoRA checkpoints with up to 8,192 tokens per state-plus-question row. See
[native Kev support](docs/backends.md#kev-f2-on-candle) for the loading contract
and current limits. The retained CPU FP32 held-out profile currently fails the
probability-delta gate and is not accepted for serving.

Inspect a native ONNX artifact (`examples/mock-model` ships a 512-dim numerical
fixture with no observed-outcome acceptance). Build with `--features onnx` (fetches a prebuilt ONNX
Runtime at build time):

```bash
cargo run --release -p huncho-cli --features onnx -- \
  bench --manifest examples/mock-model/huncho-model.json --iterations 5

cargo run --release -p huncho-cli --features onnx -- conform \
  --manifest examples/mock-model/huncho-model.json \
  --golden examples/mock-model/golden.json
```

The same commands with `--backend mock` drive the in-process reference and need
no ONNX build. For real models also pass `hf` (Hub resolution) and `tokenizers`
(byte-identical reference prompting): `--features onnx,hf,tokenizers`.

Serve a **real Hugging Face checkpoint without any external runner** by loading
the `.safetensors` directly with `candle` (HF's Rust framework). Build with
`--features candle` and point at a package that lays out `config.json` +
`model.safetensors` next to the manifest:

```bash
cargo run --release -p huncho-cli --features candle -- \
  serve --manifest my-laya/huncho-model.json \
  --qualification-golden 'MODEL_NAME=/path/to/pinned-labeled-golden.json'
```

`CandleBackend` loads `convaiinnovations/laya`'s ModernBERT encoder (remapping
its `encoder.*` key prefix and normalizing its `rope_parameters` config) and
feeds the option-marker hidden states into the F1 head. If you've downloaded
Laya's checkpoint, `huncho convert --backend candle --source <checkout>` builds
the package for you (handling `encoder/config.json` and sharded safetensors).
See [docs/backends.md](docs/backends.md#candle-feature).

Run the offline conformance harness against the mock reference:

```bash
cargo run --release -p huncho-cli -- conform \
  --golden examples/mock-model/golden.json
```

## Model package

Cloudflare Clef has a native Rust/Candle backend for text and JSON:

```bash
cargo build --release -p huncho-cli --features clef
./target/release/huncho serve --model Cloudflare/clef \
  --qualification-golden 'MODEL_NAME=/path/to/pinned-labeled-golden.json'
```

No Python, pip dependencies, or model conversion is required. The command above
builds for CPU; `--features cuda` enables NVIDIA support. Both score all
questions jointly. See [Clef setup and limits](docs/clef.md), including the
current validation scope and full-model memory requirements.

A single `huncho-model.json` manifest pins the family, backbone, head, prompt
contract, and per-backend calibration. See [docs/model-package.md](docs/model-package.md).

Experimental CPU Kev Q8_0/Q4_0 conversion uses the optional `quantization`
feature. Packages remain pending until a variant refit and fresh labeled
conformance pass. See [docs/quantization.md](docs/quantization.md).

## Testing

The default build is offline and testable without weights:

```bash
cargo test                       # core + hub + api + cli
cargo test -p huncho-backend --features onnx   # ONNX backend integration
cargo test -p huncho-backend --features candle # candle (safetensors) backend
cargo test -p huncho-core --features tokenizers --test hf_tokenizer  # CORE-02
```

Coverage includes the `/v1/systemone` HTTP contract (choice/noul/score, 422,
auth, extensions, `/metrics`), `convert`/`calibrate`/`bench`, F1–F4 prompt
building, the ONNX conformance suite, and the candle backend (load, position
extraction, determinism, and manifest-package resolution).

## Operation / packaging

A multi-stage `Dockerfile` builds a slim image with `onnx,hf,tokenizers` and a
non-root user; a hardened `deploy/huncho.service` + `deploy/huncho.env` example
cover systemd. `serve` reads `HUNCHO_BIND`/`HUNCHO_BACKEND`/`HUNCHO_DTYPE`/
`HUNCHO_CACHE_DIR`/`HUNCHO_AUTH_TOKEN`. See [docs/operations.md](docs/operations.md).

## Documentation

- [API contract](docs/API.md)
- [Model package format](docs/model-package.md)
- [Backends](docs/backends.md)
- [Calibration & confidence](docs/calibration.md)
- [Operations](docs/operations.md)
- [Optimization roadmap & evidence](docs/optimization-roadmap.md)
- [CPU browser runtime](browser/README.md)

## License

Apache-2.0.
