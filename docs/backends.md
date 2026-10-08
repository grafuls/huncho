# Backends

Every backend implements the same [`Backend`](../crates/huncho-core/src/backend.rs)
trait:

- `forward(tokens, output_positions, kv_handle?) -> hidden_states | logits` —
  returns tensors **only at the requested positions**.
- `fork(kv_handle) -> kv_handle` — copies KV (and any recurrent state) for F2
  prefix fan-out. Optional capability flag otherwise.
- `capabilities()` — reports supported families, dtypes, fork support, LoRA
  support, and maximum context.

The core never assumes a backend supports custom attention masks. For F2, the
block-causal approach is realized as: prefill the state **once**, `fork` the KV
cache per question, and prefill each branch.

## Available backends

Clef uses a whole-request backend (`forward_request`) because its joint head
scores all questions together. See [Clef setup](clef.md).

| Backend | Crate | Status | Notes |
|---|---|---|---|
| `MockBackend` | `huncho-backend` | ✅ built-in | Deterministic, dependency-free. The **offline reference** for the conformance harness and demos. |
| `ClefBackend` | `huncho-backend` | `clef` | Native Rust/Candle Qwen3.5 and trained F5 joint schema head; CPU, optional CUDA. No Python runtime. |
| `NullBackend` | `huncho-backend` | ✅ built-in | Reports an unloaded-backend error; placeholder for unavailable models. |
| `OnnxBackend` | `huncho-backend` | ⚙️ feature-gated | ONNX Runtime on CPU today; CUDA EP registration and device I/O binding remain pending. Built with `onnx`. |
| `CandleBackend` | `huncho-backend` | ⚙️ feature-gated | Native ModernBERT/Laya, FP32 on CPU by default; explicit optional CUDA path requires labeled qualification. Built with `candle`; GPU kernels require `cuda`. |
| `Qwen3_5Backend` | `huncho-backend` | ⚙️ feature-gated | Kev F2 pointers and F3 candidate logits over Qwen3.5/merged LoRA. CPU, optional CUDA; F3 GPU selection is explicit and requires labeled qualification. |

`MockBackend` emits a `Features` (hidden-state) output so the engine's
feature-projection heads (F1/F2/F4) and the mean-fallback projection are all
exercised without real weights. Each position's hidden vector is a sparse,
deterministic activation derived from the token id at that position, so results
are reproducible across runs and platforms.

## Capability flags

Capabilities are per-backend and reported through `capabilities()`. They include:

- `id` — backend id.
- `dtype` — the dtype this instance serves.
- `max_context` — maximum supported context length.
- `supports_fork` — whether KV/recurrent-state forking is available (required
  for F2 fan-out).
- `supports_lora` — whether multi-LoRA is supported.
- `families` — which decision families the backend can serve.

## ONNX feature

The ONNX backend is off by default to keep the default build dependency-free.
Enable it with:

```bash
cargo build --release -p huncho-cli --features onnx
```

With the feature enabled, `huncho serve --manifest ...` will load the ONNX artifact
declared in the manifest for the requested dtype. Without the feature, an
ONNX-only package reports a build error. Use `--backend mock` explicitly for an
offline demo.

The ONNX feature also enables `ort`'s `download-binaries` and `tls-native`
features, so a build with this feature:

- downloads a prebuilt ONNX Runtime at build time (needs network access), and
- links against the system TLS stack (native-tls / OpenSSL) for the download
  transport.

For offline/vendored builds or a fully static binary (OPS-01), provide
onnxruntime yourself: set `ORT_LIB_PATH` and disable `download-binaries`, or
build with a different `ort` provider. The default (`no features`) build is
unaffected.

## Candle feature

The `candle` backend removes the need for any external runner (ONNX artifact,
Python, or a separate server process) for real HF checkpoints. `candle`
(Hugging Face's Rust framework) loads `.safetensors` weights directly and runs
the ModernBERT encoder on CPU, producing per-token hidden states that the
engine's feature-projection heads (F1/F2/F4) consume.

Enable it with:

```bash
cargo build --release -p huncho-cli --features candle
# for real HF tokenizer + Hub download support, add hf,tokenizers
```

> ⚠️ **Real models need the `tokenizers` feature.** A manifest-declared
> `backbone.tokenizer` is loaded with the official Hugging Face `tokenizers`
> crate only when that feature is enabled. Without it `load_tokenizer` now
> **errors** (it used to silently fall back to a crude `SimpleTokenizer`, which
> fed real models garbage token ids and produced near-uniform answers). For a
> real `convaiinnovations/laya` model from the Hub, build with:
>
> ```bash
> cargo build --release -p huncho-cli --features hf,candle,tokenizers
> ```

`CandleBackend` loads a package laid out as:

```
my-laya/              # package dir
├── huncho-model.json # manifest (artifact `model.safetensors`, dtype fp32)
├── config.json       # ModernBERT config (see below)
├── model.safetensors # the backbone weights (F16 is fine; load converts → F32)
└── tokenizer.json    # optional, if backbone.tokenizer is set
```

At load time it:

- remaps weight keys `encoder.*` → `model.*` (matching `convaiinnovations/laya`'s
  layout) and drops non-encoder tensors (`temperature`, `act_head.*`);
- converts F16/BF16 weights to F32 on CPU before optional device transfer;
- normalizes a transformers-5.0 `rope_parameters` config object into the flat
  `global_rope_theta` / `local_rope_theta` fields candle expects;
- runs the base `ModernBert` encoder (not `ForMaskedLM`) via
  `ModernBert::load`, then runs retained trained Laya `head.*`, `type_emb.*`
  and `scorer.*` tensors when present. Bare encoders return selected hidden
  states with `index_select`.

ModernBERT and F3 Qwen retain CPU defaults, including `HUNCHO_DEVICE=auto`.
`HUNCHO_DEVICE=cuda` or `cuda:N` explicitly selects their newly wired CUDA
paths in a `cuda` build; unavailable devices fail instead of falling back.
ModernBERT still executes FP32 and rejects other dtype labels. F3 honors its
supported requested dtype, keeps the vocabulary head on the same device, and
stages source conversion/LoRA merging on CPU. BF16 GPU execution requires
native BF16 support. Both paths report `device_path` execution metadata and
require a complete labeled startup suite in CLI serving. Actual T4 fixtures
pass BF16 source staging, candidate weights on device and batch row checks
([retained CUDA log](verification/kev-t4-20261007/stage5-cuda-fixtures.log)).
Fixture coverage does not qualify released Laya/Nimble checkpoints; those
full-model GPU checks remain pending. See [qualification controls](operations.md).

`huncho convert --backend candle` writes a manifest whose artifact is
`model.safetensors`. The weights/config are fetched separately (e.g. `hf
download convaiinnovations/laya model.safetensors` and copy
`encoder/config.json` to `config.json`), then `huncho serve --manifest
huncho-model.json` serves it. Because Laya's checkpoint is
~842 MB, this is intentionally kept out of the automated test suite.

### Assembling a package from a local checkpoint

If you have already downloaded a raw HF checkpoint (a directory containing
the chain's `config.json` (or `encoder/config.json` for Laya), the safetensors
weights (or shards), and optionally a `tokenizer.json`), `huncho convert` can
lay it out into a servable package in one step — no external converter needed:

```bash
huncho convert --backend candle \
  --hf-repo convaiinnovations/laya \
  --source ./laya-checkout \
  --out ./my-laya

huncho serve --manifest ./my-laya/huncho-model.json
```

`--source` copies `config.json` (or `encoder/config.json`) to `config.json`, the
`model.safetensors` weights (honoring a `model.safetensors.index.json` shard
map), and the manifest-declared tokenizer (`tokenizer.json`, from the checkpoint
root or a `tokenizer/` subdir) into the package. For Laya, first download the
checkpoint, e.g.:

```bash
hf download convaiinnovations/laya --local-dir ./laya-checkout
```

`--source` is only valid for the `candle` backend.

### Kev (F2) on Candle

With `--features hf,candle,tokenizers`, `serve --model jaredpalmer/kev-4b` loads
Kev directly. The Hub resolver distinguishes Kev's
PEFT `FEATURE_EXTRACTION` adapter from Nimble's candidate-logit adapter.
`schema_config.json` is a Nimble file and is not required for Kev.

The loader reads `head.pt` with Candle's restricted pickle reader, loads its
trained query/key projections in fp32, and applies its fitted temperature once
in the core calibration layer. The base model and revision come from that
checkpoint's metadata. All adapter files are pinned to the adapter snapshot;
base weights are pinned independently to the resolved base commit.

The `kev-v1` prompt contract uses Kev's state/question/option/decision tokens,
JSON rendering, caller-text escaping, option order, and yes/no descriptions.
Each question runs as an independent causal row, preserving question isolation.
The backend returns one raw pointer logit per option. Score confidence follows
Kev's ordered-level formula; choice confidence uses the existing peak formula.

Qwen3.5 LoRA checkpoints run on CPU (fp32 by default) or CUDA (fp16 by default),
with an FP32 pointer head on the selected device and a limit of 8,192 tokens
per complete row. Independent questions remain the default. Native KV/recurrent
prefix fan-out and exact-length question batching are opt-in and require
qualification; the native full Kev/T4 FP16 variants failed the tighter paired
gate. Experimental projection/attention profiles have separate identities and
gates. See [optimization validation](kev-optimization-validation.md).
Qwen3 checkpoints,
full-weight Kev releases, option isolation, and trained special embeddings are
rejected explicitly. This implementation does not claim the upstream server's
64k state window. Offline reference fixtures test prompt tokens and calibrated
probabilities against PyTorch on CPU and CUDA; see
[`tiny_kev`](../crates/huncho-backend/tests/fixtures/tiny_kev/README.md).

## Backend selection in the CLI

`serve`, `bench`, and `conform` default to `--backend auto`. Each model is
selected independently, so one server can load packages using different
runtimes without per-model flags:

```bash
huncho serve --model convaiinnovations/laya --model Cloudflare/clef
```

The resolver reads package artifacts and the decision family. Raw Hub releases
are recognized by their metadata, including Clef's joint-head configuration
and Kev/Nimble's adapter configuration; repository names are not hard-coded.
Raw local Clef directories are also recognized without a manifest.

Selection prefers Clef for F5 and Candle for supported safetensors models, then
ONNX for F1 exports. It considers the compiled runtimes and an explicit
`--dtype`. When a package supports both Candle and ONNX, Candle wins if available.
A missing runtime produces an error with build instructions. Automatic selection
never falls back to mock inference. Startup logs report the selected backend
and dtype for each model.

`--backend onnx`, `--backend candle`, or `--backend clef` overrides selection
for all supplied models. `HUNCHO_BACKEND` sets the same override for `serve`.
`--backend auto` restores automatic selection. Use `serve --mock` for a built-in
demo or `--backend mock` to run a package with the offline reference.
`bench` and `conform` without a model continue to use the built-in mock.

`convert --backend` still names an output artifact format, and
`calibrate --backend` identifies the runtime that produced the supplied logits.

## Adding a backend

Implement [`Backend`](../crates/huncho-core/src/backend.rs) and register it in the
cli's `load` helper. A new family requires a manifest, a head implementation,
and golden vectors — nothing else.

## Portability target

Refer to the PRD for the full matrix: CUDA, Metal, x86/ARM CPU (Raspberry Pi as
the floor), and browser (WebGPU/WASM) from one codebase. v1 uses existing
runtimes for the backbone forward pass; the engine itself writes no custom
kernels.

Clef also exposes an opt-in grouped head profile via
`HUNCHO_CLEF_VECTOR_HEAD=1` (default off). It groups option and residual projections
while preserving schema routing and candidate order. Changed GEMM/reduction
shapes require labeled startup qualification; metadata records
`joint_head_execution=vectorized-v1`. CPU fixture parity passes at fp32/fp16;
full released-model and GPU acceptance remain open.
