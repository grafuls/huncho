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

| Backend | Crate | Status | Notes |
|---|---|---|---|
| `MockBackend` | `huncho-backend` | ✅ built-in | Deterministic, dependency-free. The **offline reference** for the conformance harness and demos. |
| `NullBackend` | `huncho-backend` | ✅ built-in | Always-empty output, useful for tests / shelling out. |
| `OnnxBackend` | `huncho-backend` | ⚙️ feature-gated | ONNX Runtime (CPU/CUDA via EPs). Built with the `onnx` feature (off by default). |
| `CandleBackend` | `huncho-backend` | ⚙️ feature-gated | Loads Hugging Face **safetensors** directly via `candle` (CPU). Built with the `candle` feature (off by default). The primary path for real `convaiinnovations/laya`. |

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
declared in the manifest for the requested dtype. Without the feature, manifest
loads fall back to the mock backend, which is why `huncho serve --manifest ...`
demos work without weights.

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
- converts F16 weights to F32 for CPU inference;
- normalizes a transformers-5.0 `rope_parameters` config object into the flat
  `global_rope_theta` / `local_rope_theta` fields candle expects;
- runs the base `ModernBert` encoder (not `ForMaskedLM`) via
  `ModernBert::load`, then extracts hidden states at the requested option-marker
  positions with `index_select`.

`huncho convert --backend candle` writes a manifest whose artifact is
`model.safetensors`. The weights/config are fetched separately (e.g. `hf
download convaiinnovations/laya model.safetensors` and copy
`encoder/config.json` to `config.json`), then `huncho serve --manifest
huncho-model.json --backend candle` serves it. Because Laya's checkpoint is
~842 MB, this is intentionally kept out of the automated test suite.

## Backend selection in the CLI

- `huncho serve --mock` — serves a built-in deterministic mock model (no weights).
- `huncho serve --manifest <path> --backend mock` — serves a manifest using the mock
  backend (offline demo).
- `huncho serve --manifest <path> --backend onnx` — serves a manifest using ONNX.
- `huncho serve --manifest <path> --backend candle` — serves a manifest using the
  candle CPU backend (real `.safetensors` checkpoints).
- `huncho conform --backend mock` — runs conformance against the mock reference.

## Adding a backend

Implement [`Backend`](../crates/huncho-core/src/backend.rs) and register it in the
cli's `load` helper. A new family requires a manifest, a head implementation,
and golden vectors — nothing else.

## Portability target

Refer to the PRD for the full matrix: CUDA, Metal, x86/ARM CPU (Raspberry Pi as
the floor), and browser (WebGPU/WASM) from one codebase. v1 uses existing
runtimes for the backbone forward pass; the engine itself writes no custom
kernels.
