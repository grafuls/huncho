# Backends

Every backend implements the same [`Backend`](../crates/s1-core/src/backend.rs)
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
| `MockBackend` | `s1-backend` | ✅ built-in | Deterministic, dependency-free. The **offline reference** for the conformance harness and demos. |
| `NullBackend` | `s1-backend` | ✅ built-in | Always-empty output, useful for tests / shelling out. |
| `OnnxBackend` | `s1-backend` | ⚙️ feature-gated | ONNX Runtime (CPU/CUDA via EPs). Built with the `onnx` feature (off by default). |

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
cargo build --release -p s1-cli --features onnx
```

With the feature enabled, `s1 serve --manifest ...` will load the ONNX artifact
declared in the manifest for the requested dtype. Without the feature, manifest
loads fall back to the mock backend, which is why `s1 serve --manifest ...`
demos work without weights.

## Backend selection in the CLI

- `s1 serve --mock` — serves a built-in deterministic mock model (no weights).
- `s1 serve --manifest <path> --backend mock` — serves a manifest using the mock
  backend (offline demo).
- `s1 serve --manifest <path> --backend onnx` — serves a manifest using ONNX.
- `s1 conform --backend mock` — runs conformance against the mock reference.

## Adding a backend

Implement [`Backend`](../crates/s1-core/src/backend.rs) and register it in the
cli's `load` helper. A new family requires a manifest, a head implementation,
and golden vectors — nothing else.

## Portability target

Refer to the PRD for the full matrix: CUDA, Metal, x86/ARM CPU (Raspberry Pi as
the floor), and browser (WebGPU/WASM) from one codebase. v1 uses existing
runtimes for the backbone forward pass; the engine itself writes no custom
kernels.
