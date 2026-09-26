# PRD: Huncho

A portable serving engine for Jev-style "System One" decision models: state + typed questions in, calibrated probabilities out, no text generation. It aims to be what llama.cpp/vLLM are for generative LLMs, for this new model class, on every tier of hardware from datacenter GPUs to CPUs and browsers.

---

## 1. Problem

Jev-style decision models (Laya, Kev, Nimble, OpenThai-SystemOne) arrived in September 2026. Each ships its own ad-hoc Python server, tied to one model family and usually one hardware path. There is no shared runtime that:

- serves every family behind one API (the Jev `/v1/systemone` wire contract),
- runs on CUDA, Apple Silicon, CPU/ARM, and browser from one codebase,
- **guarantees that calibrated probabilities stay calibrated** across backends and quantizations,
- supports multi-tenant LoRA serving for per-client fine-tunes.

Generative engines (vLLM, llama.cpp, MLX) run the backbones well but know nothing about decision heads, option scoring, prefix fan-out, or calibration.

## 2. Goals

1. One engine and one API for all current open decision-model families.
2. Portable: datacenter GPU, workstation GPU, Apple Silicon, x86/ARM CPU (Pi 4/5 as the floor), browser (WebGPU/WASM).
3. Correctness is defined as **probability fidelity** against a reference implementation, and it is enforced by a conformance suite in CI.
4. Drop-in compatible with the TypeSafe SDK and existing Jev clients (change the base URL only).
5. Efficient fan-out: one state, many questions, with the state prefilled once.

## 3. Non-goals

- Text generation of any kind.
- Training or fine-tuning (the engine loads adapters; it does not produce them).
- Writing custom kernels in v1. We use existing runtimes for the backbone forward pass.
- Microcontrollers (ESP32 etc.). Those are clients of a gateway, not hosts.
- Image, audio, or video input (Jev doesn't support it either).

## 4. Background: the four model families

The engine's core abstraction is **backbone + head + calibration**. The families differ as follows:

| Family | Models (HF) | Backbone | Head / readout | Prefix sharing |
|---|---|---|---|---|
| F1 Encoder | `convaiinnovations/laya` (root, `multilingual`, `typed-decisions` subfolders) | ModernBERT-large / mmBERT-base | Scores candidates at option-marker positions | N/A (bidirectional, single pass) |
| F2 Pointer | `jaredpalmer/kev-{0.6b,0.8b,4b,8b,9b,27b}` | Qwen3 / Qwen3.5 base + LoRA r=16 | Pointer head over option-boundary tokens; per-checkpoint fitted temperature | Block-causal: shared state prefix, one isolated branch per question |
| F3 Candidate-logit | `bespokelabs/Bespoke-Nimble-9B`, `-9B-v2` | Qwen3.5-9B + LoRA | Softmax over one-token answer codes using the existing LM head | Single pass per field |
| F4 Slot head | `iapp/OpenThai-SystemOne` | Qwen3.5-0.8B tower | 256-slot decision head + per-type temperatures (fp32) | Single forward pass |

Key facts that drive the design:

- All families are **prefill-only**. There is no decode loop.
- Heads are tiny compared with backbones. Head math can run in portable core code.
- Kev's block-causal mask is mathematically equivalent to: prefill the state, **fork the KV cache (and recurrent state, for hybrid layers)** per question, prefill each branch. Forking is widely supported; arbitrary attention masks are not.
- Qwen3.5 backbones contain linear-attention (DeltaNet-style) layers with recurrent state, and runtime support for those is uneven. Qwen3 dense and ModernBERT are the most portable.

## 5. Users

- **Integrators / agent builders** (primary): put a fast decision gate in front of LLM agents for routing, triage, and escalation. They need latency, a stable API, and trustworthy probabilities for thresholds.
- **Platform operators**: serve many client-specific fine-tunes on shared hardware. They need multi-LoRA, metrics, and auth.
- **Edge / app developers**: run a small decision model on-device or in-browser. They need a small binary and quantized models with known calibration.

## 6. Architecture

```
            ┌──────────────────────────────────────────────┐
 HTTP/gRPC  │  API layer: /v1/systemone (Jev contract),    │
 ─────────► │  /health, /v1/models, /metrics, auth         │
            ├──────────────────────────────────────────────┤
            │  Core (Rust)                                 │
            │   - Model package loader (manifest)          │
            │   - Tokenizer (HF tokenizers)                │
            │   - Prompt/contract builder per family       │
            │   - Scheduler: batching, prefix fan-out      │
            │   - Heads: option-marker, pointer,           │
            │     candidate-logit, slot                    │
            │   - Calibration: temperatures, softmax,      │
            │     confidence                               │
            ├──────────────────────────────────────────────┤
            │  Backend trait: forward(tokens, positions,   │
            │   kv_handle?) -> hidden_states | logits      │
            │   fork(kv_handle) -> kv_handle               │
            ├───────────┬───────────┬──────────┬───────────┤
            │ ONNX RT   │ llama.cpp │ MLX      │ vLLM      │
            │ CPU/CUDA/ │ CUDA/     │ Apple    │ plugin    │
            │ CoreML/   │ Metal/    │ Silicon  │ (DC GPU,  │
            │ WebGPU    │ Vulkan/CPU│          │ multi-LoRA)│
            └───────────┴───────────┴──────────┴───────────┘
```

### 6.1 Backend contract

Every backend implements:

- `load(package, device, dtype)`
- `forward(tokens, output_positions, kv_handle?) -> tensors` returns hidden states or logits **only at requested positions**
- `fork(kv_handle) -> kv_handle` copies KV and any recurrent state (required for F2; optional capability flag otherwise)
- `capabilities()` reports supported families, dtypes, fork support, LoRA support, and maximum context

The core never assumes a backend supports custom attention masks.

### 6.2 Model package format

A single manifest (`huncho-model.json`) plus artifacts, loadable by every backend:

- `family`: F1–F4
- `backbone`: HF repo + pinned revision, or a local path; plus per-backend converted artifacts (ONNX, GGUF, MLX)
- `adapter`: optional LoRA reference + revision
- `head`: type + weights (safetensors, always fp32)
- `prompt_contract`: template id, option-marker tokens, token budgets (e.g. head vs. state budget), max options
- `calibration`: temperatures **per (backend, dtype/quantization)** plus the eval set hash they were fitted on
- `reference`: pointer to golden conformance vectors

Nimble's `schema_config.json` is prior art for pinning the prompt contract.

### 6.3 Calibration and conformance (core differentiator)

- A reference implementation per family (the maker's official Python code, pinned) produces golden outputs: fixed inputs → probability vectors.
- Every backend × dtype × model combination must match the golden outputs within tolerance (max absolute delta on probabilities; argmax agreement; ECE drift bound).
- Quantized variants that fail the tolerance ship with **refitted temperatures** and are marked `calibration: refit`, or the variant is rejected.
- Conformance runs in CI and gates releases.

## 7. Requirements

Priority: **P0** = v1 must-have, **P1** = v1 should-have, **P2** = later.

### API

- **API-01 (P0)**: Implement `POST /v1/systemone` with request/response shapes compatible with the Jev public contract: `state` (string or JSON), `model`, `questions` map with `choice` / `score` / `noul` types, `instructions`, `criteria`.
  - AC: the stock TypeSafe Python SDK works against the engine with only a base-URL change.
  - AC: unknown model names return 422.
- **API-02 (P0)**: `GET /health`, `GET /v1/models`.
- **API-03 (P0)**: Optional bearer-token auth via env var; malformed headers return 401.
- **API-04 (P1)**: Prometheus `/metrics`: request latency (p50/p95/p99), queue depth, tokens prefilled, fork count, per-model counts.
- **API-05 (P1)**: A documented extension field for engine-specific extras (e.g. raw per-option logits, backend id, calibration status). It must be off by default so default responses stay strictly Jev-shaped.
- **API-06 (P2)**: gRPC and an in-process library API (Rust crate + Python bindings).
- **API-07 (P2)**: MCP server exposing decisions as a tool.

### Core

- **CORE-01 (P0)**: Model package loader and manifest validation (§6.2).
- **CORE-02 (P0)**: Tokenization and prompt building per family, byte-identical to the reference implementation.
  - AC: token ids match the reference on the conformance inputs.
- **CORE-03 (P0)**: Head implementations for F1 (option-marker) and F2 (pointer).
- **CORE-04 (P1)**: Head implementations for F3 (candidate-logit) and F4 (slot head).
- **CORE-05 (P0)**: Calibration layer: temperatures, softmax, confidence computation. Confidence must follow the documented Jev definition, or a per-model definition flagged in the response extension. Laya uses entropy-based confidence; Jev documents a peak-based one.
- **CORE-06 (P0)**: Prefix fan-out: prefill the state once and fork per question (F2).
  - AC: probabilities from a packed multi-question request equal the single-question results within 1e-4.
- **CORE-07 (P1)**: Scheduler with dynamic batching across requests; configurable max batch tokens and max wait.
- **CORE-08 (P1)**: Input limits enforced per model: context length, max options, and per-field budgets. Oversized input is rejected with a clear error, never silently truncated.
- **CORE-09 (P1)**: Script/language router: pick a checkpoint by detected script (e.g. Laya English vs. multilingual), configurable per deployment.
- **CORE-10 (P2)**: Multi-LoRA: many adapters on one resident base, selected per request via `model`.

### Backends

- **BE-01 (P0)**: ONNX Runtime backend: CPU (x86, ARM64) and CUDA EPs. Target: F1.
- **BE-02 (P0)**: llama.cpp backend with KV fork via sequence copy: CPU, CUDA, Metal, Vulkan. Target: F2 on Qwen3 dense (`kev-8b`), then Qwen3.5 checkpoints once recurrent-state forking is verified.
- **BE-03 (P1)**: MLX backend for Apple Silicon. Targets: F2, F4.
- **BE-04 (P1)**: vLLM plugin backend for datacenter GPUs, built on vLLM's pooling/classification path where possible. Targets: F1, F3, F2 via prefix caching.
- **BE-05 (P2)**: ONNX Runtime Web (WebGPU + WASM fallback) build for the browser.
- **BE-06 (P2)**: CoreML and DirectML execution providers via ONNX Runtime.

### Conversion tooling

- **CONV-01 (P0)**: CLI `huncho convert` producing backend artifacts (ONNX, GGUF, MLX) plus the manifest from an HF repo + revision.
- **CONV-02 (P0)**: CLI `huncho calibrate` fits temperatures for a given backend × dtype on a held-out set and writes them into the manifest.
- **CONV-03 (P1)**: Quantization presets (fp16, int8, 4-bit) with a mandatory conformance run after conversion.

### Conformance and eval

- **CONF-01 (P0)**: Golden-vector generator using the pinned reference implementation of each family.
- **CONF-02 (P0)**: `huncho conform` runs the golden vectors on any backend and reports max probability delta, argmax agreement, and ECE drift. Pass/fail thresholds live in config.
- **CONF-03 (P0)**: CI matrix: every supported (backend, model, dtype) runs conformance. A failure blocks the release.
- **CONF-04 (P1)**: `huncho bench`: latency and throughput per backend and hardware, with a fixed request mix (1, 5, and 20 questions; short and long states).
- **CONF-05 (P1)**: Option-order sensitivity test: shuffle choice options and report how often answers flip.

### Packaging and ops

- **OPS-01 (P0)**: A single static binary per platform (Linux x86_64/aarch64, macOS arm64).
- **OPS-02 (P1)**: OCI container images (CPU, CUDA). Rootless-friendly and SELinux-compatible (`:Z` volume labels documented).
- **OPS-03 (P1)**: systemd unit + env-file example.
- **OPS-04 (P1)**: Model cache directory; lazy load with an optional preload flag; idle eviction.
- **OPS-05 (P2)**: RPM spec / Fedora COPR package.

## 8. Milestones

Tracer-bullet first: a thin end-to-end slice before breadth.

### M0: Tracer bullet
Laya (F1) on the ONNX Runtime CPU backend behind `/v1/systemone`, with one conformance test passing.
- Covers: API-01 (choice + noul only), API-02, CORE-01/02/03 (F1 only), CORE-05, BE-01 (CPU), CONV-01 (ONNX only), CONF-01/02 (Laya only).
- Exit: the TypeSafe SDK gets correct, reference-matching answers from a local CPU server.

### M1: Pointer family + fork
Kev-8B (F2, Qwen3 dense) on llama.cpp with KV-fork fan-out.
- Covers: CORE-03 (F2), CORE-06, BE-02 (CPU + CUDA + Metal), score type in API-01, CONV-02.
- Exit: packed multi-question results equal single-question results (CORE-06 AC); conformance passes in fp16 on CUDA and Metal.

### M2: Breadth
All four families; MLX and vLLM backends; quantization with refit calibration.
- Covers: CORE-04, CORE-07, CORE-08, BE-03, BE-04, CONV-03, CONF-03/04/05, API-03/04.
- Exit: CI conformance matrix green across (backend × family × dtype) for the supported set.

### M3: Production
Multi-LoRA, routing, packaging, and edge.
- Covers: CORE-09, CORE-10, OPS-*, API-05, BE-05.
- Exit: one T4 box serves several Kev-4B adapters concurrently; a browser demo runs Kev-0.8B.

## 9. Success metrics

- **Fidelity**: conformance pass rate 100% for released combinations; max probability delta ≤ 1e-3 (fp16/fp32), with refit and documentation required for quantized variants.
- **Compatibility**: the unmodified TypeSafe SDK passes an integration suite.
- **Latency**: engine overhead (excluding backbone forward) < 1 ms p50 per request.
- **Fan-out efficiency**: a 20-question request costs < 2× a 1-question request on F2.
- **Portability**: M2 exit covers at least CUDA, Metal, and x86 + ARM CPU.

## 10. Test hardware (available)

- 2× bare-metal servers with a single Tesla T4 each (16 GB, Turing SM75): primary CUDA and throughput target.
- 8× A100-SXM4-40GB lab node: large-model (Kev-27B, Nimble-9B) and vLLM plugin testing.
- ThinkPad P1 Gen 7 (Fedora 44, Intel): x86 CPU backend and development.
- Raspberry Pi 4/5: ARM CPU floor.
- Apple Silicon: needed for Metal/MLX; source to be decided.

## 11. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **T4 (Turing) has no native bf16.** Several references run in bf16. | fp16 may overflow or drift; calibration mismatch | Treat fp16 as a first-class dtype with its own golden vectors and temperatures; overflow checks in conformance |
| Qwen3.5 hybrid (linear-attention) layers are unsupported or slow on some runtimes; recurrent state must be forked alongside KV | F2/F3/F4 portability gaps | Start F2 on Qwen3 dense (`kev-8b`); capability flags per backend; verify recurrent-state fork before claiming support |
| Quantization silently breaks calibration | Downstream thresholds become wrong | Mandatory conformance + refit (CONV-03, CONF-03) |
| Upstream model repos change prompt contracts | Silent accuracy loss | Pin revisions; prompt-contract hash in the manifest; CI re-check |
| Jev contract evolves (closed, external) | Compatibility drift | Contract tests against the published docs; version the API layer |
| Fast-moving ecosystem (new families weekly) | Scope creep | New families require a manifest, a head implementation, and golden vectors; nothing else |
| Reference implementations disagree with each other on confidence definitions | Confusing semantics | Document per model; expose the definition in the extension field |

## 12. Open questions (resolve before M0 planning)

1. **Distribution:** open-source project, or an internal component of the Hermes consultancy offering? This decides license, how early the plugin and model-format ecosystem matter, and multi-LoRA priority.
2. **Core language:** Rust (assumed here: small static binary, WASM target, HF `tokenizers`) vs. C++ (closer to llama.cpp) vs. Python (fastest to M0, poor for edge and browser).
3. **Apple Silicon hardware:** which machine will run Metal/MLX CI?
4. **Spanish-language priority:** should M1/M2 include a Spanish conformance and eval set (multilingual checkpoints, Qwen-based models)?
5. **Licensing of converted artifacts:** confirm each upstream license permits redistribution of converted weights (Laya, Nimble, and OpenThai are Apache-2.0; verify Kev).

## 13. References

- Jev / System One contract: https://docs.typesafe.ai
- Laya: https://huggingface.co/convaiinnovations/laya · docs https://nandhakishorm.github.io/laya
- Kev: https://github.com/jaredpalmer/kev
- Nimble: https://github.com/bespokelabsai/nimble
- OpenThai-SystemOne: https://huggingface.co/iapp/OpenThai-SystemOne
- Existing Jev-compatible servers (behavioral reference): `laya-serve` (PyPI), `nvkudva/laya-server`, `kev.serve`
