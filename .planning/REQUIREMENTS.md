# Requirements: System One Engine (`s1`)

**Defined:** 2026-09-26
**Core Value:** Guarantee that calibrated probabilities stay calibrated across every backend and quantization, served behind one API for all open decision-model families.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### API

- [ ] **API-01**: Service accepts `POST /v1/systemone` with the Jev-compatible shapes (`state` string or JSON, `model`, `questions` map with `choice`/`score`/`noul` types, `instructions`, `criteria`), and the unmodified TypeSafe Python SDK works against it with only a base-URL change; unknown model names return 422.
- [ ] **API-02**: Service exposes `GET /health` and `GET /v1/models`.
- [ ] **API-03**: Service supports optional bearer-token auth via env var; malformed or missing headers return 401.
- [ ] **API-04**: Service exposes Prometheus `/metrics` with request latency (p50/p95/p99), queue depth, tokens prefilled, fork count, and per-model counts.
- [ ] **API-05**: Service exposes a documented engine-extension field (raw per-option logits, backend id, calibration status), off by default so default responses stay strictly Jev-shaped.

### Core

- [ ] **CORE-01**: Engine loads and validates the model package (manifest) per family, backend, adapter, head, prompt contract, and calibration entries.
- [ ] **CORE-02**: Engine tokenizes and builds prompts per family byte-identical to the reference implementation; token ids match the reference on conformance inputs.
- [ ] **CORE-03**: Engine implements the F1 (option-marker) and F2 (pointer) heads.
- [ ] **CORE-04**: Engine implements the F3 (candidate-logit) and F4 (slot head) heads.
- [ ] **CORE-05**: Engine computes calibrated probabilities (temperatures, softmax, confidence) following the documented Jev or per-model confidence definition.
- [ ] **CORE-06**: Engine prefills the state once and forks per question (F2); packed multi-question probabilities equal single-question results within 1e-4.
- [ ] **CORE-07**: Engine schedules with dynamic batching across requests; configurable max batch tokens and max wait.
- [ ] **CORE-08**: Engine enforces per-model input limits (context length, max options, per-field budgets) and rejects oversized input with a clear error.
- [ ] **CORE-09**: Engine routes to a checkpoint by detected script (e.g. Laya English vs. multilingual), configurable per deployment.

### Backends

- [ ] **BE-01**: ONNX Runtime backend supports x86 and ARM64 CPU (target F1).
- [ ] **BE-02**: llama.cpp backend supports CPU, CUDA, and Metal with KV fork via sequence copy (target F2 on Qwen3 dense).
- [ ] **BE-03**: MLX backend for Apple Silicon (targets F2, F4).
- [ ] **BE-04**: vLLM plugin backend for datacenter GPUs via vLLM's pooling/classification path (targets F1, F3, F2 via prefix caching).

### Conversion

- [ ] **CONV-01**: CLI `s1 convert` produces backend artifacts (ONNX, GGUF, MLX) plus the manifest from an HF repo + revision.
- [ ] **CONV-02**: CLI `s1 calibrate` fits temperatures for a given backend × dtype on a held-out set and writes them into the manifest.
- [ ] **CONV-03**: Quantization presets (fp16, int8, 4-bit) run a mandatory conformance check after conversion.

### Conformance & Eval

- [ ] **CONF-01**: Golden-vector generator uses the pinned reference implementation of each family.
- [ ] **CONF-02**: `s1 conform` runs golden vectors on any backend and reports max probability delta, argmax agreement, and ECE drift against configurable pass/fail thresholds.
- [ ] **CONF-03**: CI matrix runs conformance for every supported (backend, model, dtype) combination; a failure blocks the release.
- [ ] **CONF-04**: `s1 bench` measures latency and throughput per backend and hardware with a fixed request mix (1, 5, 20 questions; short and long states).
- [ ] **CONF-05**: Option-order sensitivity test shuffles choice options and reports how often answers flip.

### Packaging & Ops

- [ ] **OPS-01**: Single static binary per platform (Linux x86_64/aarch64, macOS arm64).
- [ ] **OPS-02**: OCI container images (CPU, CUDA), rootless-friendly and documented as SELinux-compatible (`:Z` volume labels).
- [ ] **OPS-03**: systemd unit plus env-file example.
- [ ] **OPS-04**: Model cache directory with lazy load and optional preload flag; idle eviction.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### API

- **API-06**: gRPC and an in-process library API (Rust crate + Python bindings).
- **API-07**: MCP server exposing decisions as a tool.

### Core

- **CORE-10**: Multi-LoRA — many adapters on one resident base, selected per request via `model`.

### Backends

- **BE-05**: ONNX Runtime Web (WebGPU + WASM fallback) build for the browser.
- **BE-06**: CoreML and DirectML execution providers via ONNX Runtime.

### Packaging & Ops

- **OPS-05**: RPM spec / Fedora COPR package.

### Eval & Localization

- **EVAL-01**: Spanish-language conformance and eval set for multilingual checkpoints (Qwen-based models).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Text generation | This engine serves decision heads; no generative runtime |
| Training / fine-tuning | Engine loads adapters, does not produce them |
| Custom kernel authoring in v1 | Reuse existing runtimes for the backbone forward pass |
| Microcontroller hosts (ESP32) | Single-board computers are clients of a gateway, not hosts |
| Image / audio / video input | Matches Jev — not supported |
| Multi-LoRA in v1 | P2; deferred to production milestone |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| API-01 | Phase 1 | Pending |
| API-02 | Phase 1 | Pending |
| API-03 | Phase 3 | Pending |
| API-04 | Phase 3 | Pending |
| API-05 | Phase 4 | Pending |
| CORE-01 | Phase 1 | Pending |
| CORE-02 | Phase 1 | Pending |
| CORE-03 | Phase 1 | Pending |
| CORE-04 | Phase 3 | Pending |
| CORE-05 | Phase 1 | Pending |
| CORE-06 | Phase 2 | Pending |
| CORE-07 | Phase 3 | Pending |
| CORE-08 | Phase 3 | Pending |
| CORE-09 | Phase 4 | Pending |
| BE-01 | Phase 1 | Pending |
| BE-02 | Phase 2 | Pending |
| BE-03 | Phase 3 | Pending |
| BE-04 | Phase 3 | Pending |
| CONV-01 | Phase 1 | Pending |
| CONV-02 | Phase 2 | Pending |
| CONV-03 | Phase 3 | Pending |
| CONF-01 | Phase 1 | Pending |
| CONF-02 | Phase 1 | Pending |
| CONF-03 | Phase 3 | Pending |
| CONF-04 | Phase 3 | Pending |
| CONF-05 | Phase 3 | Pending |
| OPS-01 | Phase 4 | Pending |
| OPS-02 | Phase 4 | Pending |
| OPS-03 | Phase 4 | Pending |
| OPS-04 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 30 total
- Mapped to phases: 30
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-26*
*Last updated: 2026-09-26 after initial definition*
