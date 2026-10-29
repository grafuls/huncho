---
gsd_state_version: '1.0'
status: active
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 4
  completed_plans: 0
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-26)

**Core value:** Guarantee that calibrated probabilities stay calibrated across every backend and quantization, served behind one API for all open decision-model families.
**Current focus:** Phase 1 implemented (not formally closed); active work spans Phases 2-4 — Qualification/calibration gates, CPU batching/caching/paged attention, browser WASM, vLLM/llama.cpp backends, trial-mode serving, `huncho models` / `huncho qualify` commands, CI workflow, and packaging fixes.

## Current Position

Phase: 1 implemented (not formally closed); active work spans Phases 2-4
Plan: 0 of 1 in current phase
Status: In progress
Last activity: 2026-10-15 — Merged PR #19 (license fixes). Recent work includes qualification gates, CPU optimizations, browser WASM, vLLM/llama.cpp backends, and trial-mode serving.

Progress: [█████░░░░░] ~50% (rough estimate: Phase 1 implemented, Phases 2-4 partially implemented, no phase formally closed)

## Performance Metrics

**Velocity:**
- Total plans completed: 0 (Phase 1 plan not yet formally split/closed)
- Average duration: —
- Total execution time: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

- After init: chose Rust core, `Backend` trait abstraction, conformance-as-correctness-gate. See PROJECT.md.
- 2026-09-26: User selected Huncho as the app name; executable `huncho`, crates `huncho-*`.
- 2026-09-26: M0 built with `ort` `std`/`download-binaries`/`tls-native`; manifest-declared tokenizer loaded via HF `tokenizers` when enabled (CORE-02); `serve` config via clap `env` (HUNCHO_*).
- 2026-09-26: Real-Laya path is **candle** (not ONNX/external runner). `candle` loads Laya safetensors directly on CPU, remapping `encoder.*`→`model.*`, F16→F32, and normalizing `rope_parameters`. `huncho convert --backend candle` writes a `model.safetensors` artifact manifest. `huncho convert --backend candle --source <checkout>` assembles a servable package from a local HF checkpoint (handles `encoder/config.json`, sharded safetensors, and the tokenizer), so a downloaded Laya checkout becomes servable in one command with no external converter.

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

- GSD subagents not installed (`npx @opengsd/gsd-core@latest --global`); roadmap generated inline, research phase skipped for now.
- Apple Silicon hardware for Metal/MLX CI still to be decided (PRD open question #3). Blocks BE-03 (MLX) / BE-02 (Metal) verification.
- No real Laya/ONNX artifact is available upstream (no `huncho-model.json`, no ONNX). Serving real Laya now uses the **candle** backend (safetensors direct load), so the external `onnx` runner is no longer required for it. Fetching the ~842 MB weights is still a one-time manual step (kept out of the automated suite).
- Kev-8B on llama.cpp and Metal fp16 conformance (Phase 2 criteria) are still unvalidated.
- **Open issues** (as of 2026-10-29):
  - #2: Validate HF Hub artifact filenames before writing to snapshot cache.
  - #3: Secure `/etc/huncho/huncho.env` bearer token (0600 / systemd credentials).
  - #4: Stop defaulting runtime listener to unauthenticated `0.0.0.0`.
  - #5: Refactor per-question scoring pipeline (duplicated 6x).
  - #6: Refactor single `EvalOptions` builder; serve/bench/conform/API diverge.
  - #7: Refactor split the 2711-line engine god module and drop `supports_*` shadow fields.
  - #8: Refactor unify four duplicated byte-bounded caches and twin `PromptCache`/`ResponseCache`.
  - #9: Refactor fold backend batch-capability reporting into `Capabilities`; drop dead `retain_cache`.
  - #10: Fix EL9 `%check` must resolve `libonnxruntime` from the build tree, not the empty buildroot.
  - #12: Add CI running the cargo test matrix + coverage and run Rust tests in RPM `%check` (workflow added in PR #16, coverage gating/RPM tests open).
  - #13: Align Dockerfile `--features` with the spec and `COPY vendor/`.
  - #14: Exclude `docs/verification` and `*.log` from the RPM source tarball.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260926-sdr | Rename app to Huncho | 2026-09-26 | Working tree | [260926-sdr-rename-app-to-huncho](./quick/260926-sdr-rename-app-to-huncho/) |
| 260926-hf | Hugging Face Hub resolution (serve/calibrate/conform/bench by repo id) | 2026-09-26 | Working tree | — |
| 260926-progress | Download progress feedback when resolving HF model packages by repo id | 2026-09-26 | 8d93c40 | — |
| 260926-xet | Plain-HTTP download fallback for xet-backed Hub artifacts (fixes silent stall on large weights) | 2026-09-26 | 90f4071 | — |
| 260926-redirect | Follow relative 307 resolve-cache redirects in HTTP downloader (fixes fresh-cache config resolution) | 2026-09-26 | 67fe336 | — |
| 260926-progress | Single-line in-place progress bar (compact, block bar + % + rate + ETA) for both TTY/non-TTY | 2026-09-26 | 24dbca2 | — |
| 260926-tokenizers | Diagnose & fix uniform answers on real Laya: root cause was a silent `SimpleTokenizer` fallback (missing `tokenizers` feature), not corrupt weights; `load_tokenizer` now errors without the feature | 2026-09-26 | a900a81 | — |
| 260926-m0 | Phase 1 tracer bullet: F1 on ONNX Runtime behind `/v1/systemone`, conformance PASS | 2026-09-26 | 2ae8189 | — |
| 260926-tests | API integration tests, convert/calibrate/bench tests, F2/F4 prompt tests | 2026-09-26 | 2756c4a, 4b4be24, 3dcccbc, cf002f0, 04a8405 | — |
| 260926-core02 | Load manifest-declared HF tokenizer for byte-identical prompts | 2026-09-26 | 3d8e64a | — |
| 260926-ops | Dockerfile, systemd unit, env-file packaging | 2026-09-26 | f22a70a | — |
| 260926-candle | CandleBackend (safetensors→hidden states, Laya layout) + CLI wiring + `convert --source` assembly + tests | 2026-09-26 | ac052c5 | — |
| 261009-aut | Enforce bearer auth on every route (`/health`, `/v1/models`, `/metrics`), before routing and body parsing (GitHub #1) | 2026-10-09 | 38cc466 | [261009-aut-enforce-bearer-auth-on-all-routes](./quick/261009-aut-enforce-bearer-auth-on-all-routes/) |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Model from source | Laya / Nimble / OpenThai licenses to verify for redistributing converted weights | Open | 2026-09-26 | Init |
| Real model | ONNX export of real `convaiinnovations/laya` (superseded by candle; ONNX only if a future model needs an ONNX artifact) | Deferred | 2026-09-26 | M0 |
| Real model | Fetch & lay out Laya's ~842 MB checkpoint (`config.json`+`model.safetensors`) and verify `serve --backend candle` end-to-end | **Resolved** — `serve --model convaiinnovations/laya --backend candle` now auto-downloads via plain HTTP fallback | 2026-09-26 | M0 |
| Phase 2 | F2 pointer head + KV-fork fan-out on llama.cpp (needs Kev model + BE-02) | Open | 2026-09-26 | M1 |

## Session Continuity

Last session: 2026-10-15
Stopped at: Merged PR #19 (license fixes). Recent work includes qualification gates, CPU optimizations, browser WASM, vLLM/llama.cpp backends, and trial-mode serving.
Next step: Address open issues #2-#14, specifically focusing on security fixes (#3, #4), refactoring tasks (#5-#9), and CI/packaging improvements (#10, #12-#14).
