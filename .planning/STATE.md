---
gsd_state_version: '1.0'
status: planning
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 4
  completed_plans: 0
  percent: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-26)

**Core value:** Guarantee that calibrated probabilities stay calibrated across every backend and quantization, served behind one API for all open decision-model families.
**Current focus:** Phase 1 (Tracer Bullet / M0) — tracer bullet complete; API/CLI test coverage + tokenizer wiring done.

## Current Position

Phase: 1 of 4 (Tracer Bullet / M0)
Plan: 0 of 1 in current phase
Status: In progress
Last activity: 2026-09-26 — Completed the M0 tracer bullet (ONNX behind `/v1/systemone`, conformance PASS), added API integration tests, convert/calibrate/bench tests, HF tokenizer wiring (CORE-02), and OPS packaging (Dockerfile/systemd/env).

Progress: [██░░░░░░░░] ~25%

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

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

- GSD subagents not installed (`npx @opengsd/gsd-core@latest --global`); roadmap generated inline, research phase skipped for now.
- Apple Silicon hardware for Metal/MLX CI still to be decided (PRD open question #3). Blocks BE-03 (MLX) / BE-02 (Metal) verification.
- No real Laya/ONNX artifact is available upstream (no `huncho-model.json`, no ONNX), so `convert` needs an external runner (e.g. `optimum-cli export onnx`) and there is no system Python onnx/transformers install — real Laya convert is the next unverified step.
- Phase 2 F2 depends on a Kev-8B model + llama.cpp backend (BE-02) and exact pointer-head/fork semantics, which are not yet validated.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260926-sdr | Rename app to Huncho | 2026-09-26 | Working tree | [260926-sdr-rename-app-to-huncho](./quick/260926-sdr-rename-app-to-huncho/) |
| 260926-hf | Hugging Face Hub resolution (serve/calibrate/conform/bench by repo id) | 2026-09-26 | Working tree | — |
| 260926-m0 | Phase 1 tracer bullet: F1 on ONNX Runtime behind `/v1/systemone`, conformance PASS | 2026-09-26 | 2ae8189 | — |
| 260926-tests | API integration tests, convert/calibrate/bench tests, F2/F4 prompt tests | 2026-09-26 | 2756c4a, 4b4be24, 3dcccbc, cf002f0, 04a8405 | — |
| 260926-core02 | Load manifest-declared HF tokenizer for byte-identical prompts | 2026-09-26 | 3d8e64a | — |
| 260926-ops | Dockerfile, systemd unit, env-file packaging | 2026-09-26 | f22a70a | — |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Model from source | Laya / Nimble / OpenThai licenses to verify for redistributing converted weights | Open | 2026-09-26 | Init |
| Real model | Convert real `convaiinnovations/laya` to ONNX (needs external runner, no upstream artifact) | Open | 2026-09-26 | M0 |
| Phase 2 | F2 pointer head + KV-fork fan-out on llama.cpp (needs Kev model + BE-02) | Open | 2026-09-26 | M1 |

## Session Continuity

Last session: 2026-09-26
Stopped at: M0 tracer bullet + Phase 1/CLI test coverage committed.
Next step: wire `huncho convert` to produce a real Laya ONNX artifact + manifest via an external `--runner` and verify `serve --model convaiinnovations/laya --backend onnx`.
