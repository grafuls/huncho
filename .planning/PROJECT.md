# Huncho

## What This Is

A portable serving engine for Jev-style "System One" decision models: state + typed questions in, calibrated probabilities out, no text generation. It aims to be what llama.cpp/vLLM are for generative LLMs, running on every tier of hardware from datacenter GPUs to CPUs and browsers.

## Core Value

Guarantee that calibrated probabilities stay calibrated across every backend and quantization, served behind one API for all open decision-model families.

## Business Context

<!-- OPTIONAL — internal infrastructure/engine project. No external customer revenue model yet. -->

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Serve all four open decision-model families (F1 Encoder, F2 Pointer, F3 Candidate-logit, F4 Slot head) behind one API
- [ ] Implement the Jev `/v1/systemone` wire contract, drop-in compatible with the TypeSafe SDK and existing Jev clients
- [ ] Run on CUDA, Apple Silicon (Metal/MLX), x86/ARM CPU, and browser (WebGPU/WASM) from one codebase
- [ ] Enforce probability fidelity against a reference implementation via a conformance suite that gates releases in CI
- [ ] Support multi-tenant LoRA serving for per-client fine-tunes
- [ ] Efficient fan-out: one state, many questions, with the state prefilled once

### Out of Scope

- Text generation of any kind — this is a serving engine for decision heads, not a generative runtime
- Training or fine-tuning — the engine loads adapters; it does not produce them
- Writing custom kernels in v1 — reuse existing runtimes for the backbone forward pass
- Microcontroller hosts (ESP32 etc.) — those are clients of a gateway, not hosts
- Image, audio, or video input — Jev doesn't support it either

## Context

The engine's core abstraction is **backbone + head + calibration**. The four families differ in backbone, head/readout, and prefix-sharing behavior:

- **F1 Encoder** (`convaiinnovations/laya`): ModernBERT-large / mmBERT-base, scores candidates at option-marker positions, single bidirectional pass.
- **F2 Pointer** (`jaredpalmer/kev-*`): Qwen3/Qwen3.5 base + LoRA r=16, pointer head over option-boundary tokens, block-causal mask with shared state prefix and one isolated branch per question.
- **F3 Candidate-logit** (`bespokelabs/Bespoke-Nimble-9B`): Qwen3.5-9B + LoRA, softmax over one-token answer codes using the existing LM head.
- **F4 Slot head** (`iapp/OpenThai-SystemOne`): Qwen3.5-0.8B tower, 256-slot decision head + per-type temperatures.

Key facts driving the design:

- All families are **prefill-only** — there is no decode loop.
- Heads are tiny compared with backbones; head math can run in portable core code.
- Kev's block-causal mask is equivalent to: prefill the state, **fork the KV cache (and recurrent state, for hybrid layers)** per question, prefill each branch.
- Qwen3.5 backbones contain linear-attention (DeltaNet-style) layers with uneven runtime support; Qwen3 dense and ModernBERT are the most portable.

Available test hardware: 2× T4 (16 GB, Turing SM75), 8× A100-40GB lab node, ThinkPad P1 Gen 7 (Intel x86), Raspberry Pi 4/5, Apple Silicon (to be decided).

## Constraints

- **Backend (Turing/bf16)**: T4 has no native bf16; fp16 may overflow or drift. Treat fp16 as a first-class dtype with its own golden vectors and temperatures.
- **Recurrent-state forking**: Qwen3.5 hybrid layers must fork recurrent state alongside KV; verify before claiming support (start F2 on Qwen3 dense).
- **Calibration vs. quantization**: quantization can silently break calibration; every quantized variant requires a conformance run and possible temperature refit.
- **Upstream drift**: model repos change prompt contracts; pin revisions and hash the prompt contract in the manifest.
- **Jev contract drift**: the Jev contract is closed/external and may evolve; contract-test against published docs and version the API layer.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Name the app Huncho, with CLI `huncho` and crates `huncho-*` | User selected the name on 2026-09-26 | Confirmed |
| Backend abstraction as a `Backend` trait (`load`/`forward`/`fork`/`capabilities`) | Core never assumes a backend supports custom attention masks; fork required for F2 | — Pending |
| Correctness defined as probability fidelity vs. a pinned reference, enforced by conformance in CI | Guarantees calibrated probabilities stay calibrated across backends and quantizations | — Pending |
| Core language assumed Rust (small static binary, WASM target, HF `tokenizers`) | Portable to edge/browser; C++/Python remain candidates | — Pending (open question) |
| Prefill-only architecture with per-question KV fork | All families are prefill-only; fan-out needs forking, not decode | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-26 after initialization*
