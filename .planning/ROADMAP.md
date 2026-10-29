# Roadmap: Huncho

## Overview

From a tracer-bullet slice (Laya/F1 on ONNX Runtime CPU) to a portable serving engine that runs all four decision-model families behind the Jev `/v1/systemone` contract — on datacenter GPUs, Apple Silicon, x86/ARM CPU, and browser — while enforcing calibrated-probability fidelity across every backend and quantization.

*Roadmap refreshed on 2026-10-29 from git history.*

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Tracer Bullet (M0)** - Laya (F1) on ONNX Runtime CPU behind `/v1/systemone`, one conformance test passing (implemented, not formally closed)
- [ ] **Phase 2: Pointer Family + Fork (M1)** - Kev-8B (F2) on llama.cpp with KV-fork fan-out
- [ ] **Phase 3: Breadth (M2)** - All four families; MLX and vLLM backends; quantization with refit calibration
- [ ] **Phase 4: Production (M3)** - Multi-LoRA, routing, packaging, and edge

## Phase Details

### Phase 1: Tracer Bullet (M0)
**Goal**: A thin end-to-end slice: Laya (F1) served on CPU behind `/v1/systemone`, with one conformance test passing.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: API-01 (choice + noul), API-02, CORE-01, CORE-02, CORE-03 (F1), CORE-05, BE-01 (CPU), CONV-01 (artifact), CONF-01, CONF-02
**Success Criteria** (what must be TRUE):
  1. The stock TypeSafe Python SDK returns correct, reference-matching answers from a local CPU server with only a base-URL change.
  2. `POST /v1/systemone` returns calibrated probabilities matching the Laya reference within tolerance on the conformance inputs.
  3. `GET /health` and `GET /v1/models` respond; unknown model names return 422.
  4. `huncho convert` produces an artifact + manifest from the Laya HF repo + pinned revision (via `--runner`, or directly for the candle path, including `--source <checkout>` to assemble a local HF download).
**Plans**: TBD

Plans:
- [ ] 01-01: [TBD]

> Note: M0 starter used ONNX on CPU. The real-Laya forward path is now **candle**
> (`--features candle`), which loads Laya's `.safetensors` directly and removes the
> external ONNX/Python runner. The ONNX backend is retained for models that ship an
> ONNX artifact.

### Phase 2: Pointer Family + Fork (M1)
**Goal**: Kev-8B (F2, Qwen3 dense) served on llama.cpp with KV-fork fan-out; add score-type questions and per-backend calibration.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: CORE-03 (F2), CORE-06, API-01 (score), BE-02 (CPU + CUDA + Metal), CONV-02
**Success Criteria** (what must be TRUE):
  1. Packed multi-question results equal the single-question results within 1e-4 (CORE-06).
  2. Conformance passes in fp16 on CUDA and Metal.
  3. `huncho calibrate` fits per-backend temperatures into the manifest.
  4. KV-fork forking prefills each question's isolated branch from a shared state prefix.
**Plans**: TBD

Plans:
- [ ] 02-01: [TBD]

### Phase 3: Breadth (M2)
**Goal**: All four families; MLX and vLLM backends; quantization with refit calibration; scheduling, auth, and metrics.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: CORE-04, CORE-07, CORE-08, BE-03, BE-04, CONV-03, CONF-03, CONF-04, CONF-05, API-03, API-04
**Success Criteria** (what must be TRUE):
  1. All four families serve behind `/v1/systemone`.
  2. CI conformance matrix is green across (backend × family × dtype) for the supported set.
  3. Quantized variants carry refitted temperatures and pass tolerance, or the variant is rejected.
  4. A 20-question request costs less than 2× a 1-question request on F2.
**Plans**: TBD

Plans:
- [ ] 03-01: [TBD]

### Phase 4: Production (M3)
**Goal**: Multi-LoRA serving, script/language routing, packaging (static binary, OCI, systemd), and browser edge.
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: CORE-09, CORE-10 (v2), OPS-01, OPS-02, OPS-03, OPS-04, API-05, BE-05 (v2)
**Success Criteria** (what must be TRUE):
  1. One T4 box serves several Kev-4B adapters concurrently.
  2. A browser demo runs Kev-0.8B.
  3. A single static binary per platform, OCI container images, and a working systemd unit + env-file example.
  4. The script/language router picks the correct checkpoint by detected script.
**Plans**: TBD

Plans:
- [ ] 04-01: [TBD]

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Tracer Bullet (M0) | 0/1 | Implemented (not formally closed) | Tracer bullet committed (CPU `/v1/systemone`, conformance PASS); API/convert/calibrate/bench tests; tokenizer wiring; candle backend (real-Laya path) |
| 2. Pointer Family + Fork (M1) | 0/1 | In progress | 8a0cb8a native Candle pointer-head; 205efc5 pinned CPU llama.cpp decision readouts; d549e2c native Kev prefix fan-out; c005373 collate Kev question forks from one immutable prefix; 571f93c Kev CUDA inference on T4 |
| 3. Breadth (M2) | 0/1 | In progress | e965457 Clef native joint-schema; f2f65d4 F3 Qwen3.5+LoRA; 69394cd vLLM CPU Kev raw pooling; 8f09121 Kev Q8/Q4 artifacts with strict calibration gates; 1db5305 `huncho models`/`huncho qualify`; 8b0c656 CI workflow (PR #16). MLX: no evidence. |
| 4. Production (M3) | 0/1 | In progress | 1edd7b8 CPU FP32 LoRA over shared Qwen bases; c9d27a8 browser CPU WASM inference; 7c2ca99 RPM package; 8aa0702 COPR; a7b6f1c CPU+CUDA in one package. Script/language routing: no evidence. |
| 4.1 Packaging (OPS-01/02/03) | 0/1 | Partial | Dockerfile, systemd unit, env-file added; RPM/COPR packaging (7c2ca99, 8aa0702) |
