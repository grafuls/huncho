# Huncho Architecture

Huncho is a portable serving engine for Jev-style "System One" decision models. It accepts state and typed questions, returning calibrated probabilities without text generation. The architecture separates the HTTP interface, core logic, and hardware-specific backends to ensure probability fidelity across diverse hardware tiers, from datacenter GPUs to CPUs and browsers.

## Crate Structure

The project is organized into five crates under `crates/`, each with a specific responsibility:

*   **`huncho-core`**: The portable engine core. It implements the Jev `/v1/systemone` wire contract, model packaging, per-family prompt building, decision heads, and the calibration layer. It depends on `tokenizers` (optional) for native tokenization.
*   **`huncho-backend`**: Concrete implementations of the `Backend` trait. It provides inference runtimes (Mock, ONNX, Candle, llama.cpp, vLLM) and depends on `huncho-core`.
*   **`huncho-api`**: The HTTP API layer. It handles routing, authentication, metrics, and state management, depending on both `huncho-core` and `huncho-backend`.
*   **`huncho-hub`**: Hugging Face Hub resolution for model packages. It allows referencing models by repository ID and depends on `huncho-core` and optionally `huncho-backend`.
*   **`huncho-cli`**: The command-line interface. It provides commands like `serve`, `convert`, `calibrate`, and `conform`, depending on all other crates.

### Dependency Diagram

```text
       +----------------+
       |  huncho-cli    |
       +-------+--------+
               |
       +-------v--------+       +----------------+
       |  huncho-api    |       |  huncho-hub    |
       +-------+--------+       +-------+--------+
               |                        |
       +-------v------------------------v--------+
       |              huncho-backend             |
       +-------------------+---------------------+
                           |
                   +-------v--------+
                   |  huncho-core   |
                   +----------------+
```

`huncho-cli` also depends directly on `huncho-core`, `huncho-backend` and (optionally) `huncho-hub`; `huncho-hub` depends on `huncho-backend` only when its `candle` feature is enabled.

## Key Abstractions

### Backends
The `Backend` abstraction in `crates/huncho-core/src/backend.rs` defines the contract for inference runtimes. Concrete implementations live in `huncho-backend`:
*   **Mock**: `crates/huncho-backend/src/mock.rs` (deterministic, for tests).
*   **ONNX**: `crates/huncho-backend/src/onnx.rs` (CPU/CUDA, feature-gated).
*   **Candle**: `crates/huncho-backend/src/candle.rs` (HF safetensors).
*   **Qwen3.5**: `crates/huncho-backend/src/qwen3_5.rs` (Qwen3.5 runtime, feature-gated).
*   **Clef**: `crates/huncho-backend/src/clef.rs` (native whole-request inference).
*   **llama.cpp**: `crates/huncho-backend/src/llamacpp.rs` (CPU/CUDA/Metal, feature-gated).
*   **vLLM**: `crates/huncho-backend/src/vllm.rs` (datacenter GPU, feature-gated).

### Decision Heads (F1-F5)
The core implements decision heads for model families defined in the PRD. Logic resides in `crates/huncho-core/src/head.rs` and `crates/huncho-core/src/prompt/`:
*   **F1 (Encoder)**: Option-marker scoring (Laya).
*   **F2 (Pointer)**: Pointer head over option-boundary tokens (Kev).
*   **F3 (Candidate-logit)**: Softmax over one-token answer codes (Nimble).
*   **F4 (Slot head)**: 256-slot decision head (OpenThai-SystemOne).
*   **F5 (Joint schema)**: see the PRD.

### Calibration and Qualification
Calibration ensures probability fidelity across backends and quantizations.
*   **Core Logic**: `crates/huncho-core/src/calibration.rs` handles temperatures, softmax, and confidence computation.
*   **Conformance**: `crates/huncho-core/src/conformance.rs` and `crates/huncho-core/src/engine/qualification.rs` manage golden-vector checks and qualification gates.
*   **CLI Tools**: `crates/huncho-cli/src/calibrate.rs` fits temperatures, while `crates/huncho-cli/src/conform.rs` and `crates/huncho-cli/src/qualify.rs` run verification gates.

## Request Flow

1.  **HTTP API**: `crates/huncho-api/src/routes.rs` receives the request and `crates/huncho-api/src/state.rs` manages the model registry.
2.  **Core Engine**: `crates/huncho-core/src/engine.rs` orchestrates the request, using `crates/huncho-core/src/prompt.rs` to build prompts and `crates/huncho-core/src/tokenizer.rs` for tokenization.
3.  **Backend**: `crates/huncho-core/src/backend.rs` dispatches the forward pass to the concrete backend (e.g., `crates/huncho-backend/src/candle.rs`).
4.  **Head**: `crates/huncho-core/src/head.rs` processes hidden states/logits to produce scores.
5.  **Calibration**: `crates/huncho-core/src/calibration.rs` applies temperatures and computes confidence.
6.  **Response**: `crates/huncho-api/src/routes.rs` formats the JSON response.

## Where to Go Next

*   [Backends](backends.md): Details on runtime selection and feature flags.
*   [Calibration](calibration.md): Guide to fitting temperatures and qualification.
*   [PRD](PRD-huncho.md): Product requirements and model family definitions.
*   [Contributing](../CONTRIBUTING.md): Development setup and guidelines.
