# s1 — System One Engine

`System One Engine` (`s1`) is a portable serving engine for Jev-style "System
One" decision models. It takes a **state** and a set of **typed questions** and
returns **calibrated probabilities** — no text generation. It is to decision
models what llama.cpp / vLLM are to generative LLMs, with one non-negotiable
guarantee: *calibrated probabilities stay calibrated* across every backend and
quantization.

```
state + typed questions ──► s1 ──► calibrated probabilities (no decode loop)
```

The engine implements the **Jev `/v1/systemone` wire contract**, so an unmodified
TypeSafe Python SDK works against `s1` with only a base-URL change.

## Highlights

- **Four decision-head families** behind one contract: F1 (option-marker), F2
  (pointer), F3 (candidate-logit), F4 (slot head).
- **Calibration as a first-class guarantee**: per-backend × dtype temperatures,
  softmax, and confidence (Jev peak-based, or per-model definitions).
- **Conformance gating**: an offline harness (`s1 conform`) compares any backend
  against golden vectors and gates releases on probability fidelity, argmax
  agreement, and ECE drift.
- **Offline-first**: a dependency-free deterministic `MockBackend` lets the whole
  pipeline (prompt → head → calibration → conformance) run with no weights.
- **Static, portable**: pure-Rust core with optional ONNX Runtime.

## Crates

| Crate | Purpose |
|---|---|
| `s1-core` | Wire contract, model packaging, prompt building, heads, calibration, conformance, engine. |
| `s1-backend` | Backend implementations: `MockBackend` (offline reference), `NullBackend`, optional `OnnxBackend`. |
| `s1-api` | HTTP API: `/v1/systemone`, `/health`, `/v1/models`, `/metrics`, auth. |
| `s1-cli` | `s1 serve`, `convert`, `calibrate`, `conform`, `bench`. |

## Quick start

Serve a built-in deterministic mock model (no weights required):

```bash
cargo run --release -p s1-cli -- serve --mock --bind 127.0.0.1:8080
```

Query it:

```bash
curl -s http://127.0.0.1:8080/v1/models

curl -s -X POST http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The customer wants a refund because the shoes are too small.",
    "model": "mock",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team handles this?",
        "criteria": { "returns": "Refund request", "billing": "Charging issue" }
      },
      "is_refund": {
        "type": "noul",
        "instructions": "Is the customer requesting a refund?"
      }
    }
  }'
```

Registering a model package from a manifest:

```bash
cargo run --release -p s1-cli -- serve --manifest examples/mock-model/s1-model.json
```

Run the offline conformance harness against the mock reference:

```bash
cargo run --release -p s1-cli -- conform \
  --golden examples/mock-model/golden.json
```

## Model package

A single `s1-model.json` manifest pins the family, backbone, head, prompt
contract, and per-backend calibration. See [docs/model-package.md](docs/model-package.md).

## Documentation

- [API contract](docs/API.md)
- [Model package format](docs/model-package.md)
- [Backends](docs/backends.md)
- [Calibration & confidence](docs/calibration.md)
- [Operations](docs/operations.md)

## License

Apache-2.0.
