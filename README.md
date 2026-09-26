# Huncho

**Huncho** is a portable serving engine for Jev-style "System One" decision
models. It takes a **state** and a set of **typed questions** and
returns **calibrated probabilities** — no text generation. It is to decision
models what llama.cpp / vLLM are to generative LLMs, with one non-negotiable
guarantee: *calibrated probabilities stay calibrated* across every backend and
quantization.

```
state + typed questions ──► huncho ──► calibrated probabilities (no decode loop)
```

The engine implements the **Jev `/v1/systemone` wire contract**, so an unmodified
TypeSafe Python SDK works against Huncho with only a base-URL change.

## Highlights

- **Four decision-head families** behind one contract: F1 (option-marker), F2
  (pointer), F3 (candidate-logit), F4 (slot head).
- **Calibration as a first-class guarantee**: per-backend × dtype temperatures,
  softmax, and confidence (Jev peak-based, or per-model definitions).
- **Conformance gating**: an offline harness (`huncho conform`) compares any backend
  against golden vectors and gates releases on probability fidelity, argmax
  agreement, and ECE drift.
- **Offline-first**: a dependency-free deterministic `MockBackend` lets the whole
  pipeline (prompt → head → calibration → conformance) run with no weights.
- **Static, portable**: pure-Rust core with optional ONNX Runtime.
- **Hugging Face-native**: a model package can be resolved by repo id
  (`--model owner/repo`) so manifests, weights, head, and golden vectors are
  pulled straight from the Hub — the same run-time resolution model vLLM uses.

## Crates

| Crate | Purpose |
|---|---|
| `huncho-core` | Wire contract, model packaging, prompt building, heads, calibration, conformance, engine. |
| `huncho-backend` | Backend implementations: `MockBackend` (offline reference), `NullBackend`, optional `OnnxBackend`. |
| `huncho-api` | HTTP API: `/v1/systemone`, `/health`, `/v1/models`, `/metrics`, auth. |
| `huncho-hub` | Hugging Face Hub resolution of model packages by repo id (feature `hf`). |
| `huncho-cli` | `huncho serve`, `convert`, `calibrate`, `conform`, `bench`. |

## Quick start

Serve a built-in deterministic mock model (no weights required):

```bash
cargo run --release -p huncho-cli -- serve --mock --bind 127.0.0.1:8080
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
cargo run --release -p huncho-cli -- serve --manifest examples/mock-model/huncho-model.json
```

Registering a model package from the Hugging Face Hub by repo id (build with
`--features hf`):

```bash
cargo run --release -p huncho-cli --features hf -- serve --model my-org/laya --backend onnx
```

The manifest and every artifact it references are fetched into the HF cache,
pinned to the resolved commit, so `serve --model` is deterministic across runs.

Serve a real ONNX artifact (`examples/mock-model` ships a 512-dim encoder and
its conformance golden). Build with `--features onnx` (fetches a prebuilt ONNX
Runtime at build time):

```bash
cargo run --release -p huncho-cli --features onnx -- \
  serve --manifest examples/mock-model/huncho-model.json --backend onnx

cargo run --release -p huncho-cli --features onnx -- conform \
  --manifest examples/mock-model/huncho-model.json \
  --backend onnx --golden examples/mock-model/golden.json
```

The same commands with `--backend mock` drive the in-process reference and need
no ONNX build. For real models also pass `hf` (Hub resolution) and `tokenizers`
(byte-identical reference prompting): `--features onnx,hf,tokenizers`.

Run the offline conformance harness against the mock reference:

```bash
cargo run --release -p huncho-cli -- conform \
  --golden examples/mock-model/golden.json
```

## Model package

A single `huncho-model.json` manifest pins the family, backbone, head, prompt
contract, and per-backend calibration. See [docs/model-package.md](docs/model-package.md).

## Testing

The default build is offline and testable without weights:

```bash
cargo test                       # core + hub + api + cli
cargo test -p huncho-backend --features onnx   # ONNX backend integration
cargo test -p huncho-core --features tokenizers --test hf_tokenizer  # CORE-02
```

Coverage includes the `/v1/systemone` HTTP contract (choice/noul/score, 422,
auth, extensions, `/metrics`), `convert`/`calibrate`/`bench`, F1–F4 prompt
building, and the ONNX conformance suite.

## Operation / packaging

A multi-stage `Dockerfile` builds a slim image with `onnx,hf,tokenizers` and a
non-root user; a hardened `deploy/huncho.service` + `deploy/huncho.env` example
cover systemd. `serve` reads `HUNCHO_BIND`/`HUNCHO_BACKEND`/`HUNCHO_DTYPE`/
`HUNCHO_CACHE_DIR`/`HUNCHO_AUTH_TOKEN`. See [docs/operations.md](docs/operations.md).

## Documentation

- [API contract](docs/API.md)
- [Model package format](docs/model-package.md)
- [Backends](docs/backends.md)
- [Calibration & confidence](docs/calibration.md)
- [Operations](docs/operations.md)

## License

Apache-2.0.
