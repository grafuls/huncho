# Operations

## Build

The default build is dependency-free and works offline (the mock backend is
always available):

```bash
cargo build --release -p s1-cli
```

Enable the ONNX Runtime backend for real weights:

```bash
cargo build --release -p s1-cli --features onnx
```

## Run

Serve a built-in deterministic mock model:

```bash
s1 serve --mock --bind 127.0.0.1:8080
```

Serve one or more model manifests:

```bash
s1 serve --manifest ./models/laya/s1-model.json --backend onnx \
  --manifest ./models/kev/s1-model.json --backend onnx
```

Serve a manifest using the mock backend (offline demo, no weights):

```bash
s1 serve --manifest ./examples/mock-model/s1-model.json --backend mock
```

### serve flags

| Flag | Meaning |
|---|---|
| `--bind` | Address to bind (default `127.0.0.1:8080`). |
| `--auth-token` | Require this bearer token on every request. |
| `--mock` | Serve a built-in mock model (no weights). |
| `--mock-model` | Register the mock under extra names (repeatable). |
| `--manifest` | Load a model manifest (repeatable). |
| `--backend` | Backend for manifest models: `mock` (default) or `onnx`. |
| `--dtype` | Override dtype for manifest models (default `fp32`). |
| `--extensions` | Enable engine extensions by default (API-05). |
| `--cache-dir` | Model cache directory (placeholder for OPS-04). |

### Health and discovery

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

## Auth

Run with a bearer token:

```bash
s1 serve --mock --auth-token "secret"
```

Then request:

```bash
curl -s http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer secret'
```

Requests without a valid token return `401`. The comparison is constant-time.

## Metrics

```bash
curl -s http://127.0.0.1:8080/metrics
```

Prometheus metrics include request latency, per-model request counts and token
usage, prefilled tokens, queue depth, and fork counts.

## Calibration

Fit a temperature for a backend × dtype on a held-out set:

```bash
s1 calibrate \
  --manifest ./models/laya/s1-model.json \
  --backend onnx --dtype fp32 \
  --data ./calibration/fit.json
```

`fit.json`:

```json
{ "rows": [[0.2, 1.4, -0.3], [...]], "targets": [1, ...] }
```

`--save false` performs a dry run.

## Conformance

Run the offline conformance harness against the golden vectors:

```bash
s1 conform --golden ./examples/mock-model/golden.json
```

To compare a real manifest-backed backend:

```bash
s1 conform \
  --manifest ./models/laya/s1-model.json \
  --backend onnx --dtype fp32 \
  --golden ./models/laya/golden.json --json
```

Exit code is non-zero when the suite fails (probability delta, argmax
agreement, or ECE drift threshold).

## Benchmark

```bash
s1 bench --questions 5 --iterations 200 --long-state
```

- `--questions` — number of questions per request (`1`, `5`, or `20`).
- `--iterations` — number of timed iterations (default `50`).
- `--long-state` — use a ~1500-char state instead of a short one.
- `--manifest` / `--backend` — run against a manifest; defaults to `mock`.

(CONF-04) reports mean/p50/p95/p99 latency and requests/sec per backend.

## Model cache (OPS-04)

`--cache-dir` is accepted as a placeholder. Models are loaded lazily; a preload
flag and idle eviction are planned.

## Container / systemd

OCI images and systemd units are P1. The binary is single-file and
rootless-friendly; for SELinux hosts use the `:Z` volume label.

## Release gates

Releases are gated on the conformance matrix: every supported
(backend, model, dtype) must pass `s1 conform`. A failure blocks the release.
Quantized variants that fail ship with refit temperatures or are rejected.
