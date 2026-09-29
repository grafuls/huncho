# Operations

## Build

The default build is dependency-free and works offline (the mock backend is
always available):

```bash
cargo build --release -p huncho-cli
```

Enable the ONNX Runtime backend for real weights:

```bash
cargo build --release -p huncho-cli --features onnx
```

Enable Hugging Face Hub resolution (download a model package by repo id):

```bash
cargo build --release -p huncho-cli --features hf
```

The `hf` feature pulls in `huncho-hub` (and `hf-hub`) so a model package can be
resolved by `owner/repo`. The `tokenizers` feature loads a manifest-declared
`backbone.tokenizer` with the official Hugging Face `tokenizers` crate, so
prompt encoding is byte-identical to the reference implementation (CORE-02).

Enable everything for real models:

```bash
cargo build --release -p huncho-cli --features onnx,hf,tokenizers
```

> `onnx` fetches a prebuilt ONNX Runtime at build time (needs network), and
> `hf`'s TLS provider needs system OpenSSL (`libssl-dev`/`pkg-config`).

## Run

Serve a built-in deterministic mock model:

```bash
huncho serve --mock --bind 127.0.0.1:8080
```

Serve one or more model manifests:

```bash
huncho serve --manifest ./models/laya/huncho-model.json --backend onnx \
  --manifest ./models/kev/huncho-model.json --backend onnx
```

Serve a manifest using the mock backend (offline demo, no weights):

```bash
huncho serve --manifest ./examples/mock-model/huncho-model.json --backend mock
```

Serve a model package by Hugging Face repo id. The manifest
(`huncho-model.json`) and every artifact it references are pulled into the HF
cache, and all artifacts are pinned to the exact resolved commit:

```bash
huncho serve --model my-org/laya --backend onnx --dtype fp32 --bind 127.0.0.1:8080
```

While the repo is being resolved, a per-file progress bar is drawn to stderr
for every artifact fetched (manifest, weights, head, tokenizer, and golden
suite). The serve listener binds only after all models are loaded, so a
first-time run may spend a while downloading before the endpoint is live.

Artifacts that the Hub serves through the Xet CDN (large weight files such as
`model.safetensors`) can otherwise stall at 0 bytes. `huncho-hub` therefore
falls back to a plain-HTTP transfer of the `/resolve/<revision>/<filename>`
endpoint, following the CDN redirect manually and streaming the body into the
HF snapshot directory as a regular file. Every file is pinned to the exact
resolved commit, and an already-present snapshot file is reused (so a
second run does not re-download). Progress is reported on stderr in both
TTY (live bar) and non-TTY (throttled one-line) environments.

You can also reference a local package path or a manifest file through
`--model`; it is resolved without any network access:

```bash
huncho serve --model ./models/laya --backend onnx
```

> Requires building with `--features hf`. A local package is resolved whether or
> not the feature is enabled; only `owner/repo` ids need `hf`.

### serve flags

| Flag | Meaning |
|---|---|
| `--bind` | Address to bind (default `127.0.0.1:8080`). |
| `--auth-token` | Require this bearer token on every request. |
| `--mock` | Serve a built-in mock model (no weights). |
| `--mock-model` | Register the mock under extra names (repeatable). |
| `--manifest` | Load a model manifest (repeatable). |
| `--model` | Resolve a model reference: a local package dir/path or an HF repo id (`owner/repo`); repeatable. Requires `hf`. |
| `--revision` | Git revision to resolve HF `--model` refs at (default: repo default branch). |
| `--token` | HF access token (defaults to `HF_TOKEN` / login cache). |
| `--backend` | Backend for models: `mock` (default) or `onnx`. |
| `--dtype` | Override dtype for models (default `fp32`). |
| `--extensions` | Enable engine extensions by default (API-05). |
| `--cache-dir` | Model cache directory; also seeds HF resolution (OPS-04). |

### Health and discovery

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

## Auth

Run with a bearer token:

```bash
huncho serve --mock --auth-token "secret"
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

Prometheus metrics use the `huncho_` prefix and include request latency,
per-model request counts and token usage, prefilled tokens, queue depth, and
fork counts.

## Calibration

Fit a temperature for a backend × dtype on a held-out set:

```bash
huncho calibrate \
  --manifest ./models/laya/huncho-model.json \
  --backend onnx --dtype fp32 \
  --data ./calibration/fit.json
```

Calibrate a package resolved from the Hub (the resolved, cached manifest is
updated in place):

```bash
huncho calibrate \
  --model my-org/laya --backend onnx --dtype fp32 \
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
huncho conform --golden ./examples/mock-model/golden.json
```

To compare a real manifest-backed backend:

```bash
huncho conform \
  --manifest ./models/laya/huncho-model.json \
  --backend onnx --dtype fp32 \
  --golden ./models/laya/golden.json --json
```

Conform against a package resolved from the Hub; the golden suite is taken from
the manifest's `reference.golden` automatically:

```bash
huncho conform --model my-org/laya --backend onnx --dtype fp32 --json
```

Exit code is non-zero when the suite fails (probability delta, argmax
agreement, or ECE drift threshold).

## Benchmark

```bash
huncho bench --questions 5 --iterations 200 --long-state
```

- `--questions` — number of questions per request (`1`, `5`, or `20`).
- `--iterations` — number of timed iterations (default `50`).
- `--long-state` — use a ~1500-char state instead of a short one.
- `--manifest` / `--backend` — run against a local manifest; defaults to `mock`.
- `--model` / `--backend` — resolve and run against an HF repo id or local package.

(CONF-04) reports mean/p50/p95/p99 latency and requests/sec per backend.

## Model cache (OPS-04)

`--cache-dir` seeds the Hugging Face cache directory when resolving `--model`
refs (defaults to `HF_HUB_CACHE`). Models are loaded lazily; a preload flag and
idle eviction are planned.

## Container (OPS-02)

A multi-stage `Dockerfile` at the repo root builds a slim, single-binary image
with `onnx,hf,tokenizers` and a non-root `huncho` user:

```bash
docker build -t huncho .
docker run --rm -p 8080:8080 huncho serve --mock --bind 0.0.0.0:8080
```

The ONNX Runtime is linked statically, so the runtime image only needs glibc
deps (`libssl3`, `ca-certificates`). Mount a model package and override the
`CMD` for real weights; the HF model cache lives under `HUNCHO_CACHE_DIR`
(`/var/lib/huncho`), which is writable by the service user.

## systemd (OPS-03)

`deploy/huncho.service` runs the server under a hardened, unprivileged
`huncho` user with a `EnvironmentFile=` for secrets and config, and
`deploy/huncho.env` is the template (copy to `/etc/huncho/huncho.env`, chmod
600, set `HUNCHO_AUTH_TOKEN`):

```bash
sudo install -m 0644 deploy/huncho.service /etc/systemd/system/huncho.service
sudo install -m 0600 deploy/huncho.env /etc/huncho/huncho.env
sudo systemctl daemon-reload && sudo systemctl enable --now huncho
```

The serve flags (`--bind`, `--backend`, `--dtype`, `--cache-dir`, and
`--auth-token`) are also read from `HUNCHO_BIND`, `HUNCHO_BACKEND`,
`HUNCHO_DTYPE`, `HUNCHO_CACHE_DIR`, and `HUNCHO_AUTH_TOKEN`, so secrets never
have to appear on the command line. The binary is single-file and
rootless-friendly; for SELinux hosts use the `:Z` volume label.

## Release gates

Releases are gated on the conformance matrix: every supported
(backend, model, dtype) must pass `huncho conform`. A failure blocks the release.
Quantized variants that fail ship with refit temperatures or are rejected.
