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

Enable all supported CPU runtimes for real models:

```bash
cargo build --release -p huncho-cli --features onnx,clef
```

> `onnx` fetches a prebuilt ONNX Runtime at build time (needs network), and
> `hf`'s TLS provider needs system OpenSSL (`libssl-dev`/`pkg-config`).
> The `clef` feature includes Candle, Hub resolution, and tokenizers. The `candle` backend
> loads Hugging Face `safetensors` weights directly with
> `candle-core` (no ONNX export, no extra system libs) and is the primary
> real-model path for F1/ModernBERT packages such as `convaiinnovations/laya`.

## Run

The unified RPM supplies one `huncho` command with automatic CUDA/CPU selection
for Kev and Clef. CPU hosts need no NVIDIA libraries. Set `HUNCHO_DEVICE=cpu` to
force CPU or `cuda` / `cuda:N` to require a GPU (`HUNCHO_CLEF_DEVICE` remains an alias). The packaged systemd unit reads
this setting from `/etc/huncho/huncho.env`. See [GPU setup](gpu-setup.md) for
driver requirements and fallback behavior.

During startup, `huncho serve` logs the actual device for each loaded model,
including CPU fallback. These messages appear at the default `info` log level:

```text
registered model `clef` on GPU (CUDA device 0)
registered model `laya` on CPU
```

Serve a built-in deterministic mock model:

```bash
huncho serve --mock --bind 127.0.0.1:8080
```

Serve one or more model manifests:

```bash
huncho serve --manifest ./models/laya/huncho-model.json \
  --manifest ./models/kev/huncho-model.json
```

Huncho chooses a compatible backend for each model from its metadata and the
runtimes included in the build. `--backend` is an optional override for all
models in the command. A missing runtime produces a build hint.

Serve a manifest using the mock backend (offline demo, no weights):

```bash
huncho serve --manifest ./examples/mock-model/huncho-model.json --backend mock
```

Serve a model package by Hugging Face repo id. The manifest
(`huncho-model.json`) and the selected backend's artifacts are pulled into the HF
cache, and all artifacts are pinned to the exact resolved commit:

```bash
huncho serve --model convaiinnovations/laya --bind 127.0.0.1:8080
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
huncho serve --model ./models/laya
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
| `--backend` | Optional override: `auto` (default), `onnx`, `candle`, `clef`, or `mock`. |
| `--dtype` | Override the model-specific default precision. |
| `--extensions` | Enable engine extensions by default (API-05). |
| `--max-queued-per-model` | Waiting requests per model, excluding the running request; defaults to 32. Overload returns HTTP 503. |
| `--max-prepared-per-model` | Optional F1–F4 preparation/ready slots per model; defaults to zero. Also `HUNCHO_MAX_PREPARED_PER_MODEL`. Requires startup qualification. |
| `--result-cache-bytes` | Optional per-model exact-result retention budget; defaults to zero. Also `HUNCHO_RESULT_CACHE_BYTES`. |
| `--coalesce-bytes` | Optional per-model metadata budget for identical in-flight request sharing; defaults to zero. Also `HUNCHO_COALESCE_BYTES`. |
| `--prefix-cache` | Opt-in request-local Kev prefix fan-out; requires qualification for the loaded device/precision. |
| `--max-batch-tokens` | Opt-in exact-length question batches, bounded by submitted token positions. Conflicts with prefix reuse. |
| `--candidate-readout` | Opt-in F3 candidate-only projection, after qualification. |
| `--qualification-golden MODEL=PATH` | Independent pinned suite for each optimized model; checked before the listener opens. Repeatable. |
| `--cache-dir` | Model cache directory; also seeds HF resolution (OPS-04). |

Prefix reuse, native batching, upfront preparation and F3 candidate-only projection are off by default
in HTTP serving. Their startup gate requires the external delta/argmax/ECE
checks plus paired independent-forward probability delta <=1e-4 and complete
argmax agreement. The suite must actually exercise each enabled fork/batch/preparation path.
Fixture success does not qualify a full checkpoint or another device/dtype.
Full Kev-4B/T4 FP16 prefix and batch paths failed that qualification; keep them
disabled for that execution variant. See [implementation status](optimization-roadmap.md).
An explicitly supplied qualification binding is also checked for independent
serving when no optimization flag is enabled; it is never silently skipped.
CLI serving rejects pending calibration. Refitted entries require an explicit
qualification suite with observed target labels for every question, even when
no numerical optimization is enabled. Keep fitting inputs separate from those
held-out cases; the CLI checks coverage and drift, not dataset provenance.
Existing entries marked `fit` remain trusted unless an explicit suite or a
numerical optimization triggers qualification. General execution certificates
and equivalent enforcement for library callers remain open.

`--max-prepared-per-model N` moves F1–F4 formatting/tokenization ahead of
model execution on blocking workers. One slot covers a running preparation or
ready request; it is released when execution starts, allowing the next request
to prepare concurrently. The existing admission bound still covers all jobs.
F5 ignores this knob and keeps joint preparation in its backend. Packets own
frozen inputs/options and cannot be used by a different engine. Model forwards,
heads, temperatures and logical usage retain their existing arithmetic.
Preparation validates every question before any forward; a later invalid
question therefore submits no model work. Default zero retains the original
interleaving. Slots bound request counts, not exact memory; size them cautiously
for large question sets. Cancellation during preparation holds capacity until
the blocking job ends, while cancellation of a ready packet frees capacity.

`huncho_preparation_wait_seconds` records slot wait and
`huncho_preparation_seconds` records actual preparation-worker time.
`huncho_requests_preparing`, `huncho_prepared_waiting` and
`huncho_questions_prepared` distinguish running CPU preparation, ready packets
and completed per-question preparation (including later canceled requests).
Cache-hit counters also count work completed during preparation even if its
caller later cancels. Model queue wait excludes preparation-slot wait; model
evaluation latency excludes upfront preparation when the pipeline is enabled.
HTTP latency still includes the whole request. Qualification uses
`huncho conform --prepare-all`, bypasses retention and requires unchanged
external gates plus paired delta <=1e-4, complete argmax agreement and nonzero
prepared-question work. Measure unique requests to establish any speed benefit.

Native Qwen F2/F3 loaders have two experimental, default-disabled kernel
profiles: `HUNCHO_PROJECTION_CHUNK_ROWS=1..4096` makes backbone linear calls
use fixed row counts with local zero padding; `HUNCHO_ATTENTION_FP32=1` computes
dense attention matmuls/softmax in FP32 while retaining the original weight,
activation and KV storage dtype. Zero/false disables the respective setting.
Padding is discarded inside each projection and never enters attention,
positions or logical token usage. Fixed chunks add launches/weight traffic;
FP32 attention adds compute and temporary memory. Neither implies a speedup.
These settings do not apply to ModernBERT, ONNX or Clef.

ModernBERT and F3 Qwen have explicit CUDA loaders in a `cuda` build. Their
default and `HUNCHO_DEVICE=auto` paths remain CPU; `cuda` / `cuda:N` selects GPU
execution and fails if it is unavailable. ModernBERT still computes FP32.
CPU staging handles BF16 checkpoints on older GPUs, and backbone/head tensors
use the same device. Their `device_path` metadata also triggers complete
labeled startup qualification. These loaders are not full-model Laya/Nimble
GPU acceptance or a performance claim.

Conformance/benchmark JSON records `execution_metadata` for the actual loaded
profile. CLI serving requires a complete labeled held-out qualification suite
for either changed profile, including when the existing calibration is marked
`fit`; caches cannot bypass it. Kernels cannot be changed while native retained
prefix handles are live. The 64-row-only Kev/T4 profile passes the six-case
numerical batch comparison but fails the 1,536-case labeled probability-delta
gate (maximum 0.012506247; 85 cases exceed 1e-3). Adding FP32 attention still
fails six-case prefix parity. Keep both profiles disabled for serving. Speed
measurements remain separate. Do not infer qualification from those
regression cases or from the diagnostic test.

`scripts/qualify_kev_runtime.py` runs the fixed external delta/argmax/ECE and
paired gates, checks the actual device/dtype/profile and exercised work, and
retains reports/logs plus binary, script, package and unchanged-golden hashes
in a new output directory. It never writes temperatures or packages. Labeled
acceptance requires observed targets for every question; `--numerical-only`
records diagnostic evidence and never marks a profile qualified. Base revision
pins are recorded without rehashing all base shards; dataset separation and
training overlap still require provenance review. For example, on a qualified
CUDA build with a local pinned package and separate labeled suite:

```bash
python3 scripts/qualify_kev_runtime.py \
  --binary /path/to/huncho --package /path/to/kev-package \
  --golden /path/to/held-out-golden.json --output /path/to/new-audit \
  --device cuda --dtype fp16 --projection-chunk-rows 64 --modes independent
```

Prefix/batch modes require representative multi-question cases that exercise
those paths; a single-question labeled corpus alone cannot qualify fan-out.

`HUNCHO_TOKEN_CACHE_BYTES` optionally retains exact tokenizer encodings for
manifest-loaded F1–F4 models. Zero, the default, disables retention. Text and
the special-token flag form the key; each immutable model owns its cache.
The FIFO cache charges retained text/token payload and entry overhead against
the configured budget and retains at most 1,024 entries. Oversized entries
bypass it. It retains input text in process memory until eviction/model drop.
Clef owns its joint-schema tokenizer and does not use this setting.

`HUNCHO_PROMPT_CACHE_BYTES` optionally retains complete prepared F1–F4 prompts:
tokens, candidates, readout positions, prefix boundaries and question types.
Zero, the default, avoids retention, key serialization and cache locks. Exact
state/question contents, candidate order and optional-description presence form
engine-local keys; model/tokenizer/formatter identity belongs to the immutable
engine. Question IDs and execution/extension options do not change these
per-question prompts. Every hit still runs model inference, applies the current
input budgets and produces the current response fields. `work.prompt_cache_hits`
and `huncho_prompt_cache_hits` count reused preparation separately from result
reuse. The FIFO charges retained input/token/candidate payload plus overhead,
with at most 1,024 entries; oversized prompts and preparation errors bypass
retention. Inputs stay in process memory until eviction/model drop. Returned
prompts are owned copies because prefix/batch paths modify them. F5 owns its
whole-request preparation and ignores this setting. Conformance bypasses this
cache; concurrent misses can prepare twice. Cache hits do not change floating
point arithmetic or require a temperature refit.

`--result-cache-bytes` / `HUNCHO_RESULT_CACHE_BYTES` optionally retains successful
whole-request responses in each immutable engine. The complete serialized
request, optional-description presence and execution options form the key,
preserving question/option order, effective context overrides and extension
visibility. Presence bits distinguish Rust library inputs containing `None`
from `Some(Value::Null)`, which serialize alike but can format differently.
Hits clone the original
calibrated response, including raw logits/legend/logical usage, without floating
point recomputation. Errors are not retained. The FIFO holds at most 1,024
entries and charges owned input/output payload and container overhead against
the configured budget; this is not an exact allocator/RSS limit. Oversized
entries bypass retention. Model reload creates a separate cache; input text
remains in memory until eviction or model drop. Default zero avoids retention.
Serving authentication and admission still apply to hits. Concurrent library
misses can perform duplicate work.
Conformance always executes fresh prompt preparation and uncached forwards.

`--coalesce-bytes` / `HUNCHO_COALESCE_BYTES` enables exact in-flight request
sharing in HTTP serving. Full ordered requests and execution/extension options
form instance-local keys. Identical admitted callers await one result; completed
results are removed immediately. Errors are shared only with current waiters
and subsequent requests retry. Keys and conservative channel/node overhead
are charged against the budget, with at most 1,024 active keys; oversized/full
tables bypass sharing. This bounds retained metadata rather than response
memory or allocator RSS. No inference arithmetic or temperature changes.
Every caller passes authentication/validation and consumes admission capacity,
so a full queue still returns HTTP 503 for identical requests. One caller's
cancellation leaves other waiters intact; when every caller cancels, queued
work is abandoned. Running blocking work keeps its permits until completion.
This is duplicate suppression, not cross-request tensor batching. It retains
input text while the shared job is active; default zero disables sharing.

CLI lifecycle logs go to stderr, so `bench --json` and `conform --json` emit a
standalone JSON object on stdout even with `RUST_LOG=info`.

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
queue wait, evaluation time, waiting/admitted requests, submitted forward/prefill
token positions, native batches, forks, reused prefix positions, prepared-prompt
and exact-result cache hits, coalesced callers and callers waiting for shared results. Submitted
positions include attempted work and jobs whose HTTP caller disconnects.
Wire `usage.input_tokens` remains logical prompt usage. Cancellation releases
waiting admission; a running job retains its execution slot until completion.

## Calibration

Fit a temperature for a backend × dtype using fitting data. Reserve separate
held-out data for final calibration evaluation:

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
Add `--json` to retain the fitted entry, fitting NLL, row count and save status
as a report. For example, `huncho calibrate --manifest ./models/kev/huncho-model.json
--backend candle --dtype fp16 --data ./calibration/fit.json --save false --json`
reports the candidate fit without writing the package. A fit is not held-out
qualification.

## Conformance

Run the offline conformance harness against the golden vectors:

```bash
huncho conform --golden ./examples/mock-model/golden.json
```

To compare a real manifest-backed backend:

```bash
huncho conform \
  --manifest ./models/laya/huncho-model.json \
  --golden ./models/laya/golden.json --json
```

Conform against a package resolved from the Hub; the golden suite is taken from
the manifest's `reference.golden` automatically:

```bash
huncho conform --model my-org/laya --json
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
- `--manifest` — run against a local manifest with automatic backend selection.
- `--model` — resolve and run against an HF repo id or local package.
- `--backend` — optionally override the selected runtime. Without a model, the
  benchmark uses the built-in mock.
- `--result-cache-bytes` — exact-result retention budget, default zero. Use with
  `--repeat-inputs` for deliberate duplicate traffic; warmup populates that
  request, so timed cache hits report zero physical work. Distinct inputs still
  measure actual inference and include cache-key/miss overhead.

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

## RPM package

For RPM distros (Fedora, RHEL, Rocky), `packaging/rpm/build-rpm.sh` builds a
binary + source RPM with ONNX and Candle (`onnx,hf,tokenizers,candle`) and ships a
systemd unit, an env file, a man page, and the mock model package. See
[`packaging/rpm/README.md`](../packaging/rpm/README.md).

Release builds are published to **Fedora COPR**. COPR's `make srpm` SCM build
method invokes the `srpm` target of `.copr/Makefile`, which assembles the
source RPM into COPR's `outdir`; the root `make srpm` just delegates to the same
file and drops the source RPM into the repo root. COPR then compiles the
package on Fedora builders and serves a dnf repository. See the COPR section in
[`packaging/rpm/README.md`](../packaging/rpm/README.md).

## Release gates

Releases are gated on the conformance matrix: every supported
(backend, model, dtype) must pass `huncho conform`. A failure blocks the release.
Quantized variants require refitted temperatures and successful qualification
against independent vectors and held-out outcomes before release.


### CPU thread tuning

Tune `RAYON_NUM_THREADS` and `CANDLE_NUM_THREADS` together for the serving
machine and replica count; avoid multiplying full host thread budgets across
replicas. The full Kev FP32 pilot on the dual-Xeon lab host, using sixteen fixed
physical NUMA-node-1 cores, measures 39.382 s versus 35.476 s per five-question
request at four versus sixteen threads. Both pass the unchanged numerical
gate. This small, ordered comparison is hardware/workload-specific and has
concurrent work without frequency isolation. It does not change the default
or prove production p99 or held-out statistical calibration. [Retained CPU pilot](verification/kev-t4-20261008/stage5-cpu-thread-pilot.json)
records the complete runs and limits.

```sh
HUNCHO_DEVICE=cpu RAYON_NUM_THREADS=16 CANDLE_NUM_THREADS=16 \
  huncho bench --model /path/to/kev-package --dtype fp32 \
  --questions 5 --workload mixed --iterations 20 --json
```

Repeat with representative question counts, state lengths and concurrency;
keep package, temperature, device/dtype and numerical gates fixed while tuning.

### Cross-request batches

`--batch-max-requests` enables bounded collation across callers on native batch
backends (2–64 requests, default disabled). It requires `--max-batch-tokens` and
cannot be combined with prefix reuse. `--batch-wait-ms` bounds collection time
from the first ready request (default 2 ms); waiting behind earlier inference is
additional. Environment equivalents are `HUNCHO_BATCH_MAX_REQUESTS` and
`HUNCHO_BATCH_WAIT_MS`. Preparation capacity is raised to the collation size,
while the existing admission limit still bounds all active/waiting callers.

Startup requires a pinned suite that actually combines rows from multiple
requests and passes independent probability parity. Qualify with
`huncho conform --batch-max-requests N --max-batch-tokens B ...`. Equal-length
questions share tensors without padding; F5 remains whole-request execution.
`huncho_cross_request_batch_count` counts actual mixed batches. Group failures
reach every affected caller; disconnected queued callers submit no work, while
running jobs retain admission until completion. CPU fixture success does not
qualify full released checkpoints or GPU paths.

### Execution receipts

Build with optional `qualification` to retain a content-addressed conformance
receipt using `huncho conform --write-qualification PATH ...`. The output must
be new. `huncho serve --qualification-record MODEL=PATH` verifies that receipt
against loaded artifact bytes, executable/runtime identity, calibration,
execution options and exact golden bytes. A matching `--qualification-golden`
is still required, and serving performs fresh conformance before opening the
listener. Keep the executable, thread/affinity configuration and inference
options identical. Store ONNX receipts outside the graph directory tree, which
is included conservatively to cover external tensor data.

Receipts distinguish numerical evidence from observed-outcome checks. They do
not establish fitting/evaluation provenance or universal calibration, and do
not provide cached authorization for serving. Linux runtime/library identity is
more complete than on other platforms; GPU hardware is not independently
inventoried. Weight hashing adds startup I/O only when this opt-in workflow is
used. The default build and normal model loads retain their existing behavior.

### Persistent Kev prefixes

`--persistent-prefix-bytes B` / `HUNCHO_PERSISTENT_PREFIX_BYTES` adds a charged
native snapshot budget to qualified `--prefix-cache` serving. Default zero keeps
retention disabled. Immutable snapshots contain all attention, recurrent and
convolution state; each hit returns a fresh caller-owned handle. FIFO eviction
is bounded by bytes and sixteen snapshots. Active forks remain valid through
clear/eviction, and model/kernel changes cannot reuse these snapshots.

Qualification clears and warms each case, requires actual snapshot hits, counts
warm work and retains independent probability gates. An insufficient budget
cannot qualify. `huncho_persistent_prefix_hits` counts hits, while physical token
counters exclude their avoided prefill positions. The budget covers snapshots,
not live forks or peak inference memory. Library callers can clear retained
state with `Engine::clear_prefix_cache`; otherwise unused snapshots live until
eviction or model unload. Full released-model and GPU qualification remain open.
### ONNX compact readouts and bounded output reuse

The optional `onnx` build keeps its existing CPU/full-sequence output behavior by
default. `HUNCHO_ONNX_COMPACT_READOUT=1` requires an explicit graph contract:
`huncho_readout_positions: int64[rows]` and
`huncho_features: float32[1,rows,hidden]` (or `[rows,hidden]`), with dynamic row
count and fixed positive hidden width. Positions preserve order and duplicates.
The backend checks bounds before inference and verifies the returned row count.
This is an F1 feature readout; it does not interpret vocabulary logits as trained
candidate scores.

`scripts/compact_onnx_readout.py SOURCE DESTINATION --output last_hidden_state`
adds a graph-side Gather and prunes unrelated outputs. It requires the optional
Python `onnx` tooling, creates a new graph/external-data file without overwriting,
and leaves manifest updates, temperatures and qualification to the operator.
Select the new artifact explicitly and retain the original reference package.
Large exports require enough host RAM to materialize their external tensors.

`HUNCHO_ONNX_OUTPUT_BUFFER_BYTES=N` retains at most one exact-shape CPU output
allocation through ORT I/O binding. Zero disables it. Changed shapes replace the
buffer; unknown widths, empty outputs and allocations above the payload limit
bypass retention and release the previous buffer. The limit covers retained
output payload only, excluding ORT workspaces, input tensors and ordinary host
readout copies. Graph-side gathering reduces the host output from sequence rows
to selected rows; buffer reuse avoids repeated output allocation. Neither has a
production latency claim or a device-resident head in this increment.

Both profiles require fresh `--qualification-golden MODEL=PATH` at server startup
and are included in persisted execution identity. Exact CPU fixture rows and
unchanged mock probability goldens pass; real Laya exports still require their
own pinned qualification. These profiles do not refit or enable rejected variants.

### Explicit ONNX provider and thread selection

`HUNCHO_ONNX_EP` defaults to `cpu`; `cuda:N` requires building with the separate
`onnx-cuda` feature and a compatible ONNX Runtime CUDA distribution and its
CUDA/cuDNN libraries. This does not enable Candle CUDA. CUDA selection disables
TF32, fails registration errors, and disables CPU fallback. A graph with
unsupported CUDA operators is rejected. Session selection excludes inherited
global ORT execution providers. Actual GPU checks are deferred; this
path is unqualified for released models. Consult the runtime distribution's
[CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
for its supported library versions.

`HUNCHO_ONNX_THREADS=0..256` controls intra-op threads; zero keeps ORT's automatic
choice. Positive values use an independent session thread pool; ORT builds with
OpenMP may instead require `OMP_NUM_THREADS`. Positive values and CUDA selection require a pinned labeled
`--qualification-golden MODEL=PATH` before HTTP startup, with probability,
argmax and observed calibration gates. They do not select a new temperature.
Choose thread counts from measured workload evidence; no universal setting is
promoted. Both settings are part of optional persisted qualification identity.

### Native ONNX tensor batches

`HUNCHO_ONNX_NATIVE_BATCH=1` opts into F1 graphs with `input_ids` and optional
`attention_mask`, `token_type_ids`, and `position_ids`, all int64 with dynamic
`[batch,seq]` dimensions. The output must be `last_hidden_state`, float32 with
dynamic `[batch,seq]` and a fixed positive hidden width. The backend verifies
this schema at load. Mask values are one, token-type values zero, and explicit
position IDs run from zero to sequence length minus one. Other required inputs,
fixed-batch graphs and the separate compact-readout contract are rejected.

Use `--max-batch-tokens N` to batch equal-length questions and, optionally,
`--batch-max-requests 2..64` to collate prepared requests. Native calls take at
most 64 equal-length rows, preserve each row's selected positions, and never
pad or truncate. The serving scheduler bounds admission, preparation, collation
and total tensor tokens. This opt-in profile requires labeled pinned goldens
and fresh startup conformance; enabled scheduling must perform actual batch work
and pass the tighter independent-forward parity gate. CPU deterministic
fixtures pass; real exported Laya graphs and CUDA batching remain unqualified.

### Buffered CPU Gated DeltaNet recurrence

`HUNCHO_CPU_DELTA_RULE=1` opts Kev/F2, Qwen/F3 and Clef/F5 into a CPU-only recurrence
profile. The profile retains FP32 recurrent state and the original ascending
key-reduction order. It casts inputs once, fuses decay with the memory
projection, and fuses the state update with the output projection. A small
scratch vector replaces per-token tensor intermediates. Noncontiguous input
views and prefix state are supported; initial state remains immutable.

The mode defaults off, rejects non-CPU devices, and cannot change while live
forks or persistent snapshots remain. `delta_rule_execution=cpu-buffered-v1`
is part of execution identity and requires a complete labeled
`--qualification-golden MODEL=PATH` before serving, even with fitted calibration.
The upstream temperature is retained unless an independently fitted and gated
variant is selected. CPU fp32/fp16 fixture probabilities and cache/batch parity
pass, and Clef whole-request fixture logits retain exact bits. Released
Kev/Nimble/Clef acceptance remains open.

A retained [recurrence microbenchmark](verification/delta-cpu-20261008/recurrence-microbenchmark.json)
measures about 10 times faster recurrence for one local CPU shape, with equal
output/state float bits. It excludes all projections, dense attention, heads,
tokenization and serving; it is not a model latency or cost claim. Reproduce
the CPU-only kernel comparison with:

```sh
cargo test --offline --release -p huncho-backend --features candle --lib \
  delta_cpu::tests::recurrence_cpu_timing -- --ignored --nocapture
```
