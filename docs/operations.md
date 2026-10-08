# Operations

CPU ModernBERT F1, Qwen F2/F3 and masked native ONNX feature batching can
merge different prompt lengths:

```sh
huncho conform --model /path/to/kev-package --golden /path/to/pinned-heldout.json \
  --max-batch-tokens 2048 --max-batch-padding-percent 25 --json
huncho serve --model /path/to/kev-package --max-batch-tokens 2048 \
  --max-batch-padding-percent 25 \
  --qualification-golden 'REGISTERED_NAME=/path/to/pinned-labeled-heldout.json'
```

Zero padding percent keeps exact lengths. The limit counts padded positions
within the complete physical tensor; logical wire usage excludes padding.
Every served model must support the selected CPU padding algorithm. Prefix reuse cannot
combine with these batches. Cross-request collation uses the same policy and
existing request/wait bounds. Qualification needs actual mixed lengths as well
as unchanged numerical/paired and complete observed-label gates.
`huncho_padded_batch_count` and `huncho_padded_tokens` expose extra physical
work; native/device-resident ragged attention remains future work.

For F1, the backbone receives a real row-specific attention mask. Padded rows
are removed before the trained Laya bidirectional head; masking only the
backbone would change head scores. This composes with final-layer marker-query
selection and CPU shared-weight replicas. The original head tokens, qtype,
readout order and temperatures remain. Equal-length batches retain their
original execution. `padded_batch_execution=cpu-right-mask-unpad-head-v1`
records the available CPU algorithm; requested options and actual padded work
are required in conformance evidence. Every native serving context still needs
fresh complete observed labels and unchanged external/independent gates.
Padding adds physical work and may increase attention memory; throughput and
released Laya calibration need workload qualification. CPU Clef F5 can also
collate whole schemas as described below; actual device checks remain deferred.

CPU Kev can opt into fair scheduling between prefix chunks with
`HUNCHO_PREFILL_CHUNK_TOKENS=64 huncho serve ... --prefix-cache --cooperative-prefill`.
Supply the same complete labeled `--qualification-golden MODEL=PATH` binding
used by other native serving profiles. Startup requires actual interleaving of
split prefixes from distinct requests, external conformance and independent
parity. One context, no batching and at most 62 queued requests are supported.
This remains experimental: current released CPU Kev profiles have not passed
the full held-out gate. [Implementation and limits](optimization-roadmap.md#resumable-cpu-prefill-scheduling-2026-10-08).

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
  --manifest ./models/kev/huncho-model.json \
  --qualification-golden 'LAYA_MODEL_NAME=/path/to/laya-labeled-golden.json' \
  --qualification-golden 'KEV_MODEL_NAME=/path/to/kev-labeled-golden.json'
```

Huncho chooses a compatible backend for each model from its metadata and the
runtimes included in the build. `--backend` is an optional override for all
models in the command. A missing runtime produces a build hint.
Use each manifest's registered name in its qualification binding.

Serve a manifest using the mock backend (offline demo, no weights):

```bash
huncho serve --manifest ./examples/mock-model/huncho-model.json --backend mock
```

Serve a model package by Hugging Face repo id. The manifest
(`huncho-model.json`) and the selected backend's artifacts are pulled into the HF
cache, and all artifacts are pinned to the exact resolved commit:

```bash
huncho serve --model convaiinnovations/laya --bind 127.0.0.1:8080 \
  --qualification-golden 'MODEL_NAME=/path/to/pinned-labeled-golden.json'
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
huncho serve --model ./models/laya \
  --qualification-golden 'MODEL_NAME=/path/to/pinned-labeled-golden.json'
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
| `--replicas` | CPU contexts per model, 1–8 (default 1); also `HUNCHO_REPLICAS`. Shares weights; every context needs complete labeled startup qualification. |
| `--max-prepared-per-model` | Optional F1–F4 preparation/ready slots per model; defaults to zero. Also `HUNCHO_MAX_PREPARED_PER_MODEL`. Requires startup qualification. |
| `--result-cache-bytes` | Optional per-model exact-result retention budget; defaults to zero. Also `HUNCHO_RESULT_CACHE_BYTES`. |
| `--coalesce-bytes` | Optional per-model metadata budget for identical in-flight request sharing; defaults to zero. Also `HUNCHO_COALESCE_BYTES`. |
| `--prefix-cache` | Opt-in request-local Kev prefix fan-out; requires qualification for the loaded device/precision. |
| `--max-batch-tokens` | Opt-in exact-length question batches. CPU Kev can combine these with prefix reuse, charging complete contexts against the workspace budget. |
| `--candidate-readout` | Opt-in F3 candidate-only projection, after qualification. |
| `--qualification-golden MODEL=PATH` | Independent pinned labeled suite for every real runtime; checked before the listener opens. Repeatable. |
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
Every native Candle, Clef, ONNX, llama.cpp and vLLM runtime also requires complete
labeled startup qualification with `fit` metadata and no optimization flags.
Temperature fallback does not authorize a different backend/device/precision.
The explicit offline mock retains its demo path. General signed execution
certificates and equivalent enforcement for library callers remain open.

`--max-prepared-per-model N` moves F1–F4 formatting/tokenization ahead of
model execution on blocking workers. One slot covers a running preparation or
ready request; it is released when execution starts, allowing the next request
to prepare concurrently. The existing admission bound still covers all jobs.
F5 keeps joint tokenization in its backend. When cross-request batching is
enabled, a bounded worker freezes/validates its whole-request packets; the
standalone preparation knob does not move F5 tokenization. Packets own
frozen inputs/options and stay within their immutable model/replica group. Model forwards,
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
refs (defaults to `HF_HUB_CACHE`). Serving eagerly loads by default. Optional
CPU lazy loading, preloading and idle residency eviction are described below.

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
questions share tensors without padding. Supported CPU padding can merge mixed
lengths. CPU Clef F5 batches **whole requests**, retaining the complete joint
schema per row and removing backbone padding before the bidirectional head.
F5 batching requires `--batch-max-requests`; a per-question token budget alone
is rejected. Its all-group encoding finishes before the first model forward,
inside the backend. Complete observed labels and actual whole-request batches
(and actual padding when requested) remain mandatory.
`huncho_cross_request_batch_count` counts actual mixed batches. Group failures
reach every affected caller; disconnected queued callers submit no work, while
running jobs retain admission until completion. CPU fixture success does not
qualify full released checkpoints or GPU paths.

`huncho bench --batch-max-requests N --max-batch-tokens B ...` measures
actual groups per client for native backends, including CPU Clef. This combines
with CPU replicas; each client binds to one context and submits complete groups.
Each request's latency is its **entire group's completion time**, including
preparation, and throughput counts original requests. A requested grouped run
with no real mixed-request batch fails unless exact cached results avoided
execution. Oversized singletons remain intact. These microbenchmarks do not
establish released calibration or a general speedup.

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
### CPU FP32 runtime LoRA

`HUNCHO_DEVICE=cpu HUNCHO_RUNTIME_LORA=1` enables standard inference-only
Qwen F2/F3 A/B projections with `--dtype fp32`. Default off retains the existing
merged adapter path. With optional `shared-base` and `HUNCHO_BASE_CACHE_BYTES=B`,
separate registry models share immutable dense base tensors even at adapted
projections. Each model keeps its own A/B tensors, trained readout and execution
state; model routing selects that immutable adapter context. The existing lazy
residency and replica bounds apply. The base-cache bound covers retained bases,
not all live adapter models, caches or activations.

The update is `base(x) + B(A(x)) * alpha/r`, matching standard inference LoRA's
structure. [PEFT's pinned implementation](https://github.com/huggingface/peft/blob/v0.18.1/src/peft/tuners/lora/layer.py)
uses this structure and disables dropout in inference. Supported adapters have
constant rank 1..256 matching all A/B shapes (and trained Kev head metadata),
finite positive alpha and finite floating weights. Nonstandard variants,
base-changing initialization, biased updates, modules-to-save, rank/alpha
patterns, missing/unconsumed targets and embedding/norm/LM-head targets are
rejected. Literal `target_modules` arrays require every matching base target;
regex/all-linear declarations and layer filters are unsupported. F3 vocabulary
readout itself keeps its existing projection and candidate selection.

Runtime and merged paths have different floating-point reductions. Serving
requires fresh complete observed-label qualification for
`adapter_execution=cpu-fp32-runtime-lora-v1`; receipts include the option and
actual target count. Use independent fitting capture/refits and held-out gates
before releasing a profile. Source temperatures are not modified by loading.
It composes with CPU scalar/native batches, replicas and existing kernel options;
Kev supports its existing prefix/chunk/page paths. Two extra low-rank matmuls
per targeted projection trade arithmetic for less duplicate dense storage and
merge workspace. Mixed-adapter tensor collation, reduced/packed/device profiles,
released outcome calibration and measured RSS/latency remain separate work.
Apple work and actual GPU checks are deferred.

### CPU Kev KV pages

`HUNCHO_DEVICE=cpu HUNCHO_KV_PAGE_TOKENS=16` enables immutable native Kev pages.
Page sizes are powers of two in 16..256; default 0 retains flat storage. Only
Candle Kev F2 with full-attention layers supports this option. Configure it before
creating caches or replicas. Full pages share across forks; a changed partial
tail is copied and failed continuations publish no changed pages. Pages own
compact tensor allocations, including short tails. Recurrent/convolution state
uses its existing implementation.

Attention still materializes complete contiguous KV in original token order,
so this does not provide paged-attention kernels or a peak-memory bound. It can
trade smaller persistent fork storage for extra page/cat copies and metadata.
The existing 64 live-handle limit and retained snapshot byte/16-entry bounds
apply; snapshot accounting charges shared payloads conservatively per snapshot.
The snapshot budget excludes active forks and transient attention workspaces.

Serving requires `--prefix-cache` and fresh complete observed-label conformance,
including actual forks and independent paired probability gates. Qualification
identity records `kv_storage=cpu-cow-pages-materialize-v1`, page size and the
configuration environment. CPU fixture bitwise parity does not release Kev or
establish speed/RSS. Direct paged kernels, cached-branch batches and tenant
retention policy remain separate work. Apple work and actual GPU checks are
deferred by the user.

### Optional ONNX CUDA device I/O and graph replay

Build `onnx-cuda` and explicitly select `HUNCHO_ONNX_EP=cuda:N` with `--dtype
fp32`. `HUNCHO_ONNX_DEVICE_IO_BYTES=B` optionally retains stable device
inputs/output and CPU readback buffers within a charged 0..512 MiB budget.
Default zero preserves ordinary execution. Optional
`HUNCHO_ONNX_CUDA_GRAPH=1` requires a nonzero device-I/O budget and enables
capture/replay on that strict CUDA session with TF32 and CPU fallback disabled.
CPU output-buffer reuse/shared initializer profiles and CPU integrated-head
graphs cannot combine with this path. Raw feature/compact graphs require a
fixed positive float32 output width; trained host heads and core temperatures
stay in their existing paths.

Each of at most 32 exact I/O shape profiles retains its original bound device
addresses for the entire session lifetime. New data copies synchronously into
those addresses. Full output copies to an owned CPU readback tensor, then
requested core rows copy to independently owned results. Captured buffers are
never evicted/reallocated under a graph ID: exhausting bytes/slots rejects new
shapes. A copy/run/readback/nonfinite failure invalidates the context until
reload. The budget includes device input/output, CPU readback and conservative
slot metadata; it excludes transient caller inputs, ORT arenas/workspaces and
captured graph allocations. It is not a total host/device memory limit. Current
copy helpers use ORT identity sessions and synchronization; no throughput or
latency benefit is assumed without measurements.

[ORT's CUDA graph requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview)
include stable tensor shapes/addresses, CUDA placement of every node, no
control-flow operators and serialized calls to each session. Huncho owns one
session/slot set per backend behind its existing execution mutex; unsupported
graphs fail through the native runtime. The first native run includes capture
setup/replay work. Serving/bench counters count submitted forwards, not internal
capture kernel executions. New profile metadata/environment binds fresh
complete numerical, argmax, labeled ECE-drift and independent paired gates;
receipts do not bypass fresh qualification.

This path is compile-checked with CPU ownership/ordinary-readout regressions.
Actual GPU initialization, data transfer, capture/replay, arithmetic/calibration
and performance are unverified and deferred by the user. The ignored CUDA
regression is future qualification work and was not run. CPU substitution tests
cannot establish CUDA behavior or release a model. Apple work stays skipped.

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

The retained binding owns the original output tensor and reuses its actual
storage address. Inputs are rebound for each call and cleared on both success
and validation/execution errors, so the binding does not retain a preceding
request's tokens. This fixes the earlier implementation's use of
`Tensor::clone()`, which makes a deep allocation/copy in the pinned ORT crate;
the previous counter alone did not establish reuse. Native address/ownership
and error-cleanup checks now cover it. All calibration gates remain unchanged.

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

### Native integrated F1 ONNX heads

Build with `--features onnx,tokenizers` and use explicit CPU FP32 selection:

```sh
HUNCHO_ONNX_EP=cpu HUNCHO_ONNX_THREADS=1 HUNCHO_ONNX_INTEGRATED_HEAD=1 \
  huncho conform --model /path/to/package --backend onnx --dtype fp32 \
  --golden /path/to/pinned-complete-labeled-golden.json
```

This default-disabled profile executes the actual graph head and returns raw
scores directly to the shared Rust temperature/softmax path. The loader
requires F1, scalar option-marker width one, `laya-v1`, a declared tokenizer
and the `tokenizers` feature. The graph must declare exactly `tokens` int64
`[1,S]`, `positions` int64 `[N]`, `qtype` int64 `[1]`, optional
`attention_mask` int64 `[1,S]`, and one `scores` float32 `[N,1]` output.
Sequence/marker dimensions must be dynamic; batch one and scalar width one
must be fixed. Qtype is the actual choice/score/noul value 0/1/2. Markers retain
requested order and duplicates. Scores must be finite; they are neither
activated nor calibrated inside the graph. An optional mask is all ones.

This is the CPU graph ABI used by the separate browser SDK. No export of a
released trained model is inferred from compatible names/shapes. The graph
must contain the correct backbone and trained typed head. Fresh complete
observed-label conformance and an explicit fitted/refitted `onnx:fp32` entry
are required before serving; a fitted default entry alone cannot authorize it.
Temperature and fixed delta/argmax/ECE gates are unchanged. Capturing raw
scores from a pending package for fitting remains possible offline.

`HUNCHO_ONNX_OUTPUT_BUFFER_BYTES` can reuse one bounded CPU output buffer;
returned scores own their data. `onnx-shared` plus
`HUNCHO_ONNX_SHARED_INITIALIZERS=1` enables independent shared-source CPU
replicas, including after source replacement or primary drop. Native batching,
compact feature gathering, cached/prefix forwards, other precisions and GPU
providers are rejected. The profile and environment flag bind qualification
receipts. Synthetic native/browser fixtures establish implementation parity,
not released-model calibration, throughput or memory savings. See the
[fixture and reproduction limits](../crates/huncho-backend/tests/fixtures/integrated_f1/README.md).

### Native ONNX tensor batches

`HUNCHO_ONNX_NATIVE_BATCH=1` opts into F1 graphs with `input_ids` and optional
`attention_mask`, `token_type_ids`, and `position_ids`, all int64 with dynamic
`[batch,seq]` dimensions. The output must be `last_hidden_state`, float32 with
dynamic `[batch,seq]` and a fixed positive hidden width. The backend verifies
this schema at load. Independent/equal-length masks are one, token-type values zero, and explicit
position IDs run from zero to sequence length minus one. Other required inputs,
fixed-batch graphs and the separate compact-readout contract are rejected.

Use `--max-batch-tokens N` to batch equal-length questions and, optionally,
`--batch-max-requests 2..64` to collate prepared requests. Native calls take at
most 64 rows and preserve each row's selected positions. CPU feature graphs
that explicitly declare `attention_mask` can also use
`--max-batch-padding-percent 1..100`. This profile right-pads token zero and
builds masks from original lengths, including valid real token zeros. Position
IDs restart at zero in every row. Original readouts must fall within the
original row length, and no padding readout is returned. The scheduler charges
the full batch-by-maximum-length rectangle against the token budget, caps the
fraction of that rectangle used by padding and preserves oversized singletons
without truncation. Padding is default-disabled. Graphs without a mask, ordinary
non-batch graphs, integrated/compact graphs and GPU profiles do not expose it.
`padded_batch_execution=onnx-cpu-right-mask-v1` records availability; active
options and physical work bind conformance/receipts. The graph signature alone
cannot prove it actually honors the mask.

The serving scheduler bounds admission, preparation, collation
and total tensor tokens. This opt-in profile requires labeled pinned goldens
and fresh startup conformance; enabled scheduling must perform actual batch work
and pass the tighter independent-forward parity gate. CPU deterministic
fixtures pass, including a mask-sensitive graph compared to independent scalar
features, real token zero, duplicated/reordered positions, output reuse and
shared concurrent CPU sessions. Complete synthetic typed gates preserve wire
usage and count actual padding; incomplete/drifted suites fail. These are generic
feature/mean-head fixtures, not trained Laya releases. Native CLI tests exercise
real rectangles/replicas and refuse unlabeled startup. Released graph exports,
labeled calibration, throughput/RSS and CUDA batching remain unqualified.

### Qualification of all real serving runtimes

Every actual Candle, Clef, ONNX, llama.cpp and vLLM runtime requires
`--qualification-golden MODEL=PATH`, including an unoptimized source package
marked `fit`. The suite must supply observed outcomes for every question and
pass the unchanged external probability, argmax and ECE-drift gates before
the listener opens. Replica pools gate every context concurrently; enabled
cache/batch/preparation work still has its separate nonvacuous parity checks.
Execution receipts retain `native_execution`, and an unsigned retained record
cannot replace fresh conformance. Explicit offline mock demos remain available.

This is an intentional startup compatibility change. Temperature lookup can
still use a source `default`, but that metadata no longer authorizes a new
backend/device/precision. The full Kev CPU FP32 held-out rejection demonstrates
why the unoptimized path also needs the gate. A refit must pass the same
unchanged thresholds; do not replace goldens with the variant's outputs.
`bench`, raw-logit capture and diagnostic `conform` do not grant serving approval.

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


### Fused CPU MLP gate

`HUNCHO_CPU_FUSED_GATE=1` opts the native Candle Qwen3.5 backbone into a fused
SiLU/multiply operation for Kev/F2, Qwen/F3 and Clef/F5. It defaults off. The
kernel allocates one output instead of a separate SiLU intermediate and output,
and uses the original typed scalar SiLU and multiply operations. FP16 preserves
the intermediate FP16 rounding; there is no approximate exponential, FMA or
reduced precision introduced by this kernel. Noncontiguous equal-shape inputs
are supported; broadcasting and BF16 are rejected.

Configure it before creating replicas or retaining prefix state. Non-CPU devices
are rejected. Execution receipts record `mlp_gate_execution=cpu-fused-silu-mul-v1`
and the environment flag. Serving requires fresh complete labeled startup
conformance with the existing thresholds, even for a fitted source package.
Typed kernel and native fp32/fp16 fixture tests preserve original float bits,
including Kev batches/prefixes and whole-request Clef logits. These comparisons
cover Huncho's default Candle scalar CPU build; enabling a downstream vector
math provider is a separate execution profile and needs its own comparison.

The operation removes one allocation and one intermediate memory write/read.
It does not eliminate projection matmuls, establish a lower peak RSS, or qualify
released-model latency/calibration. An optional kernel-only comparison is:

```sh
cargo test --release -p huncho-backend --features candle --lib \
  gate_cpu::tests::gate_cpu_timing -- --ignored --nocapture
```

## Bounded shared-weight CPU replicas

`--replicas N` leases one independently locked context per complete model job,
with a maximum of eight. Native CPU ModernBERT/Laya, Qwen F2/F3, packed CPU Kev,
Clef/F5, CPU llama.cpp, optional shared-initializer ONNX and the offline mock
support it. Ordinary ONNX loads, GPU contexts and unsupported backends fail
explicitly when more than one replica is requested. The default
remains one. CLI cross-request collation cannot be combined with replicas;
per-request native batches, prefix reuse and bounded preprocessing can be used
when separately qualified.

Backbone/trained-head tensor storage, tokenizer, formatter, core head and exact
prompt/result caches are shared. Mutable KV, recurrence, convolution history,
active handles and persistent snapshots belong to each context. A snapshot hit
on one context does not make another context warm. The per-model persistent
prefix byte budget is divided evenly across contexts; a small share that cannot
retain representative prefixes fails its nonvacuous qualification. Temporary
activations and active state can grow with concurrency. Shared CPU workers can
also contend for cores and memory bandwidth. Candle packed dot products share
a process-wide barrier pool which serializes concurrent packed kernel calls;
other inference work can overlap. Replica count therefore needs workload
measurements; no automatic throughput gain is claimed.

Admission covers N running jobs plus `--max-queued-per-model` waiting jobs.
Canceled queued requests release capacity; a running blocking job retains its
context and admission until it finishes. Preprocessing packets can transfer
only within the same immutable replica group. Every actual context runs the
complete pinned suite concurrently before the listener opens, requiring
observed targets for every question and unchanged probability thresholds. A
primary pass cannot hide drift in another context. Existing single-context
receipts still describe one execution identity; they do not certify pool
throughput or replace these fresh checks.

The in-process API exposes `Engine::replica()` and
`ModelRegistry::set_replicas(N)` for current registrations before serving.
They do not authorize calibration themselves; library callers own that gate.
Registry pool construction is atomic on failure. The cross-request library
worker remains serial; use the supported CLI combinations when measuring
replica throughput. Full released-model pool qualification and CPU affinity per
context remain outstanding.

`huncho bench --replicas N --concurrency C` measures the native CPU contexts
directly, with N <= min(C, iterations) and a maximum of eight. Every context
warms before timing; clients bind by worker index modulo N. Latencies include
any wait on that assigned context. JSON reports `replica_work` with each
context's timed request count and physical work, alongside aggregate counters.
The global persistent-prefix budget is divided across contexts as in serving;
exact prompt/result caches remain shared. Changing N while keeping a fixed
host thread budget can expose oversubscription. This is a warm closed-loop
engine benchmark, with no HTTP admission, idle-context dispatch, CPU affinity
management, peak-RSS measurement or calibration approval.

## Isolated CPU ONNX sessions with shared initializers

Build with `--features onnx-shared` and set
`HUNCHO_ONNX_SHARED_INITIALIZERS=1` (default off). This opt-in loader captures
the original graph and dense initializer bytes once, injects preallocated
CPU tensors with ORT's `AddInitializer`, and shares its prepack container.
`--replicas 2..8` creates separate real ORT sessions, thread pools and output
buffers. Session creation is serialized for prepacking; inference has no shared
session mutex. The primary can be dropped without invalidating other contexts.
Later replica construction uses the immutable snapshot rather than rereading
source files.

The supported profile is a flat graph with FP32, FP64, INT32, INT64 and BOOL
dense initializers. Little-endian raw and typed repeated data are supported.
External initializer files must stay inside the graph directory; offset/length,
dimensions, payload counts and unique names are validated. External tensors are
also supplied through `AddExternalInitializers` to replace file references before
graph validation; that ORT step copies data into the graph. Subgraphs, local
functions, sparse/segmented initializers, external tensor attributes, other
dtypes and GPU providers fail explicitly. The ordinary loader remains available
for graphs outside this profile.

`onnx_initializer_residency=immutable-cpu-v1` and
`onnx_model_snapshot_sha256` bind the execution to original graph bytes and full
selected external files. `onnx_shared_initializer_bytes` describes tensor payload,
excluding retained graph bytes, temporary external copies, optimized graph
constants, packed weights, thread pools, activations and metadata. ORT can still
duplicate transformed constants and allocate per-session state. This is not a
peak-RSS bound or a promise that every operator shares its transformed weights.
Snapshotting inline graphs adds retained graph storage and startup copying/hash
work; choose replica counts using real workload/RSS measurements.

Serving requires fresh complete observed-label conformance at the unchanged
thresholds, concurrently on every actual context, even for fitted source
packages. Receipts include the flag and snapshot identity. CPU deterministic
fixtures pass raw/probability comparisons, selected-row/batch isolation and
concurrent execution after source mutation and primary drop. Those fixtures do
not qualify a released Laya graph or establish throughput improvements. The
new parser/hash dependencies are optional; the default build is unchanged.

```sh
HUNCHO_ONNX_SHARED_INITIALIZERS=1 HUNCHO_ONNX_THREADS=1 \
  huncho bench --model /path/to/onnx-package --replicas 2 --concurrency 2 --json
HUNCHO_ONNX_SHARED_INITIALIZERS=1 HUNCHO_ONNX_THREADS=1 \
  huncho serve --model /path/to/onnx-package --replicas 2 \
  --qualification-golden 'REGISTERED_MODEL_NAME=/path/to/pinned-labeled-golden.json'
```

## Immutable CPU base residency across adapters

Build with `--features shared-base` and set `HUNCHO_BASE_CACHE_BYTES` to opt into
one process-wide charged-byte budget (default zero). Safetensors Qwen loaders
retain immutable CPU base tensors before LoRA merging. Separate adapters clone
tensor references, share untargeted storage and allocate their own merged target
weights. Kev/F2, Qwen/F3 and CPU Clef/F5 use this loader; ModernBERT and already
packed GGUF artifacts do not. This differs from execution replicas, which share
one complete already-merged model.

Every lookup hashes full base-shard bytes, dtype and vocabulary-head inclusion.
Identical relocated shards can share; changed files cannot select stale weights.
Sources are checked again before returning hits or publishing a materialized
base. Shard enumeration is deterministic and duplicate canonical names fail.
Source files must remain immutable while loading/qualifying. SHA-256 I/O remains
on the startup path, and concurrent startup materialization is serialized; this
cache does not hold a lock during inference.

At most sixteen bases are retained with LRU eviction. The byte charge includes
loaded tensor payloads and conservative per-entry metadata; it does not bound
allocator overhead, activations, live models or load/merge temporaries. An
oversized base bypasses retention. Eviction/clear releases only cache-owned
references. Keeping an unmerged base plus merged targets can increase memory
for a single adapter, so size this option against your actual adapter mix.
The CLI budget is fixed after first use; restart to change it.

The backend records `base_weight_cache=content-checked-cpu-v1`, which requires
fresh numerical startup qualification against pinned goldens. Existing refit,
arithmetic-profile and replica rules still require complete observed outcomes.
The cache does not change head math, tensors or temperatures and does not grant
calibration approval. Receipts bind the option and source artifacts. Library
callers can use `BaseWeightCache`, `Qwen3_5Backend::load_kev_with_base_cache`,
`stats()` and `clear()` with explicit budgets/lifetimes.

CPU tests prove actual shared embedding storage, independent adapter merges and
cache handles, bit-identical fp32/fp16 raw/probability outputs, concurrent miss
deduplication, source-change rejection, eviction and budget bypass. CLI tests
exercise relocated sharing and keep pending calibration rejected. Full released
adapter-mix RSS/startup measurements, lazy model-registry loading/eviction,
on-the-fly multi-LoRA batches and device residency remain open. Actual GPU
checks and Apple work are deferred.

## CPU attention working memory

`HUNCHO_DEVICE=cpu HUNCHO_ATTENTION_QUERY_ROWS=64` opts native Qwen F2/F3
or Clef F5 into 64-row attention query blocks. Zero is the default dense path;
accepted block sizes are 1–4096. Each block retains all causal keys/values and
uses absolute prefix positions. This bounds each score/probability intermediate
by `batch * heads * min(block_rows, query_length) * key_length` elements and its
mask by `min(block_rows, query_length) * key_length`. The setup cache omits the
full quadratic mask. K/V, hidden activations, block outputs and other model
allocations remain; these bounds are not a peak-RSS estimate or a speed claim.

Configure before prefixes or replicas. Other backends/families and explicit
non-CPU selection are rejected. Exact block size and arithmetic profile are
included in qualification records, and fresh complete held-out observed-label
conformance is required before serving. `scripts/qualify_kev_runtime.py
--attention-query-rows 64 --device cpu ...` records the same explicit profile
and rejects substituted sizes/metadata. Existing prefix/batch gates still apply
when those modes are enabled. No temperature or golden threshold is changed.


## Optional CPU lazy residency

A `qualification` build can use `huncho serve --lazy` with an explicit real
`--backend`, explicit `--dtype`, `HUNCHO_DEVICE=cpu` and a complete observed-label
`--qualification-golden MODEL=PATH` for every registered model. CPU ONNX requires
its CPU execution provider. Manifest/Hub resolution, artifact hashing and suite
validation happen before listening; weights load on first authenticated, valid
inference. `--preload MODEL` performs the same cold-load gates before listening.
Preload names must be unique and fit the configured slots.

`--resident-models 1..64` (default 1) bounds resident/loading lazy model groups,
including all replicas in each group. Cold loads for one model share a worker,
with at most `max_queued_per_model + replicas` waiting callers. Canceling HTTP
waiters releases their waiting capacity but retains the worker's model slot
until loading/qualification finishes. If every slot is busy/loading, new cold
requests return 503. Capacity pressure evicts the least recently accessed idle
group. `--idle-evict-secs` (default 300) also evicts groups after that interval
since last access, provided no caller, preparation/execution job, admission
permit or external engine reference owns them. Idle batch workers hold weak
context references. Eager registrations are separate from the lazy slot bound.

Each actual cold load rechecks the pinned manifest, tokenizer, backbone,
adapter/head artifacts, executable, golden/receipt files, explicit runtime
library files and execution environment before/after loading and fresh complete
conformance. Prefix/readout/batch/cooperative and every replica's existing gates
also apply. A previous pass or retained receipt never replaces fresh outcomes.
Substituted engines/devices and failed/panicking factories remain unavailable;
restart after correcting a failed package. `/v1/models` lists `cold`, `loading`,
`resident` or `failed` for lazy registrations; listing/health never load weights.

The limit counts model groups, not estimated bytes or peak RSS. Model size,
replica contexts, temporary loading/qualification allocations and any separately
bounded immutable base cache still matter; eviction does not flush a configured
shared-base cache or guarantee an immediate allocator RSS decrease. Cold gates
can dominate first-use latency and their inference work is separate from normal
serving request counters. CPU frozen Clef fixtures cover real cold reloads and
failures; synthetic fixture labels test gate plumbing, not released calibration.
Runtime LoRA dispatch, released residency/RSS benchmarks and GPU residency remain
open. Apple work and actual GPU checks remain deferred by the user's scope.


## CPU grouped-query attention

`HUNCHO_DEVICE=cpu HUNCHO_GROUPED_GQA=1` enables grouped query matmuls for native
Qwen F2/F3 and Clef F5. It preserves the query-head-to-K/V-head mapping, all
causal keys, original scale/masks/rotary/gates, trained readouts and temperatures.
K/V forward workspace uses the original K/V head count; contiguous copies can
still be necessary for strided projections. Compared with dense repetition,
this workspace is smaller by the query-head/K/V-head ratio. Prefix caches already
retain unexpanded K/V and do not gain that factor again. Score/probability memory
and arithmetic remain quadratic without the separate bounded-query profile.

It composes with `HUNCHO_ATTENTION_QUERY_ROWS`, native prefixes/chunks, CPU
replicas, padded batches and packed Q8/Q4 projections. Changing it after live,
pending or retained prefixes, or shared replicas, is rejected. Unsupported
families/devices and malformed flags fail before device initialization. Metadata
and qualification receipts bind `cpu-grouped-queries-v1`; the CPU runner uses
`--grouped-gqa` and clears unrequested ambient settings. Matrix shapes change,
so complete fresh observed-label serving conformance remains mandatory. No
released speed, peak RSS or calibration acceptance follows from fixture parity.

## Optional CPU browser deployment

The [separate browser SDK](../browser/README.md) supports F1 FP32 ONNX graphs
with integrated scalar option heads. Packaging pins asset bytes and requires a
new destination; every actual browser session reruns complete labeled shared
conformance before public evaluation. It preserves Rust prompts/calibration
and Jev answers without a decode loop. Native default builds remain unchanged.
Artifact/context/queue limits, copies, application trust, single-thread CPU
execution and unsigned report limitations are explicit. Released exports and
calibration remain qualification work. Apple is skipped and WebGPU/actual GPU
checks remain deferred.

### Optional CPU vLLM execution

The `vllm` build feature uses an explicitly pinned local CPU pooling worker for
Kev F2, with BF16 backbone and FP32 raw pointer scores. It omits the vocabulary
head and decode loop. Offline conversion keeps source temperatures and creates
an explicit Pending `vllm:bf16` entry; `capture-logits` can collect independent
fitting rows before calibration. Serving requires that exact fitted/refitted
entry and fresh complete labeled startup gates. Equal-length native batching
uses existing budgets and paired gates. CPU prefixes, replicas, quantization,
other families and device execution are unsupported. See
[environment, export, fitting and limits](vllm.md).

`HUNCHO_VLLM_TENSOR_PARALLEL=2` optionally shards this same CPU backbone across
two local spawned ranks. Loaded projection ownership and per-rank forward
counts are checked. Thread/KV settings apply per rank; total rank threads are
bounded at 64. The full reduced hidden states feed replicated FP32 pointer
heads before shared calibration. Rank count/runtime/layout bind fresh receipts
and are distinct arithmetic profiles requiring fitting and held-out gates.
This increment does not implement pipeline, multi-node or GPU execution.

## CPU Kev batches from a shared prefix

`--prefix-cache --max-batch-tokens N` now combines request-local state prefill
with equal-length native CPU Kev question batches. One immutable parent retains
full-attention KV, GDN recurrence and causal-convolution state. Native rows start
from private copies of that state, and the original trained pointer head reads
each row's original markers and final decision. The parent never advances.
No temporary branch handles or suffix cache survive the call, including errors.

A group charges `B * (prefix_tokens + suffix_tokens)` against `N`, even though
only the suffixes are recomputed. This bounds the private complete-KV workspace;
it is not a byte/RSS limit. Equal complete contexts are bucketed separately from
independent prompts, with at most 63 temporary forks and the existing 64 live
handle cap. Oversized singletons remain intact. Another active prefix may exhaust
the remaining handle capacity; the backend refuses admission explicitly.
Logical wire usage counts every original complete prompt. Physical counters
count actual prefill chunks and submitted suffixes, including failed attempts.
`fork_batch_calls` in reports and `huncho_fork_batch_count` in metrics count
actual multiple-row prefix batches, separately from independent batches.

The profile is CPU Kev only and defaults off. Mixed-length padding,
cooperative scheduling and cross-request collation cannot combine with it.
Retained prefixes, chunks, immutable KV pages, CPU kernels, standard FP32 runtime
LoRA and separately refitted packed CPU backbones keep their own qualification
requirements. `conform`/`serve` require actual forks **and actual native cached
batches**; a too-small budget or suite without equal-length suffixes cannot
qualify. Serving still needs a complete fresh observed-label suite and unchanged
external delta/full argmax/ECE plus paired independent 1e-4 gates. Fixtures with
duplicated original typed questions retain frozen probabilities; synthetic
fixture labels exercise gate plumbing only. Released CPU acceptance and
workload latency/RSS remain unqualified/unmeasured. No actual GPU checks ran.
See [CPU evidence](verification/fork-batch-cpu-20261008/README.md).
