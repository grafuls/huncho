# Optimization implementation roadmap

This tracks the staged implementation of [the research report](optimization.md). Source work began from `571f93c`. The report remains a dated analysis; this file records current implementation status. No optimization is considered qualified for a new model/device/precision merely because a fixture passes.

## Current scope and remaining work

The user has deferred Apple Silicon work and all checks against actual GPUs.
CPU implementation and qualification continue. "Available" below describes
code and fixture coverage; it does not promote a released model/precision.
The CPU Kev held-out gate rejected FP32, and selected-case FP16 diagnostics
also show drift. Lab access resumed: the staged CPU Q8 job completed all 768 fitting rows
and saved its separate refit; the independent 1,536-case held-out process was
active at 20:30:54 UTC, with Q4 still waiting. Fitting alone grants no release.
[CPU progress and audit limits](verification/kev-quant-cpu-20261008/README.md).
These variants remain unqualified.

| Area | Implementation status | Remaining work or qualification |
|---|---|---|
| O01 readout deduplication | Available | New model/profile coverage |
| O02 buffers/allocations | Available in readouts, heads, CPU gate and selected ONNX outputs | Broader activation-buffer reuse and released workload measurements |
| O03 CPU kernels/threads | Thread/build profiles and optional OpenBLAS available | Platform fitting, held-out gates and throughput |
| O04 result reuse/coalescing | Available, bounded, default off | Tenant-specific retention policy |
| O05 F3 selected projection | Available | Released Nimble qualification |
| O06 execution/preparation workers | Available | Released workload capacity/latency measurements |
| O07 tokenizer/prompt caching | Available, bounded, default off | Model-specific hit rates and memory measurements |
| O08 ONNX output/readout path | CPU gather/output reuse/scalar and batched integrated F1 raw heads and compile-checked CUDA device I/O available | Released graph exports; actual device qualification and device-resident head |
| O09 fused/bounded compute | CPU SiLU/multiply, causal query blocks and unexpanded grouped K/V available | FlashAttention and further fused native/device kernels |
| O10 precision/calibration gates | Every real CLI/core/HTTP serving context requires fresh labeled conformance; opaque attempt tokens and cache generations prevent stale authorization | Signed portable proofs/dataset provenance and released CPU acceptance remain open/rejected/pending; no GPU checks |
| O11 native Kev prefix forks | Available | Released CPU paired/labeled acceptance |
| O12 dynamic batching | Equal-length collation and bounded CPU Qwen/ModernBERT/masked ONNX feature/whole-schema Clef/integrated raw F1 right-padding and CPU Kev equal/padded shared-prefix batches available | Released graph calibration and device qualification |
| O13 family head work | CPU Clef vectorized heads/grouped spans, optional Laya final marker queries and integrated ONNX F1 heads available | Released graph exports and Clef/Laya labeled qualification |
| O14 concurrent execution contexts | CPU shared-weight replicas, including optional ONNX initializers, available | Released concurrent workloads, memory and affinity measurements |
| O15 device buffers/graph replay | Optional stable CUDA ONNX I/O and bounded graph replay implemented; CPU ownership checks and CUDA compilation pass | Actual GPU capture/execution/calibration and released graph/performance qualification deferred |
| O16 Candle device propagation | Optional CUDA loaders available | Actual GPU checks deferred |
| O17 recurrent/conv buffers | CPU implementations available | Released profile acceptance rejected; parallel device kernels open |
| O18 quantization | Experimental Candle and llama.cpp CPU Q8/Q4 packages available | CPU Q8 refit complete, held-out active at the recorded snapshot; Q4 waiting; other families open |
| O19 retained/paged prefixes | Bounded immutable CPU snapshots, copy-on-write CPU KV pages, optional direct CPU FP32 page attention and equal/padded-suffix branch collation available | Mixed-prefix/multi-row direct page kernels, device kernels and tenant policies |
| O20 Metal | Skipped by user | Apple work deferred |
| O21 llama.cpp | CPU F2/F3, full-state forks/chunks, bounded native batches and pending FP32/FP16/Q8/Q4 exports available | Released fitting/qualification and graph-side readout efficiency |
| O22 shared bases/residency | Immutable CPU bases, lazy residency/preload/idle eviction and standard CPU FP32 F2/F3 runtime LoRA available | Released residency/calibration measurements, mixed-adapter collation and other profiles |
| O23 vLLM custom readouts | Pinned actual CPU Kev F2 raw pooling and equal-length batches available | Released fitting/held-out acceptance; other families, prefixes, quantization and device profiles |
| O24 distributed/sharded inference | Actual two-rank local CPU vLLM tensor sharding available | Released fitting/acceptance, capacity/latency; pipeline/multi-node/device implementation and multi-GPU qualification |
| O25 prefill scheduling | Resumable CPU prefix/question and bounded equal/padded cached question-group scheduling with live child capacity available | Released labeled qualification/latency, cross-request cached groups and device scheduling open |
| O26 MLX | Skipped by user | Apple work deferred |
| O27 browser WASM/WebGPU | Separate CPU WASM F1 scalar/native equal/masked row/marker heads, packaging, dedicated bounded CPU workers and fresh per-session labeled/paired gates available | Released exports/calibration, cross-request batches/prefixes, other families and multithread tuning; WebGPU deferred |

## Implemented increments

### Cooperative CPU Kev suffix groups with live child capacity

O12/O19/O25 now compose resumable prefix chunks with bounded equal/padded
request-local native question groups. Each step releases the backend lock
after one chunk/group. Immutable original prompts/typed readouts remain paired
through length sorting, prefix-relative positions and native scatter; parent
KV/recurrence/convolution never advance. Live `fork_batch_limits` subtracts
all active/partial parents, and regrouping under the submitting lock preserves
padding/complete-context limits. Pressure may reduce a group to one row while
the existing 64-handle cap remains enforced.

Fresh gates require real split-prefix interleaving and native/padded cached
work plus unchanged labeled and paired thresholds. Batch qualification reserves
room with at most 32 active parents; scalar qualification retains 63. Actual
CPU FP32/FP16 kernel/page/direct-page/runtime-LoRA cases preserve frozen
probabilities. Sixty-two competing partial prefixes, repeated cancellation,
large cohorts, HTTP fairness/coalescing/failures and a real qualified CPU Kev
CLI HTTP server cover the runtime boundaries. Released CPU acceptance and
latency/RSS remain separate. Cross-request cached groups and device scheduling
remain open; Apple and actual GPU checks stay deferred.
[Evidence and limits](verification/cooperative-batch-cpu-20261008/README.md).

### Native CPU browser row/marker batches

O12/O27 now collate whole F1 question rows in actual CPU WASM graph calls, using
the same distinct dynamic row/marker ABI as native ONNX heads. Core-derived
immutable groups bound physical rectangles, padding, rows and marker counts;
owned raw scores scatter to original typed calibration and logical usage.
Fresh complete labeled gates also run every question independently at batch
one, require actual tensor/padding work and compare unchanged paired 1e-4
probabilities/full argmax. Public page and dedicated-worker sessions cannot
inherit scalar acceptance or change their hashed batch profile per evaluation.

The original four synthetic cases submit four optimized calls for twelve
questions, including three padded calls (672 physical tokens, 27 padding).
Equal-row fixture cases preserve copied original vectors/labels and exercise
ten calls for sixteen questions without padding. These counts do not establish
released calibration or latency/throughput/RSS improvement. Original weights,
probabilities, temperatures and thresholds are unchanged; new graphs read the
frozen weights without regenerating references. Released exports, cross-request
collation, prefixes, other families and multithreading remain open. Apple and
actual GPU checks stay deferred.
[Evidence and limits](verification/browser-batch-cpu-20261008/README.md).

### Dedicated CPU browser inference workers

O06/O27 now move actual model loading, tokenizer preparation, conformance and
CPU WASM graph execution off the page thread. Packaged descriptors pin the
worker module and SDK; verified immutable Blob snapshots start the worker.
Every session runs its own fresh complete labeled gate before an engine is
returned. Frozen JSON submissions, eight pending evaluations, typed response
ownership, drain/dispose, fatal failure and explicit termination are bounded
and checked against actual CPU browser execution. Native Rust references and
bitwise page/worker wire answers keep their original temperatures/goldens.

This improves scheduling/isolation without changing arithmetic. Worker
instances add complete runtime/session residency and message serialization;
fixture success establishes no released calibration, isolated speedup or RSS.
Native browser batches/prefixes, shared-weight worker pools, other families and
multithread tuning remain open. Apple and actual GPU/WebGPU checks stay
deferred. [Evidence](verification/browser-worker-cpu-20261008/README.md).

### Fresh qualification for core and HTTP serving

O10 now enforces fresh complete labeled qualification in the core production
entry points and the public HTTP boundary, including lazy factories and every
actual replica. Fixed numerical/outcome/paired thresholds are unchanged.
Opaque tokens bind the exact attempt/context/options/runtime; revocation or a
later successful attempt cannot authorize work started under an older proof.
Response-cache generations prevent delayed diagnostic plans from restoring
stale entries after a fresh gate. Constructor identity and exact quantized,
vLLM and integrated-head calibration rules apply to programmatic serving too.

Default mock demos and explicit raw diagnostic interfaces remain available.
Serialized reports/unsigned receipts cannot authorize a fresh context; these
in-process proofs do not attest dataset provenance. CPU native runtime/HTTP
fixtures exercise the gate without changing goldens, labels or temperatures.
Released Kev remains rejected/pending. Signed proofs, independent dataset
policy, Apple work and actual GPU checks remain open/deferred.
[Evidence and limits](verification/library-qualification-20261008/README.md).

### Direct CPU FP32 attention over immutable Kev pages

O09/O19 now offer default-disabled direct page QK/PV with bounded query blocks
and unexpanded grouped K/V. Every original causal key enters one ordered
softmax; persistent single-row prefixes and suffixes avoid complete-KV
concatenation. Independent forwards and private multi-row branch batches use
the explicitly recorded flat fallback. Pagewise matmul shapes/PV summation
change rounding, so the profile requires fresh complete labeled prefix/fork
and paired qualification rather than inheriting the storage-only profile.

Native tests verify actual direct calls with zero materialization, causal/head
mapping, all four page sizes, frozen typed probabilities, chunk boundaries,
runtime LoRA and kernels, snapshots, replicas, guards and cancellation. CPU
fixture evidence does not release Kev or establish speed/peak RSS. Direct
multi-row/mixed-prefix and device kernels remain open; Apple and actual GPU
checks are deferred. [Evidence](verification/paged-attention-cpu-20261008/README.md).


### Native CPU batches of integrated F1 raw ONNX heads

O08/O12/O13 now collate strict dynamic raw-head graphs using explicit row/marker
coordinates and per-row question types. Equal-length and explicitly masked CPU
right-padded rows share a native tensor call; each original question receives
owned scalar raw scores before unchanged typed calibration. The graph ABI is
distinct from scalar heads and generic/compact features. Existing token/readout
bounds, request scatter, shared initializer sessions and owned output reuse
compose with fresh actual batch/label/paired gates.

New synthetic graphs retain existing frozen independent Rust weights, scores,
probabilities and temperatures. Original mixed typed requests and cross-request
batches pass fixture gates; duplicates/empty markers, output ownership, strict
contracts/limits/types, nonfinite results and isolated shared sessions are
covered. Synthetic targets prove gate plumbing only. Released Laya exports,
calibration and controlled latency/RSS remain open; no Apple or actual GPU
checks ran. [CPU evidence](verification/onnx-head-batch-cpu-20261008/README.md).


### Bounded CPU Kev mixed-length prefix batches

O12/O19 now extend shared-prefix native collation to causal right-padded suffixes.
Original typed marker/final-decision positions remain relative to each real
suffix, and all padded continuation state is discarded. Complete contexts bound
private KV workspace while padding limits count only newly submitted suffix
positions; retained prefix storage cannot disguise padding overhead. Dedicated
padded-prefix counters and unchanged fresh labeled/paired gates require actual
mixed cached work.

Original frozen typed Kev requests qualify at fixture FP32/FP16 without changing
their question lengths or probabilities. Native pages/chunks/kernels/standard
FP32 runtime LoRA, replica/retention semantics and packed-profile paired checks
retain their earlier limits. Released calibration/performance remains rejected,
pending or unmeasured. Mixed-prefix/cross-request collation and cooperative
branch scheduling remain open. No Apple or actual GPU checks ran.
[Evidence and limits](verification/fork-padding-cpu-20261008/README.md).


### CPU Kev native batches from one immutable prefix

O12/O19 now compose request-local prefill/fork reuse with equal-length CPU Kev
question batches. A native tensor call collates complete prefix KV, recurrence
and convolution into private rows, processes only new suffix tokens, then runs
the existing trained readout and unchanged temperature. Temporary handles clean
up on every path, and the parent and returned outputs remain independent.
Complete-context token budgets and the 64-handle cap bound admission; physical
work counters report actual suffix computation. A dedicated counter and gate
require real cached-branch batching, so scalar forks plus unrelated independent
batches cannot qualify this profile.

Frozen typed Kev references, full argmax and paired 1e-4 gates remain unchanged
across CPU FP32/FP16, pages, chunks, kernels, replicas and standard runtime LoRA.
Packed Q8/Q4 scheduling is compared to its own unqualified packed scalar profile,
not promoted to upstream calibration. CLI checks keep fresh labeled startup
requirements and refuse vacuous/incompatible profiles. Synthetic labels prove
plumbing only. Mixed-length/mixed-prefix/cross-request cached collation,
cooperative branch scheduling and released latency/RSS/calibration remain open.
No actual GPU check ran. [Evidence and limits](verification/fork-batch-cpu-20261008/README.md).


### Stable ONNX device I/O and bounded CUDA graph replay

O15 now has optional strict CUDA FP32 I/O slots and graph replay, separate from
CPU output reuse/shared initializers. Every exact input/output shape owns stable
device allocations and a CPU readback buffer; fresh inputs copy into their
original bound addresses. Up to 32 slots and a configured 512 MiB maximum
charged budget are retained without eviction. Captured graph IDs/addresses
therefore stay valid for the native session lifetime. Budget/slot exhaustion
refuses new shapes; copy/run/readback/nonfinite failures invalidate the context.
Allocator lifetime is explicit: a CPU ownership regression initially crashed
because ORT tensors outlived their allocator, and the retained allocator/drop
order fixes that crash. Returned core readouts remain separately owned.

Optional CUDA builder/profile code and the future ignored device regression
compile. Actual CPU address/data/ownership, limits, failure invalidation and
ordinary/compact/integrated/masked/batched/shared readout gates pass. CLI CPU
profile refusal and qualification regression checks pass. No GPU inventory,
execution, capture/replay or calibration check was run. The new path is default
off and requires fresh complete labeled qualification before serving. CPU
substitution tests establish ownership logic only; they do not qualify device
arithmetic, captures or performance. The charged budget excludes caller input
buffers, native arenas/workspaces and captured graphs. Device-resident trained
heads, released exports and actual GPU checks remain open/deferred.
[Verification and allocator failure evidence](verification/onnx-device-io-20261008/README.md).

### Immutable CPU bases with runtime LoRA adapters

O22 now keeps standard low-rank updates separate from CPU FP32 Qwen F2/F3
backbone projections. The optional shared-base cache therefore shares even
adapter-targeted dense tensors across independent models. Each model owns its
immutable A/B projections and readout; no adapter switch mutates base weights or
reuses another model's prefix state. Loaded models survive base-cache eviction.
Native scalar, candidate-only F3, equal/right-padded CPU batches, replicas and
Kev prefix/chunk/page paths keep their existing ownership and calibration paths.
Unsupported/incomplete adapter semantics and reduced/packed/device backbones
are rejected; all declared targets must have complete, consumed A/B weights.

Runtime and merged LoRA use different floating-point operation orders, so the
new default-disabled execution profile binds fresh complete labeled gates and
receipt/environment identity. Frozen independent Kev probabilities and fixed
full argmax gates remain unchanged; native F2/F3 scalar/batch/selected/fork
comparisons retain paired 1e-4 probability checks at unchanged temperatures.
Synthetic labels prove startup plumbing only. Runtime updates add two low-rank
projections per adapted call: expected savings concern resident dense weights
and startup merge temporaries, not guaranteed latency. Mixed-adapter tensor
collation, released calibration and measured RSS/throughput remain open. No
actual GPU check was run. [CPU evidence and limits](verification/runtime-lora-cpu-20261008/README.md).

### Immutable CPU KV pages for Kev forks

O19 now has default-disabled native Kev KV pages. Full compact CPU blocks share
across forks; appending a suffix replaces only a partial tail and creates new
blocks. Attention materializes all original keys and values in their original
order and retains its existing reductions. Recurrent/convolution state and
transactional offset/handle behavior are unchanged. Retained snapshots charge
shared page payloads conservatively within the existing byte/FIFO limits.
Changing the page profile with live/partial/retained caches or shared model
replicas is rejected. CLI preflight rejects incompatible profiles before device
selection, and receipts bind page size/storage identity. Serving requires actual
prefix/fork qualification and complete fresh labeled gates.

Native FP32/FP16 scores are bitwise equal to flat cached storage across four
page sizes, page boundaries, typed rows and short/shared prefixes. Frozen
independent PyTorch probabilities, fixed full argmax and paired 1e-4 gates stay
unchanged. CPU tests exercise retained hits/eviction, cancellation, malformed
continuations, context/handle bounds, replicas and profile guards; synthetic
labeled CLI gates exercise plumbing only. This is storage sharing, not a direct
paged kernel: attention still allocates contiguous complete KV workspaces.
Released acceptance, speed/RSS, branch collation and tenant policy remain open.
No actual GPU check was run. [Verification](verification/kv-pages-cpu-20261008/README.md).

### Actual ONNX output allocation ownership

O02/O08's retained-output path now binds the original preallocated tensor once
and keeps its owning ORT I/O binding. The previous implementation bound
`Tensor::clone()` on each call; in the pinned crate that deep-copies storage,
so its reuse counter overstated actual allocation reuse. A native address
regression failed on that implementation and passes with the owning binding.
Changing input values uses the same output address while previously returned
core tensors remain independent. Request inputs clear on every return path;
changed shapes/oversize/empty outputs keep their existing byte-bounded behavior.

Actual CPU generic/compact/integrated/masked/batched/shared-session checks and
the three affected CLI qualification groups preserve frozen results and the
unchanged probability/argmax/labeled/parity gates. No temperature, reference
or threshold changes. Default-disabled options and default dependency profile
are retained. CPU fixture correctness does not establish released calibration,
speed/RSS or device qualification. No actual GPU check was run.
[Retained reproduction and verification](verification/onnx-binding-cpu-20261008/README.md).

### Actual local CPU tensor sharding

O24 now uses vLLM's pinned two-process CPU tensor-parallel executor for Kev.
Loaded MLP dimensions prove shard ownership; normalized full hidden states
feed replicated FP32 pointer projections and unchanged core calibration.
Every rank must report exactly one complete native forward per group.
Actual rank layout, thread/runtime/library identities and configured rank count
bind fresh qualification. Thread and KV options are per rank with an explicit
total thread bound. Graceful native shutdown and owned-group failure cleanup
cover spawned workers. No default runtime dependency is enabled.

Independent frozen PyTorch probabilities/argmax and single-rank agreement use
the existing fixed 1e-3 bound, while native batch parity within the two-rank
profile uses the unchanged 1e-4 gate. CLI CPU processes exercise Pending raw
capture, complete synthetic labeled qualification, receipts, HTTP serving and
refusal paths. Synthetic labels cannot release Kev. Rank counts are separate
arithmetic profiles requiring independent fitting/held-out gates; one-rank
acceptance never authorizes another. More ranks, other families, pipeline,
multi-node/device execution and released capacity/performance remain open.
Actual GPU checks are deferred. [Operation and bounds](vllm.md) and
[retained CPU evidence](verification/vllm-cpu-sharding-20261008/README.md).

### Actual CPU vLLM raw pointer pooling

O23 now executes Kev/Qwen3.5 in a pinned optional local CPU vLLM worker with
BF16 backbone and FP32 trained pointer. The custom model omits the LM head,
sampler and decode loop; marker/final-position scores feed unchanged core
typed calibration. Equal-length groups perform exactly one verified native
forward. Strict artifact/runtime identities, owned bounded IPC, CPU-only
imports and failure invalidation prevent silent backend substitution.

Offline conversion merges complete standard LoRA into new FP32 storage,
preserves temperatures/overrides and writes Pending `vllm:bf16`. Actual
Pending fitting capture is supported; serving requires an explicit fitted
entry and fresh complete observed-label conformance. Default builds retain
their dependency profile. Fixed independent PyTorch fixtures pass probability,
argmax, native paired-batch and synthetic labeled gates. CLI processes cover
capture without source mutation, HTTP/receipts/work and qualification refusal.
Synthetic labels do not release Kev. Prefix/forks, chunking/padding, replicas,
runtime adapters, other families, quantization, distributed execution and
device profiles remain open. No actual GPU check or 4B performance/acceptance
claim is made. [CPU operation](vllm.md) and
[verification](verification/vllm-cpu-20261008/README.md).

### Whole-schema CPU Clef request batches

O12 now includes actual CPU Clef/F5 backbone collation of complete schemas.
All requests are encoded before the first model call, length-bucketed within
physical token/padding bounds and scattered in original order. Every row is
unpacked to its original length **before** the bidirectional joint head.
Questions within one schema are never split. Raw typed heads, temperatures,
logical usage and wire extensions stay in their existing paths. CPU replicas
share immutable weights and own execution state; HTTP replica routing and
cross-request collation still cannot be combined.

`--batch-max-requests 2..64 --max-batch-tokens B` enables this opt-in serving
path. Optional padding percent permits mixed lengths. The bounded preparation
queue freezes whole requests; F5 tokenization occurs inside the backend before
execution, so this increment does not move F5 tokenization onto a parallel
worker. Oversized singletons run whole and do not count as native batches.
`request_batch_execution=cpu-causal-right-pad-unpad-joint-v1` records CPU
availability. Actual forwarding/padding/preparation work is retained even on
failure, and new result-cache entries publish only after all outputs validate.

Frozen independent synthetic PyTorch fixtures cover FP32, FP16, grouped/vector
heads, equal and mixed lengths, malformed groups/context limits, typed answers,
cache/replica ownership and unchanged fixed numerical, argmax, paired 1e-4 and
complete labeled gates. A padded FP16 raw score differs by about 0.000131; its
raw check uses the fixture's existing FP16 tolerance (0.005), while probability
and conformance thresholds are unchanged. Synthetic labels prove gate plumbing,
not released calibration. Actual CPU CLI processes check fresh receipts, HTTP
collation/physical counters, absent/unlabeled qualification refusal and grouped
benchmarking across two replicas. Bench reports each request's full group
completion latency; it refuses a requested benchmark with no actual batch
unless exact result reuse avoided the work. Default workspace and affected
CPU checks are recorded with this increment. Released held-out acceptance,
performance measurements, GPU execution and cached-branch batching remain open.
[CPU verification and its limits](verification/clef-batching-cpu-20261008/README.md).

### Masked CPU ONNX feature batches

O12 now also collates mixed-length CPU native feature graphs that explicitly
declare `attention_mask`. Token zero padding is masked by original lengths;
each row retains its positions/order and explicit position IDs restart at zero.
Cache retention, branches and vocabulary-code hints are rejected. Complete
rectangles are charged by the existing bounded scheduler and actual padding
work is recorded. Output reuse and shared initializer replicas compose without
shared inference buffers. Non-batch/maskless graphs, integrated/compact
contracts and GPU profiles remain unsupported for this path.

A context-sensitive graph checks independent scalar features, all question
types, original usage, real token zero, duplicate/out-of-order readouts, empty
readouts, contexts 1..128, oversized/invalid inputs, equal-length identity and
concurrent shared contexts. Deliberately unmasked padding changes its valid
features. Whole-engine fixed synthetic labeled/parity gates count real batches
and reject incomplete/drifted suites. CLI processes bind work/replicas and reject
missing/unlabeled serving. Synthetic generic feature/mean heads are not released
Laya exports. The graph must really honor its mask and fresh observed-label
qualification remains mandatory; no speed/RSS or released acceptance follows.
F5, integrated graph heads, cached branches and device work remain open.

### Native CPU ONNX integrated F1 raw heads

O08/O13 now include default-disabled actual CPU FP32 graph-side F1 scalar
heads. Strict dynamic token/marker/qtype signatures match the browser ABI;
unknown names/shapes/dtypes are refused. Raw finite marker scores bypass host
mean projection and feed unchanged shared typed temperatures/softmax. Native
loading is restricted to `laya-v1`, scalar F1 and the declared native tokenizer.
Serving needs an explicit fitted/refitted `onnx:fp32` entry and fresh complete
observed-label gates. Capture from pending packages stays an offline operation.
Profile metadata and persisted environment identity bind the execution.

Bounded CPU output reuse owns returned data; optional shared initializer
replicas retain immutable sources and independent sessions/buffers. Native
tests compare all question types against frozen independent synthetic scalar
arithmetic, including masked/unmasked graphs, reordered/duplicate markers,
empty readouts, nonfinite failures and concurrent replicas after primary drop
and source replacement. Actual CLI tests check complete fixed conformance,
identity and missing/unlabeled/default-only calibration refusal. No released
graph export, outcome calibration or speed/RSS result follows from synthetic
fixtures. Native batching, feature-gather contracts, prefix caches, other
precisions and GPU providers are explicitly incompatible with this increment.
Device-resident paths and released Laya export/qualification remain open.

### Masked CPU ModernBERT batches with original-length Laya heads

O12 now also accepts mixed-length native F1 CPU inputs through the existing
bounded per-/cross-request scheduler. Row-specific backbone masks exclude
padding from valid attention. Each row is restored to its original length
before the typed bidirectional Laya head, including final marker-query
selection. No head padding, criterion truncation, prefix reuse or alternate
temperature is introduced. Equal-length native execution stays unchanged.

CPU capabilities record `cpu-right-mask-unpad-head-v1`; existing options,
token/padding budgets, fresh numerical/independent parity and complete labeled
startup gates remain mandatory. Native tests include real token zero, lengths
1..128, repeated/out-of-order readouts, global/local/global backbone layers,
bare and two-layer typed heads, selected marker queries, multiple unchanged
temperatures, empty readouts, replicas and invalid token/context/cache inputs.
Whole-engine tests preserve typed usage, count real padded rectangles and
exercise cross-request work; incomplete or drifted fixture suites fail.
CLI processes bind the profile/work and refuse missing/unlabeled serving.
Synthetic heads/labels and bare mean-head fixtures are implementation evidence
only. Padding can cost additional arithmetic/workspace; released Laya acceptance
and controlled throughput/RSS remain open. F5/ONNX padding and cached-branch
collation remain later increments. No actual GPU or Apple check runs.

### Actual CPU browser execution with shared Rust calibration

O27 now has a separate WASM core and ONNX Runtime Web 1.30.0 CPU SDK. The actual
graph includes the scalar F1 head; the core owns immutable prompts, candidate
mapping, temperatures, typed answers, usage and conformance. External-score
plans are single-use and engine-bound. Native backend forwards cannot silently
stand in for browser execution. WASM selects an explicit tokenizer regex
profile while native dependencies retain their original settings. Default
native builds gain no active browser runtime dependency.

Fresh loading verifies asset bytes/hashes and strict dynamic graph signatures,
then executes every complete labeled golden question through a new actual CPU
session. Shared unchanged delta/argmax/ECE gates must pass before public eval,
even for fitted calibration. Default fallback cannot authorize this runtime.
Metadata binds core, ORT JS/loader/WASM, model, tokenizer, manifest/golden and
browser/thread profile. Bundled loading additionally checks its descriptor and
executing SDK source. Unsigned reports cannot attest dataset provenance.

CPU Chrome tests compare native/WASM tokenizer inputs exactly and all typed
answers/raw scores to independent native synthetic arithmetic. They exercise
incomplete/unlabeled/drifted/pending/nonfinite/substituted/signature refusal,
bounded queues, input snapshots, error cleanup, disposal and self-contained
packaging with fresh qualification. Synthetic labels test gate plumbing only;
recorded evidence stays `qualified=false`. Artifact/queue/context limits and
JS/WASM copy costs are explicit. Released Laya/Kev exports/calibration, other
families, worker/multithread scheduling, quantization and browser prefix/batch
execution remain open. WebGPU and actual GPU checks are deferred; Apple work
is skipped. See [operation and limits](../browser/README.md).

[Retained CPU browser report](verification/browser-cpu-wasm-20261008/summary.json)
records Chrome 151, twelve actual qualification forwards, zero fixture delta
and a separately qualified bundled session. The
[source/check identity](verification/browser-cpu-wasm-20261008/identity.json)
also records the passing WASM build, 74 native core tests and default workspace.
Both artifacts explicitly deny released-model qualification.

### CPU grouped-query attention without repeated K/V

O09/O02 now include default-disabled grouped CPU query matmuls. Query heads
retain the original contiguous groups mapped to each K/V head, while K/V keep
their original head count. CPU matmul gets dense K/V copies where projections
are strided, without query-head repetition. Masks, absolute prefix offsets,
complete key reductions, scale, rotary and output gates/heads remain. It composes
with bounded query rows; full score matrices/arithmetic remain quadratic when
that profile is disabled. Existing retained prefix K/V was already unexpanded.
No measured released-model performance/memory claim follows.

Execution changes reject shared replicas or active/partial/retained caches.
The explicit CPU option, metadata, receipts and runner bind the arithmetic
identity; every real serving runtime still needs fresh complete observed labels.
CPU FP32/FP16 fixtures cover 1/2/4 query-head groups, strided inputs, masks/prefix
offsets, mixed native padding, complete pointer probabilities, replicas, selected
F3 codes and Clef joint heads. A real padded-batch test found and verified the
contiguous K/V requirement before acceptance. Packed Q8/Q4 tests compare to the
same independent packed kernels, without source-temperature substitutions.
The complete `clef,qualification,quantization` CPU workspace and 22 Python
runner checks pass. Actual CLI conformance binds the new profile and retained
receipt, exercises real prefix interleaving, and still marks unlabeled fixture
evidence as unqualified. No actual GPU/Apple check ran.
The dependency-light default workspace also passes with this option disabled.

### Bounded qualified CPU lazy residency

O22 now has actual opt-in cold loading, preloading and idle/capacity eviction,
under the optional `qualification` CLI feature. Explicit CPU/backend/dtype
selection avoids hardware probes and hidden precision changes. Registration
pins all model inputs and complete labeled suites without materializing weights.
Every cold load performs fresh existing native/optimized/replica conformance
and receipt verification before publishing engines, with input/environment
rechecks around loading. Failed/substituted/panicking loaders stay unavailable
until restart. No temperature, golden threshold or default loading changes.

A count bound covers loading/resident groups, including replicas. Concurrent
cold callers share one worker under a bounded wait queue; cancellation leaves
its reservation intact. LRU/idle eviction requires no model/pool/engine owners
or admission permits. Native destructors run outside the registry mutex; idle
batch workers hold weak references. Health/listing/metrics do not load weights.
Cold gates cost latency; model size, temporary allocations and shared-base cache
residency remain additional to the group count. This is not a measured RSS claim
or runtime multi-LoRA dispatch.

API fixtures exercise canceled/shared loads, full queues, busy slots, external
engine references, retry refusal, authentication/validation, and idle batching
workers. Native CPU Clef tests use unchanged upstream vectors through actual
cold loads/reloads; incomplete labels, drifted goldens and changed pinned files
fail closed. Their synthetic labels establish test plumbing only, without
released-model or GPU/Apple acceptance.
Default and `clef,qualification` CPU workspace tests pass. The final native
cold-load process checks also pass after runtime pinning changes. Eager loading
remains the default, and no source temperatures/goldens were changed.

### Bounded CPU attention query workspace

`HUNCHO_DEVICE=cpu HUNCHO_ATTENTION_QUERY_ROWS=1..4096` opts native Qwen F2/F3
and Clef F5 into query blocks; zero retains the original dense call shapes.
Every block keeps all K/V rows and reduces softmax over the complete causal key
axis. Masks use absolute prefix/query offsets, and outputs concatenate in
original token order before the unchanged attention gate/output projection.
Norms, rotary, GQA expansion, recurrence, trained heads and temperatures retain
their formulas. This is query blocking, not FlashAttention or key truncation.

Score/probability intermediates have at most `B * heads * query_rows * keys`
elements, and a block mask has at most `query_rows * keys` elements. The setup
cache holds rotary data without a full quadratic mask in this profile. K/V,
hidden activations and concatenated block outputs still consume memory, and
total arithmetic remains quadratic. More launches/copies can slow short inputs;
there is no measured released-model speed or peak-RSS claim.

The profile rejects unsupported families/devices, invalid bounds, models without
full-attention layers and changes after live prefixes/shared replicas. Switching
profiles invalidates setup masks. Metadata, qualification environment and the
Python runner bind exact block size and native arithmetic identity. Complete
fresh observed-label serving gates remain required; prefix/batch options also
retain their tighter paired/nonvacuous gates.

Actual CPU FP32/FP16 checks cover nonzero attention values and masks, frozen Kev
probabilities, one/partial/oversized blocks, full-context prompts, prefix offsets,
replicas, mixed-length padded batches, selected F3 logits and complete Clef joint
heads. Tests inspect removal/invalidation of quadratic setup masks. Fixture
parity does not qualify a released checkpoint or introduce a GPU/Apple check.
The default, `clef,qualification` and `quantization,clef,qualification` CPU
workspaces pass, along with the 21 Python runner tests. Actual packed Q8/Q4
kernels also pass paired query-block checks without substituting source FP32
outputs or refitting fixture temperatures. Combined feature coverage caught and
fixed missing pending-prefix initialization in the packed loader and its unit
reference constructor.

### Resumable native llama.cpp prefixes

The pinned CPU runtime now exposes the existing bounded prefix-step interface
when `HUNCHO_PREFILL_CHUNK_TOKENS=1..4096` is configured for F2. Partial handles
retain exact tokens and full hybrid state under the 64-handle/512 MiB limits.
Each native chunk restores its own snapshot and absolute positions, and commits
only after successful serialization. Partial state cannot enter forks/readouts
or retained-prefix hits. Release/cancellation frees pending tokens and state;
replicas inherit the immutable execution profile, without active state.

The existing cooperative scheduler can alternate these CPU prefixes and whole
questions. One-shot prefixes and chunks use the same accounting; split counters
describe actual attempts and retained hits submit zero work. Native independent
calls and batches between steps leave pending snapshots unchanged. Chunk size
and native execution identity bind qualification. Existing fixed external,
paired, split/interleaving and complete observed-label serving gates remain.
Snapshot copies can outweigh prefill savings, so no performance claim follows.

Actual FP32/FP16 CPU tests cover 1/3/7-token chunks, interleaved distinct prefixes,
native batches between steps, final pointer probabilities, retained hits,
partial/foreign/completed handle rejection, cancellation/reuse bounds and
invalid settings. CLI tests exercise actual cooperative yields and distinct
request switches against unchanged upstream fixture vectors. They retain
unlabeled diagnostic status and do not qualify released Kev-4B.
The complete CPU `llamacpp,clef,qualification` workspace passes; all actual
GPU tests remain ignored, and no Apple build or hardware check was run.

### Bounded native llama.cpp batches

`HUNCHO_LLAMA_BATCH_ROWS=2..8` adds explicit isolated native sequence slots for
equal-length CPU F2/F3 batches. Default one preserves the previous context/path.
KV/recurrent and scratch allocations grow with slot count; model/head weights
remain immutable and shared with replicas. Every token uses an explicit sequence
ID and original position. Readouts scatter using original marker/decision/code
order; no pooling, generated tokens, padding or branch collation is introduced.

Backend batch limits now also constrain the engine planner. This runtime limits
the whole call to 256 charged readouts and its configured slots, in addition to
the caller's physical token budget. Groups split without discarding candidates;
large singletons remain independent. Native batch errors clear scratch state,
and immutable prefix handles survive successful or failed independent batches.
Qualification identity binds sequence count and native profile. External,
paired, nonvacuous batch and complete observed-label serving gates remain fixed.

Actual CPU FP32/FP16 fixture tests exercise four distinct full-context sequences,
pointer and full/selected F3 logits, repeated/reverse readouts, row permutation,
replicas, prefix replay and invalid/bounded calls. CLI checks use repeated
unchanged fixture cases to create cross-request equal shapes and preserve the
unlabeled diagnostic disposition. No released calibration or speed claim follows.
The full `llamacpp,clef,qualification` workspace passes, and actual native Q8/Q4
batch diagnostics pass at the unchanged fixture temperature. Packed entries
remain unrefitted numerical fixtures; these checks do not permit serving them.

### Bounded CPU mixed-length batches

`--max-batch-padding-percent 1..100` opts CPU Qwen F2/F3 into right-padded
question/request batches; zero retains exact-length buckets. It requires a
positive `--max-batch-tokens`, remains incompatible with prefix reuse, and
rejects other families/devices. The planner groups increasing lengths, bounds
native rectangles by the physical token budget and 64 rows, and limits padding
as a percentage of all submitted positions. Oversized singletons run intact
independently. Every candidate uses its original relative position, and the
pointer decision uses the original last valid token. Existing causal attention,
convolution and recurrence ensure padding cannot affect earlier valid rows;
this does not implement bidirectional/joint masks or paged/cached branches.

Wire usage keeps original prompt tokens. Work and Prometheus counters include
physical padding, with separate mixed-batch/padded-position counts even for
failed calls. Cache keys, opaque prepared options, API scheduling and retained
qualification records bind the padding policy. Conformance additionally requires
actual nonzero padding and a mixed native call, plus unchanged external/paired
and complete observed-label serving gates. A singleton/exact-only suite cannot
qualify this mode. No temperature, source golden or default execution is changed.

CPU fixture checks exercise FP32/FP16 pointer and selected F3 logits, repeated
readouts, replicas, invalid/cache-bound rows, physical-budget planning and frozen
engine conformance. A nonzero counterexample detects using the padded final row
as the pointer decision. Cross-request checks use the unchanged fixture vectors
with an explicit larger padding allowance; this does not establish released
Kev/Nimble calibration, performance or any GPU qualification.

The default workspace and `clef,qualification` workspace tests pass. HTTP
scheduler fixtures verify distinct mixed-length responses, original wire usage,
authentication and physical padding metrics; actual model arithmetic is checked
separately by native CPU Qwen fixtures. The Python runner binds the exact padding
policy and rejects substituted policies or missing mixed-work evidence. No
released profile is promoted by these checks.

### llama.cpp full hybrid prefix snapshots

Kev F2 now supports request-local prefix reuse and optional exact retained
prefixes on the pinned CPU runtime. Full native sequence serialization preserves
attention KV and recurrent/convolution state together. Forks share immutable
bytes; each suffix clears/restores its own snapshot into sequence zero and uses
explicit absolute positions. Success commits a new branch snapshot; validation,
native, readout or allocation failure leaves the previous snapshot usable.
Process-unique handles use the same allocator as Candle/Mock, preventing
cross-runtime ownership collisions. Replicas copy no live/retained state.

Each context bounds live handles to 64, retained keys to 16 and total charged
snapshot/key/entry payload to 512 MiB. Caller retention defaults off and must
fit the same limit. Native model/context memory is additional. Snapshot
copy/restore can dominate short suffixes, so no speedup is inferred. Prefix and
retention options remain off by default and require nonvacuous external/paired
plus complete labeled serving gates. Fixture FP32/FP16/Q8/Q4 checks and real CLI
conformance cover unchanged goldens, forks, continuation, failed validation,
independent interleaving, replay hits, release/eviction, bounds and replica/foreign
handle rejection. Native multi-sequence batching/paging and released model
acceptance remain open; GPU/Apple execution was not tested.

### Laya final-layer marker queries

The F1 head now has a default-disabled CPU profile,
`HUNCHO_LAYA_SELECTED_HEAD=1`. It keeps full backbone/earlier head layers and
the same final Q/K/V projection, gathers only final query/residual rows, and
applies final attention output and FFN to those rows before the existing trained
scorer. All full-sequence keys/values, typed embeddings, temperatures and ordered
markers remain. Final-layer attention storage becomes O(markers × sequence)
instead of O(sequence²); earlier layers/backbone are unaffected. Extra gather
cost can outweigh savings for short sequences or many markers.

Nonzero typed fixtures exercise all three types, one/many/all/repeated/reversed
markers, multiple attention heads, native batches and replicas at unchanged
1e-4 calibrated paired tolerance. A counterexample test proves that dropping
nonmarker keys would change outputs. Bare encoders/other runtimes and invalid
flags are rejected. Metadata and retained runtime identities bind the profile,
and every real serving runtime still requires fresh complete labeled gates.
No trained head is substituted and no source temperature is refitted here.
The CPU-only `laya_profile` example records full-package paired outputs and
alternating timings on synthetic requests, always as unqualified evidence.

The cached released checkpoint at revision `55cf4c4...` passes that six-question
CPU paired diagnostic: maximum probability delta 2.9802322e-8, raw delta
2.3841858e-7 and complete argmax agreement at unchanged package temperatures.
Four alternating warm pairs average 9.221 s baseline and 9.125 s selected for
two three-question requests. Timings overlap, so no reliable speedup is claimed.
This has no independent external golden or observed outcomes and grants no
serving acceptance. [Full outputs, timings and identities](verification/laya-selected-cpu-20261008/summary.json)
retain the source, binary, model/tokenizer/config hashes and scope. Default and
`clef,qualification` workspace checks plus CLI flag guards pass; no GPU check
was executed.

### Qualification runner profile coverage

The CPU qualification runner now requires `native_execution=candle-qwen35-v1`
and recognizes explicit fused-gate, cooperative scheduling and OpenBLAS
profiles. BLAS requires a supplied library and a JSON object containing exactly
the five expected `cpu_blas_*` identity fields; the file hash and fixed thread
budget must match, and both files are rechecked afterward. Unrequested ambient
BLAS, shared-base and cooperative profiles are cleared. Optimized scheduling
requires nonzero split-prefix yields and actual switches between distinct
requests. Existing numerical, argmax, ECE, paired and complete-label thresholds
remain unchanged; numerical-only runs never become outcome acceptance.

The real CPU CLI smoke combines BLAS and 3-token resumable chunks on the
unchanged two-layer Kev fixture. Independent and prefix modes both pass with
external delta 2.9802322e-8; prefix paired delta is also 2.9802322e-8, with
21 native prefix calls, 19 yields and 8 request switches. It is unlabeled and
retained as unqualified. [Full reports and identities](verification/current-cpu-runner-20261008/summary.json)
record the source, executable, model files, script and expected BLAS profile.

### Optional OpenBLAS projections

O03 now includes `cpu-blas`, an optional dynamic LP64 OpenBLAS path for dense
FP32 Qwen F2/F3 and Clef backbone projections. It borrows contiguous CPU storage
for row-major SGEMM and applies the existing bias operation afterward. Default
builds and unset library variables use Candle unchanged. FP16, packed weights,
OpenMP and ILP64 are rejected; single-thread libraries must use native locking
for concurrent contexts. Configuration is fixed before caches/replicas and
recorded as library hash, selected CPU kernel, build configuration and thread
budget. This is changed reduction arithmetic, so fresh complete labeled
serving conformance remains mandatory.

Real CPU tests cover strided/offset inputs, shape/bias checks, frozen Kev
probabilities, prefix and batch parity, independently executing replicas, F3
selected logits, Clef whole-request heads, unsupported precision/thread bounds
and CLI startup refusal. The tiny fixtures do not establish released outcome
calibration. A one-thread release projection microbenchmark on local x86
shows a median Candle/OpenBLAS latency ratio around 1.01 with overlapping
individual timings. No reliable speedup or released-model throughput gain is
claimed. [Retained measurements](verification/blas-cpu-20261008/microbenchmark.json)
include the binary and library identities; this path stays default-disabled.

### CPU FP16 drift diagnosis

The original CPU FP16 path was rerun on the same eight held-out diagnostic
cases selected from the failed CPU FP32 suite. Five exceed the unchanged
0.001 probability delta limit; the largest offline FP64 softmax delta at the
unchanged source temperature is 0.0317022. This is selected-case diagnosis,
not a complete conformance run or a refit. It shows that switching the CPU
baseline to FP16 does not remove the observed drift against the older frozen
CUDA FP16 reference. The cause remains unresolved, and no CPU profile is
promoted. [Raw logits, identities and analysis](verification/kev-cpu-fp16-drift-20261008/summary.json)
retain the evidence without corpus text or a new GPU check.

### Readout, allocation and calibration checks

- O01: request each readout position once, keeping candidate order in the head.
- O05: optional vocabulary-code hints and explicit selected-logit output; native F3 projects only requested trained weight/bias rows. Backends can still return full logits. Engine/bench use the compact path by default; HTTP serving retains the legacy F3 readout until `--candidate-readout` passes startup qualification.
- O02, first portion: move prompt tokens/raw logits, borrow feature rows, index sorted readouts, project scalar heads without an output allocation, expose reusable Linear output buffers, and flatten native outputs directly into host storage. Scalar accumulation order is retained.
- O02/O08 allocation portion: ONNX copies selected rows directly from ORT-owned CPU output memory, avoiding the extra complete sequence-by-hidden host Vec. Repeated/out-of-order rows retain exact float bits. Output rank/sequence coverage and positive graph width are validated; runtime feature width remains authoritative for nonempty readouts, preserving existing conformance. The legacy empty-readout width hint remains unchanged. The mock example's manifest declares 1,024 hidden features while its ONNX graph emits 512; strict manifest-width enforcement is a separate compatibility change. ORT still materializes its full output; graph-side gather, GPU EPs and device I/O binding remain pending.
- O07: bound exact single-token ID memoization to 512 short keys per HF tokenizer. Optional `HUNCHO_TOKEN_CACHE_BYTES` wraps real manifest-loaded F1–F4 tokenizers in an instance-local FIFO cache for exact text/special-token encode calls. Optional `HUNCHO_PROMPT_CACHE_BYTES` reuses complete per-question tokens/candidates/positions/prefix boundaries/type indices. Both default to zero, have charged-byte budgets and 1,024-entry bounds, and bypass oversized entries/preparation errors. Prompt keys preserve ordered state/question values and optional-description presence; IDs and execution/extension options do not affect F1–F4 prompt preparation. Hits still execute all model work and enforce current budgets. Owned copies protect retention from prefix/batch mutations. F5 retains whole-request ownership; conformance bypasses prepared-prompt retention. Tests cover unchanged raw logits/answers/legends/usage, physical work, order/identity, invalid requests, errors, eviction and native Kev prefix replay. These caches do not alter inference arithmetic or temperatures.
- Calibration checks reject nonfinite logits, invalid fitting rows/targets, empty/incomplete golden suites, malformed distributions and mismatched labels. ECE uses empirical mean confidence. Conformance resolves ties using the loaded formatter's actual candidate order (and F5's joint schema), matching inference rather than alphabetical map order. A regression case proves a tiny reference delta cannot hide a different selected label.
- O04, first portion: optional exact whole-request response retention through `--result-cache-bytes` (serve/bench), with `HUNCHO_RESULT_CACHE_BYTES` for serving. Each immutable engine owns its FIFO; full ordered requests, optional-description presence and execution/extension options form the key. Presence bits fix a reproduced library-call alias: `None` and `Some(Value::Null)` serialize alike but can produce different prompts. Wire serialization and formatting remain unchanged. Hits clone calibrated answers/raw logits/legends/usage without recomputation, and physical-work counters remain zero with a separate hit counter. Errors and oversized entries bypass retention; charged bytes and 1,024 entries bound retention. Default zero stores no cache and takes no cache lock. Authentication/admission still apply. Tenant-specific policies remain pending. Conformance bypasses all result retention.
- O04, in-flight portion: `--coalesce-bytes` / `HUNCHO_COALESCE_BYTES` enables exact HTTP request sharing in an instance-local table bounded by charged metadata and 1,024 keys. Matching ordered requests/options share one result without modifying inference math or temperatures; completed results and failures are removed. Every caller retains authentication/admission checks. Cancellation of one caller cannot abandon other waiters, all canceled queued callers release capacity, and running work keeps its permits. Oversized/full tables bypass sharing. Tests cover exact fields/extensions, overload/auth, error retry and both queued/running cancellation. This does not complete cross-request tensor scheduling.
- Golden suites optionally accept observed target labels. Labeled suites report outcome ECE/Brier; unlabeled suites identify their reference-agreement basis. Partial target coverage is rejected.
- Scalar refits no longer inherit DEFAULT temperature overrides. Optional row-aligned question types fit active type/option-count strata. Existing variant confidence definitions survive a refit; stale shared evaluation hashes are cleared.
- The actual CLI accepts the documented `calibrate --save false` dry run, with optional `--json` fitted-entry/NLL reporting. Process-level coverage verifies the manifest is byte-for-byte unchanged.
- Benchmarks support distinct or repeated inputs, mixed types, concurrent clients, loaded model names and JSON metadata. `--reference-readout` on bench/conform compares the legacy F3 path against the optimized path and the same unchanged golden vectors.

Validation completed: default workspace tests and workspace tests with `huncho-cli/clef`. The native F3 test uses the existing reference backbone/LoRA plus a temporary deterministic vocabulary projection/bias, compares fp32/fp16 at existing temperatures, and requires probability delta <=1e-4 and matching argmax. Existing Kev/Clef reference tests remain unchanged. This is fixture coverage; full Nimble/Laya/Clef model/device qualification remains required. No production speedup is inferred from fixture timings.

Prepared-prompt reuse also has a release CPU pilot on the pinned two-layer Kev fixture: three repetitions of 20 warm requests with 20 mixed questions, four Rayon/Candle threads and alternating cache order. Every run submits 400 forwards; repeated input with a 1 MiB cache reports 400 prepared-prompt hits. Mean run latency averages 14.40 ms without retention and 14.03 ms with it, but cache run means span 13.28–14.89 ms and one run is slower. Distinct input shows a similar small difference despite zero hits. This is evidence of correct work accounting, not a reliable speedup or released Kev-4B/T4 performance claim. Both unchanged upstream conformance runs pass (max delta 2.9802322e-8, complete argmax agreement), with zero cache hits during qualification. [Retained pilot](verification/prompt-cache-cpu-20261007/summary.json) and [execution script](verification/prompt-cache-cpu-20261007/run-pilot.py) record complete runs, source/binary/fixture identities and limits.

### Bounded serving foundation

- O06, first portion: release the model-registry lock before inference; use a bounded per-model admission gate and Tokio blocking execution. Waiting requests suspend asynchronously. One complete evaluation runs per served model at a time.
- `--max-queued-per-model` / `HUNCHO_MAX_QUEUED_PER_MODEL` defaults to 32 waiting requests, excluding the running request. Zero disables waiting. Overload returns HTTP 503 with `queue_full`.
- Running jobs own admission/execution permits after HTTP cancellation. Canceled waiting jobs release admission and do not execute.
- Metrics separate admitted requests, waiting requests, model-slot queue wait and engine evaluation time. Token counters now count submitted forward/prefill positions, including failed attempts and jobs whose HTTP callers disconnect. Wire `usage.input_tokens` retains logical prompt usage. Native fork/batch and reused-prefix counters describe actual engine operations.
- Tests on a single Tokio worker verify health responsiveness, registry write access, bounded overload, queued cancellation and running-job cancellation.

O06 preprocessing overlap is now opt-in with `--max-prepared-per-model` / `HUNCHO_MAX_PREPARED_PER_MODEL` (default zero). F1–F4 prompt formatting/tokenization runs on blocking workers before acquiring model execution capacity. Each preparation slot covers one running preparation or ready packet; it is released when model execution starts. Owned, opaque packets freeze the request/options/prompts and are bound to their originating immutable engine. All model call shapes/order, heads, temperatures and usage remain unchanged. F5 keeps whole-request backend preparation. This is bounded CPU preprocessing overlap, not dynamic tensor batching or a demonstrated model-throughput gain.

Preparation cancellation keeps admission/slot capacity until the blocking job finishes and submits no model work. Ready cancellation releases its slot; running model cancellation keeps model admission through completion. Tests exercise these boundaries, single-worker health responsiveness, exact answers, sharing/authentication, native prefix/batch composition and cache-bypassing conformance. CLI startup requires external and paired qualification plus nonzero prepared-question work. The bounds count requests, not exact RSS bytes; a large multi-question packet can retain substantial memory. [Preparation API](../crates/huncho-core/src/engine.rs#L254), [serving pipeline](../crates/huncho-api/src/routes.rs#L176) and [slot configuration](../crates/huncho-api/src/state.rs#L32) show the implementation.

The full Kev/T4 preparation check passes the unchanged six-case/19-question numerical suite with external delta 4.656613e-10, paired delta zero and complete argmax agreement. A four-burst HTTP pilot (eight distinct callers, three mixed questions each; order off/on/on/off) returns byte-identical answers/raw logits/usage and submits 1,344 token positions per burst in every mode. Enabled bursts prepare 24 questions, with zero cache/coalescing/fork/batch work. Burst means are 12.287 s disabled and 12.597 s enabled, about 2.5% slower in this small pilot; no throughput improvement is claimed and default zero is retained. [Numerical identity](verification/kev-t4-20261008/stage6-preparation-numerical/identity.json) and [HTTP pilot](verification/kev-t4-20261008/stage6-preparation-pilot.json) retain full evidence. The separate 1,536-case labeled GPU gate was interrupted on 2026-10-08 after the user deferred actual GPU checks. It supplies no acceptance result. [Interruption record](verification/kev-t4-20261008/stage6-user-deferred-gpu.json) and [unfinished runner identity](verification/kev-t4-20261008/stage6-preparation-labeled-interrupted.json) preserve the disposition.

Exact in-flight sharing is now validated on full Kev-4B/T4 CUDA FP16 as well as cancellation/auth fixtures. Three eight-caller loopback bursts retain byte-identical independent responses, reduce submitted positions 2,016→252, and complete in a mean 0.849 s versus 6.661 s (7.85× for identical inputs). Unique inputs still evaluate independently; result retention and tensor batching were disabled. [Pinned HTTP audit](verification/kev-t4-20261007/stage2-http-coalescing.json) records the source/binary/runtime and exact body hashes. Landing sites: [bounded flight table](../crates/huncho-api/src/coalesce.rs#L59), [sharing/cancellation](../crates/huncho-api/src/routes.rs#L120), [blocking evaluation](../crates/huncho-api/src/routes.rs#L175).

### Native Kev prefix fan-out

- O11: attention KV at unexpanded GQA width, FP32 recurrent matrices, small convolution tails and absolute rotary/mask offsets. Forks share immutable prefix tensors; continuation produces new state. Explicit handles are process-unique, model-owned, bounded to 64 live handles per backend, and released after each branch/request. Failed native forwards leave the branch unchanged.
- The engine prefills a verified exact `kev-v1` token prefix once for multi-question F2 requests, rebases suffix positions, forks each question and releases handles on errors. Single-question and unsupported paths retain independent forwards. No arbitrary concatenation or tokenization boundary change is introduced.
- `--prefix-cache` / `HUNCHO_PREFIX_CACHE` enables the path after startup qualification. It remains off by default. Persistent cross-request prefix reuse, paged KV and combined cached-branch batching remain pending.
- `eval_with_stats` reports submitted token positions and physical calls separately from logical usage. For Q questions with prefix P, physical positions become `sum(row lengths) - (Q-1)*P`; the prefix is computed once and consumed by every fork.

The pinned native Kev fixture passes CPU/CUDA fp32/fp16 probability parity <=1e-4 and matching argmax for whole suffixes, short prefixes, single-token chunks and fan-out. Tests also cover parent release, cross-model handles, context limits, invalid token IDs, branch rollback and repeated later-question errors past the retained-handle bound. Full Kev-4B/T4 FP16 prefix reuse failed qualification; see the full-model results below.

Landing sites: [prefix fan-out](../crates/huncho-core/src/engine.rs#L450), [native prefill:1665](../crates/huncho-backend/src/qwen3_5.rs#L1665), [prefix tests](../crates/huncho-backend/tests/prefix_cache.rs).

### Native batching and setup/transfer reuse

- O12, first portion: explicit `supports_batch`/`forward_batch` interfaces and native equal-length ModernBERT/Qwen backbone batches. Engine preparation groups independent questions by exact length, scatters row-specific readouts back to IDs, and limits each tensor batch to 64 rows and `--max-batch-tokens` / `HUNCHO_MAX_BATCH_TOKENS`. An oversized singleton runs independently without truncation. Row positions, F3 code subsets and typed heads stay distinct. F5 retains whole-request inference.
- Batching remains off by default and requires startup qualification. Prefix reuse and batching are currently mutually exclusive. This is per-request question batching: cross-request queue collation, wait limits, padding and continuous scheduling are still open, so CORE-07 is not complete.
- Remaining O02 groundwork: prepare immutable zero-centered norm weights once; convert convolution inputs to FP32 once per convolution. Native Qwen also reuses exact rotary tables/device masks keyed by length and absolute offset in a FIFO cache bounded to 32 entries and 4 MiB of tensor payload per model. Large masks bypass retention; dense attention remains quadratic.
- O13, first portion: concatenate Clef question logits on device and transfer once, retaining each question's head arithmetic and option order. Pooling/projection vectorization is pending.
- O10 guard: ModernBERT rejects fp16/bf16/int8/int4 labels because its current loader executes FP32. It cannot select an incompatible calibration entry while silently casting to FP32.
- O22 loading groundwork: pointer loading excludes unused LM-head weights before materialization/casting. Shared bases, adapter scheduling and lazy eviction are pending.

Native batch tests require probability parity <=1e-4 and matching argmax for ModernBERT scalar readouts, Kev pointers and F3 selected logits; Kev pointer tests also pass on CUDA. Full Kev-4B/T4 FP16 batching failed the tighter paired gate. The ModernBERT fixture is a bare encoder; it does not qualify a released trained Laya head. Existing upstream Kev/Clef golden files remain unchanged.

Landing sites: [batch preparation](../crates/huncho-core/src/engine.rs#L566), [ModernBERT batches:261](../crates/huncho-backend/src/candle.rs#L261), [Qwen batches:1602](../crates/huncho-backend/src/qwen3_5.rs#L1602), [attention setup cache:1015](../crates/huncho-backend/src/qwen3_5.rs#L1015), [Clef transfer:302](../crates/huncho-backend/src/clef_head.rs#L302).

### Optimization qualification before serving

`huncho serve` requires `--qualification-golden MODEL=PATH` for each model using prefix reuse, native batching or F3 candidate-only projection. It runs conformance on the loaded engine/device before opening the listener. Besides the unchanged external golden delta/argmax/ECE gates, conformance runs paired independent forwards and requires <=1e-4 probability delta and complete argmax agreement. A prefix/batch suite must actually exercise a fork/batch; a vacuous singleton suite cannot qualify those paths. Bindings with unknown/duplicate model names are rejected.
Explicit bindings for independent serving are checked too, including missing/invalid files and failed goldens. Result caching cannot bypass the checks.
CLI serving rejects pending calibration, and refitted entries require an explicit labeled held-out suite even for independent execution. Core conformance rejects partial target coverage. The gate still trusts existing `fit` entries without an explicit suite and cannot prove fitting/evaluation separation from caller-supplied files; this is not a persisted qualification certificate.

The gate evaluates the currently loaded artifact and temperature; it does not persist a certificate or prove the provenance/coverage of a caller-supplied golden file. Use independent pinned references and representative cases, plus observed outcomes for statistical calibration. Lower-level Engine/API configuration remains available to library callers, who must run equivalent qualification. General enforcement for every existing backend/dtype and persisted execution identities is still open.

Landing sites: [startup gate:185](../crates/huncho-cli/src/serve.rs#L185), [paired conformance:174](../crates/huncho-core/src/conformance.rs#L174), [ordered argmax:447](../crates/huncho-core/src/conformance.rs#L447), [prepared-prompt cache](../crates/huncho-core/src/engine.rs#L675), [typed cache keys](../crates/huncho-core/src/cache_key.rs#L7). Tests demonstrate a variant that meets external delta/argmax/ECE thresholds but fails the tighter independent parity gate.

### Fixture measurements

A CPU/debug smoke benchmark on 2026-10-06 used the pinned tiny Kev weights in a temporary package: 20 mixed questions, 10 timed distinct requests, one client and FP32. Compilation/loading/warmup were excluded. The numbers below preceded the attention-setup memoization; they verify the earlier work counts and are not production predictions or a statistically controlled speed comparison.

| Path | Submitted token positions | Backbone calls | Prefill calls / forks | Mean request latency |
|---|---:|---:|---:|---:|
| Independent | 17,410 | 200 | 0 / 0 | 382.54 ms |
| Request-local prefix | 11,330 | 200 suffix + 10 prefill | 10 / 200 | 327.65 ms |
| Exact-length batches, 2,048-token budget | 17,410 | 60 | 0 / 0 | 262.82 ms |

Prefix reuse removed 6,080 submitted positions (34.9%); batching reduced independent backbone invocations from 200 to 60. These local measurements used CPU fixtures. The subsequently supplied T4 host enabled full Kev qualification; Apple hardware remains unavailable.

### Full Kev-4B/T4 qualification

On 2026-10-07 the supplied host's NVIDIA driver was rebuilt for its new kernel, with user approval and no reboot. The optional CUDA/ONNX CLI build and CPU/CUDA Kev/Clef, prefix/chunk and batch fixture tests passed. Independent full Kev-4B CUDA FP16 passes against the earlier pinned CPU FP32 and CUDA FP16 binaries on six regression cases/19 questions.

**Do not enable the optimized FP16 paths for this variant.** Prefix reuse changes probabilities by up to 0.002896 against independent forwards and 0.003143 against CPU FP32, exceeding both gates. Batching stays within the external 1e-3 threshold but its paired delta 0.000372 exceeds 1e-4. Both retain complete argmax agreement. FP32 weights exceed this T4's memory and cannot serve as a GPU fallback. Full Kev CPU FP32 prefix qualification subsequently passes the unchanged external/paired gates on those six cases, with paired delta 3.576e-7 and complete argmax agreement; this does not qualify CUDA FP16.

Five-question mixed-type measurements show long-state prefix latency falling from 23.81 s to 5.63 s, but that path is unqualified. Batching is approximately neutral for short states and slightly slower for long states. A speedup cannot compensate for the failed probability checks. [Full results and retained audit](kev-optimization-validation.md) record execution identities, unchanged independent vectors and the rejection evidence.

### Measured kernel profiles and device propagation

- O09/O11/O12 investigation: an ignored offline precision diagnostic measures layer drift and compares first input projections with values held fixed. Native CUDA FP16 split projections differ by up to 0.0078125; other operations also contribute. The test reproduces actual backend probabilities but is not a qualification certificate. [Trace identity](verification/kev-t4-20261007/precision-trace-identity.json) pins the diagnostic source overlay, binary and runtime.
- An experimental `HUNCHO_PROJECTION_CHUNK_ROWS` profile applies fixed-size backbone projections, with padding confined to each linear call. Default zero delegates to the native arithmetic. `HUNCHO_ATTENTION_FP32` separately computes dense attention in FP32 without expanding all stored weights to FP32. Both profiles are default-disabled, reject changes with retained live prefixes and record execution metadata. Changed arithmetic adds work/memory and requires complete labeled startup qualification; no speedup is inferred.
- The full Kev/T4 64-row-only profile passes the unchanged six-case independent external gate (delta 0.000456929) and batch paired gate (zero delta). Prefix paired delta falls to 0.000391185 but still fails 1e-4. The separate 1,536-case labeled independent check rejects this profile: maximum probability delta 0.012506247, 85 cases above 1e-3, complete argmax agreement and ECE drift 0.0007783044. The six-case pass cannot override that rejection. [Held-out report](verification/kev-t4-20261007/stage3-held-out.json) retains every case. Adding FP32 attention also leaves prefix paired drift above the gate (0.00021833181); independent and batch pass only the six numerical cases. [Combined-profile audit](verification/kev-t4-20261007/stage4-numerical-audit/identity.json) remains diagnostic and unqualified.
- O16, loader portion: ModernBERT and F3 Qwen now have explicit device loaders. Their default/auto CLI paths remain CPU; explicit `HUNCHO_DEVICE=cuda` / `cuda:N` selects CUDA and fails on unavailability. CPU staging converts source tensors before transfer and merges Qwen LoRA on CPU. ModernBERT remains FP32; F3 vocabulary weight/bias and backbone share a device, and unsupported GPU BF16 is rejected. `device_path` metadata forces complete labeled startup qualification. Local CPU/gate tests and actual T4 ModernBERT/F3 CUDA fixture execution pass, including source-BF16 staging, candidate weights on device and batch row preservation. [CUDA fixture log](verification/kev-t4-20261007/stage5-cuda-fixtures.log) and [receipt identity](verification/kev-t4-20261007/stage345-receipt-identity.json) pin the execution. Full released Laya/Nimble qualification remains open; fixture success does not authorize those checkpoints.
- `scripts/qualify_kev_runtime.py` retains binary/script/package/golden identities, actual device/dtype/profile, complete outcome coverage and nonvacuous optimized work. It independently checks the unchanged gates and never marks numerical-only evidence qualified. Package/binary/golden mutation during a run is rejected. Fourteen Python data/qualification tests pass, including rejection of vacuous/substituted preparation and relaxed paired gates. This is reproducible validation tooling, not persisted release certification or proof of dataset separation.

- O03, measured CPU thread portion: full Kev CPU FP32 passes the unchanged six-case gate with both four and sixteen Rayon/Candle threads (delta 9.778887e-9, complete argmax agreement, ECE drift zero). Five warm distinct five-question requests average 39.382 s at four threads and 35.476 s at sixteen, about 9.9% lower latency in this pilot. Each mode submits 25 forwards/845 positions, with no reuse or batching. The affinity is the same sixteen physical cores on NUMA node 1. This is one small ordered comparison, with other NUMA work and no frequency isolation; no universal thread default, production p99 or held-out calibration claim follows. [CPU thread pilot](verification/kev-t4-20261008/stage5-cpu-thread-pilot.json) retains scope, complete timings and source/binary/golden/script identities. Optional BLAS/platform acceleration, wider workloads and architecture-specific tuning remain open.

Landing sites: [fixed projections](../crates/huncho-backend/src/qwen3_5.rs#L298), [attention compute profile](../crates/huncho-backend/src/qwen3_5.rs#L1133), [ModernBERT device loader](../crates/huncho-backend/src/candle.rs#L73), [F3 device loader](../crates/huncho-backend/src/qwen3_5.rs#L1178), [startup qualification](../crates/huncho-cli/src/serve.rs#L185).

Continuation checkpoint (2026-10-08): lab connectivity is restored and all stage 3–5 receipts have been collected. The 64-row labeled profile and both prefix kernel profiles are rejected; no altered arithmetic or refitted temperature is enabled. The full Kev CPU thread pilot has completed on a fixed node-1 affinity: both settings pass numerical gates and sixteen threads lower measured mean latency by about 9.9%. The isolated stage-6 source snapshot has built bounded preparation and passed unchanged-profile T4 numerical qualification (external delta 4.656613e-10, paired delta zero, complete argmax agreement; 19 prepared questions/forwards). The HTTP comparison passes equality/work checks with no observed speedup; the 1,536-case labeled GPU gate was interrupted at user request and supplies no acceptance; full roadmap implementation and released-model qualification remain in progress.

CLI lifecycle logs now use stderr so benchmark/conformance JSON remains parseable. A process-level test covers both commands with info logging enabled.

### Labeled fitting/evaluation preparation

The user delegated dataset selection. `scripts/prepare_kev_calibration.py` pins SST-2 choice, BoolQ noul and Yelp five-level score datasets, samples 256 fitting and 512 evaluation cases per type, and excludes evaluation inputs from fitting. It retains source file hashes, split/row IDs and label distributions. Hidden/malformed labels are rejected. `scripts/collect_kev_calibration.py` collects independent raw logits with separate fitting rows and labeled golden vectors; target indices follow the trained prompt's candidate order. Neither script updates a model temperature.

The corrected independent capture completed all 768 fitting/1,536 evaluation rows, recording kernel/boot/script identity, explicit candidate labels and checkpointed expected vectors. The external reboot and Noul-index correction are retained in the audit; old target indices are excluded. The isolated type/cardinality refit slightly lowers held-out NLL/Brier but increases aggregate ECE from 0.015302 to 0.016950; paired 95% bootstrap intervals include zero for every task/metric. It is not promoted and the upstream temperature remains 2.40605. [Dataset provenance](verification/kev-t4-20261007/dataset-provenance.json) and [held-out statistics](verification/kev-t4-20261007/held-out-calibration-v2.json) are retained without copying corpus text into the repository. These are application-specific checks, with unknown overlap against model training; offline FP64 analysis cannot replace runtime conformance.

## Subsequent increments

| Stage | Candidates | Work and acceptance |
|---|---|---|
| Scheduler and reuse | remaining O04/O12, O14, O19, O25; remaining O06 | Build cross-request collation on the native batch interfaces, add queue wait/padding policies and measure the implemented preprocessing overlap. Preserve typed/padding masks and whole-request F5 isolation. Extend in-flight sharing with workload measurements and tenant policies; add persistent-prefix caches. Replica pools and chunk admission remain workload-specific alternatives. |
| Runtime kernels and precision | remaining O03, O08, remaining O09/O10/O13/O16, O15, O17, O18 | Complete kernel-profile and full-model F1/F3 GPU qualification; profile CPU threads/matmul; wire ONNX EP/I/O; vectorize joint-head launches; integrate compatible fused kernels. Then qualify graph replay and quantized artifacts/kernels. Keep the lightweight default and reject variants that fail gates. Kev CUDA loading existed at the implementation baseline; newly wired F1/F3 loaders are not released-checkpoint acceptance. |
| Residency and additional platforms | O20, O21, remaining O22, O23, O24, O26, O27 | Add lazy residency/eviction and isolated shared-base adapters; integrate optional llama.cpp/vLLM, Metal/MLX, distributed execution and browser artifacts. Pin runtime/artifact identities and qualify real hardware. These need platform conversion and trained readouts, not enum variants or decode endpoints. |

All 27 candidates are accounted for; the full roadmap is not implemented. Partial items remain open in their later stage. Release qualification must also record actual artifact/adapter, prompt/tokenizer, execution/head/accumulator precision, quantization layout, device/runtime/kernel identity and disjoint fitting/evaluation provenance. Current backend:dtype lookup alone is insufficient for that full matrix. Persisted qualification records remain open. The user subsequently authorized a commit for each area. Apple Silicon is skipped for now, and further checks against actual GPUs are deferred; CPU implementation and validation continue.

## Running the new checks

```sh
cargo test --offline --workspace
cargo test --offline --workspace --features huncho-cli/clef

HUNCHO_PROMPT_CACHE_BYTES=1048576 huncho bench --questions 20 --repeat-inputs --json
huncho bench --model /path/to/package --questions 20 --workload mixed --concurrency 4 --json
huncho bench --model /path/to/f3-package --questions 5 --reference-readout --json
huncho conform --model /path/to/f3-package --golden /path/to/pinned-golden.json --json
huncho conform --model /path/to/f3-package --golden /path/to/pinned-golden.json --reference-readout --json

huncho conform --model /path/to/kev-package --golden /path/to/pinned-golden.json --prepare-all --json
huncho conform --model /path/to/kev-package --golden /path/to/pinned-golden.json --prefix-cache --json
huncho bench --model /path/to/kev-package --questions 20 --workload mixed --prefix-cache --json
huncho bench --model /path/to/package --questions 20 --max-batch-tokens 2048 --json

huncho serve --model /path/to/kev-package --max-prepared-per-model 1 --qualification-golden 'REGISTERED_MODEL_NAME=/path/to/pinned-golden.json'
huncho serve --model /path/to/kev-package --prefix-cache --qualification-golden 'REGISTERED_MODEL_NAME=/path/to/pinned-golden.json'
huncho serve --model /path/to/f3-package --candidate-readout --qualification-golden 'REGISTERED_MODEL_NAME=/path/to/pinned-golden.json'
huncho serve --model /path/to/package --max-batch-tokens 2048 --qualification-golden 'REGISTERED_MODEL_NAME=/path/to/pinned-golden-with-equal-length-questions.json'
```

Do not overwrite golden vectors with optimized outputs. Benchmark measurements are warm closed-loop in-process evaluations, not HTTP open-loop saturation or cold startup. Use `--repeat-inputs` to measure a deliberate reuse workload; the default varies state and question instructions.

Fitting data retains the existing rows/targets format. Add aligned types to fit covered strata:

```json
{"rows": [[2.0, 0.0], [1.0, 2.0, 0.0]], "targets": [0, 1], "qtypes": ["choice", "score"]}
```

In each golden case, optional `targets` maps every question ID to its observed candidate label. Once any case is labeled, every question in the suite must be labeled. Use score-level labels and the existing `yes`/`no` conformance labels for noul. Keep fitting and final evaluation sets separate. A scalar-only refit clears inherited type/bucket overrides; absent strata use that variant's fitted scalar, not a stale DEFAULT override. `Refit` still means fitted, not a persisted release qualification.

## Cross-request scheduling (2026-10-08)

O12 now includes opt-in cross-request collation: `--batch-max-requests 2..64`,
`--batch-wait-ms` (default 2), and a required `--max-batch-tokens` budget. One
worker per immutable engine collects bounded prepared requests, buckets their
questions by exact length, and invokes the existing native tensor batches. No
padding, prompt concatenation, head replacement or temperature change occurs.
Original question IDs, option order, typed answers, extensions and logical usage
are scattered back to each caller. F5 stays whole-request inference; prefix
reuse remains mutually exclusive. Unsupported models retain independent serving.

Admission still bounds active plus waiting callers. Preparation has at least as
many slots as the requested collation size, subject to admission capacity. The
collection deadline starts when the first ready packet is enqueued, so draining
an older queue does not add another complete wait window. This deadline bounds
collection delay, not total queue latency or RSS. A batch may contain up to 64
rows, with an oversized singleton executed independently. A backend error fails
the affected collated group. Running cancellation keeps admission until physical
work ends; canceled queued packets are discarded. Model removal closes the
worker's queue without retaining a sender/engine reference cycle.

`huncho conform --batch-max-requests N --max-batch-tokens B` now measures this
actual path against both unchanged goldens and independent forwards. It bypasses
all retention and requires a tensor batch containing rows from multiple requests.
`huncho serve` requires this separate, nonvacuous startup gate when collation is
enabled. `huncho_cross_request_batch_count` reports physical mixed batches; token
counters count their positions once. Library callers can use opaque preparations
through `Engine::eval_prepared_batch_with_stats` and must run the same gate.

CPU fp32/fp16 tiny-Kev upstream vectors pass external and paired probability
checks with request isolation. Serving tests cover distinct states, duplicate
question IDs, mixed extension settings, singleton deadlines, auth, overload,
single-worker health, queued cancellation and admission retention after running
cancellation. Default and Clef-enabled workspace tests pass. This is fixture
qualification; full Kev-4B cross-request acceptance and production throughput
are still unmeasured. No actual GPU checks were run for this increment.

```sh
huncho conform --model /path/to/package --golden /path/to/pinned-golden.json \
  --batch-max-requests 8 --max-batch-tokens 4096 --json
huncho serve --model /path/to/package --batch-max-requests 8 \
  --batch-wait-ms 2 --max-batch-tokens 4096 \
  --qualification-golden 'REGISTERED_MODEL_NAME=/path/to/pinned-golden.json'
```

Padding, mixed-length tensor masks, persistent cached-branch collation and chunk
admission remain separate increments. This implements decision-forward dynamic
batching; there is no generation/decode scheduler.

## Persisted execution evidence (2026-10-08)

The optional CLI `qualification` feature adds `conform --write-qualification
PATH` and `serve --qualification-record MODEL=PATH`. Normal loads and the
lightweight default build do not hash model weights or acquire the optional
SHA-256 dependency. Records are immutable output files: creation refuses an
existing path, including a failed or diagnostic record.

The recorder observes selected input bytes before loading and rechecks them
after loading/conformance. Native Qwen captures every base shard plus its separate
adapter, config, trained head and tokenizer. Clef captures base and joint-head
inputs; ModernBERT captures the selected weights/config/tokenizer. ONNX captures
its graph directory tree conservatively to cover external tensor data. Store
ONNX records outside that tree. Directory symlinks in that tree are rejected;
ordinary Hub file symlinks are hashed by content. Artifacts added/removed between
snapshots or changed bytes invalidate the record.

Execution identity includes the running executable, resolved manifest and
calibration, actual backend/dtype/device/kernel metadata, inference options and
collation size. Linux also records executable library mappings with matching
file inodes, CPU model/features, kernel, affinity, available parallelism and an
allowlist of thread/device/runtime environment settings. Request text, fitting
rows and corpus text are not copied into the receipt. Unchanged F1/F2/F4/F5
reference-readout flags and response extension settings are normalized because
they do not alter arithmetic. Numerical-only evidence is explicitly distinct
from complete observed-outcome gates; numerical passes cannot be relabeled as
outcome acceptance. Strict metric limits are checked independently of a stored
report's `passed` field.

Serving with a record still requires a pinned golden binding and executes fresh
conformance before listening. It then verifies current artifacts, executable,
runtime, options, golden bytes and required outcome coverage against the record.
A record never bypasses inference gates. Refit or changed-kernel serving retains
its labeled-suite requirement. This is persisted, execution-bound audit evidence,
not a signed release certificate or cached serving authorization. It cannot
prove fitting/evaluation separation, training overlap, or universal calibration;
GPU device hardware is not independently inventoried, and non-Linux runtime
library identity is incomplete. Those stronger certification requirements remain
open, with actual GPU checks deferred by the user.

Real CPU tiny-Kev process tests cover receipt creation, matching-record startup,
refusal to overwrite, changed preparation options, modified golden/config bytes,
and attempts to substitute numerical-only evidence as an outcome pass. No mock
backend can emit a real-artifact record. No actual GPU checks were run.

```sh
cargo build --release --features clef,qualification
HUNCHO_DEVICE=cpu huncho conform --model /path/to/package --dtype fp32 \
  --golden /path/to/pinned-golden.json --write-qualification /path/to/receipt.json --json
HUNCHO_DEVICE=cpu huncho serve --model /path/to/package --dtype fp32 \
  --qualification-golden 'MODEL_NAME=/path/to/pinned-golden.json' \
  --qualification-record 'MODEL_NAME=/path/to/receipt.json'
```

Use the same executable, thread/affinity settings and exact inference options
for recording and serving. Qualification hashes all selected inputs; its startup
I/O cost can be substantial for large checkpoints and is excluded from warm
inference benchmarks.

## Grouped Clef head projections (2026-10-08)

O13 now has an optional `HUNCHO_CLEF_VECTOR_HEAD=1` execution profile. Context and
lexical projections are grouped across all option rows; the question projection
runs once and gathers each question's vector for its options. Final option
normalization and residual MLP projections also run across the flattened rows.
The five affected projection calls per question become five calls per request.
Span means, evidence routing, joint-field attention, learned scales and original
question/option order remain unchanged. Variable-length span pooling and summary
reductions still use their reference operations.

This changes GEMM shapes and the lexical prior reduction, so it is default-off.
The loaded backend records `joint_head_execution=vectorized-v1`; CLI startup
requires complete labeled held-out goldens even when calibration metadata says
`fit`. Execution receipts include this mode. CPU fp32/fp16 tiny-Clef tests compare
both paths at three unchanged temperatures, require <=1e-4 probability delta and
matching argmax, preserve option labels/usage, and cover context errors. Existing
upstream scalar golden checks remain unchanged. This is real head execution
coverage, not a full Cloudflare/Clef qualification or a measured production
speedup. No GPU checks were run; the profile remains unqualified there.

## Bounded native prefix snapshots (2026-10-08)

O19 now includes exact cross-request Kev prefix retention through
`--persistent-prefix-bytes B` with `--prefix-cache` (default budget zero).
Each native backend owns an immutable FIFO of at most sixteen snapshots,
charging token keys, conservative metadata and every retained KV/recurrent/
convolution tensor. Snapshots are copied into compact detached storage so narrow
views cannot retain larger temporary projections. Oversized prefixes and failed
retention copies remain usable fresh prefixes and bypass retention. Reducing a
budget evicts snapshots when the next retained-prefill operation executes.

Every hit mints a separate caller-owned handle; question forks continue from that
handle. Clear/eviction releases snapshot ownership while active handles remain
valid. Successful and failed continuations preserve stored parents. Exact token
keys and backend ownership isolate models/adapters/devices; the full hybrid state
and absolute token count are retained. Changing projection/attention profiles
with retained snapshots is rejected. No temperatures or token boundaries change.
This is bounded prefix-snapshot reuse, not paged attention, tenant sharing,
combined cached-branch batching or a global cross-model cache.

Conformance explicitly clears snapshots per case, performs a fresh warm
prefill/continuation, then exercises a real hit. It counts all warm and retained
model work, bypasses result/prompt retention, and checks independent probability
parity at the unchanged gate. A budget too small to retain any prefix cannot
qualify. Startup runs this separate check for enabled retention; execution
receipts bind the budget. `huncho_persistent_prefix_hits` separates successful
snapshot reuse from logical usage and submitted token counters.

CPU fp32/fp16 tiny-Kev tests pass upstream, independent and retained-prefix
probability checks. They cover full hybrid state, parent/branch isolation,
clearing with live handles, byte and sixteen-entry bounds, eviction, invalid
inputs, reduced budgets, profile changes and physical-work accounting. Full
Kev-4B acceptance/throughput for this increment is unmeasured; prior rejected T4
FP16 prefix results remain rejected. No actual GPU checks were run.

```sh
huncho conform --model /path/to/kev-package --golden /path/to/pinned-golden.json \
  --prefix-cache --persistent-prefix-bytes 67108864 --json
huncho serve --model /path/to/kev-package --prefix-cache \
  --persistent-prefix-bytes 67108864 \
  --qualification-golden 'MODEL_NAME=/path/to/pinned-golden.json'
huncho bench --model /path/to/kev-package --questions 5 --repeat-inputs \
  --prefix-cache --persistent-prefix-bytes 67108864 --json
```

The budget controls retained snapshots, not active forks, allocator RSS or
quadratic attention temporaries. In library use, stopping prefix retention does
not automatically unload snapshots created by earlier requests; call
`Engine::clear_prefix_cache` or unload the engine to release them. No request
text is retained, but token IDs and model state remain in memory until eviction,
clear or unload.

## ONNX readout and allocation increment (2026-10-08)

O08 now includes an explicit F1 graph-side Gather contract and optional bounded
CPU output-buffer reuse through real ORT I/O binding. `OnnxBackend::load` retains
the previous defaults; `load_with_options` opts into `huncho_readout_positions`
and `huncho_features`, or a single exact-shape CPU allocation. The exporter
`scripts/compact_onnx_readout.py` creates a separate artifact, preserves input
semantics and prunes unrelated output branches. It does not edit manifests or
claim calibration acceptance.

Validation uses ONNX Runtime 1.28.0 on CPU: repeated/reordered/empty selections
preserve float bits, size caps and shape replacement hold, invalid input does not
corrupt the next readout, and compact/mock probability goldens have zero delta
and complete argmax agreement with and without retained output storage. Python
tests cover schema rejection, graph pruning and create-new behavior. CPU fixture
coverage establishes neither real Laya calibration nor production throughput.
Both profiles default off and require fresh startup numerical conformance; their
configuration is captured by qualification receipts. GPU checks remain deferred
at the user's request, and Apple Silicon work is skipped for now.

Remaining O08 work includes explicit execution-provider selection, native graph
batching and device-resident heads. This CPU buffer path still copies selected
features into the core host tensor. As documented by
[ORT I/O binding](https://docs.rs/ort/2.0.0-rc.13/ort/session/struct.IoBinding.html),
binding alone does not establish an acceleration benefit for CPU-to-GPU-to-CPU
pipelines; no such benefit is claimed here.

## Explicit ONNX providers and thread profiles (2026-10-08)

O08 now has CPU/default and explicit CUDA selection through
`HUNCHO_ONNX_EP=cpu|cuda:N`. CUDA requires the separate `onnx-cuda` Cargo feature,
fails provider registration rather than silently substituting CPU, disables
TF32, and rejects graphs needing CPU fallback. CPU defaults and the separate
Candle CUDA feature remain unchanged. This is a strict placement profile; graphs
using CPU-only operators may be rejected. Outputs still use host feature rows;
there is no device-resident trained head or full-model CUDA acceptance.

The O03 thread-control portion adds `HUNCHO_ONNX_THREADS=0..256` (zero preserves
ORT defaults). Positive values and CUDA provider choice are recorded execution
profiles and require fresh labeled conformance before HTTP serving, even with
an existing fitted temperature. Receipts also bind these environment settings.
CPU tests validate parser errors, feature rejection before attempting a CUDA
load, thread-profile output parity and startup gates. The CUDA path is compiled
only; actual GPU checks and provider/model qualification remain deferred. No
latency or cost improvement is inferred from compile and fixture checks.

## Native ONNX batching increment (2026-10-08)

O08/O12 now connect verified dynamic F1 ONNX graphs to the existing per-request
and cross-request bounded scheduler. `HUNCHO_ONNX_NATIVE_BATCH=1` requires
int64 dynamic `[batch,seq]` known encoder inputs and float32 dynamic
`last_hidden_state[batch,seq,hidden]`. Optional position IDs are explicit
ascending rows. Unknown inputs, fixed batch dimensions, cache handles and
compact-readout combinations fail closed. Native batches accept at most 64
equal-length rows with no padding; readouts scatter by request and position,
including repeated and empty selections. Full graph outputs remain host
tensors, and optional CPU output binding reuses an allocation for the complete
batch shape.

CPU tests verify exact row bits against independent forward calls, shape and
buffer bounds, failure recovery, actual per-request/cross-request batching,
unchanged probability goldens, nonvacuous physical-work accounting and zero
retained result/prompt hits during conformance. Profile selection requires
labeled startup qualification; scheduling adds the independent parity gate.
These are deterministic ONNX fixtures rather than trained Laya acceptance or
a production speed measurement. CUDA batching has not run on hardware. O08
still has device-resident head work open, and graph-side compact selections and
native batches intentionally use separate contracts in this increment.

## Buffered CPU delta-rule increment (2026-10-08)

O17 now includes a real optional CPU recurrence implementation in
`crates/huncho-backend/src/delta_cpu.rs`. It reads tensor storage through
validated strides, preserves FP32 state and ascending key reduction, casts
whole inputs once, and uses two matrix passes per head/token: decay plus memory
projection, then state update plus output projection. This removes the original
per-token tensor allocations without FMA substitution, parallel prefix changes
or reduced-precision state. A caller's initial state is copied rather than
mutated; retained forks and snapshots remain isolated.

`HUNCHO_CPU_DELTA_RULE=1` / `Qwen3_5Backend::with_cpu_delta_rule(true)` is default
off and CPU-only for F2/F3; Clef exposes the same CPU profile for F5. Kernel changes are blocked while any prefix state
is retained, and metadata/receipts bind the profile. Fresh complete labeled
startup conformance remains mandatory. Unit tests match original-loop output
and final-state float bits across FP32/FP16 inputs, noncontiguous views, nonzero
state and split continuation. Kev fixture independent logits match bits;
upstream probabilities, batches, forks and persistent prefixes pass existing
gates. F3 compact/full vocabulary fixture probabilities also pass, and F5 joint
logits retain their float bits. No full Kev/Nimble/Clef or GPU profile is promoted.

The local alternating release microbenchmark averages 15.057 ms for the tensor
loop and 1.445 ms for buffered recurrence at `[1,8,256,32,32]`, about 10 times
faster for that kernel, with bit-equal output/state. This is one CPU kernel
measurement, excluding the rest of model inference, without frequency isolation.
The [audit](verification/delta-cpu-20261008/recurrence-microbenchmark.json) retains
all timings, CPU/affinity/thread environment, compiler, source and binary
identities, with released-model qualification explicitly false. Remaining O17
work includes fused convolution, GPU/chunk delta kernels and their qualification.

## Compact convolution-tail retention (2026-10-08)

O02/O19 now fix a verified backing-storage retention issue in request-local
Qwen caches. `Tensor::copy()` on the narrow convolution tail copied the full
projection backing allocation and preserved its offset/strides. With a 64-token
CPU fixture it retained 2,048 elements for a visible 96-element tail. The cache
now uses `force_contiguous().detach()` to retain only the visible tail in compact
storage. This changes storage layout without changing convolution values,
recurrent arithmetic or temperatures. Persistent snapshots already made their
own compact copies; active handles now have the same compact tail property.

The regression first fails on the old allocation (2,048 versus 96), then checks
the compact allocation, zero offset and contiguous layout for fp32/fp16. Parent
release leaves forks valid, and two-candidate continuation probabilities meet
the unchanged 1e-4 paired gate at three temperatures. Existing native batch,
prefix, immutable snapshot and original upstream checks cover the complete
readout path. The generic storage-copy change has only CPU validation here;
actual GPU checks remain deferred.

## CPU runtime qualification runner (2026-10-08)

`scripts/qualify_kev_runtime.py` now accepts `--cpu-delta-rule`,
`--persistent-prefix-bytes` and `--batch-max-requests`. It verifies the exact
reported kernel metadata, cache budget, effective preparation path and batch
size, plus actual retained-prefix hits or cross-request batch work. CPU-only
profiles and invalid mode/budget combinations are rejected before accessing a
model or device. Reports retain physical-work counters, CPU affinity and thread
environment. Fixed numerical and paired thresholds remain unchanged;
`--numerical-only` still cannot produce labeled acceptance. These changes make
the new implementations testable on the pinned Kev checkpoint without using a
GPU. Ten runner tests cover profile substitution, vacuous cache/collation work,
early rejection and unchanged subprocess evidence; the broader standard-library
Python suite also passes.

## Full Kev CPU numerical qualification (2026-10-08)

The isolated `3e79e41` source build uses `clef,qualification`, excludes GPU
features and explicitly selects CPU. On the supplied lab host with sixteen
threads and fixed sixteen-core node-1 affinity, buffered recurrence passes the
unchanged six-case/19-question FP32 reference: independent maximum probability
delta 9.778887e-9, complete argmax agreement and zero ECE drift. A 256 MiB
persistent-prefix budget exercises five actual retained hits; paired and
external delta are 3.5762787e-7. Native per-request batches exercise three actual
batch calls and have zero paired delta. No response or prepared-prompt cache
hits occur during qualification. The upstream temperature stays 2.40605.

The [execution identity](verification/kev-cpu-20261008/numerical-cpu-buffered/identity.json)
and [CPU-only job](verification/kev-cpu-20261008/huncho-cpu-roadmap-20261008.sh)
retain binary/source/golden/package hashes, affinity, complete reports and work
counts. These runs lack observed labels and explicitly record `qualified=false`;
they do not establish held-out calibration, cross-request scheduling acceptance,
latency improvement or a GPU result. Elapsed times include model loading and
paired/warm-up work, so they cannot compare optimization speed directly.

## Durable CPU Kev quantization increment (2026-10-08)

O18 now has actual CPU Q8_0/Q4_0 packed projection artifacts and Candle packed
matmul kernels, with FP32 LoRA merge, activations, state, convolution, embeddings
and trained pointer head. Huncho records the full mixed precision/kernel profile;
the loader validates embedded configuration/layout and cannot silently switch
to environment-selected dense kernels. `quantize` creates a new package,
rechecks source hashes and sets calibration pending. `capture-logits` collects
independent fitting logits offline, including pending variants, with explicit
observed targets and artifact/runtime evidence. It does not export goldens or
serve unqualified probabilities. Existing `calibrate` refits the exact variant.
HTTP serving requires an explicit backend:dtype refit and fresh complete
labeled conformance at unchanged thresholds; inherited/source fit entries fail.

CPU tests run both real packed kernels on the tiny Kev fixture and verify
durable reload, changed logits, immutable forks/persistent hits, same-profile
batch parity, profile/layout/truncation rejection, no overwrite, byte-identical
source files, candidate/target order, offline dry fitting and serving rejection.
Packed input widths below 32 are locally zero-padded, including the fixture;
no padding enters attention or token usage. Conversion/initial loading still
materialize dense weights temporarily; packed projection payload counts are
not peak RSS or a speed claim. Released Kev Q8_0/Q4_0 acceptance, direct packed
model construction, other families/platforms and additional quantization layouts
remain open. CONV-03 is partial: conversion supplies no automatic calibration
approval. See [the workflow and limits](quantization.md).

## Buffered CPU causal convolution (2026-10-08)

O17 now includes `crates/huncho-backend/src/conv_cpu.rs`, a depthwise CPU
convolution that reads strided storage, casts complete inputs once and writes
one FP32 output buffer. It preserves ascending tap multiplication/addition and
the original FP32 SiLU-before-activation-cast order. The original implementation
allocated full contribution, padding and addition tensors for each tap. This
does not change token positions, retained convolution history or temperatures.

`HUNCHO_CPU_CAUSAL_CONV=1` is a default-off CPU profile for native F2/F3/F5,
including packed Kev. Changing it with retained prefixes fails; metadata and
receipts bind it, and fresh complete labeled startup conformance is mandatory.
CPU unit tests match original raw float bits for fp32/fp16 inputs, noncontiguous
weights/input views and short sequences. Full F2/F5 independent fixture logits
also match bits; F3 candidate projections and F2 cache/batch/fork paths pass
existing probability gates. Tests cover convolution alone, recurrence alone
and both together; packed Q8_0/Q4_0 cache/batch tests exercise both CPU kernels.
The full-model tests caught an omitted SiLU during development and now verify
the corrected activation order.

The [alternating CPU kernel audit](verification/conv-cpu-20261008/convolution-microbenchmark.json)
retains six samples of twenty calls at `[1,128,512]`, kernel width four, source
and binary hashes, compiler, CPU/affinity and thread environment. It excludes
SiLU, casting, model inference and serving, and has no frequency or heterogeneous
core isolation. Released checkpoint/calibration qualification remains false;
the measurements do not establish a full-model speed or cost improvement.

## Bounded CPU prefix chunking (2026-10-08)

O25 now has an optional CPU Kev prefix-compute increment:
`HUNCHO_PREFILL_CHUNK_TOKENS=0..4096` (zero keeps the original single call).
Each chunk advances dense-attention KV, FP32 recurrence and compact convolution
history together. No pointer-head readout is computed during prefill, and a
caller-owned handle is published only after the complete prefix succeeds.
Persistent snapshots retain the completed immutable prefix. Kernel/chunk
changes with retained state fail, and F3/GPU chunk profiles are rejected.

The query length of dense prefix attention is bounded by the chunk size, while
keys cover the complete past. This bounds that attention score allocation to
approximately chunk-size times total-prefix length instead of prefix length
squared; full KV/state storage remains. Multiple appends and smaller matmuls
can increase latency. This increment holds the existing engine/backend and
serving execution locks throughout the prefill; fair interleaving, pipelining
and cancellation between chunks remain open.

Core `PrefillWork` now records each native attempt and token slice, including
work preceding an error; retained snapshot hits submit zero physical work.
`EvalStats.prefill_calls` counts actual native calls and `chunked_prefills`
counts prefixes actually split across calls. Logical response usage stays
unchanged. Both HTTP execution paths expose corresponding counters. Metadata,
receipts and the qualification runner bind the chunk budget. Serving requires
prefix reuse and fresh complete labeled qualification; conformance rejects
suites whose prefix never actually exceeds the selected chunk size.

CPU tests cover fp32/fp16, one-token through full-size chunks, short convolution
history, original independent calibrated vectors, fork/snapshot isolation,
actual misses/hits/work counts, early invalid-token rejection and retained-state
profile guards. The engine check keeps unchanged upstream goldens, preserves
logical usage, verifies lower physical token work and rejects a vacuous
4,096-token chunk profile. Released-model chunk acceptance and actual peak RSS
or latency measurements remain outstanding; no GPU checks are performed.

## Resumable CPU prefill scheduling (2026-10-08)

O25 now has an opt-in HTTP scheduler, beyond the earlier memory-only chunks.
`--cooperative-prefill --prefix-cache` (or `HUNCHO_COOPERATIVE_PREFILL=1`)
requires CPU Kev and a positive `HUNCHO_PREFILL_CHUNK_TOKENS`. An owned engine
cursor executes one prefix chunk or one complete question per step. The
backend mutex and serving execution lease are released between steps. FIFO
execution waiters can run before the long request's next chunk. This is useful
for head-of-line blocking; it adds scheduling overhead and is not a measured
isolated-request speedup or continuous tensor batch.

Partial KV/recurrent/conv caches remain private to their exact execution
context. Partial handles cannot be forked or read out. A chunk commits cache
state only after native success; immutable prefix snapshots are retained only
after the complete prefix. Dropping a cursor releases partial/completed state.
HTTP cancellation preserves admission and execution capacity until the current
kernel finishes, then submits no further chunks. Coalesced jobs continue for
remaining callers and cancel when all callers disconnect. Failed native
attempts remain in physical-work metrics.

The initial scheduler supports one CPU context, no tensor collation, and at
most 62 queued requests: 63 parents plus one question branch fit the existing
64-handle bound. Prepared packets/native state are bounded by admitted request
count and model context, not an exact live-cache byte budget. Dense question
suffixes execute as one call; arbitrary bidirectional F1 or joint F5 chunking,
cached-branch batching, paging and GPU execution remain open.

`huncho conform --prefix-cache --cooperative-prefill` actually alternates
prefix steps from distinct golden requests, bypasses response/prompt retention,
then checks the unchanged external vectors, independent-forward probability
delta/argmax, and observed-outcome ECE/Brier. A single request or chunk size
that never splits a prefix cannot qualify. Serving performs that same fresh
labeled gate before binding. Receipts bind the scheduling option and coverage.
The `prefill_yields` and `prefill_interleaves` counters distinguish actual
progress/switches; evaluation/queue histograms record each scheduling step in
this mode. Logical usage, candidate order, trained heads and temperatures
remain unchanged.

CPU tests verify bit-identical raw logits against ordinary chunked prefill in
FP32/FP16, unchanged frozen fixture probabilities, actual round-robin coverage,
retained snapshot hits, context ownership, cache capacity, repeated partial
drop, single-worker health/fairness, overload, queued/running cancellation,
coalesced cancellation and failure cleanup. Process tests exercise the real
native CLI and reject vacuous startup qualification. Released Kev-4B CPU
qualification remains rejected/pending as recorded elsewhere; this scheduler
is default-disabled and no new GPU checks occurred.

## Explicit compiled CPU kernels (2026-10-08)

O03 now records optional compiled vector arithmetic across native Candle
backends. Candle 0.11's packed Q8_0/Q4_0 dot products use static AVX2/NEON/SIMD
cfg branches; host capability detection cannot turn them on in a portable
binary. A dependency-free build script records selected arithmetic features
under `cpu_kernel_build`. An isolated x86-64-v3 build enables the existing
AVX2 kernels without changing the ordinary build or adding a runtime dependency.

The shared changed-arithmetic startup/receipt gate requires fresh complete
labeled conformance for this profile even when a source manifest says `fit`.
The runner requires an exact requested identity and rejects omission or
substitution. Tests verify compilation identity and the existing unlabeled,
partial-label and probability-drift rejections. Instructions are compiled,
not dynamically dispatched: deploy only to compatible hosts. New CPU builds
and full-model fitting/qualification are staged separately; no faster kernel
is accepted based on instruction support alone. GPU checks remain deferred.


## Shared-weight CPU execution replicas (2026-10-08)

O14 now provides bounded independent CPU execution contexts instead of sending
all ordinary HTTP requests through one model mutex. `Backend::replica()` fails
by default; native CPU ModernBERT/Laya, Qwen F2/F3, packed Kev and Mock implement
it. Backbone/trained-head weights share Arc storage. Core manifest, tokenizer,
formatter, head and bounded exact prompt/result caches are shared within one
immutable group. Native cache handles and persistent snapshots start empty and
remain local. Kernel changes fail while another Qwen replica shares the model.
This shares one already-merged adapter's weights; it is not multi-LoRA or shared
base residency across distinct adapters.

`--replicas` / `HUNCHO_REPLICAS` defaults to one, bounds the pool at eight and
rejects non-CPU/unsupported paths and cross-request collation. Per-request native
batch/prefix and preparation options remain independently gated. Idle contexts
are leased through complete blocking jobs; cancellation/unwind releases them
only when native work finishes. Admission covers N running plus the configured
waiting limit. Persistent retention budget is divided across contexts instead
of silently multiplying its per-model bound; exact caches and attention setup
storage remain shared.

Startup checks every actual context concurrently on the complete unchanged
labeled suite. Partial/unlabeled data or drift in any replica rejects the pool.
Tests prove shared loaded storage, unchanged independent fp32/fp16/packed Q8/Q4
float bits under concurrent native fixture calls, nontransferable handles,
fresh snapshots, group-bound prepared packets, exact cache sharing, atomic
unsupported construction, overlapping HTTP jobs, overload and running/queued
cancellation ownership. Existing default and native prefix/batch checks pass.

Released Kev/Laya/Nimble pool qualification and workload throughput/RSS/affinity
measurements remain open. Each concurrent job retains its own activations and
state and can contend for cores/memory bandwidth. ONNX replica loading and
cross-request worker scaling remain open;
GPU and Apple checks are not performed.


## Direct packed model construction (2026-10-08)

The remaining O18 loading-memory increment now constructs Q8_0/Q4_0 backbone
projections directly from `QTensor` blocks. A private projection source supplies
validated output rows, original input widths, packed widths and FP32 biases
during the normal layer construction. Dense embeddings, normalization,
convolution, recurrence scalars and the trained pointer head use the original
builders. No dense projection model, placeholder matrix or environment-selected
dequantized kernel exists on the production packed load path. The GGUF layouts,
projection kernel, prompt, temperatures and dtype identity stay unchanged.

A test-only reference retains the previous dequantize/build/replace loader;
all fixture rows match raw float bits and calibrated probabilities at three
temperatures for both packed schemes. Tests also prove that the loaded dense
map excludes every projection, validate packed schema/rows/input widths and
check retained packed payloads. Existing native batch/fork/replica fixtures,
conversion/source immutability and offline fit/serving rejection checks pass.
Conversion still materializes merged FP32 weights; dense non-projection file
copies remain. Full-model peak RSS/load latency and labeled acceptance are not
inferred from this structural memory improvement.


## Full Kev compiled-CPU numerical evidence (2026-10-08)

The isolated x86-64-v3 binary at source `ed6d7c6` runs CPU FP32 with both buffered
recurrence and convolution on the unchanged six-case/19-question suite. All
three fixed numerical gates pass: independent external delta 2.5629997e-6;
prefix external delta 3.8146973e-6 and paired delta 1.2516975e-6; batch paired
delta zero. Argmax agreement is complete throughout. Persistent-prefix
qualification observes five actual hits and five fresh prefills; batches
observe three actual native calls. Temperatures remain unchanged at 2.40605.

[Retained identity and complete reports](verification/kev-cpu-v3-20261008/numerical/identity.json)
bind the archive/binary/package/golden hashes and the exact compiled profile.
The native CPU tests and job script are retained, including an initial mistyped
test-target invocation and its corrected rerun. No GPU command executes.
These cases have no observed outcomes and remain `qualified=false`. Elapsed
times include loads, warm-up and paired work while other CPU jobs run; they are
not an isolated speed comparison, labeled acceptance or replica/chunk profile
qualification. Separate held-out CPU and packed fitting jobs remain active.


## Complete fitting-work accounting (2026-10-08)

The full-model CPU pilot exposed that offline `capture-logits` passed its
aggregate counters into an evaluator that resets them for every request. The
final fitting audit therefore described only the last record's work. Collection
now accumulates a fresh per-record `EvalStats`; logits, targets, question order,
input identity and temperature behavior are unchanged. The CLI process test
uses two distinct record IDs, three typed questions per record, both packed
schemes and unchanged fixture token rows to require all six forwards and every
submitted token position. Old in-flight fitting binaries retain their original
identity; their partial work counters cannot support aggregate speed claims.

## Grouped Clef option gathers and summaries (2026-10-08)

The next O13 increment adds `HUNCHO_CLEF_GROUPED_POOL=1` on CPU. One compact
lexical gather replaces one gather/cast per option; individual variable-length
span means keep their original reduction axes. Routed option summaries are
bucketed by exact cardinality, scored with BMM, softmaxed within each question
and scattered into original field order. No token/candidate padding, schema
splitting, vocabulary projection or cross-request F5 batch is introduced.
This composes with the existing vectorized projections, while each option
defaults off independently.

Changed BMM/reduction shapes are an arithmetic profile, recorded as
`joint_pool_execution=grouped-spans-summary-v1` in backend identity and receipts.
The shared startup gate requires complete observed outcomes and all unchanged
numerical/ECE checks, including when the source entry says `fit`. CPU fp32/fp16
tests cover the original upstream requests and a mixed schema with 1/2/3/4
options, repeated nonadjacent cardinality groups, multi-token descriptions,
both projection profiles, three unchanged temperatures, option labels and
logical usage. Probability agreement must remain <=1e-4 with matching argmax.
This does not qualify the released Clef checkpoint or demonstrate throughput.
The larger compact lexical temporary is a memory tradeoff; shared backbone and
joint-field attention are unchanged. Actual GPU checks remain deferred.

## CPU Clef execution replicas (2026-10-08)

O14 now also covers native F5. Clef shares its loaded backbone, joint head and
tokenizer through Arc storage; lexical tensors retain their shared storage.
The existing bounded pool leases each context for the complete schema and
requires fresh complete labeled qualification for every context. Backbone
kernel configuration is frozen while replicas share it; unchanged settings
remain harmless. Per-context option/projection profiles copy their original
metadata, and no mutable activation, schema or output is retained between jobs.

Native tests prove shared ownership and profile guards, then run three contexts
concurrently on upstream requests for CPU fp32/fp16 with both buffered kernels
and grouped head profiles. Independent raw logits and logical usage retain
their float bits. Unsupported per-question forward/cache calls still fail,
and F5 remains whole-request inference. This enables overlap without loading
duplicate model weights; activation memory, shared CPU thread contention and
full released-model calibration/throughput remain workload qualification tasks.
No GPU execution or Apple work is performed.

## Per-context CPU replica benchmarks (2026-10-08)

O14 workload measurement now has `bench --replicas 1..8`, independent of
`--concurrency`. Every real context is warmed; clients bind round-robin to
contexts and JSON retains per-context timed requests and physical work.
Global work is the sum of those context counters. Shared result/prompt caches
and the divided global prefix budget match the pool's storage semantics.
Inactive context counts and invalid budgets are rejected before timing.

Process checks cover uneven client/request distribution, all typed questions,
distinct inputs versus exact shared-cache hits, counter totals, bounds and
unsupported combinations. Native F5 process coverage submits one complete
schema on each of two actual contexts and requires two whole-request forwards.
The measurement includes wait on each statically assigned context; it does not
simulate HTTP admission/idle-context dispatch or certify probability quality.
Fixed-affinity released-model throughput/RSS and fresh per-context labeled
acceptance remain open. Actual GPU checks remain deferred.

## Immutable CPU bases across adapter models (2026-10-08)

O22 now includes optional `shared-base` builds and
`HUNCHO_BASE_CACHE_BYTES` (default zero). Safetensors Qwen/F2/F3 and CPU F5
loaders retain immutable unmerged CPU tensors. Different adapters share
untargeted storage, while every LoRA target is merged into a new tensor with
the original arithmetic. Trained heads, temperatures, model activations and
native prefix state remain independent. This extends the earlier shared-weight
replicas without claiming dynamic multi-LoRA inference.

Cache keys hash every base-shard byte plus dtype and LM-head inclusion; file
contents are checked again around hits/materialization. Identical relocated
shards share; source changes cannot select stale base tensors. Deterministic
enumeration rejects duplicate canonical tensor names. LRU retention is bounded
by charged payload/metadata and sixteen bases, with oversized bypass and explicit
library clear/stats. A mutex prevents duplicate concurrent miss loads and never
covers inference. SHA-256 startup I/O and the extra unmerged target residency
are real costs; the cache budget is not a live-model or peak-RSS bound.

`base_weight_cache=content-checked-cpu-v1` requires fresh numerical startup
conformance and is retained in execution receipts. Temperature/refit/changed
kernel/replica rules remain strict. Tests prove shared native embedding storage,
independent models/heads/handles, two distinct actual adapter merges with
bit-identical independent fp32/fp16 logits and probabilities, source mutations,
concurrent miss deduplication, eviction, bypass and pending CLI rejection.
The default build adds no active runtime dependency. Full released adapter-mix
RSS/startup qualification, lazy registry residency, on-the-fly multi-LoRA
scheduling and GPU residency remain open; actual GPU checks remain deferred.

## Packed Q8 CPU instruction-profile drift (2026-10-08)

A CPU-only diagnostic collects the first eight fitting rows using the same
unchanged pending Q8_0 artifact in alternating x86-64-v3/portable/portable/v3
order. Raw logits repeat exactly within each mode, but differ across the two
compiled packed-kernel profiles. At the original FP32 temperature 2.40605, an
offline FP64 softmax comparison gives maximum probability difference 0.0123914
and complete argmax agreement on these eight rows. This is diagnostic fitting
data, not unchanged held-out conformance, a quantized refit or acceptance.

[The retained audit](verification/kev-q8-cpu-pilot-20261008/q8-pilot/summary.json)
pins binaries, artifact/manifest/input hashes, CPU affinity and all raw rows;
[the reproducible analysis](verification/kev-q8-cpu-pilot-20261008/analyze.py)
keeps `qualified=false`. End-to-end collections take roughly 264–265 seconds
on v3 and 400–401 on portable, including hashing/load/audit while other jobs
contend for the same CPUs and bandwidth. Older collectors report only the
last record's work; `75a48c6` fixes later collectors. These runs establish no
isolated inference speedup or aggregate-work rate. No corpus text, GPU checks,
manifest changes or optimized golden replacements are retained. A portable
Q8 fit cannot silently qualify AVX2; each kernel profile needs its own full
fitting and unchanged labeled acceptance gates.

### O21: pinned CPU llama.cpp execution and new-package GGUF export

Implemented an optional native CPU backend for actual dense Qwen3.5 Kev pointer
and F3 vocabulary readouts, plus shared immutable weights with isolated replica
contexts. The pinned masked hidden-state staging API avoids ordinary embeddings
mode's all-token output behavior. There is one prefill and no sampler/decode
loop. F3 candidate selection remains a host gather; F2 still computes an unused
auxiliary vocabulary projection over selected marker/decision rows.

`export-llamacpp` performs the existing CPU FP32 LoRA merge and the pinned official
GGUF conversion, preserves trained readouts/tokenizer/contracts and publishes a
new pending package only after a native load check and input rechecks. It never
inherits calibration or regenerates golden probabilities. Fitting collection
and execution receipts now accept the new runtime; fresh complete observed-label
startup gates apply even to `fit` metadata. Quantized runtime artifacts require
exact profile metadata and an explicit refit. Existing thresholds are unchanged.

Real CPU fixture tests retain frozen upstream Kev raw/probability references
(raw delta `8.94e-8`, probability delta `2.98e-8`), cover 512-token masked readouts,
state resets, F3 projection and concurrent isolated replicas. CLI checks cover
exact dtype selection, artifact receipts, refusal of unlabeled serving and
per-context work. See [runtime, export and limits](llamacpp.md). Full released
Kev fitting/held-out acceptance and controlled runtime benchmarks remain open.
Native prefix fan-out, native batching, graph-side decision-only projection,
Q8/Q4 conversion and GPU execution are later increments. No Apple or actual GPU
checks were performed.

## CPU SiLU/multiply fusion (2026-10-08)

O02/O09 now include an optional fused CPU Qwen3.5 MLP gate for F2/F3 and
whole-request Clef/F5. One output buffer replaces the separate SiLU tensor and
multiply output. The public Candle typed operations retain FP32/FP16 rounding,
with strided views supported and no broadcast/approximate math. This removes
one allocation and intermediate write/read; projection cost and peak activation
memory remain workload dependent.

`HUNCHO_CPU_FUSED_GATE=1` defaults off, rejects non-CPU devices and cannot change
after shared replicas or retained native prefixes are created. Execution
metadata and receipts bind `cpu-fused-silu-mul-v1`; serving requires fresh
complete observed-label gates even for fitted packages. Scalar/strided kernel,
Kev raw/batch/prefix and whole-request Clef fixture comparisons preserve original
float bits at FP32/FP16. Concurrent replica and packed Kev paths retain their
profile semantics. FlashAttention, other fused ops, downstream vector math
providers and released-model calibration/performance remain open. No actual
GPU or Apple checks were performed.

The [CPU gate microbenchmark](verification/gate-cpu-20261008/microbenchmark.json)
retains twelve alternating pairs at 256 x 4096 elements on a local Core Ultra
7 CPU, with identical float bits. FP32's separate/fused median ratio is about
1.32; FP16's is about 0.88 (fusion is slower). This is a shape-specific kernel
pilot without frequency isolation, excludes all projection/head/serving work,
and supplies no peak-RSS or released-model acceptance. The flag stays off by
default; allocation savings alone do not imply lower latency for every dtype.

## Complete CPU FP32 held-out rejection and drift diagnosis (2026-10-08)

The CPU buffered recurrence run completed all 1,536 frozen labeled cases
and 205,297 submitted token positions. It fails the unchanged probability-delta
gate: maximum delta 0.02354765, with 57 cases above 0.001. Argmax agreement is
complete and ECE drift is 0.0007697381; those passing metrics do not override the
failed delta. Temperature 2.40605 and the original CUDA FP16 reference vectors
remain unchanged. This profile is rejected.

A separate CPU-only diagnostic replays the eight worst failing held-out cases
through the original and buffered CPU FP32 paths. All raw logits retain exact
float bits across the two modes, including the worst 0.02355 reference delta.
This establishes that the baseline CPU path already differs from the older
reference on those cases; it does not prove equality on the entire suite or
accept any newer optimization. Different pinned binaries are retained for the
full run and diagnostic. Diagnostic cases are forbidden for fitting, and the
older capture counters describe only the final row. No throughput claim,
temperature refit, threshold change, golden replacement or new GPU execution
is made. [Reports, identities and reproducible analysis](verification/kev-cpu-heldout-20261008/summary.json)
retain the rejection without corpus text. Cross-precision acceptance and
full-profile paired CPU qualification remain open.

## CPU ONNX replica sessions (2026-10-08)

O14 now supports optional independent CPU ORT sessions under `onnx-shared` and
`HUNCHO_ONNX_SHARED_INITIALIZERS=1`. Dense initializers use managed preallocated
CPU storage and one shared prepack container; every session owns its mutable
execution/output buffers. Source graph and external values are captured once
and hashed. Replicas use that immutable snapshot, including after source file
mutation or primary drop. External graph replacement temporarily copies data
inside ORT; transformed graph constants and per-session allocations can still
duplicate storage. No total-weight or peak-RSS reduction is assumed.

The opt-in profile validates flat supported FP32/FP64/INT32/INT64/BOOL graphs,
external ranges and directory bounds, unique names, dimensions and byte counts.
Unsupported graph/storage kinds and GPU providers fail explicitly; ordinary
ONNX loading remains unchanged. Fresh complete labeled gates apply to every
actual serving context, even with fitted source calibration. CPU fixtures prove
raw bits, unchanged probability goldens, row/batch isolation and concurrent
sessions; CLI checks exercise per-context benchmark work and refuse unlabeled
serving. Released Laya graph calibration, controlled pool throughput/RSS and
GPU replicas/device I/O remain open. See [operation and limits](operations.md#isolated-cpu-onnx-sessions-with-shared-initializers).
No Apple or actual GPU checks were performed.

## Pinned CPU llama.cpp Q8/Q4 export (2026-10-08)

O18/O21 now include new-package `export-llamacpp --dtype gguf-q8_0|gguf-q4_0`.
Conversion merges the original LoRA in CPU FP32, performs pinned official GGUF
mapping to an FP32 intermediate, then calls the same bundled native CPU
quantizer. Embeddings, vocabulary output, norms, convolution and matrices with
block-incompatible rows stay FP32. Explicit exact-name overrides prevent an
implicit FP16 fallback; eligible projections use the declared Q8_0/Q4_0 type.
Quantizer worker count, runtime/kernel identity, retained shapes, intermediate
hash and final artifact are retained in provenance. New destinations cannot
overwrite existing artifacts; successful export removes owned staging only.

Calibration starts pending without inherited temperatures, strata, evaluation
hashes or golden vectors. Serving requires an exact backend:dtype refit plus
fresh complete observed-label conformance at unchanged gates. Real CPU fixture
conversion and packed projection execution retain the original pointer head,
deterministic independent/replica behavior and source bytes. The fixture is
unrefitted and supplies no observed calibration acceptance. Released model
conversion/fitting/held-out acceptance, controlled CPU performance/RSS and
other quantizers remain open. See [conversion and limits](llamacpp.md#new-package-conversion).
No actual GPU or Apple checks were performed.

## Mandatory labeled startup gates for every real runtime (2026-10-08)

O10/calibration enforcement now covers unoptimized Candle, Clef and ONNX loads,
in addition to the already gated llama.cpp and changed arithmetic profiles.
Every real implementation reports `native_execution`. `serve` requires a
complete observed-label suite and fresh unchanged delta/argmax/ECE gates even
when the loaded source entry says `fit` and no optimization flag is enabled.
All actual replica contexts are still checked concurrently. Numerical-only
receipts and default-temperature fallback cannot authorize a new execution.
Offline mock demos and diagnostic bench/capture/conform remain available.

This closes the measured gap exposed by CPU Kev drift: fitted upstream metadata
could previously authorize an unoptimized backend whose worst held-out outputs
fail the reference gate. Process regressions verify ordinary CPU Kev/F5 loads
reject missing or unlabeled suites; native numerical receipts cannot start
serving. Synthetic labeled fixture receipts still exercise successful startup,
source/golden/option mutation and outcome-flag substitution without qualifying
released models. This is an intentional startup compatibility change. General
signed certificates, dataset-provenance enforcement and equivalent mandatory
library gates remain open. No actual GPU or Apple checks were performed.
