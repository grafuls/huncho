# Kev optimization qualification — 2026-10-07

Independent Kev-4B CUDA FP16 passes the current numerical regression checks.
CUDA FP16 request-local prefix reuse and native question batching **do not qualify** on
this checkpoint/T4 variant and remain disabled by default. CPU FP32 passes both
paths on the numerical regression suite. The gates were not
relaxed. CUDA FP32 loading fails with out-of-memory.

## Execution and references

Host: `x37-h17-000-r740xd.rdu3.labs.perfscale.redhat.com`; Tesla T4, 15,360 MiB.
The performance/parity measurements below used `5.14.0-742.el9.x86_64`. NVIDIA 580.178.04 was initially
built only for the old kernel. With user approval, matching kernel-devel and
the host's bundled Red Hat signing key were installed/imported, the existing
DKMS driver was rebuilt, and its modules loaded without rebooting. No running
kernel or persistent repository configuration was changed by that repair.
The host subsequently rebooted at 14:17 UTC into `5.14.0-427.13.1.el9_4.x86_64`,
interrupting labeled capture. Later runs record their separate kernel/boot
identity; the NVIDIA driver remained healthy.

The build uses CUDA toolkit 12.8.93, `sm_75`, Rust 1.97.1 and the existing
feature-gated Candle/ONNX dependencies. The CUDA-enabled release CLI compiled
offline. CPU/CUDA Kev/Clef upstream fixtures and the new CUDA prefix/chunk and
batch tests all passed; the [test log](verification/kev-t4-20261007/cuda-tests.log)
is retained.

Pinned adapter: `jaredpalmer/kev-4b` at
`6cfce5c2fa4b4bd64026336ab649c5ca78857d52`. Pinned base:
`Qwen/Qwen3.5-4B-Base` at `1001bb4d826a52d1f399e183466143f4da7b741b`.
Temperature: the unchanged upstream `2.40605`; pointer projections/recurrent
accumulation remain FP32. No quantized execution or temperature refit was used.

Six requests/19 questions cover short, negated, structured and longer states,
all three types, a singleton, and equal-length rows with reordered options.
Reference vectors came from the earlier independent-forward binary, separately
on CPU FP32 and CUDA FP16; they were never regenerated from optimized outputs.
These cases measure numerical fidelity, not statistical outcome calibration.

The [audit summary](verification/kev-t4-20261007/summary.json) records binary and
source-snapshot SHA-256 identities, complete reports and load failures. The
[CPU vectors](verification/kev-t4-20261007/cpu-fp32-golden.json) and
[CUDA vectors](verification/kev-t4-20261007/cuda-fp16-golden.json) can be reused
with `huncho conform`. Original remote files are under
`/var/tmp/huncho-optimization-20261007`; the prior installed RPM/source directory
was preserved. The source snapshot predates the subsequent tokenizer-cache and
stderr-logging changes, which have separate local tests.

## Probability gates

Every row below executes CUDA FP16. The reference column names the golden
execution. External gates require delta <=1e-3, complete argmax agreement and
ECE drift <=0.02. Optimizations additionally require paired delta <=1e-4 and
complete argmax agreement against independent forwards on the loaded engine.

| Reference | Path | External max delta | Paired max delta | Result |
|---|---|---:|---:|---|
| CPU FP32 | Independent | 0.000512362 | — | Pass |
| Prior CUDA FP16 | Independent | 0.000000000466 | — | Pass |
| CPU FP32 | Prefix | 0.003143430 | 0.002896309 | Reject |
| Prior CUDA FP16 | Prefix | 0.002896309 | 0.002896309 | Reject |
| CPU FP32 | Batch | 0.000884831 | 0.000372469 | Reject paired gate |
| Prior CUDA FP16 | Batch | 0.000372469 | 0.000372469 | Reject paired gate |

All argmax comparisons agree, illustrating why that check alone is insufficient.
Prefix work drops from 1,951 submitted positions to 865, with five prefills and
18 forks. Batching uses 16 backbone calls versus 19, with three native batches.
GPU FP32 independently fails at weight loading for all three paths; no FP32
probability or latency result is claimed. Numerical root cause remains open;
shape-dependent reduced-precision arithmetic is a hypothesis, not a diagnosis.

## Measured performance, including unqualified paths

Warm, closed-loop, in-process release measurements: five mixed questions,
10 distinct timed requests, one client, CUDA FP16; loading/warmup excluded.
Batch budget: 4,096 token positions. Paths ran sequentially. Ten observations
are descriptive, not a production saturation study or a robust p99 estimate.

| State | Path | Mean latency | Submitted positions | Forward calls / prefills |
|---|---|---:|---:|---|
| Short | Independent | 1,613.74 ms | 1,690 | 50 / 0 |
| Short | Prefix, unqualified | 1,246.02 ms | 1,210 | 50 / 10 |
| Short | Batch, unqualified | 1,571.57 ms | 1,690 | 30 / 0 |
| Long | Independent | 23,813.58 ms | 27,040 | 50 / 0 |
| Long | Prefix, unqualified | 5,631.89 ms | 6,280 | 50 / 10 |
| Long | Batch, unqualified | 24,881.46 ms | 27,040 | 30 / 0 |

Prefix reuse removes 76.8% of long-state submitted positions and shows about
4.23× lower latency, but fails probability qualification. Short batching's
2.6% reduction is small; long batching is 4.5% slower. The generic per-token
recurrence and remaining launches need profiling before further kernel work.
The audit retains individual benchmark metadata and physical work counters.

## Observed outcomes for subsequent precision work

Dataset selection was delegated by the user. Preparation uses published
training splits for fitting and separate validation/test splits for evaluation,
with deterministic sampling and exclusion of identical inference inputs from
fitting. Each type contributes 256 fitting and 512 evaluation cases: 768/1,536
total. The source hashes, selected row IDs and label distributions are in
[dataset provenance](verification/kev-t4-20261007/dataset-provenance.json).

| Type | Dataset / pinned revision | Fit / evaluation splits |
|---|---|---|
| Choice | [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2), `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb` | train / validation; observed negative/positive labels |
| Noul | [BoolQ](https://huggingface.co/datasets/google/boolq), `35b264d03638db9f4ce671b711558bf7ff0f80d5` | train / validation; passage/question with boolean answer |
| Score | [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full), `c1f9ee939b7d05667af864ee1cb066393154bf85` | train / test; five observed rating labels mapped to score indices |

These are application-specific checks, not evidence of universal calibration or
of disjointness from Kev/base-model training. Yelp rating ambiguity remains
part of its observed-label task. Corpus text is retained in the remote analysis
directory rather than redistributed in this repository. Independent CUDA FP16
raw-logit capture completed after restarting on the later kernel. A collector
candidate-order error was also corrected: `kev-v1` Noul rows are `[no, yes]`;
generic F2/joint heads can use a different order. Old fitting target indices
are excluded from refit conclusions. Captures now retain explicit candidate
labels, check raw/wire argmax consistency and checkpoint expected vectors.
The corrected capture has all 768 fitting and 1,536 evaluation rows; its
[execution identity](verification/kev-t4-20261007/labeled-reference-v2-identity.json)
pins the reference binary, manifest, kernel/boot, collector and corpus inputs.
Fitting and evaluation artifacts remain separate. The baseline manifest hash
and exact fitting-logit bytes were verified before offline evaluation.

The temporary candidate refit uses type/cardinality temperatures 2.8377154
(choice:2), 1.9939160 (noul:2), and 2.3756089 (score:3–5). It is **not promoted**;
the released package retains 2.40605. [Fit report](verification/kev-t4-20261007/candidate-fit-v2.json)
and [held-out results](verification/kev-t4-20261007/held-out-calibration-v2.json)
use observed labels and offline FP64 probability math, not a runtime certificate.

| Held-out task (512 each) | Baseline NLL / refit | Baseline Brier / refit | Baseline peak ECE / refit |
|---|---|---|---|
| SST-2 choice | 0.167649 / 0.161806 | 0.090138 / 0.089634 | 0.027864 / 0.019002 |
| BoolQ noul | 0.230962 / 0.225940 | 0.134618 / 0.133807 | 0.036084 / 0.025578 |
| Yelp five-level score | 0.740560 / 0.740592 | 0.431260 / 0.431413 | 0.041281 / 0.044067 |
| All 1,536 | 0.379724 / 0.376113 | 0.218672 / 0.218285 | 0.015302 / 0.016950 |

Argmax accuracy is unchanged (aggregate 84.11%). All paired 95% percentile
intervals from 1,000 deterministic row bootstraps include zero, with fitting
temperatures held fixed. ECE uses 15 equal-width bins; the runtime conformance
harness uses 10. Aggregate ECE can hide type-specific errors, and this small
sample does not support a confident refit improvement or universal calibration.
No candidate temperature was served or used to excuse the failed GPU gates.

## Subsequent CPU and exact-reuse qualification

The later binary (`huncho-reuse`) runs on the older kernel after the external
reboot. [Separate summary](verification/kev-t4-20261007/reuse-summary.json)
records snapshot/runtime identities. Six unchanged numerical cases pass CPU
FP32 independent, prefix and batch execution: external deltas 9.779e-9,
3.576e-7 and 9.779e-9 respectively; paired prefix delta is 3.576e-7 and batch
delta is zero, with complete argmax agreement. CUDA FP16 independent again
passes with delta 4.657e-10. These reports precede the conformance tie-order
fix; all retained reference modes were checked to be untied.

Ten-request, five-question warm in-process CUDA FP16 measurements compare
result reuse at a 1 MiB budget with the unchanged independent path:

| Workload | Retention | Mean latency | Timed forwarded positions / hits |
|---|---|---:|---:|
| Distinct | Disabled | 1,561.143 ms | 1,690 / 0 |
| Distinct | 1 MiB | 1,559.261 ms | 1,690 / 0 |
| Repeated | Disabled | 1,559.897 ms | 1,690 / 0 |
| Repeated after warmup | 1 MiB | 0.005386 ms | 0 / 10 |

The repeated workload deliberately measures exact cache hits; these microsecond
values exclude HTTP, loading and warmup and are not model-compute speedups or
saturation rates. Distinct requests do the same work with no meaningful timing
difference established by ten observations. Exact reuse preserves all original
probabilities; conformance bypasses it. HTTP in-flight sharing has separate
cancellation/auth/queue-limit coverage and passes the full-model comparison below.

## Exact in-flight HTTP sharing

The subsequent CUDA-enabled `huncho-stage2` binary passes the same six unchanged
CUDA FP16 numerical cases with the corrected candidate-order argmax gate
(max delta 4.657e-10, complete agreement, zero ECE drift). Both HTTP servers
also pass that explicit startup binding before opening their listeners.

Three warm loopback bursts each send eight simultaneous identical three-question
requests. Burst inputs differ; result retention, prefix reuse and native tensor
batching are disabled. [Retained report](verification/kev-t4-20261007/stage2-http-coalescing.json)
pins the binary/snapshot, driver/kernel/boot, per-request latencies and exact
input/output byte hashes; the [execution script](verification/kev-t4-20261007/http-coalescing-run.py)
is retained byte-for-byte. This is a duplicate-work experiment, not open-loop
saturation or a production p99 estimate.

| HTTP mode | Mean eight-caller burst completion | Forwarded positions | Coalesced callers |
|---|---:|---:|---:|
| Independent, sharing disabled | 6,660.961 ms | 2,016 | 0 |
| Exact sharing, 1 MiB metadata budget | 848.636 ms | 252 | 21 |

All eight replies within every burst and both execution modes agree byte-for-byte,
including calibrated probabilities, confidence, legends, logical usage and
requested raw-logit extensions. Three evaluations serve 24 callers; the 7.85×
burst reduction applies to this identical-input workload. Unique requests still
use independent evaluations. The optimization defaults to zero and adds no
runtime dependencies. Later kernel experiments use separate source snapshots
and binaries; this qualification applies to the recorded stage2 identity.

## Measured FP16 drift

An ignored offline diagnostic traces independent and prefix-continuation
execution layer by layer on the pinned structured case, with temperature
2.40605. It verifies that the manual trace reproduces each actual backend path
within 1e-6 probability delta. [The trace](verification/kev-t4-20261007/precision-prefix-fp16.json)
and [execution identity](verification/kev-t4-20261007/precision-trace-identity.json)
retain all 32 layer comparisons. Passing this diagnostic test means the trace
is valid; it does not qualify the prefix path.

Holding each layer's normalized input values fixed and projecting it in one
call versus separate prefix/suffix calls produces FP16 differences in all
32 first projections for the choice and Noul rows, reaching 0.0078125. This
measures a contribution from GEMM shape/layout. The score row's first
projections have zero difference under that comparison, but its layer outputs
still drift, so this is not the sole cause.

| Structured-case question | Prefix / suffix positions | Final prefix / suffix hidden max delta | Actual probability delta |
|---|---:|---:|---:|
| color (choice) | 27 / 17 | 0.375 / 0.375 | 0.002896309 |
| blue (Noul) | 27 / 13 | 0.375 / 0.250 | 0.000297010 |
| rating (score) | 27 / 21 | 0.125 / 0.375 | 0.000301778 |

These are observed internal activation differences, not tolerances for
calibrated output. The external and paired probability gates remain unchanged.

## Fixed-row projection experiment

The separate `huncho-stage3` binary applies 64-row backbone projections with
zero padding local to the linear calls. It retains native attention, FP16
weight/activation storage and the upstream temperature. [Numerical audit](verification/kev-t4-20261007/stage3-projection-qualification.json)
pins the source snapshot, binary, runtime, script and unchanged CUDA golden.

| 64-row path | External max probability delta | Paired independent delta | Numerical outcome |
|---|---:|---:|---|
| Independent | 0.000456929 | — | Pass |
| Prefix | 0.000456929 | 0.000391185 | Reject |
| Exact-length batch | 0.000456929 | 0 | Pass |

All paths retain complete argmax agreement on the six cases/19 questions.
Prefix drift is substantially smaller but still exceeds 1e-4. The batch pass
only applies to this numerical suite; the separate 1,536-case labeled
independent evaluation **rejects** the profile (max delta 0.012506247; 85 cases
above 1e-3). Complete argmax agreement and ECE drift 0.0007783044 do not
compensate for failed probability delta. [Complete held-out report](verification/kev-t4-20261007/stage3-held-out.json)
records ECE 0.011417184 against reference 0.01063888 and Brier
0.218687489 against 0.218672209. The larger suite is captured by the native
independent reference and is never regenerated using this profile.
No profile is enabled for serving, no temperature is changed, and no speed
claim is made from these conformance timings (which include loading).

A further separate source/binary adds FP32 dense-attention computation while
retaining FP16 storage. Its CUDA numerical run is complete: independent max
delta 0.00045901537 passes; exact-length batch has paired delta zero; prefix
paired delta 0.00021833181 still rejects. All retain complete argmax agreement.
[Combined profile audit](verification/kev-t4-20261007/stage4-numerical-audit/identity.json)
is numerical-only and explicitly unqualified; no broad labeled test was run.
Local CPU fixtures validate profile readouts, cache/batch parity and rejection
of kernel changes with live retained handles. Full-model GPU and labeled gates
remain required. Newly added ModernBERT/F3 CUDA loaders likewise have separate
fixture and released-checkpoint qualification; actual T4 fixtures now pass
([execution log](verification/kev-t4-20261007/stage5-cuda-fixtures.log)). Released Laya/Nimble
acceptance remains open. The unchanged independent Kev FP16 numerical check
also passes (max delta 4.656613e-10); it does not change the recorded baseline.
[Collected receipt identity](verification/kev-t4-20261007/stage345-receipt-identity.json)
links source/binary/report hashes and exit receipts for all three stages.


## Bounded preparation (2026-10-08)

The isolated stage-6 snapshot adds owned, engine-bound prepared requests and
optional bounded F1–F4 CPU formatting/tokenization ahead of model execution.
The default remains disabled; forward shapes/order, readouts, temperature,
softmax and usage are unchanged. `huncho conform --prepare-all` compares the
optimized arm with uncached independent forwards and requires actual preparation.

On the T4, the unchanged six-case/19-question FP16 suite passes: external maximum
probability delta 4.656613e-10, paired delta zero, complete argmax agreement and
ECE drift zero. The optimized arm submits 19 forwards/1,951 token positions and
prepares 19 questions, with zero result/prompt hits, forks or batches. The paired
independent arm executes additional work that is excluded from the optimized
arm's `work` counters. [Numerical report](verification/kev-t4-20261008/stage6-preparation-numerical/independent.json)
and [runtime identity](verification/kev-t4-20261008/stage6-preparation-numerical/identity.json)
pin source, binary, script, model/tokenizer/head/temperature and device.

The separate HTTP pilot compares disabled/enabled preparation for eight distinct
short structured states with three mixed questions each, two bursts per mode in
alternating order. All responses are byte-identical, including raw logits and
logical usage. Every burst submits 1,344 model positions; enabled bursts prepare
24 questions, with zero cache/coalescing/fork/batch work. Warmup is excluded.

| Preparation slots | Burst seconds (two repetitions) | Mean burst seconds |
|---|---|---|
| 0 (disabled) | 12.27495, 12.29863 | 12.28679 |
| 1 | 12.65576, 12.53884 | 12.59730 |

Enabled preparation is about 2.5% slower in this small pilot; no speedup is
claimed and default zero is retained. This is a short-state eight-client
measurement, with a concurrent CPU pilot on other NUMA cores and no system-wide
frequency/isolation controls. [Complete HTTP pilot](verification/kev-t4-20261008/stage6-preparation-pilot.json)
records per-request durations, response hashes, physical work and scope. The
1,536-case held-out GPU preparation gate was interrupted on 2026-10-08 at the user's request to defer actual GPU checks. It has no acceptance result. The preceding completed checks used the same original temperature and
unchanged independent reference. These are separate gates; no altered kernel
profile, prefix reuse, batching or refitted temperature is enabled.

Local default and Clef-enabled workspace tests pass, including cancellation
while tokenization is blocked, ready-packet cancellation, bounded overlap,
authentication/sharing and native prefix/batch composition. ONNX/Clef Clippy
completes with existing warnings; fourteen Python qualification/data tests pass.
The user authorized commits per implementation area on 2026-10-08. Further actual GPU checks are deferred and Apple Silicon is skipped for now.


## Full-model CPU thread pilot

The retained stage-5 binary was measured with CPU FP32, unchanged upstream
T=2.40605, no projection/attention profile and no prefix/batch/result reuse.
Both runs use the same sixteen physical cores on NUMA node 1. Each thread
setting first passes the unchanged six-case numerical suite (max probability
delta 9.778887e-9, complete argmax agreement, zero ECE drift), then measures five
warm distinct requests with five mixed questions each.

| Rayon / Candle threads | Mean request seconds | Median seconds | Measured forwards / positions |
|---|---:|---:|---|
| 4 / 4 | 39.38248 | 39.38210 | 25 / 845 |
| 16 / 16 | 35.47609 | 35.47509 | 25 / 845 |

Sixteen threads lower observed mean latency by about 9.9% in this small pilot.
The four-thread run precedes the sixteen-thread run; other NUMA work, including
CUDA loading/qualification, can affect memory bandwidth and frequency. This
single comparison does not establish a portable default, production p99,
replica throughput or calibration against observed labels. No runtime thread
default is changed. [Complete CPU pilot](verification/kev-t4-20261008/stage5-cpu-thread-pilot.json)
and its [execution script](verification/kev-t4-20261008/huncho-cpu-thread-pilot.py)
pin the affinity, reference vectors, model, source/binary and work counts.
