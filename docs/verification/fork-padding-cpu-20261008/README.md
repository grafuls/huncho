# CPU Kev mixed-length cached batches, 2026-10-08

This extends [equal-length prefix batches](../fork-batch-cpu-20261008/README.md)
with causal right-padding. It is fixture evidence, not released calibration,
throughput or peak-memory qualification. No Apple or actual GPU checks ran.
The original temperatures, probabilities and acceptance thresholds are intact.

Every row starts from one immutable complete parent. The backbone processes a
suffix rectangle with zero token IDs only after each original suffix end.
Original option markers and each row's real final decision select valid hidden
rows. All resulting recurrent/convolution/KV state is private and discarded;
the parent and previously returned readouts remain independent.

The planner charges `B*(prefix+max_suffix)` to its workspace token budget and
measures padding as `padded/(B*max_suffix)`. Prefix residency cannot dilute the
padding limit. Reports and Prometheus expose dedicated padded-prefix counters,
and fresh conformance/receipt checks require actual multiple mixed-length
cached rows. A profile with only unrelated padded batches cannot qualify.

- [Native/engine tests](native.log): original frozen mixed-length typed Kev
  requests pass FP32/FP16 external 1e-3/full-argmax and paired 1e-4 gates.
  Pages, chunking, buffered/fused/grouped CPU kernels, FP32 runtime LoRA,
  ownership, handles, retained hits, replicas and original usage are exercised.
- [Backend/packed tests](backend.log): 48 unit tests pass (4 existing opt-in
  helpers ignored), including actual equal/padded native failure cleanup. The
  Q8/Q4 test compares original and shortened suffixes against each packed
  profile's own scalar execution at fixed temperatures; no upstream calibration
  acceptance is inferred from packed scheduling parity.
- [CLI](cli.log): seven CPU process groups pass. Frozen original question
  lengths/probabilities exercise actual mixed-prefix batches. Arbitrary fixture
  targets exercise labeled gate plumbing only. Unlabeled serving, vacuous
  budgets and cooperative-prefix combinations remain refused.
- [CPU BLAS](blas.log): three native/engine tests pass with actual optional
  LP64 pthread OpenBLAS on FP32; unsupported FP16 keeps ordinary CPU kernels.
- [Default workspace](default-workspace.log): regression coverage includes
  full-context budget and suffix-padding accounting at a 25% boundary.

Commands are sequential to process exit:

```sh
RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1 cargo test -p huncho-backend --features clef,quantization,shared-base --test fork_batch -- --test-threads=1
RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1 cargo test -p huncho-backend --features clef,quantization,shared-base --lib --test quantized_kev -- --test-threads=1
RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1 cargo test -p huncho-cli --features clef,qualification,quantization,shared-base --test cooperative -- --test-threads=1
HUNCHO_CPU_BLAS_LIBRARY=/usr/lib64/libopenblas.so.0 HUNCHO_CPU_BLAS_THREADS=1 RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1 cargo test -p huncho-backend --features clef,quantization,shared-base,cpu-blas --test fork_batch -- --test-threads=1
cargo test --workspace
```

Defaults remain off. Cooperative scheduling, cross-request/mixed-parent cached
collation and retained padded continuations are outside this increment. Token
budgets bound rectangles rather than total RSS; private full KV materialization
still allocates. Released CPU acceptance remains rejected/pending, and fitting
status remains unverifiable: a read-only retry to the supplied lab host's CPU
log still timed out. No remote inputs, source, binary or driver were changed.
