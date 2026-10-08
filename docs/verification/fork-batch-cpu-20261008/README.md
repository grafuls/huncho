# CPU Kev cached-branch batches, 2026-10-08

This is fixture software evidence, not released Kev calibration or a performance
claim. No Apple or actual GPU checks were run. No source temperature, frozen
probability vector or acceptance threshold changed.

The native path makes actual isolated `fork` handles, collates one complete
parent's KV/recurrent/convolution state into private batch rows, submits one
suffix backbone call, and uses the original pointer readout. All temporary
handles are removed on success and failure. The engine validates original
contexts before prefill, charges complete contexts to the batch token budget,
and preserves logical usage and typed head/temperature/scatter behavior.

- [Native and packed tests](native.log): three FP32/FP16 engine/native tests and
  one actual Q8/Q4 test pass. Frozen upstream typed probabilities use unchanged
  1e-3/full-argmax gates; native/scalar scheduling uses paired 1e-4 probability
  checks at temperatures 0.75, 1 and 2.40605. Pages, chunks, buffered kernels,
  grouped attention, runtime LoRA, ownership, handles, context budgets, replicas,
  retained hits and vacuous-gate refusal are covered. Packed comparisons use
  their own altered scalar weights; they do not qualify upstream calibration.
- [Failure cleanup](cleanup.log): the test passes input admission, forces an
  actual native embedding failure, observes one attempted batched call and two
  forks, then proves all temporary handles are gone and the parent is unchanged.
- [CLI](cli.log): seven process groups pass. The new group duplicates original
  typed fixture questions without changing their frozen probabilities, requires
  actual prefix batches, binds an execution receipt, refuses unlabeled serving
  and a vacuous token budget, and exercises labeled receipts with arbitrary
  fixture targets. Those targets prove plumbing only.
- [CPU OpenBLAS](blas.log): the same native/engine group uses the real optional
  LP64 pthread OpenBLAS provider with one provider thread for FP32; unsupported
  FP16 retains ordinary CPU kernels.
- [Default workspace](default-workspace.log): default-feature regression checks.

Commands run sequentially to process exit:

```sh
cargo test -p huncho-backend --features clef,quantization,shared-base --lib failed_native_branch -- --test-threads=1
cargo test -p huncho-backend --features clef,quantization,shared-base --test fork_batch --test quantized_kev -- --test-threads=1
cargo test -p huncho-cli --features clef,qualification,quantization,shared-base --test cooperative -- --test-threads=1
HUNCHO_CPU_BLAS_LIBRARY=/usr/lib64/libopenblas.so.0 HUNCHO_CPU_BLAS_THREADS=1 RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1 cargo test -p huncho-backend --features clef,quantization,shared-base,cpu-blas --test fork_batch -- --test-threads=1
cargo test --workspace
```

The profile is default off and limited to equal-length CPU Kev suffixes from one
parent. It does not retain suffix continuation state, directly address KV pages,
mix unrelated prefixes, pad rows or combine with cooperative/cross-request
scheduling. Prefix payloads are copied into a bounded private complete-KV
workspace; memory sharing of retained pages does not remove that allocation.
Released CPU profiles remain rejected/pending at their unchanged held-out gates.
