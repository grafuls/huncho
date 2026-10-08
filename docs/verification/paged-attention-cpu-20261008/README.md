# Direct CPU Kev page attention, 2026-10-08

No Apple or actual GPU check ran. Native CPU Candle 0.11 executes these tests.
Default dependencies, temperatures, goldens and acceptance thresholds are
unchanged. Direct execution is opt-in CPU FP32 Kev only; source storage pages
without the direct option keep their bitwise storage path.

The new profile reads immutable pages directly for persistent single-row
prefix/suffix calls. Bounded query blocks and unexpanded K/V heads avoid full
KV concatenation/expansion. One softmax includes every original causal key in
order; page matmul shapes/PV summation change arithmetic and need fresh gates.
Independent forwards and private multi-row suffix batches keep a recorded flat
fallback. Scores/softmax still allocate temporary blocks; this is not Flash
Attention, a device path or a total memory bound. Small page matmuls can increase
latency. No released speed/RSS or outcome-acceptance claim is made.

- [Kernel check](kernel.log): grouped heads, causal absolute positions, all four
  page sizes, query blocks and dtype/shape guards match full attention within
  1e-6 at the attention output. Future values cannot enter earlier queries.
- [Native unit/integration checks](native.log): 42 units pass, four unrelated
  hardware/explicit-corpus diagnostics stay ignored; three new integrations
  pass. Actual cached direct calls perform zero complete KV materializations,
  unlike the flat storage path. Frozen typed upstream probabilities and paired
  1e-4/full-argmax gates pass across short/shared/page boundaries, chunks,
  runtime LoRA, CPU kernels, transactional malformed continuation, retained
  snapshots, replica isolation/survival, fallback and configuration guards.
- [Initial CLI run](cli.log): seven existing process groups pass. The new test
  initially attempted to overwrite one receipt with a second profile; strict
  existing output protection correctly rejected it.
- [Final CLI check](final-cli.log): unique receipts fix the test setup. Frozen
  typed external and paired gates, actual forks/retained hits, new execution
  identity/environment, synthetic labeled qualification and refusal of
  unlabeled/missing-prefix/invalid dtype/prerequisite profiles pass. Synthetic
  targets prove gate plumbing only.
- [Full CPU backend regression](backend.log): Candle/Clef and experimental
  packed paths retain their prior behavior; no device test runs.
- [Optional real OpenBLAS profile](openblas.log): direct native checks use the
  installed CPU provider; the provider remains part of qualification identity.
- [Default workspace](default-workspace.log): default dependency/profile checks.

Cargo and actual CLI commands execute sequentially to process exit with
`RAYON_NUM_THREADS=1`; native regression uses `--features clef,quantization`,
CLI uses `--features clef,qualification`. The separate OpenBLAS run uses
`--features clef,cpu-blas`, `HUNCHO_CPU_BLAS_LIBRARY=/usr/lib64/libopenblas.so.0`
and `HUNCHO_CPU_BLAS_THREADS=1`. No GPU inventory, initialization or execution
is involved. Every real serving profile needs fresh complete observed labels,
unchanged external delta 1e-3, full argmax, ECE drift .02 and paired 1e-4 gates.
Released Kev CPU acceptance remains rejected/pending; these fixtures do not
release it. Multi-row/mixed-prefix direct kernels and device work remain open.
