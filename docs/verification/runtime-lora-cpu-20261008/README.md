# CPU FP32 runtime LoRA — 2026-10-08

Standard A/B projections are retained separately from immutable base weights.
The actual cache-sharing check compares storage addresses at an adapted Qwen
query projection across two independently loaded models with different B weights, then clears the base
cache and drops the first model. The second model's inference stays identical.
Prefix handles and native replicas retain their existing ownership isolation.

| CPU-only check | Evidence |
|---|---|
| Projection arithmetic, malformed/unsupported adapter rejection, shared targeted storage/eviction, native model/head/cache regressions | [library.log](library.log) |
| Native F2 frozen probabilities, existing-temperature paired gates, scalar/native batches, chunks/forks/pages/replicas/owned output and changed-adapter isolation; F3 full/selected vocabulary and native batch parity | [native.log](native.log) |
| Actual CLI original numerical and synthetic labeled gates, receipts, actual prefix work and missing/unlabeled/reduced/invalid-profile refusal | [cli.log](cli.log) |
| Existing CPU backend suite | [backend.log](backend.log) |
| Actual one-thread OpenBLAS base projections with runtime A/B updates, F2/F3 native parity | [blas.log](blas.log) |
| Default workspace | [default-workspace.log](default-workspace.log) |

Checks run sequentially. Backend features are optional `clef,shared-base,quantization,cpu-blas`;
The final native log includes updated library checks and all three F2/F3 native tests.
CLI process checks use optional `clef,shared-base,qualification` and explicit
`HUNCHO_DEVICE=cpu`, one Candle/Rayon thread. No actual GPU inventory, execution
or qualification runs. Apple work is skipped.

The original independent PyTorch Kev vectors/temperature stay unchanged. The
external 1e-3 bound, full argmax agreement, .02 ECE drift and paired 1e-4 gates
are unchanged. F3 uses the existing synthetic tied vocabulary projection/bias
recipe against the merged native reference; it does not qualify Nimble.
Arbitrary synthetic labels prove gate wiring only. Released model acceptance,
refits, held-out gates, RSS/startup/throughput and mixed-adapter tensor collation
remain open. CPU FP32 is the only new execution profile; reduced, packed and
device runtime adapters are refused. No new default dependency is introduced.
