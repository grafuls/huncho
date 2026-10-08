# Local CPU tensor sharding verification — 2026-10-08

The optional Kev raw pooler ran through vLLM's actual local two-process CPU
tensor-parallel executor. No actual GPU inventory, probe, inference or
qualification ran. Apple Silicon is deferred. The runtime is the same pinned
CPU-only Python/vLLM/Torch environment documented in [the initial increment](../vllm-cpu-20261008/README.md).

Sequential checks:

| Check | Result | Evidence |
|---|---|---|
| Actual one- and two-rank native readouts, loaded shard/head dimensions, original probability/argmax gates, native paired-batch gates | 2 passed | [native.log](native.log) |
| Actual two-rank Pending capture, unchanged package/temperature, fresh complete synthetic labeled conformance/receipt, HTTP serving, grouped bench and refusal paths | 1 passed | [cli.log](cli.log) |
| Workspace with optional `vllm` feature and rank-budget/package refusals | Passed; explicit runtime tests ignored in this ordinary run | [vllm-workspace.log](vllm-workspace.log) |
| Default workspace | Passed | [default-workspace.log](default-workspace.log) |

Commands selected the explicit CPU interpreter, `HUNCHO_DEVICE=cpu` and
isolated Intel OpenMP preload. Native tests ran
`cargo test -p huncho-backend --features vllm --test vllm_cpu real_cpu -- --ignored --test-threads=1`.
CLI tests additionally set `HUNCHO_VLLM_TENSOR_PARALLEL=2` and ran
`cargo test -p huncho-cli --features vllm --test vllm_cpu real_cpu_cli -- --ignored --test-threads=1`.

The fixture's two loaded MLP gate/up projections have shape `[256,128]` and
down projections `[128,128]` **on each rank**, half their unsharded intermediate
width. The q/k pointer weights remain `[32,128]` FP32 on both ranks. Every
submitted group is accepted only after each rank reports one actual model
forward. All fixed independent probability comparisons and single-rank
agreement stay within 1e-3, with full argmax agreement. Per-profile native batch
parity uses the unchanged 1e-4 bound at multiple original temperatures. No
reference vector, temperature, or threshold is changed to pass these checks.

The source now registers its model in spawned imports, uses picklable module
functions for collectives and verifies each rank's runtime/layout/thread
identity. Successful disposal uses native shutdown; failed protocol/startup
disposal kills the owned process group. Thread/KV budgets are per rank and
the total rank-thread bound is explicit. Work counters count one collective
forward and unique input tokens, rather than summing tokens across shards.

Synthetic first-candidate labels prove complete-label gate plumbing, not
released calibration. The fixture remains Pending; temporary CLI copies alone
are marked fitted for refusal/acceptance checks, without temperature changes.
Released Kev rank-specific fitting/held-out acceptance, controlled capacity,
speed/RSS, pipeline/multi-node/device execution and multi-GPU qualification
remain open. [Operation and bounds](../../vllm.md).
