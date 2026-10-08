# Optional CPU vLLM verification — 2026-10-08

This increment was exercised on Linux x86_64 CPU only. No actual GPU probe,
kernel, inventory or qualification was run. Apple Silicon is deferred.
The isolated Python 3.12.9 environment contains vLLM 0.31.0+cpu,
Torch 2.13.0+cpu, Transformers 5.17.0 and Triton 3.8.0+cpu.
`HUNCHO_DEVICE=cpu` and explicit CPU interpreter/OpenMP paths were used.

Sequential checks and retained outputs:

| Check | Result | Evidence |
|---|---|---|
| Actual raw pooler, short/empty/owned readouts, native batches, fixed independent probability/argmax/paired/labeled gates | 1 passed | [native.log](native.log) |
| Actual Pending capture, unchanged source/temperature, fresh receipt, HTTP serving, grouped benchmark and qualification refusal | 1 passed | [cli.log](cli.log) |
| Default workspace | Passed | [default-workspace.log](default-workspace.log) |
| Workspace with optional `vllm` feature, including pre-worker package refusals | Passed; runtime tests explicitly ignored in this ordinary run | [vllm-workspace.log](vllm-workspace.log) |
| CPU exporter completeness, source mutation, unsupported semantics, unchanged source/overrides and new-output-only checks | 4 passed | [export-tests.log](export-tests.log) |
| Independent fixture regeneration | Passed | [export-reproduction.log](export-reproduction.log) |

Actual runtime commands were
`cargo test -p huncho-backend --features vllm --test vllm_cpu real_cpu_pooling -- --ignored --test-threads=1`
and
`cargo test -p huncho-cli --features vllm --test vllm_cpu real_cpu_cli -- --ignored --test-threads=1`.
The optional exporter test uses the explicit CPU Python environment:
`python scripts/tests/test_export_kev_vllm.py -v`.

Regeneration reproduced checked-in model, tokenizer, config, manifest,
descriptor and golden bytes exactly. Provenance key ordering differs after
independent safetensors loading, but all source digests match. FP32 reference
goldens are independent eager PyTorch results with seed-711 hybrid GDN/full
attention, merged LoRA and original Kev inputs. They are never replaced by
runtime outputs to pass the gate. Measured maximum BF16 probability delta was
about 0.0006074 at the original temperature, below the fixed 1e-3 bound;
all six argmaxes and the unchanged paired 1e-4 batch gate pass.

Synthetic first-candidate labels exercise complete-label/ECE gate plumbing.
They cannot establish observed-outcome calibration of a released checkpoint.
The committed fixture and every newly exported variant stay Pending; only
temporary CLI packages use fitted status to test serving refusal/acceptance.
No temperature is refitted in these checks. Full 4B released calibration,
latency/throughput/RSS, other architectures, quantization, distributed runs and
GPU execution are not qualified. See [profile and operation](../../vllm.md).
