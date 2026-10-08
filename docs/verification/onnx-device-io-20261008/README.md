# ONNX stable I/O ownership and CUDA compilation — 2026-10-08

No actual GPU inventory, provider initialization, execution, transfer, capture
or qualification check ran. Apple work stays skipped. Optional CUDA code and
future ignored GPU tests are compiled against the explicitly selected CPU ORT
library; this is type/build verification only.

CPU-only checks use ONNX Runtime 1.28.0 with `ORT_PREFER_DYNAMIC_LINK=1` and the
explicit CPU library directory. All Cargo/process runs are sequential.

| Check | Result / evidence |
|---|---|
| Initial buffer ownership regression | Crashed on destruction: [red.log](red.log), [CPU debugger stack](crash.log) |
| Retained allocator plus exact input/output/readback addresses, changed data, owned old results, byte/32-slot bounds and failure invalidation | 2 CPU tests pass: [unit.log](unit.log) |
| Full optional CPU `onnx-shared` backend suite and incompatible-profile preflight | Passed: [backend.log](backend.log) |
| CLI `onnx-cuda,onnx-shared,tokenizers,qualification` compilation | Passed: [cuda-compile.log](cuda-compile.log) |
| Backend `onnx-cuda,onnx-shared --tests` compilation, including future ignored CUDA test | Passed: [cuda-tests-compile.log](cuda-tests-compile.log) |
| Actual CPU CLI integrated/masked/shared qualification regressions and device-option refusal | Passed: [cli.log](cli.log) |
| Default workspace | Passed: [default-workspace.log](default-workspace.log) |

The CPU state test substitutes only the allocator device inside a private test
helper and disables capture. Public options refuse CPU device-buffer/graph
profiles before session loading. It proves stable allocation ownership and
copy logic, not CUDA addresses or graph capture. The initial regression exposed
that pinned ORT's allocator-created tensors do not keep the allocator alive;
each slot now retains that allocator after all tensors/bindings in drop order.

Immutable exact-shape slots keep up to 32 permanent graph IDs/addresses, with
at most 512 MiB charged buffers/readback/metadata. No eviction silently changes
captured addresses. Oversized/new-shape admission errors preserve existing
slots; native copy/run/readback/nonfinite failures invalidate the context.
Caller input allocations, ORT workspaces/arenas and captured graphs are outside
the byte budget. Copying uses synchronized ORT identity helpers. No speed,
peak-memory, GPU execution or calibration claim follows.

Original fixtures, temperatures and 1e-3 probability, full argmax, .02 ECE drift
and paired 1e-4 gates are unchanged. Fresh complete labeled serving gates and
execution identity remain required for the optional profile. Real GPU and
released graph/model qualification must be completed after the user's deferral
is lifted; CPU fixture gates cannot release this path. Defaults and dependency
features stay unchanged.
