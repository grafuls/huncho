# Borrowed external batch planning, 2026-10-09

No default dependency/feature, prompt, weight, temperature, probability or gate
changes. No Apple or actual GPU inventory/execution/qualification checks run.
The shared grouping algorithm now accepts owned native input dimensions and
borrowed external readout dimensions. External planning no longer clones token
and marker vectors into temporary `ForwardInput` values that it then discards.
Stable order, original readout indices, physical rectangles, conservative
marker/final-row charges, padding and complete-prefix budgets are unchanged.

[Release allocation check](allocations.log) measures successful alloc/realloc
calls on only the current thread, excluding request preparation and disposal.
The historical comparison retains its cloned inputs so the compiler cannot
eliminate them; it counts just the clone stage, excluding historical grouping.
The new measurement includes complete actual borrowed group planning.

| Four question rows | Historical clone-stage calls / requested bytes | Actual borrowed planning calls / requested bytes |
|---|---:|---:|
| 14 tokens each | 9 / 704 | 4 / 256 |
| 512 tokens each | 9 / 8,672 | 4 / 256 |

These explicit shape fixtures never run inference or claim model calibration.
They verify that preparation keeps original payloads and that planning retains
all four indices/actual lengths without heap requests growing with token data.
The test-only allocator is shared with the previous owned-calibration check;
its original zero-additional-allocation and probability assertions still pass.

[Workspace](workspace.log) includes the same original native rectangle,
padding/row/readout/prefix bounds and external fresh labeled/paired gate checks.
[Native CPU cached groups](native.log) retain original frozen typed vectors,
complete-context/padding budgets, immutable parents, pressure/cooperative
cohorts, adapters and the unchanged fresh external/paired acceptance thresholds.
[Official CPU WASM build](browser-build.log) and [actual browser](browser.log)
retain scalar/page/worker answers and the original native equal/masked group
counts, physical tokens/padding and complete labeled/paired probabilities.
GPU/WebGL initialization stays explicitly disabled. Tests and builds run
sequentially to process exit.

Requested bytes exclude allocator overhead, graph tensors, serialization,
tokenizer work and all other request/runtime memory. They are not simultaneous
peak RSS, latency, throughput or cost. Original native/browser frozen vectors
and fixed complete observed-label/paired gates remain necessary. Released Kev
FP32 is rejected and Q8/Q4 outcome qualification pending. Cross-request external
collation, activation-buffer reuse and device work remain open.
