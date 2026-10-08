# CPU llama.cpp decision backend

The optional `llamacpp` feature loads standard Qwen3.5 GGUF weights for
Kev F2 pointer scoring and F3 candidate logits. It runs one finite prompt prefill,
returns raw scores and leaves temperature/softmax to the existing core. It has no
sampler or text-generation loop. The default build gains no native dependencies.
Apple and GPU execution are not included in this profile.

## Build and execution

```bash
cargo build --release -p huncho-cli --features llamacpp,tokenizers
HUNCHO_DEVICE=cpu HUNCHO_LLAMA_THREADS=4 huncho bench \
  --model /path/to/gguf-package --backend llamacpp --iterations 10 --json
```

This feature needs a C/C++ compiler, CMake and libclang for the pinned sys crate.
On distributions with multiple LLVM installations, select a matching libclang
and compiler resource directory, for example:

```bash
export LIBCLANG_PATH=/usr/lib64/llvm15/lib
export BINDGEN_EXTRA_CLANG_ARGS='-isystem /usr/lib64/llvm15/lib/clang/15.0.7/include'
```

`llama-cpp-sys-2` is pinned to `0.1.159`, with bundled llama.cpp revision
[`26394b4`](https://github.com/ggml-org/llama.cpp/tree/26394b4e6749a41c3633db040e0987500a5f7013).
Common helpers and GPU/Metal/Vulkan features are disabled. This adapter explicitly
loads only the CPU device, with no offload, FP32 KV/recurrent state and fixed
`HUNCHO_LLAMA_THREADS` (1–256, default 1). An explicit `HUNCHO_DEVICE` must be `cpu`.
Compiled GGML CPU features are reported separately from Huncho's Rust features;
AVX/AVX2/FMA/F16C/AVX512 kernels require compatible host instructions.

F2 uses the original trained pointer head in FP32. A narrow C++ shim calls the
pinned staging API in
[`src/llama-ext.h`](https://github.com/ggml-org/llama.cpp/blob/26394b4e6749a41c3633db040e0987500a5f7013/src/llama-ext.h)
to read masked, unnormalized Qwen3.5 final RMS-norm states. Ordinary embeddings
mode forces every token to be an output and is deliberately avoided. This
staging API can change: upgrading the runtime requires rebuilding the shim and
repeating numerical and observed-outcome qualification.

F3 copies raw vocabulary logits, optionally selecting only requested answer-code
columns on the host. GGML still computes a vocabulary projection. F2 likewise
has an unused auxiliary vocabulary projection over selected marker/decision
rows; removing it requires a future native graph profile and new qualification.
The loader rejects other architectures, unsupported tensor mixtures and Huncho's
custom `kev-projections-*` GGUF format. Supported exact artifact labels are
`gguf-f32`, `gguf-f16`, `gguf-q8_0` and `gguf-q4_0`. Quantized artifacts must use
`llamacpp-qwen35-q8_0-v1` or `llamacpp-qwen35-q4_0-v1`, respectively, and may mix
only FP32 tensors with that declared type. Mixed F16/quantized layouts and other
GGML quantizers are rejected, rather than assigned an ambiguous profile.

Each independent call clears the context's hybrid state. CPU replicas share
immutable weights/head but own separate contexts and buffers; `--replicas`
uses the existing pool and gates every context. Multiply native threads by
replicas when budgeting CPU capacity. Kev F2 also exposes optional prefix
prefill/forks and exact retained prefixes through the existing engine flags.
Each immutable snapshot uses the pinned
[full sequence-state API](https://github.com/ggml-org/llama.cpp/blob/26394b4e6749a41c3633db040e0987500a5f7013/include/llama.h#L877):
attention KV and recurrent/convolution state are serialized together. Forks
share immutable bytes. Before a suffix, sequence zero is cleared and restored,
then explicit absolute positions are submitted. Successful continuations commit
a new snapshot only after readout/snapshot creation; failed calls retain the
old branch. Independent calls and other forks cannot change saved bytes.

`llamacpp_prefix_state=full-hybrid-sequence-snapshot-v1` binds this profile.
There are at most 64 live handles and 512 MiB of charged snapshot/key/entry
storage per context; native model/context allocations are additional. Optional
retention uses a caller budget of 0–512 MiB and at most 16 exact token prefixes,
evicts FIFO, and defaults to zero. Clearing retention keeps live handles valid.
Oversized snapshots fail explicitly; no tokens or cache state are truncated.
This is CPU state-copy reuse, not paging or native multi-sequence attention.
Copying a large recurrent state can outweigh prefill savings. Cooperative
chunks and runtime multi-adapter attachment remain open.

`HUNCHO_LLAMA_BATCH_ROWS=2..8` explicitly allocates independent native sequence
slots; default `1` exposes no batch path and preserves the original context.
This multiplies KV/recurrent capacity and increases graph scratch memory, even
when an individual call uses fewer rows. Native equal-length batches assign
each token one sequence ID and its own zero-based position, so attention and
recurrence cannot cross requests. Selected pointer/LM readouts use their
original row positions and candidate order. Each call clears all native state;
saved immutable prefix snapshots survive batch/independent interleaving.

The engine additionally honors this runtime's configured sequence limit and
256 charged readouts across the entire call (candidate positions plus a possible
final decision per row). It splits groups before those bounds and the caller's
token budget, preserving oversized singletons. No padding or cached branch
batching is exposed. Physical work is one native prefill per batch, with unchanged
logical wire usage. The sequence count and native profile enter qualification
metadata and environment identity; batch serving still requires actual nonzero
batches, paired/external gates and complete held-out observed labels.

```bash
HUNCHO_DEVICE=cpu HUNCHO_LLAMA_BATCH_ROWS=4 huncho conform \
  --model /path/to/gguf-package --backend llamacpp \
  --golden /path/to/unchanged-heldout.json --max-batch-tokens 8192 --json
```

```bash
HUNCHO_DEVICE=cpu HUNCHO_LLAMA_THREADS=4 huncho conform \
  --model /path/to/gguf-package --backend llamacpp \
  --golden /path/to/unchanged-heldout.json --prefix-cache --json
# Optional retention also requires nonzero replay hits during conformance.
HUNCHO_DEVICE=cpu huncho conform --model /path/to/gguf-package \
  --golden /path/to/unchanged-heldout.json --prefix-cache \
  --persistent-prefix-bytes 16777216 --json
```

Prefix serving remains opt-in and requires the unchanged external, paired and
complete observed-label startup gates for every context. Numerical fixture
checks grant no serving acceptance or released-checkpoint performance claim.

## New-package conversion

Use a clean checkout at the exact pinned revision and a separate Python
installation with the upstream converter's CPU dependencies:

```bash
huncho export-llamacpp --model /path/to/original-kev-package \
  --output /path/to/new-gguf-package --dtype gguf-f32 \
  --tool-dir /path/to/pinned-llama.cpp --python /path/to/cpu-venv/bin/python
```

`gguf-f16`, `gguf-q8_0` and `gguf-q4_0` are also supported. Q8/Q4 exports first
produce FP32 GGUF, then call the same pinned native CPU quantizer with
`--quantization-threads 1..256` (default 1). Embeddings, vocabulary output, norms,
convolution and projections whose rows do not fit a 32-element block remain
FP32. Exact per-tensor overrides prevent an implicit FP16 fallback. Eligible
projections use only the declared Q8_0 or Q4_0 type; there is no importance matrix,
layer pruning or re-quantization. Native quantizer parameters request a 64 MiB
slab target; a complete row is the minimum unit and total RSS is not bounded
by that parameter. Source pins, the original
tokenizer and prompt contract remain in the new
manifest. F2 copies the trained pointer head; Bias-bearing projections, unequal linear key/value head dimensions and an F3
LM-head bias are outside this pinned export profile. F3 retains its trained vocabulary
weights inside GGUF. The exporter uses the existing CPU FP32 LoRA merge, then
the pinned official converter for GGUF tensor names, zero-centered norms,
DeltaNet parameters and grouped-to-tiled value-head order. It disables MTP and
network resolution. For F2 only, GGUF's unused auxiliary LM head shares embeddings.
Inference continues to use Huncho's original HF tokenizer, never native GGUF
retokenization.

The destination must not exist. Source/base/adapter/tokenizer inputs and tracked
converter files are hashed before and checked after conversion. Provenance also
records Python package versions, exporter/interpreter identities, native
quantizer/runtime/kernel identity, retained shape overrides, FP32 intermediate
hash and final artifact
hashes. These are unsigned audit records; they do not prove the integrity of
every installed Python module. Dense temporary HF staging is removed after a
successful native load check, together with the owned intermediate FP32 GGUF
for quantized exports. On failure partial output and `conversion.log`
remain for inspection, without a published manifest. Conversion can temporarily
retain both dense weights and GGUF; payload bytes are not peak RSS.

New calibration entries start **pending**, without inherited temperatures or
optimized goldens. Collect raw fitting logits with `capture-logits --backend
llamacpp`, refit the exact `llamacpp:gguf-*` entry using `calibrate`, then run
`conform` on separate unchanged held-out golden vectors with observed outcomes.
`serve` requires fresh complete labeled conformance for this native execution,
even if a supplied manifest labels its temperature `fit`. Quantized serving also
requires an explicit exact backend/dtype `refit` entry. Numerical tolerances,
argmax agreement and ECE-drift thresholds are unchanged. A successful conversion
or an unlabeled numerical receipt grants no serving acceptance.

## CPU validation and limits

The checked-in GGUF is generated from the existing two-layer tiny Kev fixture
with its nonzero LoRA updates. Its fixed upstream goldens remain unchanged.
CPU FP32 tests cover all existing typed requests, duplicates and out-of-order
readouts, independent resets, a full 512-token prompt, trained pointer scores,
candidate-logit selection and concurrent shared-weight contexts. The largest
frozen-suite raw-logit delta is `8.94e-8`; probability delta is `2.98e-8` at the
original temperature `2.40605`. F16 probability delta is `4.36e-6`, with matching
argmax on the same frozen fixture. These random fixture weights establish numerical
integration, not calibration or speed of released Kev-4B.

The fixture generator overrides only metadata for its tiny BPE vocabulary,
whose fingerprint is absent from upstream's registry. Production export uses
the unmodified converter. See [fixture regeneration](../crates/huncho-backend/tests/fixtures/llamacpp/README.md).
The unmodified production converter also completes FP32/F16 export and native
load checks with the released Kev tokenizer and a synthetic tiny backbone whose
embedding table is padded to that vocabulary. Both packages retain pending
calibration and unchanged source hashes; [CPU export evidence](verification/llamacpp-cpu-20261008/summary.json)
records identities. This is a packaging check, not released-model inference.

CPU Q8/Q4 tests quantize the original FP32 fixture through the real native
routine, run actual packed projections with the untouched pointer head, and
verify deterministic independent calls/replicas and rejection of overwrite,
invalid precision and thread bounds. Block-incompatible matrices stay FP32.
These unrefitted numerical diagnostics do not establish observed calibration.
[Q8/Q4 export evidence](verification/llamacpp-quant-cpu-20261008/summary.json)
also records successful official conversion/native loads with the synthetic
backbone and released tokenizer, pending entries and rejection of unrefitted
serving. This is a packaging check with only two packed fixture projections.

FP32/FP16 native batch tests cover distinct sequences, final/earlier/repeated
readouts, four full-length 512-token prompts, row reversal, isolated replicas,
pointer and selected/full F3 logits, prefix replay after batches, invalid rows
and allocation bounds. CLI conformance repeats every unchanged upstream fixture
case under a distinct ID to exercise cross-request equal shapes; it retains
unlabeled diagnostic status and rejects invalid slot configuration. This checks
native integration and arithmetic, not observed outcome calibration.

No full released-model fit/held-out qualification, CPU performance claim or
GPU/Apple check is included. Larger contexts, graph-side
candidate projection, ONNX-independent heads and additional model families
remain separate roadmap work.
