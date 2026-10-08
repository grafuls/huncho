# CPU llama.cpp decision backend

The optional `llamacpp` feature loads standard dense Qwen3.5 GGUF weights for
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
replicas when budgeting CPU capacity. This profile exposes no prefix forks,
persistent prefixes, native request batches or multi-adapter attachment.

## New-package conversion

Use a clean checkout at the exact pinned revision and a separate Python
installation with the upstream converter's CPU dependencies:

```bash
huncho export-llamacpp --model /path/to/original-kev-package \
  --output /path/to/new-gguf-package --dtype gguf-f32 \
  --tool-dir /path/to/pinned-llama.cpp --python /path/to/cpu-venv/bin/python
```

`gguf-f16` is also supported by this command. Q8/Q4 conversion through this
runtime is still open; accepting an externally prepared artifact does not qualify
it. Source pins, the original tokenizer and prompt contract remain in the new
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
records Python package versions, exporter/interpreter identities and artifact
hashes. These are unsigned audit records; they do not prove the integrity of
every installed Python module. Dense temporary HF staging is removed after a
successful native load check. On failure partial output and `conversion.log`
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

No full released-model fit/held-out qualification, CPU performance claim or
GPU/Apple check is included. Larger contexts, prefix/fork ownership, graph-side
candidate projection, ONNX-independent heads and additional model families
remain separate roadmap work.
