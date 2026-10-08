# Experimental CPU Kev quantization

Build with `--features quantization,tokenizers` (or `quantization,clef`). The
default build adds no quantization, Candle, hashing or tokenizer dependencies.
Current conversion supports F2 `kev-v1` only; GPU checks and Apple platforms
are deferred. No packed profile is qualified for the released Kev checkpoint.

`huncho quantize` writes a new merged text-backbone GGUF. The `q8_0-fp32` and
`q4_0-fp32` dtypes use Q8_0 and Q4_0 blocks for all MLP and dense/linear-attention
projection weights. Biases, embeddings, normalization, causal convolution,
activations, recurrence state and the trained `head.pt` pointer projection stay
FP32. LoRA is merged in FP32 before quantization. Projection input widths are
padded to 32 locally if necessary; zero columns never become tokens or enter
positions/attention. Embedded configuration and original widths validate layout.

The CPU path uses Candle's packed `QMatMul::QTensor` directly. Its block dot
products also quantize activation blocks internally. FP32 activation tensors
therefore do **not** imply unchanged arithmetic. Environment-controlled dense
dequantization cannot silently substitute another kernel. This is real packed
weight/kernel quantization. The GGUF profile uses Huncho metadata and tensor
names; it is not an ordinary llama.cpp generation model.
[Candle's pinned implementation](https://github.com/huggingface/candle/blob/0.11.0/candle-core/src/quantized/mod.rs)
defines the packed kernels and block layouts.

Conversion refuses an existing output directory and leaves source files
unchanged. It hashes all selected source inputs before conversion and rechecks
them before publishing a manifest. The new package carries unchanged tokenizer,
pointer head and prompt contract, a packed artifact, source pins and conversion
provenance. Only the new Candle dtype artifact remains. Calibration is reset to
`pending`, without inherited type/cardinality temperatures or an old evaluation
hash; provenance records `qualified=false`. Failure can leave partial output
without a manifest; inspect it and choose a new path for a retry.

```sh
HUNCHO_DEVICE=cpu huncho quantize --model /path/to/pinned-kev-package \
  --output /path/to/new-kev-q8-package --dtype q8_0-fp32
```

Collect variant logits using only the fitting split, then refit and run unchanged
independent goldens with a separate, complete observed-outcome evaluation split.
`capture-logits` runs offline, allows pending calibration for analysis, and
exports raw fitting logits without probability goldens or HTTP serving. Each
JSONL record needs an `id`, `request` and `targets` mapping every question ID to
its observed candidate label. It accepts fitting JSONL from
`scripts/prepare_kev_calibration.py`; records may also have multiple questions.
Choice labels follow caller order, score labels are zero-based strings, and Kev
noul follows `[no,yes]`. Collection disables response/prepared-prompt caches,
uses independent forwards, retains work and artifact/binary/runtime identities,
and rechecks inputs/execution before completing `fit.json` and the audit.
Interrupted collection can leave a raw-logit checkpoint without a final audit.

```sh
HUNCHO_DEVICE=cpu huncho capture-logits --model /path/to/new-kev-q8-package \
  --dtype q8_0-fp32 --data /path/to/fitting.jsonl --output /path/to/new-fit-audit
huncho calibrate --manifest /path/to/new-kev-q8-package/huncho-model.json \
  --backend candle --dtype q8_0-fp32 --data /path/to/new-fit-audit/fit.json
HUNCHO_DEVICE=cpu huncho conform --model /path/to/new-kev-q8-package \
  --backend candle --dtype q8_0-fp32 --golden /path/to/unchanged-heldout.json \
  --write-qualification /path/to/new-receipt.json --json
HUNCHO_DEVICE=cpu huncho serve --model /path/to/new-kev-q8-package \
  --backend candle --dtype q8_0-fp32 \
  --qualification-golden 'REGISTERED_MODEL_NAME=/path/to/unchanged-heldout.json'
```

Serving requires an exact `candle:q8_0-fp32` / `candle:q4_0-fp32` entry with
`status=refit`. An inherited default or source `fit` entry is insufficient.
Startup requires fresh complete labeled conformance: delta <=1e-3, complete
argmax agreement and ECE drift <=0.02. Scheduling/caching also require paired
independent delta <=1e-4 and complete argmax agreement. A refit that cannot pass
stays unserved; thresholds are not relaxed for a smaller artifact.
`scripts/qualify_kev_runtime.py` accepts both packed identities and verifies the
actual artifact, mixed precision and kernel metadata. Numerical-only runs remain
diagnostic. Receipts hash the packed artifact, trained head and tokenizer, but
cannot replace startup qualification or establish dataset independence alone.

Conversion and initial loading temporarily materialize dense weights.
Steady-state projection modules release dense projection tensors and retain
packed blocks; embeddings/head remain dense. Payload stats exclude peak RSS,
allocator/workspace, bias, head, tokenizer and caches. No full-model latency,
memory or cost improvement is claimed. CPU tests execute both packed kernels
on the tiny Kev fixture, verify changed logits, finite distributions, immutable
forks, persistent hits, native batch isolation, malformed artifact rejection,
source immutability, no overwrite, offline target alignment and mandatory
refit/startup gates. They do not qualify released Kev or replace upstream
reference goldens with optimized output.
