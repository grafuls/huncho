# Model package format

Experimental CPU Kev packages may carry a merged GGUF artifact with dtype
`q8_0-fp32` or `q4_0-fp32` and its exact `kev-projections-*-fp32-v1` quantization
profile. These artifacts embed text configuration and projection layout; the
package also retains the trained pointer head and tokenizer. See the
[conversion and calibration workflow](quantization.md).

Standard dense Qwen3.5 artifacts for the optional `llamacpp` runtime use separate
`gguf-f32`, `gguf-f16`, `gguf-q8_0` or `gguf-q4_0` labels. They are incompatible
with the Candle packed-projection format above. See [CPU GGUF export and pending
calibration](llamacpp.md); external quantized layouts need their exact profile.

The optional CPU `vllm` artifact uses dtype `bf16` and the pinned
`kev-pointer-vllm-cpu-bf16-v1` descriptor. FP32 merged backbone and pointer
storage execute as BF16/FP32 respectively. Offline export always writes an
explicit Pending `vllm:bf16` calibration entry; source temperatures and DEFAULT
fallback do not authorize serving. See [export, fitting and limits](vllm.md).

A model package is a single `huncho-model.json` manifest plus artifacts. Every
backend can load it. The manifest pins everything the engine needs to serve a
model deterministically: family, backbone, head, prompt contract, calibration,
and the reference conformance vectors.

```json
{
  "schema_version": "1.0",
  "name": "mock-laya",
  "family": "F1",
  "backbone": {
    "source": { "kind": "hf", "repo": "convaiinnovations/laya", "revision": "mock-reference" },
    "artifacts": { "onnx": [{ "path": "mock-model.onnx", "dtype": "fp32" }] },
    "hidden_size": 1024,
    "max_context": 8192,
    "tokenizer": "tokenizer.json"
  },
  "head": { "kind": "option-marker", "weights": "mock-head.safetensors", "width": 1 },
  "prompt_contract": {
    "template": "laya-v1",
    "option_marker_tokens": ["<option:0>"],
    "state_budget": 3072,
    "head_budget": 1024,
    "max_options": 255,
    "contract_hash": "f1-mock-contract-hash"
  },
  "calibration": {
    "default": { "temperature": 1.0, "confidence": "peak", "status": "fit" },
    "entries": { "onnx:fp32": { "temperature": 1.0, "confidence": "peak" } },
    "eval_set_hash": "mock-eval-0001"
  },
  "reference": { "family_impl": "huncho-mock-reference", "revision": "main", "golden": "golden.json" },
  "capabilities": { "supports_fork": false, "supports_multi_lora": false }
}
```

## Fields

### `family`

`F1 | F2 | F3 | F4 | F5` — selects the prompt builder and the expected head kind.

- **F1** (Encoder) — scores candidates at option-marker positions. Head kind:
  `option-marker`.
- **F2** (Pointer) — pointer head over option-boundary tokens; block-causal
  fan-out. Head kind: `pointer`.
- **F3** (Candidate-logit) — softmax over one-token answer codes using the LM
  head. Head kind: `candidate-logit`.
- **F4** (Slot head) — fixed-width decision head. Head kind: `slot`.
- **F5** (Joint schema) — one shared forward for all questions. Head kind:
  `joint-schema`. Clef packages are generated from the reference release; see
  [Clef](clef.md). Their `clef-native-v1` template and contract identifier select
  the native encoder; the source revision pins model weights. No repository
  Python code is executed.

The manifest is validated so `head.kind` matches `family`; a mismatch is
rejected at load time.

### `backbone.source`

- `{ "kind": "hf", "repo": "...", "revision": "..." }` — a pinned Hugging Face
  repo and revision. Pinning a revision is how upstream drift is contained.
- `{ "kind": "local", "path": "..." }` — a local pre-converted backbone.

### `backbone.artifacts`

Per-backend artifacts, keyed by backend id (`onnx`, `candle`, `clef`,
`llamacpp`, `mlx`, `vllm`). Each artifact has a relative `path` and the `dtype` it provides
(`fp32`, `fp16`, `int8`, `q4`). `huncho convert` produces these. The CLI uses
these entries and the decision family to select an available runtime
automatically; users can override it with `--backend`.

### `head`

- `kind`: matches the family.
- `weights`: relative path to the head weights (always fp32).
- `width`: output width of the head logits (default `1`).
- `pointer_offset` (optional): used by pointer heads.

### `prompt_contract`

- `template`: the template id (e.g. `laya-v1`, `kev-block-causal`).
- `option_marker_tokens`: marker tokens that anchor F1 candidate positions.
- `state_budget` / `head_budget`: token budgets for the shared state and the
  per-question head. These and `max_context` are enforced (CORE-08); oversized
  input is rejected, never silently truncated.
- `max_options`: maximum options the contract allows.
- `contract_hash`: a hash of the prompt contract bytes, used to detect upstream
  drift.

### `calibration`

- `default`: the backend-agnostic entry used when no `{backend}:{dtype}` key
  matches.
- `entries`: per-`{backend}:{dtype}` overrides.
- `eval_set_hash`: the hash of the eval set the temperatures were fitted on.

Each entry has:

- `temperature`: the softmax temperature.
- `per_type_temperatures` (optional): per question-type temperatures (used by
  F4 / per-type calibration).
- `confidence`: `peak` (Jev) or `entropy` (Laya), or `custom:<name>`.
- `status`: `fit`, `refit`, or `pending`. Quantized variants that fail
  conformance ship with `refit` temperatures or are rejected.

### `reference`

Pointer to golden conformance vectors. `family_impl` names the reference
implementation, `revision` pins it, and `golden` is the relative path to the
golden suite that `huncho conform` consumes.

### `capabilities`

Model-level capability flags: `supports_fork` and `supports_multi_lora`.

## Example

A runnable, deterministic mock model package lives in
[`examples/mock-model`](../examples/mock-model) (`huncho-model.json` + `golden.json`)
and is used by the offline conformance suite and the CLI demos.

## Conversion

`huncho convert` produces backend artifacts plus the manifest from an HF repo and
revision (CONV-01). `huncho calibrate` fits temperatures for a given backend × dtype
on a held-out set and writes them back into the manifest (CONV-02).
