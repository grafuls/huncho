# Optional CPU browser runtime

This separate package executes an actual ONNX graph with ONNX Runtime Web
1.30.0's CPU WASM provider. Shared `huncho-core` WASM builds prompts, applies
the package's unchanged temperatures/softmax and returns the Jev typed
response. There is no sampler or decode loop. Native Cargo's default build
does not include this package or its browser dependencies.

This increment accepts F1 FP32 packages with a **graph-integrated scalar option
head**. The graph includes the trained head; a generic hidden-state encoder
export is insufficient. This does not automatically export Laya's `act_head`
or qualify released Kev. All weights must be embedded in one ONNX file.
External weights, quantization, other families, batching and browser prefixes
remain open. Apple work is skipped; WebGPU and actual GPU checks are deferred.

## Build and package

Use Rust 1.97 or later with the `wasm32-unknown-unknown` standard library,
`wasm-bindgen` CLI **0.2.129**, and Node 22 or later:

```sh
cd browser
npm ci --ignore-scripts
sh scripts/build.sh
node scripts/package.mjs \
  --manifest /path/to/huncho-model.json \
  --tokenizer /path/to/tokenizer.json \
  --model /path/to/integrated-head.onnx \
  --golden /path/to/held-out-golden.json \
  --out /path/to/new-browser-bundle
```

The tool requires a new destination, snapshots input bytes, hashes all model,
core, SDK and runtime assets, and prints the descriptor SHA-256. Packaging sets
`qualified=false`; acceptance cannot be inherited. Serve the bundle on HTTPS
or localhost with JavaScript/WASM MIME types and a policy permitting verified
Blob module imports and WASM execution. Protect the application/SDK as trusted
code. Obtain the descriptor hash from your trusted build output.

```js
import { HunchoBrowser } from '/models/example/index.mjs';

const engine = await HunchoBrowser.loadPackage({
  url: '/models/example/config.json',
  sha256: '<descriptor SHA-256 from packaging>',
});
const response = await engine.eval({
  model: '<exact package name>', state: 'Customer wants a refund.',
  questions: { team: { type: 'choice', instructions: 'Which team?',
    criteria: { billing: 'Charges', returns: 'Refunds' } } },
});
console.log(response.answers);
await engine.dispose();
```

`loadPackage` resolves assets relative to the descriptor and checks the
executing SDK source hash. `load(config)` accepts explicitly hashed absolute
asset URLs for applications with a separately trusted/bundled SDK. One page
uses one immutable core/ORT build per SDK module. Keep the SDK a single module:
duplicating it creates separate WASM runtimes and additional residency.

## Graph and serving contract

| Name | Type and shape | Meaning |
|---|---|---|
| `tokens` | INT64 `[1, sequence]` | Complete shared-core question prompt |
| `positions` | INT64 `[markers]` | Sorted unique option-marker positions |
| `qtype` | INT64 `[1]` | Original core question-type index |
| `attention_mask` | Optional INT64 `[1, sequence]` | All ones; no padding |
| `scores` | FP32 `[markers, 1]`, sole output | Raw trained-head score at each supplied position |

Sequence/marker axes must be dynamic. Input names/types/ranks and output shape
are validated. Nonfinite scores fail before calibration. The graph owns type
routing; this ABI performs no vocabulary pooling, host MLP or mean fallback.
Only the core maps scores back to labels. Question/criterion order, score
legends/confidence, noul, per-type/option-bucket temperatures, logical input
usage and zero output tokens retain the shared behavior. `eval(request,
{extensions:true})` opts into existing extensions; `evalWithStats` additionally
returns physical work separately.

Serving requires an explicit fitted/refitted `onnx:fp32` entry and complete
observed targets/reference vectors for **every** golden question. Loading runs
every question through a fresh actual ORT session. Shared fixed gates require
delta <=0.001, complete argmax agreement and ECE drift <=0.02. No public `eval`
instance is returned on failure, including fitted packages. Qualification and
serving use no response/prompt cache; every new session reruns conformance.

`engine.qualification` returns a report copy with the browser user agent,
asset hashes, exact runtime version, CPU provider, single-thread fixed SIMD
and graph-optimization profile. Reports are unsigned and cannot attest dataset
provenance. Supply a trusted independent held-out suite with real outcomes;
synthetic labels or self-generated vectors cannot release a model. The gate
measures preservation relative to the reference. Reference drift causes
rejection, never golden replacement or widened thresholds.

Limits: context <=4096, <=64 questions, <=255 declared options, <=8 active/queued
evaluations, request/result JSON <=1 MiB. Manifest/tokenizer/model/golden limits
are 1/16/64/32 MiB; core/ORT WASM limits are 32/64 MiB. Large released checkpoints
can exceed them. Byte snapshots, JSON transport, tokenizer storage, runtime
buffers and activations increase peak memory beyond artifact size. Evaluations
are serialized and snapshot input on submission. Failures release plans/tensors.
`dispose()` rejects new work and drains existing work before session release.
There is no in-flight kernel interrupt. CPU work runs on the calling thread;
applications may host the SDK in their own Worker. An HTTP server, dedicated
worker protocol and multithread tuning are separate increments.

## CPU-only tests

With optional `onnx==1.20.1` and NumPy tooling, and an existing Chrome executable:

```sh
python tests/generate_fixture.py
cargo run --locked --manifest-path wasm/Cargo.toml --example reference -- tests/generated
sh scripts/build.sh
npm test
```

`HUNCHO_TEST_BROWSER` overrides `/usr/bin/google-chrome`. The harness disables
GPU, software rasterization and WebGL and selects only CPU WASM. It compares
actual graph execution to independent native Rust fixture arithmetic/shared
answers for all types, JSON/unicode/long states, and exact tokenizer inputs.
It checks gate/signature/hash refusal, plan cleanup, queues, snapshots,
disposal and actual bundled deployment. Generated artifacts/dependencies are
ignored. Fixed synthetic weights/labels test implementation only; they supply
no released calibration acceptance, isolated speedup or memory measurement.

Runtime interfaces follow [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
and its pinned source; glue uses
[wasm-bindgen](https://wasm-bindgen.github.io/wasm-bindgen/reference/cli.html).
