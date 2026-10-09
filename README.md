# Huncho

Huncho is a Rust server for decision models. Give it text or JSON and a set of
questions, and it returns structured answers with calibrated probabilities.
For example, you can ask which team should handle a support ticket or whether
a customer is requesting a refund.

It runs Jev-style "System One" models: models that score possible answers in a
forward pass, with no text generation or token-by-token decode loop.

```text
state + questions  ->  Huncho  ->  answers + probabilities
```

Huncho implements the [Jev `/v1/systemone` API](docs/API.md). You can use the
TypeSafe Python SDK by pointing its base URL at your Huncho server.

## Installation

### Fedora and EPEL 9

On supported x86_64 systems, install from the
[Huncho COPR repository](https://copr.fedorainfracloud.org/coprs/quadsdev/huncho/):

```bash
sudo dnf copr enable quadsdev/huncho
sudo dnf install huncho
```

The RPM includes ONNX and native CPU/CUDA runtimes. Kev and Clef use a compatible
NVIDIA GPU when available and otherwise run on CPU. CPU hosts need no NVIDIA
packages. See [GPU setup](docs/gpu-setup.md) for driver and runtime requirements,
or [RPM packaging](packaging/rpm/README.md) to build your own package.

### From source

You'll need a current stable Rust toolchain with Cargo, Git, a C/C++ compiler,
`pkg-config`, and OpenSSL development headers.

```bash
git clone https://github.com/grafuls/huncho.git
cd huncho
cargo install --locked --path crates/huncho-cli --features onnx,clef
```

This installs the CPU runtimes for ONNX, Laya, Kev, and Clef, including Hub
downloads and native tokenization. Cargo installs `huncho` in `~/.cargo/bin`;
make sure that directory is on your `PATH`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

Add that line to your shell's startup file to keep it for future terminals.
The first installation downloads dependencies and a prebuilt ONNX Runtime.
For a mock-only installation, omit `--features onnx,clef`. CUDA builds also
require the NVIDIA CUDA toolkit; see the [build guide](docs/operations.md#build)
and [backend guide](docs/backends.md) for optional runtime features.

Check that Huncho is installed:

```bash
huncho --version
```

## Quick start

After installing Huncho, you'll need `curl` for the example requests.

Start a server with the built-in mock model:

```bash
huncho serve --mock --bind 127.0.0.1:8080
```

The mock needs no model downloads and returns deterministic test values. Use
it to try the API and check your integration; its answers don't reflect the
meaning of your input.

In another terminal, check which models are available:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

Then ask two questions about a customer message:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The customer wants a refund because the shoes are too small.",
    "model": "mock",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team handles this?",
        "criteria": { "returns": "Refund request", "billing": "Charging issue" }
      },
      "is_refund": {
        "type": "noul",
        "instructions": "Is the customer requesting a refund?"
      }
    }
  }'
```

The mock returns:

```json
{
  "model": "mock",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.5034228, "returns": 0.4965772 },
      "confidence": 0.0068455935
    },
    "is_refund": { "type": "noul", "noul": 0.4965772 }
  },
  "usage": { "input_tokens": 54, "output_tokens": 0 }
}
```

Each answer uses the question ID you supplied. The API supports these types:

| Type | Use it for | Result |
|---|---|---|
| `choice` | Choosing among named options, such as a support team. | Chosen label, probabilities for all options, and confidence. |
| `noul` | A yes/no question. | Probability that the answer is yes. |
| `score` | Rating something on ordered levels, such as low/medium/high. | Per-level probabilities, a probability-weighted score, and confidence. |

See the [API guide](docs/API.md) for request formats, error responses, and
optional response details.

## Run a real model

Huncho can load a local model package or resolve one from Hugging Face. A
package's `huncho-model.json` describes its weights, decision head, prompt
format, and calibration settings. See the
[package format](docs/model-package.md) if you're preparing your own model.

The RPM and source installation above include the runtimes for the examples
below. Check the setup guide for your model:

| Model or runtime | Setup guide |
|---|---|
| ONNX package | [ONNX backend](docs/backends.md#onnx-feature) |
| Laya / ModernBERT or Kev / Qwen safetensors | [Native backends](docs/backends.md#candle-feature) |
| Cloudflare Clef | [Clef](docs/clef.md) |
| Optional CPU llama.cpp or vLLM runtime | [llama.cpp](docs/llamacpp.md), [vLLM](docs/vllm.md) |

The optional llama.cpp and vLLM backends require the installation steps in
their guides.

### Before serving: check the model's probabilities

With a calibrated model, events assigned an 80% probability should happen about
80% of the time. Huncho applies the model's calibration settings and checks
that the serving runtime preserves its reference behavior.

Every real serving runtime must pass fresh startup conformance checks. Supply
a pinned evaluation file containing reference probabilities and observed
outcomes for every question:

```text
--qualification-golden 'MODEL_NAME=/path/to/pinned-labeled-golden.json'
```

Replace `MODEL_NAME` with the name in the model manifest. Keep this evaluation
set separate from the data used to fit calibration. Huncho checks probability
differences, agreement on the chosen answer, and calibration error drift; it
refuses to serve a model that fails. A manifest marked `fit` doesn't skip these
checks. Some batching and prefix optimizations also require comparisons
against independent forward passes.

The [calibration guide](docs/calibration.md) explains fitting, evaluation files,
and acceptance limits. Numerical-only `conform` and `bench` runs are available
for analysis, but don't establish that a model is ready to serve.

### Local safetensors package

For a prepared Laya package:

```bash
huncho serve --manifest my-laya/huncho-model.json \
  --qualification-golden 'MODEL_NAME=/path/to/pinned-labeled-golden.json'
```

If you have a Laya checkpoint checkout, follow the
[local checkpoint conversion guide](docs/backends.md#assembling-a-package-from-a-local-checkpoint)
to prepare the package with `huncho convert`.

### Kev-4B from Hugging Face

```bash
huncho serve --model jaredpalmer/kev-4b \
  --qualification-golden 'kev-4b=/path/to/pinned-labeled-golden.json'
```

The first load downloads the adapter, tokenizer, pointer head, and pinned base
checkpoint. Use `"model": "kev-4b"` in requests. The native path defaults to
FP32 on CPU and FP16 on CUDA; `--dtype` overrides it. Use `--revision <commit>`
to select a specific Hub revision.

**Current limit:** the retained CPU FP32 held-out profile fails the probability
difference check and is not accepted for serving. See
[Kev support and validation](docs/backends.md#kev-f2-on-candle) before choosing
an execution profile.

### Cloudflare Clef

```bash
huncho serve --model Cloudflare/clef \
  --qualification-golden 'MODEL_NAME=/path/to/pinned-labeled-golden.json'
```

Clef uses a native Rust/Candle runtime and scores all questions together. It
needs no Python runtime or model conversion. Read the
[Clef guide](docs/clef.md) for memory requirements and current validation limits.

Huncho selects the backend from each model's package. Use `--backend` to
override that choice; if the required runtime is missing, the CLI tells you
which build feature to enable. The supported decision heads are F1
(option-marker), F2 (pointer), F3 (candidate-logit), F4 (slot), and F5 (joint
schema), all behind the same API.

## Performance

Huncho scores answers without generating tokens. Depending on the model and
backend, you can also reduce repeated work or serve requests concurrently:

- **Batch questions:** combine compatible rows in one model call, with bounded
  padding where supported. See [batch controls](docs/operations.md#serve-flags)
  and [cross-request batches](docs/operations.md#cross-request-batches).
- **Reuse shared state:** native CPU Kev can process a state once and reuse it
  across question branches or requests. See
  [Kev prefix batches](docs/operations.md#cpu-kev-batches-from-a-shared-prefix).
- **Cache repeated inputs:** opt-in caches reuse tokenization, prompts, and
  complete responses within configured memory budgets. The HTTP server can
  also combine identical in-flight requests. See
  [cache controls](docs/operations.md#serve-flags).
- **Run CPU replicas:** supported runtimes share immutable weights across
  independent serving contexts. See
  [replicas](docs/operations.md#bounded-shared-weight-cpu-replicas) and
  [shared bases across adapters](docs/operations.md#cpu-fp32-runtime-lora).
- **Tune CPU execution:** optional OpenBLAS and family-specific head and
  attention optimizations reduce computation and allocations. See
  [backend profiles](docs/backends.md).

Most tuning controls are opt-in, and each serving context must pass the
calibration checks. Performance depends on your workload. The
[optimization roadmap](docs/optimization-roadmap.md) records measurements,
implemented features, and remaining qualification work. Experimental CPU
Q8_0/Q4_0 packages require a calibration refit and fresh labeled conformance;
see [quantization](docs/quantization.md).

For inference in the browser, the separate [CPU WASM package](browser/README.md)
supports compatible F1 ONNX graphs and dedicated workers that move inference
off the page thread. It has its own packaging and qualification requirements.

## Development

Run these checks from the root of a source checkout. The default build can run
without model weights:

```bash
cargo test

# Compare the example package with its reference vectors using the mock backend.
huncho conform \
  --manifest examples/mock-model/huncho-model.json --backend mock \
  --golden examples/mock-model/golden.json
```

To check optional backends and the native tokenizer:

```bash
cargo test -p huncho-backend --features onnx
cargo test -p huncho-backend --features candle
cargo test -p huncho-core --features tokenizers --test hf_tokenizer
```

The [mock model package](examples/mock-model/README.md) includes an ONNX fixture
for testing numerical agreement. It doesn't establish real-model calibration.

| Crate | Responsibility |
|---|---|
| `huncho-core` | Request/response types, model packages, prompts, heads, calibration, and conformance. |
| `huncho-backend` | Mock and optional inference runtimes. |
| `huncho-api` | HTTP endpoints, authentication, and metrics. |
| `huncho-hub` | Hugging Face model package resolution. |
| `huncho-cli` | `serve`, `convert`, `calibrate`, `conform`, and `bench` commands. |

## Deployment and further reading

With the [RPM package](packaging/rpm/README.md), set `HUNCHO_DEVICE=cpu` to force
CPU, or `cuda` / `cuda:N` to require a GPU. See [GPU setup](docs/gpu-setup.md).

The repository also includes a Dockerfile and systemd service examples. The
[operations guide](docs/operations.md) covers deployment, environment variables,
authentication, health checks, and Prometheus metrics.

| Guide | Use it to |
|---|---|
| [API](docs/API.md) | Build requests and interpret answers. |
| [Model packages](docs/model-package.md) | Prepare and inspect model artifacts. |
| [Backends](docs/backends.md) | Choose a runtime and check its limits. |
| [Calibration](docs/calibration.md) | Fit temperatures and qualify a model for serving. |
| [Operations](docs/operations.md) | Configure, deploy, and monitor Huncho. |

## License

Apache-2.0.
