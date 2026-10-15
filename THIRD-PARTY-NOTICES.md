# Third-Party Notices

Huncho's own source code is licensed under the Apache License 2.0 (see `LICENSE`).
The distribution bundles or links components that carry their own licenses.

## Vendored and patched

- **candle-kernels 0.11.0** (`vendor/candle-kernels/`) — Hugging Face Candle
  CUDA kernels, licensed **MIT OR Apache-2.0**. Patched by us (see
  `vendor/candle-kernels/HUNCHO-PATCH.md` for the exact diff scope and the
  upstream archive SHA-256). Original license texts:
  `vendor/candle-kernels/LICENSE-APACHE` and `vendor/candle-kernels/LICENSE-MIT`.

## Bundled runtimes

- **ONNX Runtime 1.28.0** — **MIT** license. Downloaded during the build
  (prebuilt release from the ONNX Runtime GitHub organization). In EL9 RPM
  builds the shared library is installed alongside the executables.
- **tokenizers / Candle / ONNX crates and all other crates.io dependencies** —
  their respective permissive licenses; the full resolved set is recorded in
  `Cargo.lock`.

## Optional build-time backends

- **llama.cpp** — **MIT** license. Only used when the optional `llamacpp`
  runtime feature is enabled at build time.
- **vLLM** — **Apache-2.0** license. Only used when the optional `vllm`
  runtime is configured; Huncho talks to it over a private Python shim and
  never links against it.

## Branding

The Huncho brand assets under `brand/` are provided for project identification
as described in `brand/README.md`.

## Notice

The SPDX license expression used for binary packages is:

```
Apache-2.0 AND MIT
```

("Apache-2.0" for Huncho's code; "MIT" for the bundled MIT-licensed runtime
components. Dual-licensed components are used under Apache-2.0.)
