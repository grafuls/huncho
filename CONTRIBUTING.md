# Contributing to Huncho

Thank you for your interest in contributing to Huncho! This document outlines the steps to set up your development environment, build, test, and submit changes.

For general project information, see [README.md](README.md). For a deeper understanding of the system design, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Prerequisites

To build Huncho from source, you need:
- A current stable Rust toolchain with Cargo.
- Git.
- A C/C++ compiler.
- `pkg-config`.
- OpenSSL development headers.

## Building

The default build can run without model weights. To install the CLI with CPU runtimes for ONNX, Laya, Kev, and Clef (including Hub downloads and native tokenization), use:

```bash
cargo install --locked --path crates/huncho-cli --features onnx,clef
```

Optional backends are enabled via Cargo feature flags. The table below lists features defined in the crates' `Cargo.toml` files.

| Crate | Feature | Description |
|---|---|---|
| `huncho-core` | `tokenizers` | Enables native tokenization. |
| `huncho-core` | `external-scores` | Enables the external-scores engine path (`engine/external_scores.rs`). |
| `huncho-backend` | `vllm` | Enables vLLM backend support. |
| `huncho-backend` | `onnx` | Enables ONNX runtime support. |
| `huncho-backend` | `onnx-shared` | Enables shared ONNX runtime support. |
| `huncho-backend` | `onnx-cuda` | Enables ONNX runtime support with CUDA. |
| `huncho-backend` | `candle` | Enables Candle framework support for loading HF safetensors. |
| `huncho-backend` | `quantization` | Enables quantization support (requires `candle`). |
| `huncho-backend` | `shared-base` | Enables shared base support (requires `candle`). |
| `huncho-backend` | `cpu-blas` | Enables CPU BLAS support (requires `candle`). |
| `huncho-backend` | `llamacpp` | Enables llama.cpp support (requires `candle`). |
| `huncho-backend` | `clef` | Enables Cloudflare Clef support (requires `candle`). |
| `huncho-backend` | `cuda` | Enables CUDA support for Clef and Candle (implies `clef`). |
| `huncho-cli` | `vllm` | Enables vLLM support in CLI. |
| `huncho-cli` | `onnx` | Enables ONNX support in CLI. |
| `huncho-cli` | `onnx-shared` | Enables shared ONNX support in CLI. |
| `huncho-cli` | `onnx-cuda` | Enables ONNX CUDA support in CLI. |
| `huncho-cli` | `candle` | Enables Candle support in CLI. |
| `huncho-cli` | `hf` | Enables Hugging Face Hub resolution. |
| `huncho-cli` | `tokenizers` | Enables native tokenization in CLI. |
| `huncho-cli` | `clef` | Enables Clef support in CLI. |
| `huncho-cli` | `cuda` | Enables CUDA support in CLI. |
| `huncho-cli` | `qualification` | Enables qualification features. |
| `huncho-cli` | `quantization` | Enables quantization support in CLI. |
| `huncho-cli` | `shared-base` | Enables shared base support in CLI. |
| `huncho-cli` | `cpu-blas` | Enables CPU BLAS support in CLI. |
| `huncho-cli` | `llamacpp` | Enables llama.cpp support in CLI. |
| `huncho-hub` | `candle` | Enables Candle support for Hub resolution. |

## Testing

Run the default test suite from the root of the checkout:

```bash
cargo test --workspace
```

To test optional backends and native tokenization, run the following commands (as used in CI):

```bash
cargo test -p huncho-backend -p huncho-cli --features onnx
cargo test -p huncho-backend -p huncho-cli --features candle
cargo test -p huncho-backend -p huncho-cli --features clef
cargo test -p huncho-core --features tokenizers --test hf_tokenizer
```

CI also runs a golden conformance check against the example package:

```bash
cargo build -p huncho-cli --features onnx,hf,tokenizers
./target/debug/huncho conform \
  --manifest examples/mock-model/huncho-model.json \
  --golden   examples/mock-model/golden.json
```

## Formatting and Linting

CI uses a pinned toolchain version `1.97.1` for linting to ensure reproducibility. Please run these checks with the same toolchain before opening a PR:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p huncho-backend -p huncho-cli --all-targets --features onnx,candle,clef -- -D warnings
```

## Commits, Branches, and Pull Requests

- **Branch Naming**: Use prefixes like `fix/`, `ci/`, `feat/`, `docs/`, `perf/`, `test/`, `build/`, `style/`, `refactor/`, `chore/` followed by a short description (e.g., `fix/license`, `ci/add-github-actions`).
- **Commit Messages**: Use Conventional Commit prefixes such as `feat`, `fix`, `ci`, `docs`, `perf`, `test`, `build`, `style`, `refactor`, `chore`.
- **Pull Requests**: Target the `main` branch. CI should pass before a PR is merged.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0, as stated in the [LICENSE](LICENSE) file.
