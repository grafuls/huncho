#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# Install wasm32-unknown-unknown and a wasm-bindgen CLI matching Cargo.lock.
# RUSTC/CARGO_TARGET_DIR may select an isolated toolchain/build directory.
cargo build --locked --manifest-path wasm/Cargo.toml --target wasm32-unknown-unknown --release
build_root=${CARGO_TARGET_DIR:-wasm/target}
wasm-bindgen --target web --out-dir pkg --out-name huncho_browser_core \
  "$build_root/wasm32-unknown-unknown/release/huncho_browser_core.wasm"
