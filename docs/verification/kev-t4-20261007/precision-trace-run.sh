#!/bin/bash
set -eu
root=/var/tmp/huncho-optimization-20261007
trap 'printf "%s\n" "$?" > "$root/precision-trace.exit"' EXIT
mkdir "$root/precision-source"
tar -xzf "$root/huncho-stage2-20261007.tar.gz" -C "$root/precision-source"
cp "$root/qwen3_5-trace.rs" "$root/precision-source/crates/huncho-backend/src/qwen3_5.rs"
# Preserve the source snapshot belonging to the already-qualified CLI binary.
tar -xzf "$root/huncho-stage2-20261007.tar.gz" -C "$root/stage2-source" crates/huncho-backend/src/qwen3_5.rs
source /var/tmp/huncho-t4-build/environment.sh
export CARGO_TARGET_DIR=/var/tmp/huncho-t4-build/source/target
export RAYON_NUM_THREADS=16
export HUNCHO_DEVICE=cuda
export HUNCHO_TRACE_BASE=/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b
export HUNCHO_TRACE_PACKAGE=/root/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52
export HUNCHO_TRACE_GOLDEN="$root/reference-cuda-fp16/golden.json"
export HUNCHO_TRACE_DTYPE=fp16
export HUNCHO_TRACE_CASE=structured
export HUNCHO_TRACE_OUTPUT="$root/precision-prefix-fp16.json"
cd "$root/precision-source"
sha256sum crates/huncho-backend/src/qwen3_5.rs "$root/huncho-precision-run.sh"
cargo test --offline --release --locked -p huncho-backend --features cuda --lib trace_kev_prefix_precision -- --ignored --nocapture
