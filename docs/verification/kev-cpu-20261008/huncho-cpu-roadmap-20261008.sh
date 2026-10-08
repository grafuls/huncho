#!/bin/bash
set -euo pipefail
task_root=/var/tmp/huncho-cpu-roadmap-20261008
trap 'status=$?; printf "%s\n" "$status" > "$task_root/job.exit"' EXIT
export RUSTUP_HOME=/var/tmp/huncho-t4-build/rustup
export CARGO_HOME=/var/tmp/huncho-t4-build/cargo
export PATH="$CARGO_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export CARGO_BUILD_JOBS=12
unset CUDA_ROOT CUDA_COMPUTE_CAP LIBRARY_PATH LD_LIBRARY_PATH RUSTFLAGS ORT_LIB_LOCATION ORT_PREFER_DYNAMIC_LINK
cd "$task_root/source"
cargo build --offline --release -p huncho-cli --features clef,qualification > "$task_root/build.log" 2>&1
export HUNCHO_DEVICE=cpu RAYON_NUM_THREADS=16 CANDLE_NUM_THREADS=16
package=/root/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52
suite=/var/tmp/huncho-optimization-20261007/reference-cpu-fp32/golden.json
taskset -c 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31 python3 scripts/qualify_kev_runtime.py \
  --binary target/release/huncho --package "$package" --golden "$suite" \
  --source-archive "$task_root/source.tar.gz" --output "$task_root/numerical-cpu-buffered" \
  --device cpu --dtype fp32 --cpu-delta-rule --persistent-prefix-bytes 268435456 \
  --modes independent,prefix,batch --numerical-only
