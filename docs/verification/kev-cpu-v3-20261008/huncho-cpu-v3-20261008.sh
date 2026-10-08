#!/bin/bash
set -euo pipefail
task_root=/var/tmp/huncho-cpu-v3-20261008
trap 'status=$?; printf "%s\n" "$status" > "$task_root/job.exit"' EXIT
export RUSTUP_HOME=/var/tmp/huncho-t4-build/rustup
export CARGO_HOME=/var/tmp/huncho-t4-build/cargo
export PATH="$CARGO_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export CARGO_BUILD_JOBS=12 HUNCHO_DEVICE=cpu RAYON_NUM_THREADS=16 CANDLE_NUM_THREADS=16
unset CUDA_ROOT CUDA_COMPUTE_CAP LIBRARY_PATH LD_LIBRARY_PATH ORT_LIB_LOCATION ORT_PREFER_DYNAMIC_LINK
unset HUNCHO_PROJECTION_CHUNK_ROWS HUNCHO_ATTENTION_FP32 HUNCHO_PROMPT_CACHE_BYTES HUNCHO_TOKEN_CACHE_BYTES
unset HUNCHO_CPU_DELTA_RULE HUNCHO_CPU_CAUSAL_CONV HUNCHO_PREFILL_CHUNK_TOKENS
export RUSTFLAGS='-C target-cpu=x86-64-v3'
cd "$task_root/source"
tar -xzf "$task_root/source.tar.gz"
cpu_affinity=1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31
# Separate target/binary and package paths; never replace a running job's files.
taskset -c "$cpu_affinity" cargo build --offline --release -p huncho-cli --features quantization,clef > "$task_root/build.log" 2>&1
taskset -c "$cpu_affinity" cargo test --offline --release -p huncho-backend --features quantization,clef --lib --test chunked_prefill --test prefix_cache --test native_batch --test quantized_kev > "$task_root/tests.log" 2>&1
source_package=/root/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52
golden=/var/tmp/huncho-optimization-20261007/reference-cpu-fp32/golden.json
# Numerical-only diagnostics never authorize serving or replace held-out gates.
taskset -c "$cpu_affinity" python3 scripts/qualify_kev_runtime.py --binary target/release/huncho --package "$source_package" --golden "$golden" --source-archive "$task_root/source.tar.gz" --output "$task_root/numerical" --device cpu --dtype fp32 --cpu-delta-rule --cpu-causal-conv --cpu-kernel-build x86_64:avx,avx2,f16c,fma --persistent-prefix-bytes 268435456 --numerical-only
