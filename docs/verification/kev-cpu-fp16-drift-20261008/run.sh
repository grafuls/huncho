#!/bin/bash
set -euo pipefail
root=/var/tmp/huncho-cpu-fp16-drift-20261008
trap 'status=$?; printf "%s\n" "$status" > "$root/job.exit"' EXIT
export HUNCHO_DEVICE=cpu RAYON_NUM_THREADS=16 CANDLE_NUM_THREADS=16
unset CUDA_ROOT CUDA_COMPUTE_CAP LIBRARY_PATH LD_LIBRARY_PATH RUSTFLAGS ORT_LIB_LOCATION ORT_PREFER_DYNAMIC_LINK HUNCHO_CPU_DELTA_RULE HUNCHO_CPU_CAUSAL_CONV HUNCHO_CPU_FUSED_GATE HUNCHO_PREFILL_CHUNK_TOKENS HUNCHO_PROJECTION_CHUNK_ROWS HUNCHO_ATTENTION_FP32 HUNCHO_REPLICAS HUNCHO_PROMPT_CACHE_BYTES HUNCHO_TOKEN_CACHE_BYTES
cp /var/tmp/huncho-cpu-drift-audit-20261008/diagnostic-eval-rows.jsonl "$root/diagnostic-eval-rows.jsonl"
taskset -c 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31 /var/tmp/huncho-quant-cpu-20261008/source/target/release/huncho capture-logits --model /root/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52 --backend candle --dtype fp16 --data "$root/diagnostic-eval-rows.jsonl" --output "$root/baseline" > "$root/baseline.json" 2> "$root/baseline.log"
