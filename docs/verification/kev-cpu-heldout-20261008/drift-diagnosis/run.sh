#!/bin/bash
set -euo pipefail
root=/var/tmp/huncho-cpu-drift-audit-20261008
trap 'status=$?; printf "%s\n" "$status" > "$root/job.exit"' EXIT
export HUNCHO_DEVICE=cpu RAYON_NUM_THREADS=16 CANDLE_NUM_THREADS=16
unset HUNCHO_CPU_CAUSAL_CONV HUNCHO_PREFILL_CHUNK_TOKENS HUNCHO_PROJECTION_CHUNK_ROWS HUNCHO_ATTENTION_FP32 HUNCHO_REPLICAS HUNCHO_PROMPT_CACHE_BYTES HUNCHO_TOKEN_CACHE_BYTES
for mode in baseline buffered; do
 if test "$mode" = baseline; then export HUNCHO_CPU_DELTA_RULE=0; else export HUNCHO_CPU_DELTA_RULE=1; fi
 taskset -c 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31 /var/tmp/huncho-quant-cpu-20261008/source/target/release/huncho capture-logits --model /root/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52 --backend candle --dtype fp32 --data "$root/diagnostic-eval-rows.jsonl" --output "$root/$mode" > "$root/$mode.json" 2> "$root/$mode.log"
done
