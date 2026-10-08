#!/bin/bash
set -eu
root=/var/tmp/huncho-optimization-20261007
trap 'printf "%s\n" "$?" > "$root/stage3-held-out.exit"' EXIT
source /var/tmp/huncho-t4-build/environment.sh
export HUNCHO_DEVICE=cuda
export RAYON_NUM_THREADS=16
export HUNCHO_PROJECTION_CHUNK_ROWS=64
"$root/huncho-stage3" conform --model /root/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52 --dtype fp16 --golden "$root/labeled-reference-v2/golden.json" --json > "$root/stage3-held-out.json" 2> "$root/stage3-held-out-model.log"
