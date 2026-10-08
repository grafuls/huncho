#!/bin/bash
set -eu
root=/var/tmp/huncho-optimization-20261007
trap 'printf "%s\n" "$?" > "$root/stage6-qualification.exit"' EXIT
source /var/tmp/huncho-t4-build/environment.sh
export RAYON_NUM_THREADS=8
export CANDLE_NUM_THREADS=8
export HUNCHO_RESULT_CACHE_BYTES=0 HUNCHO_PROMPT_CACHE_BYTES=0 HUNCHO_TOKEN_CACHE_BYTES=0
export HUNCHO_PREFIX_CACHE=false HUNCHO_COALESCE_BYTES=0
package=/root/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52
while ! test -f "$root/stage6-build.exit"; do sleep 5; done
test "$(cat "$root/stage6-build.exit")" = 0
# Run on node 0, apart from the concurrently running CPU thread pilot.
taskset -c 0,2,4,6,8,10,12,14 python3 "$root/qualify_kev_runtime.py" --binary "$root/huncho-stage6" --package "$package" --golden "$root/reference-cuda-fp16/golden.json" --source-archive "$root/huncho-stage6-20261008.tar.gz" --device cuda --dtype fp16 --prepare-all --modes independent --numerical-only --output "$root/stage6-preparation-numerical"
taskset -c 0,2,4,6,8,10,12,14 python3 "$root/huncho-stage6-preparation-pilot.py"
taskset -c 0,2,4,6,8,10,12,14 python3 "$root/qualify_kev_runtime.py" --binary "$root/huncho-stage6" --package "$package" --golden "$root/labeled-reference-v2/golden.json" --source-archive "$root/huncho-stage6-20261008.tar.gz" --device cuda --dtype fp16 --prepare-all --modes independent --output "$root/stage6-preparation-labeled"
