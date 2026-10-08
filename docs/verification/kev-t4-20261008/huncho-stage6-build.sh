#!/bin/bash
set -eu
root=/var/tmp/huncho-optimization-20261007
trap 'printf "%s\n" "$?" > "$root/stage6-build.exit"' EXIT
mkdir "$root/stage6-source"
tar -xzf "$root/huncho-stage6-20261008.tar.gz" -C "$root/stage6-source"
source /var/tmp/huncho-t4-build/environment.sh
export CARGO_TARGET_DIR=/var/tmp/huncho-t4-build/source/target
# CPU pilot owns odd node-1 cores. Limit this build to even node-0 cores.
export CARGO_BUILD_JOBS=8
cd "$root/stage6-source"
taskset -c 0,2,4,6,8,10,12,14 cargo build --offline --release --locked -p huncho-cli --features onnx,hf,tokenizers,candle,clef,cuda --bin huncho
cp "$CARGO_TARGET_DIR/release/huncho" "$root/huncho-stage6"
sha256sum "$root/huncho-stage6" "$root/huncho-stage6-20261008.tar.gz"
