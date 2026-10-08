#!/bin/bash
set -eu
root=/var/tmp/huncho-optimization-20261007
trap 'printf "%s\n" "$?" > "$root/stage5-cuda-fixtures.exit"' EXIT
mkdir "$root/stage5-source"
tar -xzf "$root/huncho-stage5-20261007.tar.gz" -C "$root/stage5-source"
source /var/tmp/huncho-t4-build/environment.sh
export CARGO_TARGET_DIR=/var/tmp/huncho-t4-build/source/target
export RAYON_NUM_THREADS=16
export HUNCHO_DEVICE=cuda
export HUNCHO_PROJECTION_CHUNK_ROWS=0
export HUNCHO_ATTENTION_FP32=false
# Preserve the executable used for the previously retained diagnostic before
# Cargo may replace its content in the shared build directory.
cp "$CARGO_TARGET_DIR/release/deps/huncho_backend-8090b676bfe76be9" "$root/precision-diagnostic-test-retained"
sha256sum "$root/precision-diagnostic-test-retained"
cd "$root/stage5-source"
cargo build --offline --release --locked -p huncho-cli --features onnx,hf,tokenizers,candle,clef,cuda --bin huncho
cp "$CARGO_TARGET_DIR/release/huncho" "$root/huncho-stage5"
sha256sum "$root/huncho-stage5" "$root/huncho-stage5-20261007.tar.gz"
cargo test --offline --release --locked -p huncho-backend --features cuda --test candle_integration --test candidate_readout --test native_batch --no-run
while ! test -f "$root/stage4-qualification.exit"; do sleep 5; done
cargo test --offline --release --locked -p huncho-backend --features cuda --test candle_integration cuda_modernbert -- --ignored --nocapture
cargo test --offline --release --locked -p huncho-backend --features cuda --test candidate_readout cuda_candidate -- --ignored --nocapture
cargo test --offline --release --locked -p huncho-backend --features cuda --test native_batch cuda_modernbert -- --ignored --nocapture
"$root/huncho-stage5" conform --model /root/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52 --dtype fp16 --golden "$root/reference-cuda-fp16/golden.json" --json > "$root/stage5-cuda-independent.json" 2> "$root/stage5-cuda-independent.log"
