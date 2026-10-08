# CPU Kev KV page storage — 2026-10-08

Immutable full pages are shared by `Arc`; append copies only a changed partial
tail and new compact blocks. Narrow projection views cannot retain a complete
suffix allocation. Each attention call still materializes the full original KV
sequence and uses the existing arithmetic. This is not a paged kernel.

Sequential CPU-only checks:

| Check | Evidence |
|---|---|
| FP32/FP16 block ordering, full-page sharing and tail isolation | [unit.log](unit.log) |
| Native FP32/FP16 flat-cache bitwise parity, frozen typed probabilities, chunks/forks, limits, retention, cancellation and replicas; existing prefix/chunk regressions | [native.log](native.log) |
| Actual CLI cooperative/page/retained qualification, receipts, synthetic labeled gates and missing/unlabeled/vacuous/invalid-profile refusal | [cli.log](cli.log) |
| Default workspace | [default-workspace.log](default-workspace.log) |

Commands:

```sh
cargo test -p huncho-backend --features candle --lib kv_pages -- --test-threads=1
RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1 HUNCHO_DEVICE=cpu cargo test -p huncho-backend --features clef,quantization --test kv_pages --test prefix_cache --test chunked_prefill -- --test-threads=1
RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1 HUNCHO_DEVICE=cpu cargo test -p huncho-cli --features clef,qualification --test cooperative -- --test-threads=1
cargo test --workspace
```

Fixture golden vectors and their independent PyTorch provenance are unchanged.
Temperatures are unchanged: exact flat-cache raw parity and paired checks use
0.75, 1 and 2.40605; external fixture probabilities use their original 2.40605.
The 1e-3 external probability, complete argmax, 0.02 ECE drift and 1e-4 paired
gates are unchanged. Arbitrary synthetic labels prove gate plumbing only;
released Kev acceptance remains rejected/pending. No 4B speed/RSS, packed-profile
acceptance or GPU result follows. Pages are off by default; no dependency was
added. Apple work and actual GPU checks remain deferred.
