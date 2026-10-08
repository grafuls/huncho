# CPU whole-schema Clef batching verification

All inference selected CPU. No GPU inventory, execution or qualification was
performed. Frozen `tiny_clef` PyTorch vectors and existing temperatures were
kept; temporary first-option labels only exercise plumbing, not released
outcome calibration.

- Native batch tests: three passed, including FP32/FP16, exact/mixed lengths,
  typed probabilities, original joint heads/usage, preparation/cache/replica
  ownership, fixed conformance, invalid groups and oversized singletons.
- Actual CLI batch tests: two passed; fresh conformance receipts, three concurrent
  HTTP callers, physical counters, missing/unlabeled refusal, real grouped
  benchmarks on two replicas and rejection of vacuous grouped runs.
- Default workspace passed, including seven malformed native whole-schema
  reports that must not publish partial cached responses.
- Core/HF hub with tokenizers, external scores and Candle passed.
- CPU backend/reference tests and CLI tests preceding lazy loading passed in
  the broader `clef,qualification` workspace run. That run stopped at the
  existing cold-load test's 15-second socket timeout. A standalone diagnostic
  measured about 18 seconds: two hashes of the 459 MB unstripped executable
  took about eight seconds each, plus fresh qualification. The test's socket
  timeout is now 60 seconds; identity/numerical gates are unchanged. Lazy,
  native gate and receipt tests then passed in the remainder run.

The padded FP16 raw score delta of about 0.000131 is covered by the fixture's
existing FP16 raw tolerance of 0.005. Probability/argmax gates, the paired
1e-4 gate, fixed external vectors and ECE threshold are unchanged. This is not
released model acceptance or a general latency/memory result.
