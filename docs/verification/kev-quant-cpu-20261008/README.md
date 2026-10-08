# CPU Kev Q8 fitting progress, 2026-10-08

Lab access resumed. At 20:30:54 UTC, the existing isolated CPU job had captured
all 768 fitting questions, fit and saved a separate `candle:q8_0-fp32` entry,
and was executing its 1,536-case independent held-out conformance run. That
report was still empty. Q4 had not started. **Neither variant is qualified.**
No Apple or actual GPU inventory/execution/qualification check ran.

[Snapshot](summary.json), [variant refit](q8_0-fp32-refit.json),
[fitting identity](identity.json) and [conversion provenance](quantization-provenance.json)
retain the actual reports, binary/artifact/input hashes and refitted entry.
Fitting NLL is 0.3900520886; it is not an acceptance metric or a speed result.
Original source temperatures and held-out goldens were not changed. No running
job binary/source/inputs were replaced, and no additional native remote job was
launched. Only CPU job files/process status were inspected.

The older capture binary's audit correctly lists 768 questions but its physical
work fields describe only the final request. Evaluation resets per-request
stats; the current implementation already fixes accumulation. This limitation
is recorded without rewriting the historical audit or claiming aggregate work
from those counters. New execution profiles need separate fitting/gates.
Corpus text, fitting logits and model weights remain outside the repository.

Complete external delta 1e-3, full argmax, ECE drift .02 and the selected paired
1e-4 gates remain required. The earlier released CPU FP32 profile was rejected;
fitting this quantized variant cannot override that result or authorize serving.
