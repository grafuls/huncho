"""FP64 drift diagnosis from retained raw fitting logits; no acceptance gate."""
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / "q8-pilot/summary.json").read_text())
rows = summary["logits"]
assert [sample["mode"] for sample in summary["samples"]] == [
    "cpu_v3", "portable", "portable", "cpu_v3"
]
assert all(len(rows[str(index)]) == 8 for index in range(4))
temperature = 2.40605  # Source FP32 temperature, not a quantized refit.


def probabilities(row):
    assert all(math.isfinite(value) for value in row)
    scaled = [value / temperature for value in row]
    offset = max(scaled)
    exp = [math.exp(value - offset) for value in scaled]
    return [value / sum(exp) for value in exp]


first, second = rows["0"], rows["1"]
pairs = [(probabilities(a), probabilities(b)) for a, b in zip(first, second)]
report = {
    "qualified": False,
    "comparison": "Same Q8_0 artifact, portable CPU vs x86-64-v3 on eight fitting rows only",
    "temperature": temperature,
    "questions": len(first),
    "max_raw_logit_delta": max(abs(a - b) for x, y in zip(first, second) for a, b in zip(x, y)),
    "max_probability_delta": max(abs(a - b) for x, y in pairs for a, b in zip(x, y)),
    "argmax_agreement": sum(
        max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
        for a, b in pairs
    ) / len(pairs),
    "within_mode_raw_logits_equal": {
        "cpu_v3": rows["0"] == rows["3"], "portable": rows["1"] == rows["2"]
    },
    "limits": summary["limits"] + [
        "Collection predates 75a48c6; work counters describe the last fitting record only.",
        "The displayed temperature is the source FP32 temperature for drift diagnosis, not a fitted/accepted quantized entry. No held-out gate or latency qualification.",
    ],
}
(root / "probability-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
