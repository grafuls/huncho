#!/usr/bin/env python3
"""Compare frozen diagnostic logits; no fitting or probability replacement."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    reference = json.loads((ROOT / "reference.json").read_text())
    identity = json.loads((ROOT / "capture-identity.json").read_text())
    rows = [json.loads(line) for line in (ROOT / "diagnostic-logits.jsonl").read_text().splitlines()]
    assert len(rows) == len({row["id"] for row in rows}) == 8
    assert identity["identity"]["device"] == "CPU"
    assert identity["identity"]["dtype"] == "fp16"
    expected = {case["id"]: case["expected"] for case in reference["cases"]}
    assert set(expected) == {row["id"] for row in rows}
    # Original source temperature, unchanged. Diagnostic cases are held-out
    # cases selected after a failed gate and are forbidden for fitting.
    temperature = 2.40605
    comparisons = []
    for row in rows:
        logits = row["logits"]
        assert len(logits) == len(row["labels"]) and all(math.isfinite(x) for x in logits)
        scaled = [x / temperature for x in logits]
        exps = [math.exp(x - max(scaled)) for x in scaled]
        probabilities = [x / sum(exps) for x in exps]
        gold = [expected[row["id"]][row["question"]][label] for label in row["labels"]]
        comparisons.append({"id": row["id"], "cpu_fp16_vs_reference_max_prob_delta_fp64": max(abs(a-b) for a,b in zip(probabilities, gold))})
    summary = {
        "qualified": False,
        "source": "8f09121",
        "disposition": "CPU FP16 diagnostic still exceeds the unchanged probability delta limit on selected worst cases.",
        "diagnostic_temperature": temperature,
        "diagnostic_cases": comparisons,
        "max_prob_delta_fp64": max(case["cpu_fp16_vs_reference_max_prob_delta_fp64"] for case in comparisons),
        "cases_exceeding_0_001": sum(case["cpu_fp16_vs_reference_max_prob_delta_fp64"] > 0.001 for case in comparisons),
        "limits": [
            "Eight cases selected from the rejected 1,536-case held-out CPU FP32 suite; forbidden for fitting.",
            "Baseline CPU FP16, no optional kernels, no new GPU execution. Reference is the previously frozen CUDA FP16 suite.",
            "Offline FP64 softmax at the source temperature is diagnostic, not full runtime conformance or temperature refit.",
            "This does not identify the cause of the CPU/reference drift or qualify any CPU runtime, optimization or quantization.",
            "Capture accounting in this older binary records only the final row, not aggregate physical work.",
            "No model, calibration, golden vector or threshold changed. No throughput claim.",
        ],
        "evidence_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(ROOT.iterdir()) if path.is_file() and path.name not in ["summary.json", "analyze.py"]},
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
