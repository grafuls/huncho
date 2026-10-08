#!/usr/bin/env python3
"""Analyze frozen held-out evidence; never fit or rewrite model probabilities."""
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def probabilities(logits, temperature):
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    return [value / sum(exps) for value in exps]


def main():
    full = read(ROOT / "labeled-cpu-buffered/independent.json")
    diagnostic = ROOT / "drift-diagnosis"
    scope = read(diagnostic / "scope.json")
    reference = read(diagnostic / "reference.json")
    full_identity = read(ROOT / "labeled-cpu-buffered/identity.json")
    assert reference["golden_sha256"] == full_identity["golden_sha256"]
    modes = {}
    for mode in ["baseline", "buffered"]:
        rows = [json.loads(line) for line in (diagnostic / mode / "diagnostic-logits.jsonl").read_text().splitlines()]
        modes[mode] = {row["id"]: row for row in rows}
        assert len(rows) == len(modes[mode]) == len(scope["case_ids"])
        assert set(modes[mode]) == set(scope["case_ids"])
    expected = {case["id"]: case["expected"] for case in reference["cases"]}
    temperature = full_identity["resolved_calibration"]["temperature"]
    comparisons = []
    for case_id in scope["case_ids"]:
        a, b = modes["baseline"][case_id], modes["buffered"][case_id]
        assert a["question"] == b["question"] and a["labels"] == b["labels"]
        assert len(a["labels"]) == len(a["logits"]) == len(b["logits"])
        pa, pb = probabilities(a["logits"], temperature), probabilities(b["logits"], temperature)
        ref = [expected[case_id][a["question"]][label] for label in a["labels"]]
        comparisons.append({
            "id": case_id,
            "raw_float_bits_equal": all(struct.pack("<f", x) == struct.pack("<f", y) for x, y in zip(a["logits"], b["logits"])),
            "max_raw_delta": max(abs(x - y) for x, y in zip(a["logits"], b["logits"])),
            "cpu_baseline_vs_buffered_max_prob_delta_fp64": max(abs(x - y) for x, y in zip(pa, pb)),
            "cpu_baseline_vs_reference_max_prob_delta_fp64": max(abs(x - y) for x, y in zip(pa, ref)),
        })
    summary = {
        "qualified": False,
        "disposition": "Rejected: complete held-out maximum probability delta exceeds unchanged 0.001 gate.",
        "full_run": {
            "source": "3e79e41",
            "backend": full["backend"], "dtype": full["dtype"], "device": full["device"],
            "metadata": full["execution_metadata"], "work": full["work"],
            "questions": full["outcome_calibration"]["questions"],
            "max_prob_delta": full["max_prob_delta"], "argmax_agreement": full["argmax_agreement"],
            "ece_drift": full["ece"], "outcome_calibration": full["outcome_calibration"],
            "cases_exceeding_probability_delta_gate": sum(case["max_prob_delta"] > 0.001 for case in full["cases"]),
        },
        "diagnostic_temperature": temperature,
        "diagnostic_cases": comparisons,
        "limits": [
            "Diagnostic cases were selected from the failed held-out suite. They are forbidden for temperature fitting.",
            "All eight selected cases retain identical original CPU and buffered CPU raw float bits. The baseline itself differs from the older CUDA FP16 reference.",
            "The diagnostic uses a separately pinned older CPU binary (8f09121). It does not prove equality on all 1,536 cases or qualify newer kernels.",
            "Diagnostic softmax is offline FP64; the full rejected runtime report retains its actual FP32 gate values.",
            "Old diagnostic capture work records describe only the last row, not aggregate work. Collector accounting was fixed in 75a48c6.",
            "Elapsed full-run time includes load/hash and concurrent CPU contention. No inference-throughput claim is made.",
            "No fitting, threshold changes, model/calibration mutation, golden replacement or new GPU execution occurred.",
        ],
        "evidence_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(ROOT.rglob("*")) if path.is_file() and path.name not in ["summary.json", "analyze.py"]},
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
