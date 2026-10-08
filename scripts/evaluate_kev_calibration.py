#!/usr/bin/env python3
"""Evaluate isolated temperature refits on independently captured held-out logits.

This is offline statistical analysis using FP64 math, not inference conformance
or a release certificate. No fitting, model execution or manifest writes occur.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("duplicate captured record ID")
        seen.add(row["id"])
        logits, target = row["logits"], row["target"]
        if row["qtype"] not in ("choice", "score", "noul") or not logits:
            raise ValueError("unknown type or empty logit row")
        if type(target) is not int or not 0 <= target < len(logits):
            raise ValueError("target must identify an observed candidate")
        if any(type(value) not in (float, int) or not math.isfinite(value) for value in logits):
            raise ValueError("logits must be finite numbers")
        if row["qtype"] == "noul" and len(logits) != 2:
            raise ValueError("noul requires exactly two logits")
    if not rows:
        raise ValueError("captured split must contain rows")
    return rows


def calibration_entry(manifest, backend, dtype):
    config = manifest["calibration"]
    return config.get("entries", {}).get(f"{backend}:{dtype}", config["default"])


def temperature(entry, qtype, count):
    bucket = "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
    value = entry.get("temperature_by_options", {}).get(f"{qtype}:{bucket}") if entry.get("temperature_by_options") else None
    if value is None and entry.get("per_type_temperatures"):
        value = entry["per_type_temperatures"].get(qtype)
    if value is None:
        value = entry["temperature"]
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be positive and finite")
    return value


def predictions(rows, entry):
    values = []
    for row in rows:
        logits, target = row["logits"], row["target"]
        temp = temperature(entry, row["qtype"], len(logits))
        maximum = max(logits)
        scaled = [(value - maximum) / temp for value in logits]
        weights = [math.exp(value) for value in scaled]
        total = sum(weights)
        probabilities = [value / total for value in weights]
        # Use first candidate on ties, as in the inference argmax helper.
        mode = max(range(len(logits)), key=lambda i: logits[i])
        values.append({
            "confidence": probabilities[mode], "correct": int(mode == target),
            "nll": math.log(total) - scaled[target],
            "brier": sum((p - int(i == target)) ** 2 for i, p in enumerate(probabilities)),
            "temperature": temp,
        })
    return values


def metrics(values, bins=15):
    if bins < 1 or not values:
        raise ValueError("positive bin count and nonempty predictions required")
    groups = [[] for _ in range(bins)]
    for row in values:
        groups[min(int(row["confidence"] * bins), bins - 1)].append(row)
    ece = sum(abs(sum(row["confidence"] - row["correct"] for row in group)) for group in groups) / len(values)
    return {
        "cases": len(values), "accuracy": sum(row["correct"] for row in values) / len(values),
        "nll": sum(row["nll"] for row in values) / len(values),
        "brier": sum(row["brier"] for row in values) / len(values), "ece": ece,
        "mean_max_probability": sum(row["confidence"] for row in values) / len(values),
    }


def comparison(rows, baseline, refit, repetitions=1000, seed=20261007):
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    before, after = predictions(rows, baseline), predictions(rows, refit)
    baseline_metrics, refit_metrics = metrics(before), metrics(after)
    rng = random.Random(seed)
    samples = {key: [] for key in ("nll", "brier", "ece")}
    for _ in range(repetitions):
        indices = [rng.randrange(len(rows)) for _ in rows]
        old = metrics([before[i] for i in indices])
        new = metrics([after[i] for i in indices])
        for key, draws in samples.items():
            draws.append(new[key] - old[key])
    intervals = {}
    for key, draws in samples.items():
        draws.sort()
        intervals[key] = {"delta": refit_metrics[key] - baseline_metrics[key], "percentile_95": [draws[int(.025 * (repetitions - 1))], draws[int(.975 * (repetitions - 1))]]}
    return {
        "baseline": baseline_metrics, "refit": refit_metrics,
        "paired_deltas": intervals,
        "effective_temperatures": sorted({row["temperature"] for row in after}),
        "bootstrap": {"repetitions": repetitions, "seed": seed, "method": "paired row percentile; refit temperatures held fixed"},
    }


def evaluate(fit_path, eval_path, baseline_path, refit_path, backend, dtype, repetitions):
    fitting, evaluation = read_rows(fit_path), read_rows(eval_path)
    if {row["id"] for row in fitting} & {row["id"] for row in evaluation}:
        raise ValueError("fitting/evaluation record IDs overlap")
    baseline = calibration_entry(json.loads(baseline_path.read_text()), backend, dtype)
    refit = calibration_entry(json.loads(refit_path.read_text()), backend, dtype)
    report = {
        "schema_version": "1.0", "backend": backend, "dtype": dtype,
        "basis": "observed outcomes on independent held-out raw logits; offline FP64 analysis",
        "scope": "application-specific; not a runtime/optimization release qualification",
        "ece_bins": 15, "baseline_entry": baseline, "refit_entry": refit,
        "inputs": {name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in [("fit", fit_path), ("eval", eval_path), ("baseline_manifest", baseline_path), ("refit_manifest", refit_path)]},
        "fit": {}, "evaluation": {},
        "limitations": ["unknown overlap with model training", "bootstrap excludes fitting-temperature uncertainty", "fixed prompts/cardinalities and sample distribution; no universal calibration claim", "numerical conformance still requires actual independent model execution"],
    }
    for qtype in ("choice", "noul", "score"):
        fit_rows = [row for row in fitting if row["qtype"] == qtype]
        eval_rows = [row for row in evaluation if row["qtype"] == qtype]
        if not fit_rows or not eval_rows:
            raise ValueError("both splits must cover every typed task")
        report["fit"][qtype] = {"cases": len(fit_rows), "targets": dict(Counter(row["target"] for row in fit_rows)), "baseline": metrics(predictions(fit_rows, baseline)), "refit": metrics(predictions(fit_rows, refit))}
        report["evaluation"][qtype] = comparison(eval_rows, baseline, refit, repetitions)
    report["evaluation"]["all"] = comparison(evaluation, baseline, refit, repetitions)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-logits", required=True, type=Path)
    parser.add_argument("--eval-logits", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--refit-manifest", required=True, type=Path)
    parser.add_argument("--backend", default="candle")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--bootstrap", default=1000, type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.fit_logits, args.eval_logits, args.baseline_manifest, args.refit_manifest, args.backend, args.dtype, args.bootstrap), indent=2, allow_nan=False))
