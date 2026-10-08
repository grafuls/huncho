#!/usr/bin/env python3
"""Collect raw logits from an independently launched Kev reference server.

Launch the pinned reference binary with --extensions. This script makes no
temperature updates, and writes fitting rows separately from held-out goldens.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request


def labels_for(record):
    request = record["request"]
    question = request["questions"]["question"]
    qtype = question["type"]
    if qtype == "choice":
        labels = list(question["criteria"])
    elif qtype == "score":
        labels = [str(i) for i in range(len(question["criteria"]))]
    elif qtype == "noul":
        labels = ["no", "yes"]  # kev-v1 trained prompt/head order (not generic F2).
    else:
        raise ValueError("unsupported question type")
    return qtype, labels


def captured_row(record, logits):
    qtype, labels = labels_for(record)
    if len(logits) != len(labels) or not all(type(value) in (float, int) and math.isfinite(value) for value in logits):
        raise ValueError("reference logits must be finite and cover every candidate")
    return {
        "id": record["id"], "source": record["source"], "qtype": qtype,
        "labels": labels, "logits": logits,
        "target": labels.index(record["targets"]["question"]),
    }


def extract(record, response):
    request = record["request"]
    if response["model"] != request["model"]:
        raise ValueError("response model differs from the requested execution")
    qtype, labels = labels_for(record)
    logits = response["extensions"]["raw_logits"]["question"]
    row = captured_row(record, logits)
    answer = response["answers"]["question"]
    if answer["type"] != qtype:
        raise ValueError("reference answer type differs from the requested head")
    probabilities = (
        {"yes": answer["noul"], "no": 1.0 - answer["noul"]}
        if qtype == "noul" else answer["probabilities"]
    )
    if set(probabilities) != set(labels) or any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 1e-4:
        raise ValueError("invalid reference probability distribution")
    mode = max(range(len(logits)), key=lambda i: logits[i])
    if probabilities[labels[mode]] < max(probabilities.values()) - 1e-6:
        raise ValueError("raw candidate order disagrees with calibrated response")
    return row, {"id": record["id"], "request": request, "expected": {"question": probabilities}, "targets": record["targets"]}


def reuse_fitting(path, records):
    """Reindex targets from the original observed labels; never trust old indices."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(records):
        raise ValueError("reused fitting rows must cover the complete original fitting split")
    result = []
    for row, record in zip(rows, records):
        if row["id"] != record["id"] or row["source"] != record["source"] or row["qtype"] != labels_for(record)[0]:
            raise ValueError("reused fitting provenance differs from pinned fitting inputs")
        result.append(captured_row(record, row["logits"]))
    return result


def collect(url, fit_file, eval_file, output, execution_identity, reuse_fit=None, reuse_identity=None):
    def http(path, body=None):
        request = urllib.request.Request(
            url.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.load(response)

    output.mkdir(parents=True, exist_ok=False)
    identity = json.loads(execution_identity.read_text())
    if reuse_fit is not None:
        if reuse_identity is None or json.loads(reuse_identity.read_text()) != identity:
            raise ValueError("fitting reuse requires the same declared pinned execution identity")
        identity["reused_fitting"] = {"sha256": hashlib.sha256(reuse_fit.read_bytes()).hexdigest(), "target_indices": "reconstructed from pinned observed labels and kev-v1 candidate order"}
    identity["served_models"] = http("/v1/models")
    fit_rows = []
    golden = {"schema_version": "1.0", "family": "F2", "cases": []}
    seen = set()
    for phase, file in [("fit", fit_file), ("eval", eval_file)]:
        records = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        started = time.monotonic()
        reused = reuse_fitting(reuse_fit, records) if phase == "fit" and reuse_fit is not None else None
        with (output / (phase + "-logits.jsonl")).open("w") as stream:
            for index, record in enumerate(records):
                if record["id"] in seen:
                    raise ValueError("duplicate fitting/evaluation record ID")
                seen.add(record["id"])
                if reused is not None:
                    row, case = reused[index], None
                else:
                    response = http("/v1/systemone", record["request"])
                    row, case = extract(record, response)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if phase == "fit":
                    fit_rows.append(row)
                else:
                    golden["cases"].append(case)
                    # Keep independent expected probabilities through interruption.
                    with (output / "eval-cases.jsonl").open("a") as checkpoint:
                        checkpoint.write(json.dumps({"row": row, "case": case}) + "\n")
                if (index + 1) % 16 == 0 or index + 1 == len(records):
                    print(json.dumps({"phase": phase, "completed": index + 1, "total": len(records), "elapsed_seconds": time.monotonic() - started}), flush=True)
        identity[phase] = {"cases": len(records), "inputs_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        if phase == "fit":
            fitting = {"rows": [row["logits"] for row in fit_rows], "targets": [row["target"] for row in fit_rows], "qtypes": [row["qtype"] for row in fit_rows]}
            (output / "fit.json").write_text(json.dumps(fitting) + "\n")
            (output / "identity.pending.json").write_text(json.dumps(identity, indent=2) + "\n")
    fitting = {"rows": [row["logits"] for row in fit_rows], "targets": [row["target"] for row in fit_rows], "qtypes": [row["qtype"] for row in fit_rows]}
    (output / "fit.json").write_text(json.dumps(fitting) + "\n")
    vectors = json.dumps(golden, indent=2) + "\n"
    (output / "golden.json").write_text(vectors)
    identity["golden_sha256"] = hashlib.sha256(vectors.encode()).hexdigest()
    identity["basis"] = "observed outcomes; independent raw fitting and evaluation rows; no optimized outputs used"
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-identity", type=Path, required=True)
    parser.add_argument("--reuse-fit-logits", type=Path)
    parser.add_argument("--reuse-fit-identity", type=Path)
    args = parser.parse_args()
    collect(args.url, args.fit, args.eval, args.output_dir, args.execution_identity, args.reuse_fit_logits, args.reuse_fit_identity)
