#!/usr/bin/env python3
"""Prepare pinned, disjoint labeled Kev qualification data; never fit a model.

Only the preparation environment needs pyarrow. Huncho's runtime/default build
has no Python dependency. Dataset contents stay in the chosen cache/output dir.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import urllib.request


DATASETS = {
    "choice": {
        "repo": "stanfordnlp/sst2",
        "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb",
        "fit": "data/train-00000-of-00001.parquet",
        "eval": "data/validation-00000-of-00001.parquet",
    },
    "noul": {
        "repo": "google/boolq",
        "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
        "fit": "data/train-00000-of-00001.parquet",
        "eval": "data/validation-00000-of-00001.parquet",
    },
    "score": {
        "repo": "Yelp/yelp_review_full",
        "revision": "c1f9ee939b7d05667af864ee1cb066393154bf85",
        "fit": "yelp_review_full/train-00000-of-00001.parquet",
        "eval": "yelp_review_full/test-00000-of-00001.parquet",
    },
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def question(qtype, row):
    if qtype == "choice":
        label = row["label"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError("SST-2 requires observed labels 0/1; hidden test labels are invalid")
        return row["sentence"], {
            "type": "choice",
            "instructions": "Classify the sentiment of this movie review.",
            "criteria": {"negative": None, "positive": None},
        }, ["negative", "positive"][label]
    if qtype == "noul":
        answer = row["answer"]
        if type(answer) is not bool:
            raise ValueError("BoolQ requires an observed boolean answer")
        return row["passage"], {
            "type": "noul",
            "instructions": "Answer this yes/no question using only the passage: " + row["question"],
        }, "yes" if answer else "no"
    if qtype == "score":
        label = row["label"]
        if type(label) is not int or not 0 <= label <= 4:
            raise ValueError("Yelp requires an observed five-class rating label 0..4")
        return row["text"], {
            "type": "score",
            "instructions": "Predict the star rating expressed by this review.",
            "criteria": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
        }, str(label)
    raise ValueError("unknown question type")


def content_key(qtype, row):
    state, prompt, _ = question(qtype, row)
    # Labels and row IDs cannot conceal identical inference inputs.
    value = json.dumps([state, prompt], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def select_rows(qtype, fit_rows, eval_rows, fit_count, eval_count, seed):
    if fit_count <= 0 or eval_count <= 0:
        raise ValueError("sample counts must be positive")
    evaluation_inputs = {content_key(qtype, row) for row in eval_rows}
    eligible = [i for i, row in enumerate(fit_rows) if content_key(qtype, row) not in evaluation_inputs]
    if len(eligible) < fit_count or len(eval_rows) < eval_count:
        raise ValueError("insufficient disjoint labeled rows for the requested sample sizes")
    rng = random.Random(f"{seed}:{qtype}")
    fit_indices = sorted(rng.sample(eligible, fit_count))
    eval_indices = sorted(rng.sample(range(len(eval_rows)), eval_count))
    return [(i, fit_rows[i]) for i in fit_indices], [(i, eval_rows[i]) for i in eval_indices]


def download(dataset, name, cache, offline):
    path = cache / dataset["repo"].replace("/", "--") / dataset["revision"] / name
    if not path.is_file():
        if offline:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/datasets/{dataset["repo"]}/resolve/{dataset["revision"]}/{name}'
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(path)
    return path


def prepare(cache, output, fit_count, eval_count, seed, offline=False):
    import pyarrow.parquet as parquet

    # Never replace an existing fitting/evaluation artifact.
    output.mkdir(parents=True, exist_ok=False)
    records = {"fit": [], "eval": []}
    provenance = {
        "schema_version": "1.0", "seed": seed, "datasets": {},
        "scope": "application-specific choice/noul/score calibration; not universal Kev calibration",
        "separation": "published train versus validation/test; identical inference inputs excluded from fitting",
        "model_training_overlap": "unknown; disjointness is only between these fitting and evaluation inputs",
    }
    for qtype, dataset in DATASETS.items():
        paths = {split: download(dataset, dataset[split], cache, offline) for split in ("fit", "eval")}
        rows = {split: parquet.read_table(path).to_pylist() for split, path in paths.items()}
        selected = select_rows(qtype, rows["fit"], rows["eval"], fit_count, eval_count, seed)
        metadata = dict(dataset, source_sha256={split: digest(path) for split, path in paths.items()})
        metadata["selected"] = {}
        for split, chosen in zip(("fit", "eval"), selected):
            distribution = Counter()
            for index, row in chosen:
                state, prompt, target = question(qtype, row)
                distribution[target] += 1
                records[split].append({
                    "id": f"{qtype}-{split}-{index}",
                    "source": {"repo": dataset["repo"], "revision": dataset["revision"], "file": dataset[split], "row": index},
                    "request": {"model": "kev-4b", "state": state, "questions": {"question": prompt}},
                    "targets": {"question": target},
                })
            metadata["selected"][split] = {"rows": [i for i, _ in chosen], "labels": dict(distribution)}
        provenance["datasets"][qtype] = metadata
    for split, values in records.items():
        path = output / (split + ".jsonl")
        with path.open("w") as stream:
            for record in values:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        provenance[split] = {"cases": len(values), "sha256": digest(path)}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({split: provenance[split] for split in ("fit", "eval")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-per-type", type=int, default=256)
    parser.add_argument("--eval-per-type", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    prepare(args.cache_dir, args.output_dir, args.fit_per_type, args.eval_per_type, args.seed, args.offline)
