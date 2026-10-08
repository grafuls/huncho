#!/usr/bin/env python3
"""Run unchanged Kev probability gates and retain reproducible execution evidence.

No model, temperature, source or golden files are written. Numerical-only runs
are diagnostic; labeled acceptance is scoped to the supplied suite, not a
universal calibration certificate or proof of fitting/evaluation separation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

PACKED_PROFILES = {
    "q8_0-fp32": "kev-projections-q8_0-fp32-v1",
    "q4_0-fp32": "kev-projections-q4_0-fp32-v1",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_suite(suite, numerical_only):
    cases = suite.get("cases", [])
    if suite.get("family") != "F2" or not cases:
        raise ValueError("a nonempty pinned F2 golden suite is required")
    ids = set()
    for case in cases:
        if case["id"] in ids:
            raise ValueError("duplicate golden case ID")
        ids.add(case["id"])
        questions = case["request"]["questions"]
        expected = case["expected"]
        if not questions or questions.keys() != expected.keys():
            raise ValueError("every golden question needs an unchanged expected distribution")
        targets = case.get("targets", {})
        if not numerical_only and questions.keys() != targets.keys():
            raise ValueError("labeled acceptance requires an observed target for every question")
        for question, probabilities in expected.items():
            if not probabilities or any(type(p) not in (int, float) or not math.isfinite(p)
                    or not 0 <= p <= 1 for p in probabilities.values()):
                raise ValueError("expected probabilities must be finite distributions")
            if abs(sum(probabilities.values()) - 1) > 1e-4:
                raise ValueError("expected distributions must sum to one")
            if targets and targets.get(question) not in probabilities:
                raise ValueError("observed targets must identify actual candidates")


def verify_report(report, args, suite, mode):
    metadata = report.get("execution_metadata", {})
    expected_metadata = {}
    if args.projection_chunk_rows:
        expected_metadata["projection_chunk_rows"] = str(args.projection_chunk_rows)
    if args.fp32_attention:
        expected_metadata["attention_compute_dtype"] = "fp32"
    if args.cpu_delta_rule:
        expected_metadata["delta_rule_execution"] = "cpu-buffered-v1"
    if args.cpu_causal_conv:
        expected_metadata["causal_conv_execution"] = "cpu-buffered-v1"
    if args.prefill_chunk_tokens:
        expected_metadata["prefill_chunk_tokens"] = str(args.prefill_chunk_tokens)
    if args.dtype in PACKED_PROFILES:
        expected_metadata.update(weight_quantization=PACKED_PROFILES[args.dtype],
            activation_dtype="fp32", recurrent_state_dtype="fp32", pointer_head_dtype="fp32",
            projection_kernel="candle-packed-cpu-v1")
    if metadata != expected_metadata:
        raise ValueError("reported kernel profile differs from the requested execution")
    device = report.get("device", "")
    if (args.device == "cpu" and device != "CPU") or (args.device.startswith("cuda") and "CUDA" not in device):
        raise ValueError("reported device differs from the explicit requested device")
    if report.get("backend") != "candle" or report.get("dtype") != args.dtype:
        raise ValueError("reported backend/dtype differs from the requested execution")
    if [case["id"] for case in report["cases"]] != [case["id"] for case in suite["cases"]]:
        raise ValueError("report did not evaluate the complete unchanged golden suite")
    if report.get("prefix_cache") != (mode == "prefix") or report.get("max_batch_tokens") != (
            args.batch_tokens if mode == "batch" else None):
        raise ValueError("report did not evaluate the requested optimization")
    if report.get("prepare_all", False) != (args.prepare_all or (mode == "batch" and args.batch_max_requests is not None)):
        raise ValueError("report did not evaluate the requested preparation path")
    if report.get("persistent_prefix_bytes", 0) != (args.persistent_prefix_bytes if mode == "prefix" else 0):
        raise ValueError("report did not evaluate the requested persistent prefix budget")
    if report.get("cross_request_max_requests") != (args.batch_max_requests if mode == "batch" else None):
        raise ValueError("report did not evaluate the requested cross-request batch size")
    if not args.numerical_only and (report.get("outcome_calibration") or {}).get("questions") != sum(
            len(case["request"]["questions"]) for case in suite["cases"]):
        raise ValueError("complete observed-outcome metrics are required")
    if mode != "independent" or args.prepare_all:
        if not report.get("optimization_parity"):
            raise ValueError("optimized execution requires independent-forward parity")
    if args.prepare_all and report.get("work", {}).get("prepared_questions", 0) <= 0:
        raise ValueError("suite did not exercise request preparation")
    if mode == "prefix" and args.persistent_prefix_bytes > 0 and report.get("work", {}).get("persistent_prefix_hits", 0) <= 0:
        raise ValueError("suite did not exercise persistent prefix reuse")
    if mode == "prefix" and args.prefill_chunk_tokens and report.get("work", {}).get("chunked_prefills", 0) <= 0:
        raise ValueError("suite did not exercise an actual split prefix")
    if mode == "batch" and args.batch_max_requests is not None and report.get("work", {}).get("cross_request_batches", 0) <= 0:
        raise ValueError("suite did not exercise cross-request collation")
    if mode != "independent":
        counter = "cache_forks" if mode == "prefix" else "batch_calls"
        if report.get("work", {}).get(counter, 0) <= 0:
            raise ValueError("suite did not exercise the requested native optimization")
    parity = report.get("optimization_parity")
    values = [report["max_prob_delta"], report["argmax_agreement"], report["ece"]]
    if parity:
        values.extend([parity["max_prob_delta"], parity["argmax_agreement"]])
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("qualification metrics must be finite numbers between zero and one")
    # Recheck the fixed policy even if an unexpected binary reports `passed`.
    passed = report["max_prob_delta"] <= 1e-3 and report["argmax_agreement"] == 1.0 and report["ece"] <= .02
    if parity:
        passed = passed and parity["max_prob_delta"] <= 1e-4 and parity["argmax_agreement"] == 1.0
    if report.get("passed") is not passed:
        raise ValueError("reported acceptance does not match the unchanged qualification policy")
    return passed


def run(args):
    if (args.cpu_delta_rule or args.cpu_causal_conv) and args.device != "cpu":
        raise ValueError("buffered recurrence and convolution are CPU-only")
    if args.dtype in PACKED_PROFILES and args.device != "cpu":
        raise ValueError("packed Kev artifacts are CPU-only")
    if not 0 <= args.prefill_chunk_tokens <= 4096 or (args.prefill_chunk_tokens and args.device != "cpu"):
        raise ValueError("prefix chunk size must be CPU-only and 0..4096")
    if args.persistent_prefix_bytes < 0:
        raise ValueError("persistent prefix byte budget must be nonnegative")
    if args.batch_max_requests is not None and not 2 <= args.batch_max_requests <= 64:
        raise ValueError("cross-request batch size must be 2..64")
    modes = args.modes.split(",")
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"independent", "prefix", "batch"}:
        raise ValueError("modes must be distinct independent, prefix or batch entries")
    if args.persistent_prefix_bytes and "prefix" not in modes:
        raise ValueError("persistent prefix budget requires the prefix mode")
    if args.prefill_chunk_tokens and "prefix" not in modes:
        raise ValueError("prefix chunk size requires the prefix mode")
    if args.batch_max_requests is not None and "batch" not in modes:
        raise ValueError("cross-request batch size requires the batch mode")
    suite = json.loads(args.golden.read_text())
    validate_suite(suite, args.numerical_only)
    manifest_path = args.package / "huncho-model.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("family") != "F2" or manifest["prompt_contract"]["template"] != "kev-v1":
        raise ValueError("package must be the Kev F2 pointer contract")
    calibration = manifest["calibration"]
    entry = calibration.get("entries", {}).get(f"candle:{args.dtype}", calibration["default"])
    if args.dtype in PACKED_PROFILES and not args.numerical_only and calibration.get("entries", {}).get(f"candle:{args.dtype}", {}).get("status") != "refit":
        raise ValueError("packed acceptance requires an explicit backend:dtype refit")
    if not args.numerical_only and entry.get("status") not in ("fit", "refit"):
        raise ValueError("labeled acceptance requires an explicitly fitted calibration entry")
    if not 0 <= args.projection_chunk_rows <= 4096 or args.batch_tokens <= 0:
        raise ValueError("projection rows must be 0..4096 and the batch token budget must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be new or empty; previous evidence is never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    golden_hash, manifest_hash = sha256(args.golden), sha256(manifest_path)
    tracked = [manifest_path, args.package / manifest["head"]["weights"]]
    if manifest["backbone"].get("tokenizer"):
        tracked.append(args.package / manifest["backbone"]["tokenizer"])
    if args.dtype in PACKED_PROFILES:
        artifacts = [a for a in manifest["backbone"].get("artifacts", {}).get("candle", []) if a.get("dtype") == args.dtype]
        if len(artifacts) != 1 or artifacts[0].get("quantization") != PACKED_PROFILES[args.dtype]:
            raise ValueError("an exact packed artifact with the requested profile is required")
        tracked.append(args.package / artifacts[0]["path"])
    else:
        tracked.extend(path for path in [args.package / "adapter_config.json", args.package / "adapter_model.safetensors"] if path.exists())
    identity = {
        "schema_version": "1.0", "started_utc": datetime.now(timezone.utc).isoformat(),
        "binary": str(binary), "binary_sha256": sha256(binary),
        "script_sha256": sha256(__file__), "golden_sha256": golden_hash,
        "package": str(args.package.resolve()), "package_files": {str(path.relative_to(args.package)): sha256(path) for path in tracked},
        "adapter": manifest.get("adapter"), "backbone_source": manifest["backbone"].get("source"),
        "calibration": calibration, "resolved_calibration": entry, "device_selection": args.device,
        "dtype": args.dtype, "projection_chunk_rows": args.projection_chunk_rows,
        "fp32_attention": args.fp32_attention, "kernel": platform.release(),
        "cpu_delta_rule": args.cpu_delta_rule,
        "cpu_causal_conv": args.cpu_causal_conv,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "persistent_prefix_bytes": args.persistent_prefix_bytes,
        "batch_max_requests": args.batch_max_requests,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "prepare_all": args.prepare_all,
        "numerical_only": args.numerical_only, "results": [], "qualified": False,
        "limits": ["Acceptance applies to the supplied independent vectors and observed outcomes only.",
            "The runner records pins/hashes but cannot prove training overlap or fit/evaluation separation.",
            "No speed or universal calibration claim; base revisions are recorded, base shards are not rehashed.",
            "Single-question suites cannot qualify prefix or batch paths; use separate representative fan-out cases."],
    }
    if args.source_archive:
        identity["source_archive_sha256"] = sha256(args.source_archive)
    boot = Path("/proc/sys/kernel/random/boot_id")
    if boot.exists():
        identity["boot_id"] = boot.read_text().strip()
    if args.device.startswith("cuda"):
        identity["gpu_inventory"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip()
    env = os.environ.copy()
    env.update(HUNCHO_DEVICE=args.device, HUNCHO_PROJECTION_CHUNK_ROWS=str(args.projection_chunk_rows),
        HUNCHO_ATTENTION_FP32=str(args.fp32_attention).lower(),
        HUNCHO_CPU_DELTA_RULE=str(args.cpu_delta_rule).lower())
    env["HUNCHO_CPU_CAUSAL_CONV"] = str(args.cpu_causal_conv).lower()
    env["HUNCHO_PREFILL_CHUNK_TOKENS"] = str(args.prefill_chunk_tokens)
    identity["rayon_num_threads"] = env.get("RAYON_NUM_THREADS")
    identity["thread_environment"] = {key: env[key] for key in
        ["RAYON_NUM_THREADS", "CANDLE_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"] if key in env}
    try:
        for mode in modes:
            command = [str(binary), "conform", "--model", str(args.package.resolve()), "--backend", "candle",
                "--dtype", args.dtype, "--golden", str(args.golden.resolve()), "--json"]
            if args.prepare_all:
                command.append("--prepare-all")
            if mode == "prefix":
                command.append("--prefix-cache")
                if args.persistent_prefix_bytes:
                    command.extend(["--persistent-prefix-bytes", str(args.persistent_prefix_bytes)])
            if mode == "batch":
                command.extend(["--max-batch-tokens", str(args.batch_tokens)])
                if args.batch_max_requests is not None:
                    command.extend(["--batch-max-requests", str(args.batch_max_requests)])
            start = time.monotonic()
            with (args.output / f"{mode}.json").open("w") as output, (args.output / f"{mode}.log").open("w") as log:
                result = subprocess.run(command, env=env, stdout=output, stderr=log)
            report = json.loads((args.output / f"{mode}.json").read_text())
            passed = verify_report(report, args, suite, mode)
            if result.returncode != (0 if passed else 1):
                raise ValueError("conformance process exit disagrees with its acceptance report")
            record = {"mode": mode, "passed": passed, "exit_code": result.returncode,
                "elapsed_seconds_including_load": time.monotonic() - start,
                "report_sha256": sha256(args.output / f"{mode}.json"), "max_prob_delta": report["max_prob_delta"],
                "argmax_agreement": report["argmax_agreement"], "ece_drift": report["ece"],
                "optimization_parity": report.get("optimization_parity"), "outcome_calibration": report.get("outcome_calibration")}
            record["work"] = report.get("work", {})
            identity["results"].append(record)
            print(json.dumps(record), flush=True)
        if sha256(args.golden) != golden_hash or sha256(manifest_path) != manifest_hash or sha256(binary) != identity["binary_sha256"]:
            raise ValueError("golden, manifest or binary changed during qualification")
        if any(sha256(path) != identity["package_files"][str(path.relative_to(args.package))] for path in tracked):
            raise ValueError("model package changed during qualification")
        identity["qualified"] = not args.numerical_only and all(record["passed"] for record in identity["results"])
    except BaseException as error:
        identity["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        identity["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["binary", "package", "golden", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--dtype", choices=["fp32", "fp16", *PACKED_PROFILES], required=True)
    parser.add_argument("--projection-chunk-rows", type=int, default=0)
    parser.add_argument("--fp32-attention", action="store_true")
    parser.add_argument("--cpu-delta-rule", action="store_true")
    parser.add_argument("--cpu-causal-conv", action="store_true")
    parser.add_argument("--prefill-chunk-tokens", type=int, default=0)
    parser.add_argument("--persistent-prefix-bytes", type=int, default=0)
    parser.add_argument("--batch-max-requests", type=int)
    parser.add_argument("--prepare-all", action="store_true")
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--modes", default="independent,prefix,batch")
    parser.add_argument("--numerical-only", action="store_true")
    args = parser.parse_args()
    result = run(args)
    raise SystemExit(0 if all(record["passed"] for record in result["results"]) else 1)


if __name__ == "__main__":
    main()
