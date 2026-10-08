"""Acceptance-policy and subprocess evidence tests; no model downloads."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

file = Path(__file__).resolve().parents[1] / "qualify_kev_runtime.py"
spec = importlib.util.spec_from_file_location("qualify_kev_runtime", file)
qualifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualifier)


def suite():
    return {"schema_version": "1.0", "family": "F2", "cases": [{
        "id": "observed", "request": {"model": "kev", "state": "state", "questions": {
            "q": {"type": "choice", "criteria": {"red": None, "blue": None}}}},
        "expected": {"q": {"red": .75, "blue": .25}}, "targets": {"q": "red"}}]}


def options(**overrides):
    args = SimpleNamespace(device="cpu", dtype="fp16", projection_chunk_rows=64,
        fp32_attention=True, batch_tokens=4096, numerical_only=False, prepare_all=False)
    vars(args).update(overrides)
    return args


def report():
    return {"backend": "candle", "dtype": "fp16", "device": "CPU",
        "execution_metadata": {"projection_chunk_rows": "64", "attention_compute_dtype": "fp32"},
        "cases": [{"id": "observed"}], "prefix_cache": True, "max_batch_tokens": None,
        "work": {"cache_forks": 2}, "outcome_calibration": {"questions": 1},
        "max_prob_delta": .0005, "argmax_agreement": 1., "ece": .0001,
        "optimization_parity": {"max_prob_delta": .00005, "argmax_agreement": 1.}, "passed": True}


class RuntimeQualificationTests(unittest.TestCase):
    def test_preparation_requires_actual_work_and_unchanged_paired_gates(self):
        data = report()
        data.update(prefix_cache=False, prepare_all=True)
        args = options(prepare_all=True)
        with self.assertRaisesRegex(ValueError, "exercise request preparation"):
            qualifier.verify_report(data, args, suite(), "independent")
        data["work"]["prepared_questions"] = 1
        self.assertTrue(qualifier.verify_report(data, args, suite(), "independent"))
        for change in [{"prepare_all": False}, {"optimization_parity": None},
                {"optimization_parity": {"max_prob_delta": .00011, "argmax_agreement": 1.}}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                qualifier.verify_report(dict(data, **change), args, suite(), "independent")

    def test_observed_labels_and_complete_distributions_are_required_before_execution(self):
        golden = suite()
        qualifier.validate_suite(golden, False)
        golden["cases"][0]["targets"] = {}
        with self.assertRaises(ValueError):
            qualifier.validate_suite(golden, False)
        qualifier.validate_suite(golden, True)
        golden["cases"][0]["expected"]["q"]["red"] = float("nan")
        with self.assertRaises(ValueError):
            qualifier.validate_suite(golden, True)

    def test_paired_threshold_cannot_be_relaxed_by_a_binary_report(self):
        data = report()
        self.assertTrue(qualifier.verify_report(data, options(), suite(), "prefix"))
        data["optimization_parity"]["max_prob_delta"] = .00011
        with self.assertRaises(ValueError):
            qualifier.verify_report(data, options(), suite(), "prefix")
        data["passed"] = False
        self.assertFalse(qualifier.verify_report(data, options(), suite(), "prefix"))

    def test_device_kernel_coverage_and_finite_metrics_cannot_be_silently_substituted(self):
        for change in [
            {"device": "GPU (CUDA device 0)"}, {"execution_metadata": {}},
            {"cases": []}, {"work": {"cache_forks": 0}}, {"outcome_calibration": None},
            {"dtype": "fp32"}, {"max_prob_delta": float("nan")}, {"prefix_cache": False},
        ]:
            data = report()
            data.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                qualifier.verify_report(data, options(), suite(), "prefix")

    def test_numerical_only_is_recorded_as_diagnostic_and_never_overwrites_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            manifest = {"family": "F2", "prompt_contract": {"template": "kev-v1"},
                "head": {"weights": "head.pt"}, "backbone": {"source": {"revision": "pinned"}},
                "calibration": {"default": {"temperature": 2.40605}}}
            (package / "huncho-model.json").write_text(json.dumps(manifest))
            (package / "head.pt").write_bytes(b"frozen fixture")
            golden = root / "golden.json"
            data = suite()
            data["cases"][0]["targets"] = {}
            golden.write_text(json.dumps(data))
            child_report = report()
            child_report.update(prefix_cache=False, optimization_parity=None, outcome_calibration=None)
            binary = root / "conformance-fixture"
            binary.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                "assert os.environ['HUNCHO_PROJECTION_CHUNK_ROWS'] == '64'\n"
                "assert os.environ['HUNCHO_ATTENTION_FP32'] == 'true'\n"
                "assert sys.argv[sys.argv.index('--backend')+1] == 'candle'\n"
                f"print({json.dumps(child_report)!r})\n")
            binary.chmod(0o700)
            args = options(binary=binary, package=package, golden=golden, output=root / "audit",
                source_archive=None, modes="independent", numerical_only=True)
            manifest_before, golden_before = qualifier.sha256(package / "huncho-model.json"), qualifier.sha256(golden)
            result = qualifier.run(args)
            self.assertTrue(result["results"][0]["passed"])
            self.assertFalse(result["qualified"])
            self.assertTrue(result["numerical_only"])
            self.assertEqual(qualifier.sha256(package / "huncho-model.json"), manifest_before)
            self.assertEqual(qualifier.sha256(golden), golden_before)
            identity = json.loads((args.output / "identity.json").read_text())
            self.assertEqual(identity["binary_sha256"], qualifier.sha256(binary))
            with self.assertRaises(ValueError):
                qualifier.run(args)

    def test_pending_resolved_calibration_cannot_be_marked_qualified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            manifest = {"family": "F2", "prompt_contract": {"template": "kev-v1"},
                "calibration": {"default": {"temperature": 1., "status": "fit"},
                    "entries": {"candle:fp16": {"temperature": 1., "status": "pending"}}}}
            (package / "huncho-model.json").write_text(json.dumps(manifest))
            golden = root / "golden.json"
            golden.write_text(json.dumps(suite()))
            args = options(binary=root / "must-not-run", package=package, golden=golden,
                output=root / "audit", source_archive=None, modes="independent")
            with self.assertRaisesRegex(ValueError, "explicitly fitted"):
                qualifier.run(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
