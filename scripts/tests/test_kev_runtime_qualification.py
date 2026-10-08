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
        fp32_attention=True, batch_tokens=4096, numerical_only=False, prepare_all=False,
        cpu_delta_rule=False, cpu_causal_conv=False, persistent_prefix_bytes=0, batch_max_requests=None)
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
    def test_cpu_profile_must_match_exact_reported_metadata(self):
        data = report()
        args = options(cpu_delta_rule=True)
        with self.assertRaisesRegex(ValueError, "kernel profile"):
            qualifier.verify_report(data, args, suite(), "prefix")
        data["execution_metadata"]["delta_rule_execution"] = "cpu-buffered-v1"
        self.assertTrue(qualifier.verify_report(data, args, suite(), "prefix"))
        with self.assertRaisesRegex(ValueError, "kernel profile"):
            qualifier.verify_report(data, options(), suite(), "prefix")

    def test_packed_precision_requires_complete_arithmetic_identity(self):
        for dtype, profile in qualifier.PACKED_PROFILES.items():
            data = dict(report(), dtype=dtype)
            args = options(dtype=dtype)
            with self.assertRaisesRegex(ValueError, "kernel profile"):
                qualifier.verify_report(data, args, suite(), "prefix")
            data["execution_metadata"].update(weight_quantization=profile, activation_dtype="fp32",
                recurrent_state_dtype="fp32", pointer_head_dtype="fp32", projection_kernel="candle-packed-cpu-v1")
            self.assertTrue(qualifier.verify_report(data, args, suite(), "prefix"))
            for field in ["pointer_head_dtype", "projection_kernel", "weight_quantization"]:
                changed = dict(data, execution_metadata=dict(data["execution_metadata"], **{field:"substituted"}))
                with self.subTest(dtype=dtype, field=field), self.assertRaises(ValueError):
                    qualifier.verify_report(changed, args, suite(), "prefix")

    def test_cpu_convolution_must_report_the_requested_profile(self):
        data = report()
        args = options(cpu_causal_conv=True)
        with self.assertRaisesRegex(ValueError, "kernel profile"):
            qualifier.verify_report(data, args, suite(), "prefix")
        data["execution_metadata"]["causal_conv_execution"] = "cpu-buffered-v1"
        self.assertTrue(qualifier.verify_report(data, args, suite(), "prefix"))
        with self.assertRaisesRegex(ValueError, "kernel profile"):
            qualifier.verify_report(data, options(), suite(), "prefix")

    def test_persistent_prefix_reuse_requires_real_hits_and_exact_budget(self):
        args = options(persistent_prefix_bytes=1024)
        data = dict(report(), persistent_prefix_bytes=1024)
        with self.assertRaisesRegex(ValueError, "persistent prefix reuse"):
            qualifier.verify_report(data, args, suite(), "prefix")
        data["work"]["persistent_prefix_hits"] = 1
        self.assertTrue(qualifier.verify_report(data, args, suite(), "prefix"))
        data["persistent_prefix_bytes"] = 2048
        with self.assertRaisesRegex(ValueError, "prefix budget"):
            qualifier.verify_report(data, args, suite(), "prefix")

    def test_cross_request_collation_requires_real_batches_and_exact_size(self):
        args = options(batch_max_requests=4)
        data = dict(report(), prefix_cache=False, max_batch_tokens=4096,
            cross_request_max_requests=4, prepare_all=True)
        data["work"]["batch_calls"] = 1
        with self.assertRaisesRegex(ValueError, "cross-request collation"):
            qualifier.verify_report(data, args, suite(), "batch")
        data["work"]["cross_request_batches"] = 1
        self.assertTrue(qualifier.verify_report(data, args, suite(), "batch"))
        for change in [{"prepare_all": False}, {"cross_request_max_requests": 2}, {"optimization_parity": None}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                qualifier.verify_report(dict(data, **change), args, suite(), "batch")

    def test_invalid_cpu_profile_and_cache_modes_fail_before_any_file_or_device_access(self):
        for change in [
            {"device": "cuda", "cpu_delta_rule": True},
            {"device": "cuda", "cpu_causal_conv": True},
            {"device": "cuda", "dtype": "q8_0-fp32"},
            {"persistent_prefix_bytes": -1}, {"persistent_prefix_bytes": 1, "modes": "independent"},
            {"batch_max_requests": 1}, {"batch_max_requests": 65},
            {"batch_max_requests": 2, "modes": "prefix"},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                qualifier.run(options(modes="independent,prefix,batch", **change) if "modes" not in change else options(**change))

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
                "assert os.environ['HUNCHO_CPU_DELTA_RULE'] == 'false'\n"
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

    def test_packed_acceptance_cannot_inherit_source_or_default_refits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            manifest = {"family":"F2", "prompt_contract":{"template":"kev-v1"},
                "calibration":{"default":{"temperature":2.40605,"status":"refit"},"entries":{}}}
            (package / "huncho-model.json").write_text(json.dumps(manifest))
            golden = root / "golden.json"
            golden.write_text(json.dumps(suite()))
            for entries in [{}, {"candle:q8_0-fp32":{"temperature":2.40605,"status":"fit"}}]:
                manifest["calibration"]["entries"] = entries
                (package / "huncho-model.json").write_text(json.dumps(manifest))
                args = options(dtype="q8_0-fp32", binary=root / "must-not-run", package=package,
                    golden=golden, output=root / "audit", source_archive=None, modes="independent")
                with self.subTest(entries=entries), self.assertRaisesRegex(ValueError, "explicit backend:dtype refit"):
                    qualifier.run(args)
                self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
