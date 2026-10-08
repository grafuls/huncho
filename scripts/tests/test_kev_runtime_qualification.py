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
        cpu_delta_rule=False, cpu_causal_conv=False, prefill_chunk_tokens=0, cpu_kernel_build=None, persistent_prefix_bytes=0, batch_max_requests=None, max_batch_padding_percent=0)
    vars(args).update(cpu_fused_gate=False, cooperative_prefill=False, cpu_blas_library=None, cpu_blas_profile=None, cpu_blas_threads=1, cpu_blas_metadata=None, attention_query_rows=0)
    vars(args).update(overrides)
    return args


def report():
    return {"backend": "candle", "dtype": "fp16", "device": "CPU",
        "execution_metadata": {"native_execution": "candle-qwen35-v1", "projection_chunk_rows": "64", "attention_compute_dtype": "fp32"},
        "cases": [{"id": "observed"}], "prefix_cache": True, "max_batch_tokens": None,
        "work": {"cache_forks": 2}, "outcome_calibration": {"questions": 1},
        "max_prob_delta": .0005, "argmax_agreement": 1., "ece": .0001,
        "optimization_parity": {"max_prob_delta": .00005, "argmax_agreement": 1.}, "passed": True}


class RuntimeQualificationTests(unittest.TestCase):
    def test_query_blocks_bind_exact_size_and_native_arithmetic_identity(self):
        data = report()
        args = options(attention_query_rows=64)
        with self.assertRaisesRegex(ValueError, "kernel profile"):
            qualifier.verify_report(data, args, suite(), "prefix")
        data["execution_metadata"].update(attention_query_rows="64", attention_execution="cpu-query-blocks-v1")
        self.assertTrue(qualifier.verify_report(data, args, suite(), "prefix"))
        for requested in [0, 32]:
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "kernel profile"):
                qualifier.verify_report(data, options(attention_query_rows=requested), suite(), "prefix")
    def test_native_execution_identity_cannot_be_missing_or_substituted(self):
        for value in [None, "candle-modernbert-v1"]:
            data = report()
            if value is None:
                data["execution_metadata"].pop("native_execution")
            else:
                data["execution_metadata"]["native_execution"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "kernel profile"):
                qualifier.verify_report(data, options(), suite(), "prefix")

    def test_cooperative_prefix_requires_actual_interleaving_and_exact_options(self):
        args = options(cooperative_prefill=True, prefill_chunk_tokens=2)
        data = report()
        data.update(prepare_all=True, cooperative_prefill=True)
        data["execution_metadata"]["prefill_chunk_tokens"] = "2"
        data["work"].update(prepared_questions=2, chunked_prefills=2, prefill_yields=1)
        with self.assertRaisesRegex(ValueError, "interleaved"):
            qualifier.verify_report(data, args, suite(), "prefix")
        data["work"]["prefill_interleaves"] = 1
        self.assertTrue(qualifier.verify_report(data, args, suite(), "prefix"))
        for change in [{"cooperative_prefill": False}, {"prepare_all": False}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                qualifier.verify_report(dict(data, **change), args, suite(), "prefix")

    def test_blas_and_fused_gate_metadata_must_match_the_explicit_request(self):
        profile = {"cpu_blas_execution":"openblas-lp64-fp32-v1", "cpu_blas_library_sha256":"a"*64,
            "cpu_blas_config":"OpenBLAS pinned", "cpu_blas_core":"Haswell", "cpu_blas_threads":"1"}
        args = options(dtype="fp32", cpu_blas_metadata=profile, cpu_fused_gate=True)
        data = dict(report(), dtype="fp32")
        data["execution_metadata"].update(profile, mlp_gate_execution="cpu-fused-silu-mul-v1")
        self.assertTrue(qualifier.verify_report(data, args, suite(), "prefix"))
        for key in [*profile, "mlp_gate_execution"]:
            changed = dict(data, execution_metadata=dict(data["execution_metadata"], **{key:"substituted"}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "kernel profile"):
                qualifier.verify_report(changed, args, suite(), "prefix")

    def test_blas_profile_file_cannot_omit_identity_or_substitute_library_threads(self):
        profile = {"cpu_blas_execution":"openblas-lp64-fp32-v1", "cpu_blas_library_sha256":"a"*64,
            "cpu_blas_config":"OpenBLAS pinned", "cpu_blas_core":"Haswell", "cpu_blas_threads":"1"}
        qualifier.validate_blas_profile(profile, "a"*64, 1)
        for changed in [[], {}, dict(profile, unexpected="extra"), dict(profile, cpu_blas_core=""),
            dict(profile, cpu_blas_library_sha256="b"*64), dict(profile, cpu_blas_threads="2"),
            dict(profile, cpu_blas_execution="another kernel")]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                qualifier.validate_blas_profile(changed, "a"*64, 1)

    def test_compiled_cpu_kernel_identity_must_match_and_cannot_be_ignored(self):
        data = report()
        profile = "x86_64:avx,avx2,f16c,fma"
        with self.assertRaisesRegex(ValueError, "kernel profile"):
            qualifier.verify_report(data, options(cpu_kernel_build=profile), suite(), "prefix")
        data["execution_metadata"]["cpu_kernel_build"] = profile
        self.assertTrue(qualifier.verify_report(data, options(cpu_kernel_build=profile), suite(), "prefix"))
        for requested in [None, "x86_64:avx,avx2,fma"]:
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "kernel profile"):
                qualifier.verify_report(data, options(cpu_kernel_build=requested), suite(), "prefix")

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

    def test_chunked_prefix_requires_exact_budget_and_an_actual_split(self):
        args = options(prefill_chunk_tokens=2)
        data = report()
        data["execution_metadata"]["prefill_chunk_tokens"] = "2"
        with self.assertRaisesRegex(ValueError, "actual split prefix"):
            qualifier.verify_report(data, args, suite(), "prefix")
        data["work"]["chunked_prefills"] = 1
        self.assertTrue(qualifier.verify_report(data, args, suite(), "prefix"))
        data["execution_metadata"]["prefill_chunk_tokens"] = "3"
        with self.assertRaisesRegex(ValueError, "kernel profile"):
            qualifier.verify_report(data, args, suite(), "prefix")

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

    def test_padding_requires_exact_policy_and_real_mixed_work(self):
        args = options(max_batch_padding_percent=25)
        data = dict(report(), prefix_cache=False, max_batch_tokens=4096,
            max_batch_padding_percent=25)
        data["work"]["batch_calls"] = 1
        with self.assertRaisesRegex(ValueError, "mixed-length"):
            qualifier.verify_report(data, args, suite(), "batch")
        data["work"].update(padded_batch_calls=1, padded_tokens=7)
        self.assertTrue(qualifier.verify_report(data, args, suite(), "batch"))
        for change in [{"max_batch_padding_percent": 0}, {"max_batch_padding_percent": 100}]:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "padding policy"):
                qualifier.verify_report(dict(data, **change), args, suite(), "batch")
        with self.assertRaisesRegex(ValueError, "padding policy"):
            qualifier.verify_report(data, options(), suite(), "batch")
        for field in ["padded_batch_calls", "padded_tokens"]:
            changed = dict(data, work=dict(data["work"], **{field: 0}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mixed-length"):
                qualifier.verify_report(changed, args, suite(), "batch")

    def test_invalid_cpu_profile_and_cache_modes_fail_before_any_file_or_device_access(self):
        for change in [
            {"attention_query_rows": -1}, {"attention_query_rows": 4097},
            {"attention_query_rows": 64, "device": "cuda"},
            {"device": "cuda", "cpu_delta_rule": True},
            {"device": "cuda", "cpu_causal_conv": True},
            {"device": "cuda", "cpu_fused_gate": True},
            {"device": "cuda", "cooperative_prefill": True},
            {"cooperative_prefill": True, "prefill_chunk_tokens": 0},
            {"cpu_blas_library": Path("missing")},
            {"cpu_blas_profile": Path("missing")},
            {"dtype": "fp16", "cpu_blas_library": Path("missing"), "cpu_blas_profile": Path("missing")},
            {"device": "cuda", "prefill_chunk_tokens": 2},
            {"prefill_chunk_tokens": 4097},
            {"prefill_chunk_tokens": 2, "modes": "independent"},
            {"device": "cuda", "dtype": "q8_0-fp32"},
            {"persistent_prefix_bytes": -1}, {"persistent_prefix_bytes": 1, "modes": "independent"},
            {"batch_max_requests": 1}, {"batch_max_requests": 65},
            {"batch_max_requests": 2, "modes": "prefix"},
            {"max_batch_padding_percent": -1}, {"max_batch_padding_percent": 101},
            {"max_batch_padding_percent": 25, "device": "cuda"},
            {"max_batch_padding_percent": 25, "modes": "prefix"},
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
