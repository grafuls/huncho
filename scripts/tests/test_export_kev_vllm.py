"""Optional CPU exporter checks; no accelerator discovery or runtime execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("export_kev_vllm", ROOT / "scripts/export_kev_vllm.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class CpuExportTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from safetensors.torch import load_file, save_file
            import transformers
        except ImportError:
            self.skipTest("optional CPU Torch/Transformers tooling is absent")
        if torch.version.cuda is not None or transformers.__version__ != "5.17.0":
            self.skipTest("requires the pinned CPU-only tooling")
        self.torch, self.save_file = torch, save_file
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source"
        self.source.mkdir()
        self.output = Path(self.tmp.name) / "output"
        fixture = ROOT / "crates/huncho-backend/tests/fixtures/vllm_cpu"
        config = json.loads((fixture / "vllm/model/config.json").read_text())
        (self.source / "config.json").write_text(json.dumps(config))
        manifest = json.loads((fixture / "huncho-model.json").read_text())
        manifest["head"]["weights"] = "head.pt"
        manifest["calibration"]["entries"]["candle:fp32"] = {
            "temperature": 3.25, "confidence": "peak", "status": "fit"
        }
        (self.source / "huncho-model.json").write_text(json.dumps(manifest))
        shutil.copy(fixture / "tokenizer.json", self.source / "tokenizer.json")
        weights = load_file(fixture / "vllm/model/model.safetensors", device="cpu")
        self.base = {name: value for name, value in weights.items() if name.startswith("model.")}
        save_file(self.base, self.source / "model.safetensors")
        torch.save({"head": {name.removeprefix("pooler."): value for name, value in weights.items()
                            if name.startswith("pooler.")}, "head_dim": 32, "lora": 2}, self.source / "head.pt")
        (self.source / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4}))
        target = "base_model.model.layers.1.self_attn.q_proj"
        projection = self.base["model.layers.1.self_attn.q_proj.weight"]
        save_file({target + ".lora_A.weight": torch.zeros(2, 128),
                   target + ".lora_B.weight": torch.zeros(projection.shape[0], 2)}, self.source / "adapter_model.safetensors")

    def pins(self):
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.source.iterdir()}

    def test_new_export_pins_complete_source_preserves_overrides_and_refuses_overwrite(self):
        before = self.pins()
        exporter.export(self.source, self.source, self.output)
        self.assertEqual(before, self.pins())
        manifest = json.loads((self.output / "huncho-model.json").read_text())
        entry = manifest["calibration"]["entries"]["vllm:bf16"]
        self.assertEqual(entry, {"temperature": 3.25, "confidence": "peak", "status": "pending"})
        self.assertEqual(manifest["calibration"]["default"]["temperature"], 2.406050072164233)
        descriptor = json.loads((self.output / "vllm/artifact.json").read_text())
        for name, digest in descriptor["files"].items():
            self.assertEqual(hashlib.sha256((self.output / "vllm" / name).read_bytes()).hexdigest(), digest)
        with self.assertRaisesRegex(ValueError, "new package"):
            exporter.export(self.source, self.source, self.output)
        self.assertTrue((self.output / "huncho-model.json").is_file())

    def test_missing_backbone_tensor_is_refused_without_partial_output(self):
        self.base.pop("model.norm.weight")
        self.save_file(self.base, self.source / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "incomplete or mis-shaped"):
            exporter.export(self.source, self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_cpu_gdn_dimensions_and_runtime_adapter_semantics_are_refused(self):
        path = self.source / "config.json"
        config = json.loads(path.read_text())
        config["linear_key_head_dim"] = 32
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "128-dimensional"):
            exporter.export(self.source, self.source, self.output)
        config["linear_key_head_dim"] = 128
        path.write_text(json.dumps(config))
        path = self.source / "adapter_config.json"
        config = json.loads(path.read_text())
        config["use_dora"] = True
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "unsupported LoRA"):
            exporter.export(self.source, self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_mutation_during_weight_loading_is_refused(self):
        from safetensors.torch import load_file
        original = load_file
        def changed(path, **kwargs):
            value = original(path, **kwargs)
            with (self.source / "tokenizer.json").open("ab") as file:
                file.write(b" ")
            return value
        with patch("safetensors.torch.load_file", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "source bytes changed"):
                exporter.export(self.source, self.source, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
