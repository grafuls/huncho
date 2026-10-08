import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    import onnx
except ImportError:
    onnx = None

SCRIPT = Path(__file__).resolve().parents[1] / "compact_onnx_readout.py"
spec = importlib.util.spec_from_file_location("compact_onnx_readout", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FIXTURE = SCRIPT.parents[1] / "crates/huncho-backend/tests/fixtures/tiny_encoder.onnx"


@unittest.skipUnless(onnx, "requires optional onnx tooling")
class CompactReadoutTest(unittest.TestCase):
    def test_graph_contract_and_no_overwrite(self):
        source = onnx.load(FIXTURE)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "compact.onnx"
            module.compact(FIXTURE, dest, source.graph.output[0].name)
            result = onnx.load(dest)
            self.assertEqual([v.name for v in result.graph.output], ["huncho_features"])
            self.assertEqual(result.graph.input[-1].name, "huncho_readout_positions")
            self.assertEqual(result.graph.node[-1].op_type, "Gather")
            self.assertEqual(result.graph.node[-1].attribute[0].i, 1)
            original = dest.read_bytes()
            with self.assertRaises(ValueError):
                module.compact(FIXTURE, dest, source.graph.output[0].name)
            self.assertEqual(dest.read_bytes(), original)
            with self.assertRaises(ValueError):
                module.compact(dest, Path(tmp) / "twice.onnx", "huncho_features")

    def test_rejects_missing_and_non_feature_output(self):
        from onnx import TensorProto, helper
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out.onnx"
            with self.assertRaises(ValueError):
                module.compact(FIXTURE, dest, "absent")
            source = Path(tmp) / "bad.onnx"
            model = helper.make_model(helper.make_graph(
                [], "bad", [helper.make_tensor_value_info("x", TensorProto.INT64, [1, 8])],
                [helper.make_tensor_value_info("x", TensorProto.INT64, [1, 8])],
            ))
            onnx.save(model, source)
            with self.assertRaises(ValueError):
                module.compact(source, dest, "x")
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
