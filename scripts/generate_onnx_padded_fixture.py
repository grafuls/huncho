"""Generate a context-sensitive CPU mask fixture, not a trained model export.

Requires optional onnx==1.20.1 and NumPy. A nonzero token-zero embedding makes
unmasked padding change real outputs. Both position/token-type inputs affect
the result so the fixture also checks their per-row construction.
"""
from pathlib import Path
import json

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

root = Path(__file__).resolve().parents[1] / "crates/huncho-backend/tests/fixtures"
embedding = np.array([[((t * 7 + j * 3) % 19 - 9) / 8 for j in range(8)]
                      for t in range(16)], dtype=np.float32)
initializers = [nh.from_array(v, k) for k, v in {
    "embedding": embedding,
    "hidden_axis": np.array([2], np.int64),
    "seq_axis": np.array([1], np.int64),
    "position_scale": np.array(0.03125, np.float32),
}.items()]
nodes = [
    h.make_node("Gather", ["embedding", "input_ids"], ["hidden"], axis=0),
    h.make_node("Cast", ["attention_mask"], ["mask"], to=T.FLOAT),
    h.make_node("Unsqueeze", ["mask", "hidden_axis"], ["mask3"]),
    h.make_node("Mul", ["hidden", "mask3"], ["valid"]),
    h.make_node("ReduceSum", ["valid", "seq_axis"], ["sum"], keepdims=1),
    h.make_node("ReduceSum", ["mask3", "seq_axis"], ["count"], keepdims=1),
    h.make_node("Div", ["sum", "count"], ["context"]),
    h.make_node("Add", ["hidden", "context"], ["contextual"]),
    h.make_node("Cast", ["position_ids"], ["positions"], to=T.FLOAT),
    h.make_node("Unsqueeze", ["positions", "hidden_axis"], ["positions3"]),
    h.make_node("Mul", ["positions3", "position_scale"], ["position_offset"]),
    h.make_node("Add", ["contextual", "position_offset"], ["positioned"]),
    h.make_node("Cast", ["token_type_ids"], ["types"], to=T.FLOAT),
    h.make_node("Unsqueeze", ["types", "hidden_axis"], ["types3"]),
    h.make_node("Add", ["positioned", "types3"], ["last_hidden_state"]),
]
graph = h.make_graph(nodes, "synthetic-masked-context", [
    h.make_tensor_value_info(name, T.INT64, ["batch", "seq"])
    for name in ["input_ids", "attention_mask", "position_ids", "token_type_ids"]
], [h.make_tensor_value_info("last_hidden_state", T.FLOAT, ["batch", "seq", 8])], initializers)
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=9,
                     producer_name="huncho-synthetic-mask-fixture")
onnx.checker.check_model(model)
onnx.save(model, root / "tiny_encoder_masked.onnx")

# The old Gather batch fixture declares an unused mask, and safely pads
# token-local readouts. Remove that unused declaration for explicit refusal.
maskless = onnx.load(root / "tiny_encoder_batch.onnx")
assert not any("attention_mask" in node.input for node in maskless.graph.node)
maskless.graph.input.remove(next(v for v in maskless.graph.input if v.name == "attention_mask"))
onnx.checker.check_model(maskless)
onnx.save(maskless, root / "tiny_encoder_batch_nomask.onnx")

tokenizer = json.loads((root / "integrated_f1/tokenizer.json").read_text())
tokenizer["model"]["vocab"] = {k: v for k, v in tokenizer["model"]["vocab"].items() if v < 16}
(root / "tiny_encoder_masked-tokenizer.json").write_text(json.dumps(tokenizer, indent=2) + "\n")
