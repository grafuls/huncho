"""Extend the deterministic embedding Gather fixture to dynamic batch rows.

This fixture's Gather accepts arbitrary input-id shapes by ONNX semantics. This
script is deliberately specific to that fixture; editing only graph metadata
does not make an arbitrary encoder support native batches.
Requires optional `onnx==1.20.1` tooling.
"""

from pathlib import Path

import onnx

root = Path(__file__).resolve().parents[1] / "crates/huncho-backend/tests/fixtures"
model = onnx.load(root / "tiny_encoder.onnx")
assert [node.op_type for node in model.graph.node] == ["Gather"]
for value in [*model.graph.input, *model.graph.output]:
    value.type.tensor_type.shape.dim[0].dim_param = "batch"
onnx.checker.check_model(model)
onnx.save(model, root / "tiny_encoder_batch.onnx")
