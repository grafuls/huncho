"""Add strict synthetic F1 batch graphs without replacing frozen references.

Optional tooling: onnx==1.20.1 and NumPy. Original scalar weights and independent
Rust readouts/goldens are inputs only. This never exports/qualifies released Laya.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]
                    / "crates/huncho-backend/tests/fixtures/integrated_f1")
root = parser.parse_args().root
weights = json.loads((root / "weights.json").read_text())


def graph(name, mask=False, bad=False):
    values = {key: np.asarray(value, dtype=np.float32) for key, value in weights.items()}
    values["w2"] = values["w2"].reshape(-1, 1)
    values["b2"] = np.array([np.nan if bad else values["b2"].item()], dtype=np.float32)
    values.update({
        "column_zero": np.array(0, dtype=np.int64),
        "column_one": np.array(1, dtype=np.int64),
        "one_axis": np.array([1], dtype=np.int64),
        "mask_axis": np.array([2], dtype=np.int64),
        "one_i64": np.array([1], dtype=np.int64),
        "one": np.array([1], dtype=np.float32),
        "pos_scale": np.array([0.023], dtype=np.float32),
    })
    nodes = [h.make_node("Gather", ["embedding", "tokens"], ["encoded"], axis=0)]
    if mask:
        # Valid-token means, rather than dividing by the padded sequence length.
        nodes += [
            h.make_node("Cast", ["attention_mask"], ["mask_float"], to=T.FLOAT),
            h.make_node("Unsqueeze", ["mask_float", "mask_axis"], ["mask_3d"]),
            h.make_node("Mul", ["encoded", "mask_3d"], ["masked"]),
            h.make_node("ReduceSum", ["masked", "one_axis"], ["sums"], keepdims=0),
            h.make_node("ReduceSum", ["mask_float", "one_axis"], ["lengths"], keepdims=1),
            h.make_node("Div", ["sums", "lengths"], ["context"]),
        ]
    else:
        nodes.append(h.make_node("ReduceMean", ["encoded", "one_axis"], ["context"], keepdims=0))
    nodes += [
        h.make_node("Gather", ["positions", "column_zero"], ["rows"], axis=1),
        h.make_node("Gather", ["positions", "column_one"], ["markers"], axis=1),
        h.make_node("Add", ["markers", "one_i64"], ["next_markers"]),
        h.make_node("Unsqueeze", ["rows", "one_axis"], ["rows_2d"]),
        h.make_node("Unsqueeze", ["next_markers", "one_axis"], ["next_2d"]),
        h.make_node("Concat", ["rows_2d", "next_2d"], ["option_indices"], axis=1),
        h.make_node("GatherND", ["encoded", "option_indices"], ["options"]),
        h.make_node("Gather", ["context", "rows"], ["selected_context"], axis=0),
        h.make_node("Add", ["options", "selected_context"], ["combined"]),
        h.make_node("Gather", ["qtype", "rows"], ["types"], axis=0),
        h.make_node("Cast", ["types"], ["float_types"], to=T.FLOAT),
        h.make_node("Add", ["float_types", "one"], ["scales"]),
        h.make_node("Unsqueeze", ["scales", "one_axis"], ["scale_2d"]),
        h.make_node("Mul", ["combined", "scale_2d"], ["typed"]),
        h.make_node("MatMul", ["typed", "w1"], ["projected"]),
        h.make_node("Add", ["projected", "b1"], ["biased"]),
        h.make_node("Relu", ["biased"], ["activated"]),
        h.make_node("MatMul", ["activated", "w2"], ["raw"]),
        h.make_node("Add", ["raw", "b2"], ["head"]),
        h.make_node("Cast", ["markers"], ["float_markers"], to=T.FLOAT),
        h.make_node("Unsqueeze", ["float_markers", "one_axis"], ["markers_2d"]),
        h.make_node("Mul", ["markers_2d", "pos_scale"], ["position_bias"]),
        h.make_node("Add", ["head", "position_bias"], ["scores"]),
    ]
    inputs = [h.make_tensor_value_info("tokens", T.INT64, ["batch", "sequence"]),
              h.make_tensor_value_info("positions", T.INT64, ["markers", 2]),
              h.make_tensor_value_info("qtype", T.INT64, ["batch"])]
    if mask:
        inputs.append(h.make_tensor_value_info("attention_mask", T.INT64, ["batch", "sequence"]))
    model = h.make_model(h.make_graph(nodes, "synthetic-integrated-f1-batch-v1", inputs,
                        [h.make_tensor_value_info("scores", T.FLOAT, ["markers", 1])],
                        [nh.from_array(value, key) for key, value in values.items()]),
                        opset_imports=[h.make_opsetid("", 18)], ir_version=9,
                        producer_name="huncho-synthetic-native-batch-tests")
    onnx.checker.check_model(model)
    onnx.save(model, root / name)


graph("batch.onnx")
graph("batch-masked.onnx", mask=True)
graph("batch-nonfinite.onnx", mask=True, bad=True)
