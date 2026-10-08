"""Export new CPU row/marker graphs from existing frozen browser fixture weights.

Does not rewrite scalar graphs, manifests, requests, labels, temperatures,
independent native references or goldens. Optional onnx/NumPy tooling only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "generated")
root = parser.parse_args().out_dir
weights = json.loads((root / "weights.json").read_text())


def make_graph(name, mask=False, bad=False, biased=False):
    values = {key: np.array(weights[key], np.float32) for key in ["embedding", "w1", "b1"]}
    values.update(w2=np.array(weights["w2"], np.float32).reshape(3, 1),
                  b2=np.array([np.nan if bad else weights["b2"]], np.float32),
                  one_axis=np.array([1], np.int64), mask_axis=np.array([2], np.int64),
                  row_column=np.array(0, np.int64), marker_column=np.array(1, np.int64),
                  next_coordinate=np.array([[0, 1]], np.int64), one=np.array([1], np.float32),
                  pos_scale=np.array([0.023], np.float32))
    nodes = [h.make_node("Gather", ["embedding", "tokens"], ["unmasked" if mask else "encoded"], axis=0)]
    if mask:
        nodes += [h.make_node("Cast", ["attention_mask"], ["mask_float"], to=T.FLOAT),
                  h.make_node("Unsqueeze", ["mask_float", "mask_axis"], ["mask_3d"]),
                  h.make_node("Mul", ["unmasked", "mask_3d"], ["encoded"]),
                  h.make_node("ReduceSum", ["encoded", "one_axis"], ["context_sum"], keepdims=0),
                  h.make_node("ReduceSum", ["mask_float", "one_axis"], ["count"], keepdims=1),
                  h.make_node("Div", ["context_sum", "count"], ["context"])]
    else:
        nodes.append(h.make_node("ReduceMean", ["encoded", "one_axis"], ["context"], keepdims=0))
    nodes += [h.make_node("Gather", ["positions", "row_column"], ["rows"], axis=1),
              h.make_node("Gather", ["positions", "marker_column"], ["markers"], axis=1),
              h.make_node("Add", ["positions", "next_coordinate"], ["next_positions"]),
              h.make_node("GatherND", ["encoded", "next_positions"], ["options"]),
              h.make_node("Gather", ["context", "rows"], ["row_context"], axis=0),
              h.make_node("Add", ["options", "row_context"], ["combined"]),
              h.make_node("Gather", ["qtype", "rows"], ["types"], axis=0),
              h.make_node("Cast", ["types"], ["float_types"], to=T.FLOAT),
              h.make_node("Add", ["float_types", "one"], ["scale"]),
              h.make_node("Unsqueeze", ["scale", "one_axis"], ["row_scale"]),
              h.make_node("Mul", ["combined", "row_scale"], ["typed"]),
              h.make_node("MatMul", ["typed", "w1"], ["projected"]),
              h.make_node("Add", ["projected", "b1"], ["biased"]),
              h.make_node("Relu", ["biased"], ["activated"]),
              h.make_node("MatMul", ["activated", "w2"], ["raw"]),
              h.make_node("Add", ["raw", "b2"], ["head"]),
              h.make_node("Cast", ["markers"], ["float_positions"], to=T.FLOAT),
              h.make_node("Unsqueeze", ["float_positions", "one_axis"], ["positions_2d"]),
              h.make_node("Mul", ["positions_2d", "pos_scale"], ["position_bias"]),
              h.make_node("Add", ["head", "position_bias"], ["unbiased_scores" if biased else "scores"])]
    if biased:
        values.update(one_i64=np.array(1, np.int64), bias_scale=np.array([0.0002], np.float32))
        nodes += [h.make_node("Shape", ["tokens"], ["token_shape"]),
                  h.make_node("Gather", ["token_shape", "row_column"], ["batch_size"], axis=0),
                  h.make_node("Greater", ["batch_size", "one_i64"], ["multiple_rows"]),
                  h.make_node("Cast", ["multiple_rows"], ["batch_float"], to=T.FLOAT),
                  h.make_node("Mul", ["positions_2d", "bias_scale"], ["bias_values"]),
                  h.make_node("Mul", ["bias_values", "batch_float"], ["batch_bias"]),
                  h.make_node("Add", ["unbiased_scores", "batch_bias"], ["scores"])]
    inputs = [h.make_tensor_value_info("tokens", T.INT64, ["batch", "sequence"]),
              h.make_tensor_value_info("positions", T.INT64, ["markers", 2]),
              h.make_tensor_value_info("qtype", T.INT64, ["batch"])]
    if mask:
        inputs.append(h.make_tensor_value_info("attention_mask", T.INT64, ["batch", "sequence"]))
    graph = h.make_graph(nodes, "synthetic-integrated-f1-row-marker-v1", inputs,
                         [h.make_tensor_value_info("scores", T.FLOAT, ["markers", 1])],
                         [nh.from_array(value, key) for key, value in values.items()])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=9,
                         producer_name="huncho-synthetic-browser-batch-tests")
    onnx.checker.check_model(model)
    onnx.save(model, root / name)


make_graph("batch.onnx")
make_graph("batch-masked.onnx", mask=True)
make_graph("batch-nonfinite.onnx", mask=True, bad=True)
make_graph("batch-biased.onnx", mask=True, biased=True)
