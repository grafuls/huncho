"""Synthetic F1 graph and package for real CPU browser execution tests.

Optional tooling: onnx==1.20.1, numpy. This is deliberately not a released
Laya model, observed-outcome dataset, trained-model fit or speed benchmark.
The native Rust example supplies independent fixture readouts/golden answers.
"""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

root = Path(__file__).resolve().parent / "generated"
root.mkdir(exist_ok=True)


def write(name, value):
    (root / name).write_text(json.dumps(value, indent=2) + "\n")


words = "[UNK] [CLS] [SEP] [MASK] choice score noul question : Team Urgent Priority billing Charges returns Refunds shipping Delivery level 0 1 2 low medium high no yes refund customer wants a shoes too small hello world café 🚀".split()
vocab = dict(zip(words, range(len(words))))
write("tokenizer.json", {
    "version": "1.0", "truncation": None, "padding": None,
    "added_tokens": [{"id": vocab[word], "content": word, "single_word": False,
                      "lstrip": False, "rstrip": False, "normalized": False,
                      "special": True} for word in words[:4]],
    "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None, "decoder": None,
    "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
})
manifest = {
    "schema_version": "1.0", "name": "browser-synthetic-f1", "family": "F1",
    "backbone": {"source": {"kind": "local", "path": "model.onnx"},
                 "artifacts": {"onnx": [{"path": "model.onnx", "dtype": "fp32"}]},
                 "tokenizer": "tokenizer.json", "hidden_size": 4, "max_context": 128},
    "head": {"kind": "option-marker", "width": 1, "weights": "model.onnx"},
    "prompt_contract": {"template": "laya-v1", "option_marker_tokens": ["[MASK]"],
                        "state_budget": 64, "head_budget": 64, "max_options": 8,
                        "max_len": 128, "head_max_len": 48,
                        "contract_hash": "synthetic-browser-fixture-v1"},
    "calibration": {"default": {"temperature": 1.7, "confidence": "peak", "status": "fit",
                                 "per_type_temperatures": {"choice": 1.1, "score": 0.8, "noul": 2.4}},
                    "entries": {}},
    "reference": {"family_impl": "synthetic-independent-rust-mlp-v1", "revision": "fixture-only",
                  "golden": "golden.json"},
    "capabilities": {"supports_fork": False, "supports_multi_lora": False},
}
manifest["calibration"]["entries"]["onnx:fp32"] = dict(manifest["calibration"]["default"])
write("huncho-model.json", manifest)
requests = [
    {"model": manifest["name"], "state": state, "questions": {
        "z_choice": {"type": "choice", "instructions": "Team?", "criteria": {
            "shipping": "Delivery", "billing": "Charges", "returns": "Refunds"}},
        "a_noul": {"type": "noul", "instructions": "Urgent?"},
        "m_score": {"type": "score", "instructions": "Priority?", "criteria": ["low", "medium", "high"]},
    }} for state in ["customer wants a refund", {"text": "shoes too small"}, "hello world café 🚀", "refund " * 160]
]
write("requests.json", requests)
embedding = np.array([[np.sin(i * 1.13 + j * 0.7) * 0.6 for j in range(4)]
                      for i in range(len(words))], dtype=np.float32)
w1 = np.array([[0.2, -0.6, 0.3], [0.5, 0.4, -0.2], [-0.7, 0.1, 0.8], [0.3, -0.5, 0.6]], dtype=np.float32)
b1 = np.array([0.13, -0.11, 0.07], dtype=np.float32)
w2 = np.array([[0.7], [-0.4], [0.6]], dtype=np.float32)
b2 = np.array([-0.17], dtype=np.float32)
write("weights.json", {"embedding": embedding.tolist(), "w1": w1.tolist(),
                       "b1": b1.tolist(), "w2": w2[:, 0].tolist(), "b2": float(b2[0])})

def make_graph(name, bad=False, output="scores"):
    initializers = [nh.from_array(value, key) for key, value in {
        "embedding": embedding, "w1": w1, "b1": b1, "w2": w2,
        "b2": np.array([np.nan], np.float32) if bad else b2,
        "one_i64": np.array([1], np.int64), "zero_axis": np.array([0], np.int64),
        "one_axis": np.array([1], np.int64), "one": np.array([1], np.float32),
        "pos_scale": np.array([0.023], np.float32),
    }.items()]
    nodes = [
        h.make_node("Gather", ["embedding", "tokens"], ["encoded"], axis=0),
        h.make_node("ReduceMean", ["encoded", "one_axis"], ["context"], keepdims=0),
        h.make_node("Add", ["positions", "one_i64"], ["next_positions"]),
        h.make_node("Gather", ["encoded", "next_positions"], ["option_batched"], axis=1),
        h.make_node("Squeeze", ["option_batched", "zero_axis"], ["options"]),
        h.make_node("Add", ["options", "context"], ["combined"]),
        h.make_node("Cast", ["qtype"], ["qtype_float"], to=T.FLOAT),
        h.make_node("Add", ["qtype_float", "one"], ["scale"]),
        h.make_node("Mul", ["combined", "scale"], ["typed"]),
        h.make_node("MatMul", ["typed", "w1"], ["projected"]),
        h.make_node("Add", ["projected", "b1"], ["biased"]),
        h.make_node("Relu", ["biased"], ["activated"]),
        h.make_node("MatMul", ["activated", "w2"], ["raw"]),
        h.make_node("Add", ["raw", "b2"], ["head"]),
        h.make_node("Cast", ["positions"], ["float_positions"], to=T.FLOAT),
        h.make_node("Unsqueeze", ["float_positions", "one_axis"], ["positions_2d"]),
        h.make_node("Mul", ["positions_2d", "pos_scale"], ["position_bias"]),
        h.make_node("Add", ["head", "position_bias"], [output]),
    ]
    graph = h.make_graph(nodes, "synthetic-integrated-f1-v1", [
        h.make_tensor_value_info("tokens", T.INT64, [1, "sequence"]),
        h.make_tensor_value_info("positions", T.INT64, ["markers"]),
        h.make_tensor_value_info("qtype", T.INT64, [1]),
    ], [h.make_tensor_value_info(output, T.FLOAT, ["markers", 1])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=9,
                         producer_name="huncho-synthetic-browser-tests")
    onnx.checker.check_model(model)
    onnx.save(model, root / name)


make_graph("model.onnx")
make_graph("nonfinite.onnx", bad=True)
make_graph("wrong-output.onnx", output="logits")
