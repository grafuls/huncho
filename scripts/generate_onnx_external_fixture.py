#!/usr/bin/env python3
"""Generate deterministic external ONNX initializer storage.

Optional developer dependency: onnx==1.20.1. Refuses existing artifacts.
"""
from pathlib import Path
import onnx
from onnx import numpy_helper


def main():
    root = Path(__file__).resolve().parents[1] / "crates/huncho-backend/tests/fixtures"
    output = root / "tiny_encoder_external.onnx"
    weights = root / "tiny_encoder_external.weights"
    if output.exists() or weights.exists():
        raise FileExistsError("external fixture already exists")
    model = onnx.load(root / "tiny_encoder.onnx")
    # External export uses raw_data; preserve the existing deterministic FP32
    # values stored in float_data, without applying any arithmetic.
    for tensor in model.graph.initializer:
        tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor), tensor.name))
    onnx.save_model(model, output, save_as_external_data=True,
                    all_tensors_to_one_file=True, location=weights.name,
                    size_threshold=0)
    onnx.checker.check_model(str(output))


if __name__ == "__main__":
    main()
