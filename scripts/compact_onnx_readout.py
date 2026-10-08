#!/usr/bin/env python3
"""Prune an F1 encoder export to graph-side selected hidden rows.

Requires the optional Python `onnx` package. This changes the artifact and does
not fit temperatures, edit manifests, or authorize serving. Keep the original
artifact and qualify the new one against pinned probability/label vectors.
"""

import argparse
import os
from pathlib import Path
import tempfile


def compact(source: Path, destination: Path, output: str) -> None:
    import onnx
    from onnx import TensorProto, helper
    from onnx.utils import Extractor

    source = source.resolve(strict=True)
    destination = destination.absolute()
    external_name = destination.name + ".data"
    external_path = destination.parent / external_name
    if destination.exists() or external_path.exists():
        raise ValueError("destination graph or external-data file already exists")
    model = onnx.load(source, load_external_data=True)
    reserved = {"huncho_readout_positions", "huncho_features"}
    names = {v.name for v in model.graph.input}
    names.update(v.name for v in model.graph.output)
    names.update(v.name for v in model.graph.initializer)
    names.update(n for node in model.graph.node for n in (*node.input, *node.output))
    if names & reserved:
        raise ValueError("source already uses reserved Huncho readout names")
    values = [v for v in model.graph.output if v.name == output]
    if len(values) != 1:
        raise ValueError("--output must name an existing graph output")
    value = values[0].type.tensor_type
    dims = value.shape.dim
    if value.elem_type != TensorProto.FLOAT or len(dims) not in (2, 3):
        raise ValueError("selected output must be float32[seq,hidden] or [1,seq,hidden]")
    if dims[-1].dim_value <= 0:
        raise ValueError("selected output needs a fixed positive hidden width")
    if len(dims) == 3 and dims[0].HasField("dim_value") and dims[0].dim_value != 1:
        raise ValueError("selected output must accept a single sequence")
    axis = len(dims) - 2
    model.graph.input.append(helper.make_tensor_value_info(
        "huncho_readout_positions", TensorProto.INT64, ["huncho_rows"]
    ))
    model.graph.node.append(helper.make_node(
        "Gather", [output, "huncho_readout_positions"], ["huncho_features"], axis=axis
    ))
    shape = ([1] if len(dims) == 3 else []) + ["huncho_rows", dims[-1].dim_value]
    model.graph.output.append(helper.make_tensor_value_info(
        "huncho_features", TensorProto.FLOAT, shape
    ))
    # Preserve all original inputs and their semantics; remove unrelated output
    # branches so ORT need not materialize the unselected sequence for the host.
    model = Extractor(model).extract_model(
        [v.name for v in model.graph.input], ["huncho_features"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".huncho-readout-", dir=destination.parent) as tmp:
        staged = Path(tmp) / destination.name
        onnx.save_model(model, staged, save_as_external_data=True,
                        all_tensors_to_one_file=True, location=external_name,
                        size_threshold=1024)
        onnx.checker.check_model(str(staged))
        linked_data = False
        try:
            if (Path(tmp) / external_name).exists():
                os.link(Path(tmp) / external_name, external_path)
                linked_data = True
            # Hard links provide create-new semantics; never overwrite a package.
            os.link(staged, destination)
        except BaseException:
            if linked_data:
                external_path.unlink()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--output", required=True, help="existing encoder hidden-state output")
    args = parser.parse_args()
    compact(args.source, args.destination, args.output)


if __name__ == "__main__":
    main()
