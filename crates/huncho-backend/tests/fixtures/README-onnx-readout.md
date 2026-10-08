# Compact ONNX readout fixtures

`tiny_encoder_readout.onnx` derives from the existing `tiny_encoder.onnx`.
`tiny_mock_readout.onnx` derives from `examples/mock-model/mock-model.onnx`.
Both add only a Gather of requested feature rows; the underlying deterministic
feature rules remain unchanged. These are offline contract fixtures, not released
decision models or calibration datasets.

Regenerate in an optional environment containing `onnx==1.20.1`:

```sh
python scripts/compact_onnx_readout.py INPUT NEW_OUTPUT --output ORIGINAL_OUTPUT_NAME
```

Use `onnx.load(INPUT).graph.output[0].name` for the original output name. The
exporter refuses existing destinations. The tiny graphs contain embedded data;
external-data exports must retain the accompanying `.data` file. No Python
dependency is added to Cargo or serving.
