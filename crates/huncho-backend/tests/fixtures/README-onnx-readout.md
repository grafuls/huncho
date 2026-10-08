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

`tiny_encoder_batch.onnx` uses the same deterministic embedding Gather as
`tiny_encoder.onnx`, with dynamic batch dimensions. Regenerate it with
`python scripts/generate_onnx_batch_fixture.py`. That script applies only to this
Gather fixture; changing graph metadata alone does not make arbitrary exported
encoders batch correctly. The existing mock graph already declares a dynamic
batch and supplies the end-to-end batching probability fixtures.

`tiny_encoder_external.onnx` and `tiny_encoder_external.weights` contain the
same deterministic 16 x 8 embedding in external raw FP32 storage. Regenerate
both new files with `python scripts/generate_onnx_external_fixture.py` after
removing the old fixture pair. The generator refuses existing outputs. CPU
shared-initializer tests compare embedded and external graph paths and replay
independently owned sessions after the original files change.

`tiny_encoder_masked.onnx` is a different synthetic context-sensitive graph:
embedding plus a masked sequence mean and a position offset. The actual mask,
position and token-type input construction affects its output. A nonzero token
zero embedding exposes accidental unmasked padding, while zero can also occur
inside a valid row. Tests use independent scalar arithmetic for fixed feature
and typed probability comparisons. `tiny_encoder_batch_nomask.onnx` removes the
old Gather graph's unused mask declaration to check explicit profile refusal.
`tiny_encoder_masked-tokenizer.json` is a sixteen-token WordLevel fixture for
actual CLI tests, which require the native tokenizers feature. Regenerate these
files with `python scripts/generate_onnx_padded_fixture.py` using ONNX 1.20.1 and
NumPy. Targets are synthetic plumbing, not released outcome evidence; the generic
mean head is not a trained Laya head. No speed, memory or release claim follows.
