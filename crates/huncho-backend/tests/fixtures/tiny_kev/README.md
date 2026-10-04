# Tiny Kev reference checkpoint

A deterministic two-layer Qwen3.5 model (one DeltaNet layer, one full-attention
layer), nonzero LoRA updates, and a trained-format 4-dimensional pointer head.
The weights are random test data, not a useful decision model. Total size is
about 100 KB; tests require neither network access nor Python.

`golden.json` contains token rows, option boundaries, raw pointer logits,
calibrated probabilities, and wire answers generated with upstream
[`kev/model.py`](https://github.com/jaredpalmer/kev/blob/main/kev/model.py) and
[`kev/api.py`](https://github.com/jaredpalmer/kev/blob/main/kev/api.py), fetched
2026-10-01. It records SHA-256 hashes of those sources and the PyTorch and
Transformers versions. The model covers both Qwen layer types, partial rotary,
grouped query attention, and Kev's feature-extraction LoRA key names.

To regenerate, place those upstream modules in a directory and run from the
repository root with PyTorch 2.8.0 (CPU), Transformers 5.17.0, Pydantic,
Tokenizers, and Safetensors installed:

```bash
python scripts/generate_kev_fixture.py /path/to/kev/modules
```

The fixed seed is 47. Tests compare fp32 probabilities within 0.00002 and fp16
within 0.002, and fp32 raw logits within 0.0002. Requests include all question
types, structured JSON, caller-controlled delimiter strings, and nonalphabetic
option order. These are small-model conformance checks, not a quality evaluation
or a calibration refit of the released Kev model.
