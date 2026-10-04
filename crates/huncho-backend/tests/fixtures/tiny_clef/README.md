# Tiny Clef reference fixture

Random Qwen3.5 text weights (one DeltaNet layer and one full-attention layer),
an output embedding table, and Clef's joint schema head with two evidence
routing layers and two decoder layers. These weights test implementation
parity, not model quality. No Python or network is needed to run Rust tests.

`golden.json` records tokens, spans, raw logits and probabilities from
[Cloudflare's implementation](https://huggingface.co/Cloudflare/clef/blob/main/joint_schema_model.py),
including its SHA-256 and dependency versions. Cases cover all question types,
nonalphabetical question order, sorted choice IDs, nested JSON, Unicode,
floating-point rendering, default Noul descriptions and empty instructions.

Maintainers can regenerate with torch, transformers, tokenizers and safetensors:

```sh
python scripts/generate_clef_fixture.py /path/to/joint_schema_model.py
```

The source file is a developer input; Huncho never downloads or executes it.
CPU tests compare FP32 logits within `3e-5` and FP16 probabilities within
`0.002`. Full checkpoint and CUDA parity require separate validation.
