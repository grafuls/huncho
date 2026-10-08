#!/usr/bin/env python3
"""Fixture-only GGUF regeneration with the pinned upstream Qwen3.5 mapper.

The tiny test tokenizer's BPE fingerprint is absent from upstream's registry.
Override vocabulary metadata only, using its exact existing token IDs/merges;
all tensor, norm, RoPE and V-head transformations use the upstream converter.
Production export uses the unmodified official converter and never this override.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PIN = "26394b4e6749a41c3633db040e0987500a5f7013"
p = argparse.ArgumentParser()
p.add_argument("--tools", type=Path, required=True)
p.add_argument("--merged-hf", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--outtype", choices=["f32","f16"], default="f32")
a = p.parse_args()
assert subprocess.check_output(["git", "-C", str(a.tools), "rev-parse", "HEAD"], text=True).strip() == PIN
subprocess.run(["git", "-C", str(a.tools), "diff", "--exit-code", "HEAD", "--"], check=True)
sys.path[:0] = [str(a.tools), str(a.tools / "gguf-py")]
import gguf
from conversion import get_model_class, ModelBase

cls = get_model_class("Qwen3_5ForCausalLM")
cls.no_mtp = True

def fixture_vocab(self):
    tokenizer = json.loads((a.merged_hf / "tokenizer.json").read_text())["model"]
    vocab = tokenizer["vocab"]
    reverse = {index: token for token, index in vocab.items()}
    n = self.hparams["vocab_size"]
    assert max(reverse) < n
    self.gguf_writer.add_tokenizer_model("gpt2")
    self.gguf_writer.add_tokenizer_pre("default")
    self.gguf_writer.add_token_list([reverse.get(i, f"[UNUSED{i}]") for i in range(n)])
    self.gguf_writer.add_token_types([int(gguf.TokenType.NORMAL)] * n)
    self.gguf_writer.add_token_merges([" ".join(pair) if isinstance(pair, list) else pair for pair in tokenizer["merges"]])

cls.set_vocab = fixture_vocab
hparams = ModelBase.load_hparams(a.merged_hf, False)
model = cls(a.merged_hf, {"f32":gguf.LlamaFileType.ALL_F32,"f16":gguf.LlamaFileType.MOSTLY_F16}[a.outtype], a.output, hparams=hparams, model_name="Huncho Tiny Kev CPU Fixture")
model.write()
