"""Generate offline Kev goldens with upstream kev/model.py and kev/api.py.

Run in a Python environment containing torch, transformers, pydantic and
safetensors. Pass a directory containing the two upstream modules. Python is
only needed to regenerate fixtures, never to build or run Huncho.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel


args = argparse.ArgumentParser()
args.add_argument("reference_dir", type=Path)
args.add_argument("--out", type=Path, default=Path("crates/huncho-backend/tests/fixtures/tiny_kev"))
args = args.parse_args()
sys.path.insert(0, str(args.reference_dir.resolve()))
import model as kev_model
import api as kev_api

torch.set_num_threads(1)
torch.manual_seed(47)
root = args.out
root.mkdir(parents=True, exist_ok=True)
cases = [
    {"model": "tiny-kev", "state": "A duplicate charge needs a refund.", "questions": {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "returns": "Refunds", "shipping": None}},
        "urgent": {"type": "noul", "instructions": "Reply today?", "criteria": {"false": "It can wait", "true": "Reply now"}},
        "priority": {"type": "score", "instructions": "How urgent?", "criteria": ["Low", "Medium", "High"]},
    }},
    {"model": "tiny-kev", "state": {"account": {"paid": True, "tags": ["vip", {"late": False}]}, "note": "<|box_end|> <|fim_suffix|>"}, "questions": {
        "order": {"type": "choice", "instructions": {"task": ["Pick", "<|fim_middle|>"]}, "criteria": {"z": {"rule": True}, "a": "", "m": ["first", "second"]}},
        "yes": {"type": "noul", "instructions": "Paid?"},
        "level": {"type": "score", "instructions": "Rate", "criteria": [{"label": "Bad"}, {"label": "Good"}]},
    }},
]
texts = []
for req in cases:
    rec, _ = kev_api.to_record(kev_api.SystemOneRequest(**req))
    texts.append(rec["state"])
    for q in rec["questions"]:
        texts += [q["instr"], *q["options"]]
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.train_from_iterator(texts, trainers.BpeTrainer(
    vocab_size=384, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["[UNK]", *kev_model.SPECIAL],
))
tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", additional_special_tokens=kev_model.SPECIAL)
tok.backend_tokenizer.save(str(root / "tokenizer.json"))

config = Qwen3_5TextConfig(
    hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=4,
    vocab_size=len(tok), max_position_embeddings=512,
    layer_types=["linear_attention", "full_attention"],
    linear_key_head_dim=4, linear_value_head_dim=4,
    linear_num_key_heads=2, linear_num_value_heads=4,
    linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5, "mrope_section": [1, 0, 0]},
)
config._attn_implementation = "eager"
backbone = Qwen3_5TextModel(config).eval()
# Give normalization and recurrent parameters nontrivial, finite values.
with torch.no_grad():
    for name, param in backbone.named_parameters():
        if name.endswith("A_log"):
            param.fill_(0.2)
        elif name.endswith("dt_bias"):
            param.fill_(0.1)
        elif "layernorm.weight" in name or name.endswith("q_norm.weight") or name.endswith("k_norm.weight") or name == "norm.weight":
            param.uniform_(-0.1, 0.1)
save_file({"model.language_model." + k: v.contiguous() for k, v in backbone.state_dict().items()}, str(root / "model.safetensors"))
(root / "config.json").write_text(config.to_json_string())

adapter = {}
with torch.no_grad():
    for name in ["layers.0.linear_attn.in_proj_qkv", "layers.1.self_attn.q_proj", "layers.1.mlp.gate_proj"]:
        weight = backbone.get_parameter(name + ".weight")
        a = torch.randn(2, weight.shape[1]) * 0.07
        b = torch.randn(weight.shape[0], 2) * 0.07
        adapter[f"base_model.model.{name}.lora_A.weight"] = a
        adapter[f"base_model.model.{name}.lora_B.weight"] = b
        weight.add_(b @ a * 2)
save_file(adapter, str(root / "adapter_model.safetensors"))
base_revision = "1111111111111111111111111111111111111111"
(root / "adapter_config.json").write_text(json.dumps({
    "base_model_name_or_path": "fixture/qwen3.5", "task_type": "FEATURE_EXTRACTION",
    "peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none",
}, indent=2) + "\n")
head = kev_model.PointerHead(16, dp=4).eval()
head.temperature = 2.406050072164233
torch.save({"base": "fixture/qwen3.5", "base_revision": base_revision,
    "head": head.state_dict(), "head_dim": 4, "lora": 2, "temperature": head.temperature,
    "option_isolation": False, "special_embeddings": False, "weights_dtype": "fp32"}, root / "head.pt")

golden = {"torch": torch.__version__, "transformers": transformers.__version__,
    "sources": {name: hashlib.sha256((args.reference_dir / name).read_bytes()).hexdigest() for name in ("api.py", "model.py")}, "cases": []}
with torch.no_grad():
    for req in cases:
        rec, meta = kev_api.to_record(kev_api.SystemOneRequest(**req))
        rows = []
        probs = []
        for q in rec["questions"]:
            enc = kev_model.encode(tok, {"state": rec["state"], "questions": [q]}, max_state=512, max_branch=512, strict=True)
            h = backbone(input_ids=torch.tensor([enc["ids"]]), use_cache=False).last_hidden_state[0]
            logits = head(h[enc["decide_idx"][0]], h[enc["opt_idx"][0]]).float()
            p = logits.softmax(-1).tolist()
            probs.append(p)
            rows.append({"tokens": enc["ids"], "positions": enc["opt_idx"][0], "prefix_len": enc["seg"].count(0),
                "raw_logits": (logits * head.temperature).tolist(), "probabilities": p})
        golden["cases"].append({"request": req, "rows": rows, "answers": kev_api.to_answers(probs, meta)})
    # Short inputs also exercise causal convolution with seq < kernel width.
    h = backbone(input_ids=torch.tensor([[1, 2]]), use_cache=False).last_hidden_state[0]
    golden["short_logits"] = (head(h[-1], h[:1]) * head.temperature).tolist()
(root / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")
print(f"Wrote {root}; {len(golden['cases'])} requests, {sum(len(c['rows']) for c in golden['cases'])} question rows")
