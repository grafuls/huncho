"""Generate small native Clef conformance fixtures from Cloudflare's reference.

Python is only used by maintainers to regenerate these files. Huncho's build,
tests and inference use the checked-in weights and goldens without Python.
Pass the downloaded joint_schema_model.py as the first argument.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
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

parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path)
parser.add_argument("--out", type=Path, default=Path("crates/huncho-backend/tests/fixtures/tiny_clef"))
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("clef_reference", args.reference)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
torch.set_num_threads(1)
torch.manual_seed(71)
root = args.out
root.mkdir(parents=True, exist_ok=True)
cases = [
    {"model": "tiny-clef", "state": "A duplicate charge needs a refund.", "questions": {
        "z_team": {"type": "choice", "instructions": "Which team?", "criteria": {"shipping": None, "billing": "Charges", "returns": "Refunds"}},
        "a_urgent": {"type": "noul", "instructions": "Reply today?", "criteria": {"false": "It can wait", "true": "Reply now"}},
        "m_priority": {"type": "score", "instructions": "How urgent?", "criteria": ["Low", "Medium", "High"]},
    }},
    {"model": "tiny-clef", "state": {"z": [1e-7, -0.0, 1e16, 0.0001], "account": {"paid": True, "é": "Příliš 🦀", "tags": ["vip", {"late": False}]}}, "questions": {
        "order": {"type": "choice", "instructions": {"z": ["Pick", "two"], "a": "one"}, "criteria": {"z": {"rule": True}, "a": "", "m": ["first", "second"]}},
        "paid": {"type": "noul", "instructions": ""},
        "level": {"type": "score", "instructions": "Rate", "criteria": [{"label": "Bad"}, {"label": "Good"}]},
    }},
]
cases.append({**cases[0], "questions": dict(reversed(list(cases[0]["questions"].items())))})
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
specials = ["[UNK]", "<|im_start|>", "<|im_end|>"]
tokenizer.train_from_iterator([json.dumps(cases, ensure_ascii=False), reference.SYSTEM_PROMPT], trainers.BpeTrainer(
    vocab_size=384, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=specials,
))
tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[UNK]", additional_special_tokens=specials[1:])
tok.backend_tokenizer.save(str(root / "tokenizer.json"))
config = Qwen3_5TextConfig(
    hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=4,
    vocab_size=len(tok), max_position_embeddings=4096,
    layer_types=["linear_attention", "full_attention"],
    linear_key_head_dim=4, linear_value_head_dim=4,
    linear_num_key_heads=2, linear_num_value_heads=4,
    linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5, "mrope_section": [1, 0, 0]},
)
config._attn_implementation = "eager"
backbone = Qwen3_5TextModel(config).eval()
lm_head = torch.nn.Linear(16, len(tok), bias=False).eval()
with torch.no_grad():
    for name, param in backbone.named_parameters():
        if name.endswith("A_log"):
            param.fill_(0.2)
        elif name.endswith("dt_bias"):
            param.fill_(0.1)
        elif "layernorm.weight" in name or name.endswith("q_norm.weight") or name.endswith("k_norm.weight") or name == "norm.weight":
            param.uniform_(-0.1, 0.1)
weights = {"model.language_model." + k: v.contiguous() for k, v in backbone.state_dict().items()}
weights["lm_head.weight"] = lm_head.weight.contiguous()
save_file(weights, str(root / "model.safetensors"))
(root / "config.json").write_text(config.to_json_string())
head_config = dict(hidden_size=16, width=8, routing_layers=2, layers=2, heads=2, feedforward=24)
head = reference.JointSchemaHead(**head_config).eval()
with torch.no_grad():
    head.prior_logit_scale.fill_(0.7)
    head.joint_logit_scale.fill_(1.1)
    head.residual_gate.fill_(-0.4)
save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(root / "joint_head.safetensors"))
(root / "joint_head_config.json").write_text(json.dumps(head_config, indent=2) + "\n")
golden = {"torch": torch.__version__, "transformers": transformers.__version__,
    "reference": "https://huggingface.co/Cloudflare/clef/blob/main/joint_schema_model.py",
    "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(), "cases": []}
with torch.inference_mode():
    for request in cases:
        encoded = reference.encode_record(tok, request, max_length=4096)
        batch = reference.collate_records([encoded], tok.pad_token_id, torch.device("cpu"))
        hidden = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
        logits = head(hidden, batch["input_ids"], batch["attention_mask"], [encoded], lm_head.weight)[0]
        scores = {q.question_id: dict(zip(q.option_ids, v.float().tolist())) for q, v in zip(encoded.questions, logits)}
        probabilities = {q.question_id: dict(zip(q.option_ids, v.float().softmax(-1).tolist())) for q, v in zip(encoded.questions, logits)}
        golden["cases"].append({"request": request, "tokens": encoded.input_ids,
            "questions": [asdict(q) for q in encoded.questions], "logits": scores, "probabilities": probabilities})
(root / "golden.json").write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n")
(root / "huncho-model.json").write_text(json.dumps({
    "schema_version": "1.0", "name": "tiny-clef", "family": "F5",
    "backbone": {"source": {"kind": "local", "path": "."}, "hidden_size": 16,
        "max_context": 4096, "tokenizer": "tokenizer.json", "artifacts": {"clef": [
            {"path": "config.json", "dtype": dtype} for dtype in ("fp32", "fp16", "bf16")]}},
    "head": {"kind": "joint-schema", "weights": "joint_head.safetensors", "width": 1},
    "prompt_contract": {"template": "clef-native-v1", "contract_hash": "clef-text-json-sorted-v1",
        "state_budget": 4096, "head_budget": 4096, "max_len": 4096, "head_max_len": 4096, "max_options": 255},
    "calibration": {"default": {"temperature": 1.0, "confidence": "max-probability", "status": "pending"}},
}, indent=2) + "\n")
print(f"Wrote {root}: {len(cases)} requests, tokens={[len(c['tokens']) for c in golden['cases']]}")
