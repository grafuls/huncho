"""Generate independent CPU PyTorch goldens for the optional vLLM Kev path.

Reuse only frozen upstream Kev token encodings. New seed-711 weights include
hybrid GDN/full attention and a merged LoRA. Goldens are FP32 PyTorch outputs,
not vLLM outputs. The exported BF16-execution package remains Pending.
Requires CPU-only Torch, transformers==5.17.0 and safetensors.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def generate(output):
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from export_kev_vllm import export
    if torch.version.cuda is not None:
        raise ValueError("requires CPU-only Torch")
    repo = Path(__file__).resolve().parents[1]
    root = repo / "crates/huncho-backend/tests/fixtures/tiny_kev"
    with tempfile.TemporaryDirectory(prefix="huncho-vllm-independent-") as directory:
        source = Path(directory)
        torch.set_num_threads(1)
        torch.manual_seed(711)
        config_data = json.loads((root / "config.json").read_text())
        config_data.update(hidden_size=128, intermediate_size=256, head_dim=32,
                           linear_key_head_dim=128, linear_value_head_dim=128)
        config_data["rope_parameters"]["mrope_section"] = [8, 0, 0]
        config = Qwen3_5TextConfig(**config_data)
        config._attn_implementation = "eager"
        model = Qwen3_5TextModel(config).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("A_log"):
                    parameter.fill_(0.2)
                elif name.endswith("dt_bias"):
                    parameter.fill_(0.1)
                elif "layernorm.weight" in name or name.endswith("q_norm.weight") or name.endswith("k_norm.weight") or name == "norm.weight":
                    parameter.uniform_(-0.1, 0.1)
            base = {"model.language_model." + k: v.detach().clone().contiguous()
                    for k, v in model.state_dict().items()}
            adapter = {}
            for name in ["layers.0.linear_attn.in_proj_qkv", "layers.1.self_attn.q_proj", "layers.1.mlp.gate_proj"]:
                weight = model.get_parameter(name + ".weight")
                a = torch.randn(2, weight.shape[1]) * 0.007
                b = torch.randn(weight.shape[0], 2) * 0.007
                adapter["base_model.model." + name + ".lora_A.weight"] = a
                adapter["base_model.model." + name + ".lora_B.weight"] = b
                weight.add_(b @ a * 2)
            q, k = torch.nn.Linear(128, 32), torch.nn.Linear(128, 32)
            head = {"q.weight":q.weight.detach(), "q.bias":q.bias.detach(),
                    "k.weight":k.weight.detach(), "k.bias":k.bias.detach()}
            inputs = json.loads((root / "golden.json").read_text())
            cases = copy.deepcopy(inputs["cases"])
            for case in cases:
                case.pop("answers", None)
                for row in case["rows"]:
                    h = model(input_ids=torch.tensor([row["tokens"]]), use_cache=False).last_hidden_state[0]
                    logits = k(h[row["positions"]]) @ q(h[-1]) / (32**0.5)
                    row["raw_logits"] = logits.tolist()
                    row["probabilities"] = (logits / 2.406050072164233).softmax(-1).tolist()
            h = model(input_ids=torch.tensor([[1,2]]), use_cache=False).last_hidden_state[0]
            short_logits = (k(h[:1]) @ q(h[-1]) / (32**0.5)).tolist()
        save_file(base, source / "model.safetensors")
        save_file(adapter, source / "adapter_model.safetensors")
        torch.save({"head":head, "head_dim":32, "temperature":2.406050072164233,
            "base":"fixture/vllm-qwen35", "base_revision":"1111111111111111111111111111111111111111",
            "lora":2, "option_isolation":False, "special_embeddings":False, "weights_dtype":"fp32"}, source / "head.pt")
        (source / "config.json").write_text(config.to_json_string())
        shutil.copy(root / "adapter_config.json", source / "adapter_config.json")
        shutil.copy(root / "tokenizer.json", source / "tokenizer.json")
        manifest = json.loads((repo / "examples/mock-model/huncho-model.json").read_text())
        manifest.update(name="tiny-vllm-kev", family="F2")
        manifest.pop("reference", None)
        manifest["head"].update(kind="pointer", weights="head.pt", width=32)
        manifest["backbone"].update(source={"kind":"local","path":"."}, hidden_size=128,
            max_context=512, tokenizer="tokenizer.json", artifacts={"candle":[{"path":"model.safetensors","dtype":"fp32"}]})
        manifest["prompt_contract"].update(template="kev-v1", state_budget=512, head_budget=512, contract_hash="frozen-kev-v1-vllm-fixture")
        manifest["calibration"] = {"default":{"temperature":2.406050072164233,"confidence":"peak","status":"pending"},"entries":{}}
        (source / "huncho-model.json").write_text(json.dumps(manifest))
        output = export(source, source, output)
        for case in cases:
            case["request"]["model"] = "tiny-vllm-kev"
        golden = {"reference":{"torch":torch.__version__, "transformers":transformers.__version__,
            "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_fixture_sha256":hashlib.sha256((root / "golden.json").read_bytes()).hexdigest(),
            "seed":711,"backbone_dtype":"fp32","head_dtype":"fp32"},"cases":cases,"short_logits":short_logits}
        (output / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")
        # Retain content provenance without nondeterministic temporary directory names.
        provenance = output / "vllm/provenance.json"
        data = json.loads(provenance.read_text())
        data["source_sha256"] = {Path(name).name:value for name,value in data["source_sha256"].items()}
        provenance.write_text(json.dumps(data, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("crates/huncho-backend/tests/fixtures/vllm_cpu"))
    print(generate(parser.parse_args().out))
