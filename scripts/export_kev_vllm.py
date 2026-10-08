"""Export a local pinned Kev/Qwen3.5 LoRA package for CPU vLLM raw pooling.

Optional CPU-only Torch/safetensors tooling; creates a NEW package. It never
fits temperatures, edits the source package or authorizes serving. Conversion
keeps FP32 merged storage, selects BF16 backbone execution/FP32 pointer readout
and writes an explicit Pending vllm:bf16 calibration entry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def export(package, base, output):
    import torch
    from safetensors.torch import load_file, save_file
    if torch.version.cuda is not None:
        raise ValueError("use CPU-only Torch for this exporter")
    package, base, output = map(Path, (package, base, output))
    if output.exists():
        raise ValueError("output must be a new package directory")
    def digest(path):
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()
    manifest_file = package / "huncho-model.json"
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["family"] != "F2" or manifest["prompt_contract"]["template"] != "kev-v1":
        raise ValueError("requires a pinned Kev F2 package")
    config_bytes = (base / "config.json").read_bytes()
    config = json.loads(config_bytes)
    config = config.get("text_config", config)
    if config.get("model_type") != "qwen3_5_text" or config.get("quantization_config"):
        raise ValueError("requires unquantized Qwen3.5 text weights")
    if any(layer == "linear_attention" for layer in config["layer_types"]) and (
        config["linear_key_head_dim"] != 128 or config["linear_value_head_dim"] != 128
    ):
        raise ValueError("vLLM 0.31.0 CPU GDN requires 128-dimensional key/value heads")
    if manifest["backbone"]["hidden_size"] != config["hidden_size"]:
        raise ValueError("source manifest and backbone dimensions disagree")
    head_file = package / manifest["head"]["weights"]
    tokenizer = package / manifest["backbone"]["tokenizer"]
    shards = sorted(p for p in base.glob("*.safetensors")
                    if p.name not in {"adapter_model.safetensors", "joint_head.safetensors"})
    if not shards:
        raise ValueError("no pinned base safetensors shards")
    sources = [head_file, package / "adapter_config.json", package / "adapter_model.safetensors", tokenizer, *shards]
    pins = {str(path.resolve()): digest(path) for path in sources}
    pins[str(manifest_file.resolve())] = hashlib.sha256(manifest_bytes).hexdigest()
    pins[str((base / "config.json").resolve())] = hashlib.sha256(config_bytes).hexdigest()
    head = torch.load(head_file, weights_only=True, map_location="cpu")
    if head.get("option_isolation", False) or head.get("special_embeddings", False) or head.get("weights", "lora") != "lora":
        raise ValueError("unsupported Kev head/embedding/adapter semantics")
    projections = head["head"]
    if set(projections) != {"q.weight", "q.bias", "k.weight", "k.bias"}:
        raise ValueError("requires complete trained q/k pointer projections")
    d, h = head["head_dim"], config["hidden_size"]
    for name, tensor in projections.items():
        if tuple(tensor.shape) != ((d, h) if name.endswith("weight") else (d,)):
            raise ValueError("invalid pointer projection shape")
    adapter_config = json.loads((package / "adapter_config.json").read_text())
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("bias", "none") != "none" or any(
        adapter_config.get(key) for key in ("use_dora", "use_rslora", "fan_in_fan_out", "rank_pattern", "alpha_pattern", "modules_to_save")
    ) or adapter_config["r"] != head["lora"] or adapter_config["r"] <= 0:
        raise ValueError("unsupported LoRA scaling/storage")
    scale = float(adapter_config["lora_alpha"]) / int(adapter_config["r"])
    weights = {}
    def normalized(name):
        for prefix in ("base_model.model.", "model.language_model.", "language_model.model.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name
    for shard in shards:
        for name, value in load_file(shard, device="cpu").items():
            name = normalized(name)
            if not (name.startswith("layers.") or name.startswith("embed_tokens.") or name.startswith("norm.")):
                if name.startswith(("lm_head.", "visual.", "mtp.")):
                    continue
                raise ValueError("unsupported base weight name: " + name)
            key = "model." + name
            if key in weights:
                raise ValueError("duplicate base tensor: " + key)
            weights[key] = value.float().contiguous()
    # Meta construction verifies complete backbone names/shapes without allocating
    # another multi-billion-parameter model or invoking any accelerator.
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    hf_config = Qwen3_5TextConfig(**config)
    hf_config._attn_implementation = "eager"
    with torch.device("meta"):
        expected = Qwen3_5TextModel(hf_config).state_dict()
    if set(weights) != {"model." + key for key in expected} or any(
        tuple(weights["model." + key].shape) != tuple(value.shape) for key, value in expected.items()
    ):
        raise ValueError("incomplete or mis-shaped Qwen3.5 backbone")
    adapter = load_file(package / "adapter_model.safetensors", device="cpu")
    visited = set()
    for name, a in adapter.items():
        if not name.endswith(".lora_A.weight"):
            continue
        b_name = name.replace(".lora_A.weight", ".lora_B.weight")
        b = adapter[b_name]
        target = "model." + normalized(name.removesuffix(".lora_A.weight")) + ".weight"
        if target not in weights or a.ndim != 2 or b.ndim != 2 or a.shape[0] != adapter_config["r"] or b.shape[1] != adapter_config["r"]:
            raise ValueError("invalid or unsupported adapter target: " + target)
        update = b.float() @ a.float() * scale
        if update.shape != weights[target].shape:
            raise ValueError("adapter shape mismatch")
        weights[target] = (weights[target] + update).contiguous()
        visited.update((name, b_name))
    if not visited or visited != set(adapter):
        raise ValueError("unmatched or unsupported adapter tensors")
    weights.update({"pooler." + name: value.float().contiguous() for name, value in projections.items()})
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("nonfinite merged weights or head")
    if any(digest(Path(name)) != pin for name, pin in pins.items()):
        raise ValueError("source bytes changed during conversion")
    output.mkdir(parents=True)
    try:
        root = output / "vllm"
        model = root / "model"
        model.mkdir(parents=True)
        config["architectures"] = ["HunchoKevForPooling"]
        config["huncho_pointer_dim"] = d
        config.pop("auto_map", None)
        (model / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        save_file(weights, model / "model.safetensors")
        shutil.copyfile(tokenizer, output / "tokenizer.json")
        (root / "artifact.json").write_text(json.dumps({"schema_version":1,
            "profile":"kev-pointer-vllm-cpu-bf16-v1", "model_dir":"model",
            "files":{str(path.relative_to(root)):digest(path) for path in sorted(model.iterdir())}}, indent=2) + "\n")
        # Resolve source overrides without erasing them. Conversion is not fitting.
        calibration = manifest["calibration"]
        calibration.setdefault("entries", {})["vllm:bf16"] = dict(calibration["entries"].get("candle:fp32", calibration["default"]), status="pending")
        manifest["backbone"]["source"] = {"kind":"local", "path":"vllm/model"}
        manifest["backbone"]["artifacts"] = {"vllm":[{"path":"vllm/artifact.json", "dtype":"bf16"}]}
        manifest["backbone"]["tokenizer"] = "tokenizer.json"
        manifest["backbone"].pop("adapter", None)
        manifest["head"]["weights"] = "vllm/model/model.safetensors"
        (output / "huncho-model.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if any(digest(Path(name)) != pin for name, pin in pins.items()):
            raise ValueError("source bytes changed during export")
        (root / "provenance.json").write_text(json.dumps({"schema_version":1,
            "source_sha256":pins, "torch":torch.__version__, "temperature_fit":"not performed",
            "backbone_execution":"bf16", "head_execution":"fp32", "decode":"disabled"}, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.package, args.base, args.out))
