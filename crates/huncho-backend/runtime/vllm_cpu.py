"""Pinned optional CPU vLLM pooling worker. No decoder or probability transform."""
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
from pathlib import Path
import sys

PROTOCOL_OUTPUT = None
if __name__ == "__main__":
    PROTOCOL_OUTPUT = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

# Set CPU before importing the runtime. This short-circuits accelerator plugin
# discovery in vLLM 0.31.0; CPU-only Torch is mandatory, not a fallback.
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if importlib.metadata.version("vllm") != "0.31.0+cpu":
    raise RuntimeError("requires pinned vllm==0.31.0+cpu")
if importlib.metadata.version("torch") != "2.13.0+cpu":
    raise RuntimeError("requires pinned torch==2.13.0+cpu")
if importlib.metadata.version("transformers") != "5.17.0":
    raise RuntimeError("requires pinned transformers==5.17.0")
import torch
from torch import nn
if torch.version.cuda is not None:
    raise RuntimeError("requires CPU-only Torch")
from vllm.platforms import current_platform
if not current_platform.is_cpu():
    raise RuntimeError("requires CPU platform")
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase, Qwen3_5Model
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

class PointerPooler(Pooler):
    def __init__(self, h, d):
        super().__init__()
        self.q = nn.Linear(h, d, dtype=torch.float32)
        self.k = nn.Linear(h, d, dtype=torch.float32)
        self.scale = math.sqrt(d)
    def get_supported_tasks(self):
        return {'classify'}
    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill() or not all(cursor.finished_mask):
            raise ValueError('partial prefills are unsupported')
        out = []
        for i, params in enumerate(pooling_metadata.pooling_params):
            if params.use_activation is not False:
                raise ValueError('raw readout needs use_activation=False')
            begin = int(cursor.first_token_indices_gpu[i])
            length = int(pooling_metadata.prompt_lens[i])
            positions = params.extra_kwargs['positions']
            h = hidden_states[begin:begin+length].float()
            q = self.q(h[-1])
            k = self.k(h[positions])
            out.append(k @ q / self.scale)
        return out

class HunchoKevForPooling(Qwen3_5ForCausalLMBase):
    is_pooling_model = True
    supports_lora = False
    supports_eagle3 = False
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={'model.language_model.':'model.'})
    def __init__(self, *, vllm_config, prefix=''):
        nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = vllm_config.model_config.hf_text_config
        self.scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config
        self.model = Qwen3_5Model(vllm_config=vllm_config, prefix=(prefix+'.model' if prefix else 'model'))
        self.pooler = PointerPooler(self.config.hidden_size, self.config.huncho_pointer_dim)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self.huncho_forward_calls = 0
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        self.huncho_forward_calls += 1
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
    def compute_logits(self, hidden_states):
        raise RuntimeError('decode is unsupported')
    def compute_logits_local(self, hidden_states):
        raise RuntimeError('decode is unsupported')
    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)


def runtime_identity():
    """Hash installed core runtime bytes and loaded executable libraries."""
    paths = {Path(sys.executable).resolve()}
    for name in ("cc", "c++", "gcc", "g++", "clang", "clang++", "ld"):
        if compiler := shutil.which(name):
            paths.add(Path(compiler).resolve())
    for name in ("vllm", "torch", "transformers", "triton", "numpy", "safetensors"):
        dist = importlib.metadata.distribution(name)
        for file in dist.files or ():
            if str(file).endswith((".pyc", ".pyo")):
                continue
            path = Path(dist.locate_file(file)).resolve()
            if path.is_file():
                paths.add(path)
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        if len(fields) == 6 and "x" in fields[1] and fields[5].startswith("/"):
            paths.add(Path(fields[5]))
    digest = hashlib.sha256()
    digest.update(json.dumps({name: os.environ.get(name, "") for name in
        ("PATH", "HOME", "TMPDIR", "LD_PRELOAD", "LD_LIBRARY_PATH", "OMP_NUM_THREADS")},
        sort_keys=True).encode())
    for path in sorted(paths):
        digest.update(str(path).encode() + b"\0")
        with path.open("rb") as file:
            while block := file.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def main():
    # Reserve a protocol FD, then redirect OS-level runtime/extension output.
    # A Python context manager alone misses native prints and child inspectors.
    output = PROTOCOL_OUTPUT
    from vllm import LLM, ModelRegistry, PoolingParams
    from vllm.config import PoolerConfig
    model_dir, context, batch_rows, threads, kv_bytes = sys.argv[1:]
    context, batch_rows, threads, kv_bytes = map(int, (context, batch_rows, threads, kv_bytes))
    # The legacy CPU variable accepts integer GiB and overrides byte limits.
    # Use the supported exact byte option; never round a configured budget.
    os.environ.pop("VLLM_CPU_KVCACHE_SPACE", None)
    os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "nobind"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    ModelRegistry.register_model("HunchoKevForPooling", "huncho_vllm_cpu:HunchoKevForPooling")
    llm = LLM(model=model_dir, runner="pooling", dtype="bfloat16", trust_remote_code=False,
              skip_tokenizer_init=True, max_model_len=context, max_num_seqs=batch_rows,
              max_num_batched_tokens=context * batch_rows, enable_chunked_prefill=False,
              enable_prefix_caching=False, enforce_eager=True, tensor_parallel_size=1,
              pipeline_parallel_size=1, kv_cache_memory_bytes=kv_bytes, pooler_config=PoolerConfig(task="classify", use_activation=False))
    config = json.loads((Path(model_dir) / "config.json").read_text())
    def emit(value):
        output.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    emit({"protocol": 1, "kind": "ready", "runtime_sha256": runtime_identity(),
          "vllm": "0.31.0+cpu", "torch": "2.13.0+cpu", "device": "CPU", "dtype": "bf16",
          "context": context, "batch_rows": batch_rows, "head_dtype": "fp32"})
    for line in sys.stdin:
        if len(line.encode()) > 32 * 1024 * 1024:
            raise ValueError("protocol input exceeds 32 MiB")
        request = json.loads(line)
        if set(request) != {"seq", "inputs"} or type(request["seq"]) is not int:
            raise ValueError("invalid protocol request")
        inputs = request["inputs"]
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= batch_rows:
            raise ValueError("invalid batch rows")
        for item in inputs:
            if set(item) != {"tokens", "positions"}:
                raise ValueError("invalid row fields")
            tokens, positions = item["tokens"], item["positions"]
            if not isinstance(tokens, list) or not 1 <= len(tokens) <= context:
                raise ValueError("invalid sequence length")
            if any(type(x) is not int or not 0 <= x < config["vocab_size"] for x in tokens):
                raise ValueError("invalid token")
            if not isinstance(positions, list) or len(positions) > 255 or any(type(x) is not int or not 0 <= x < len(tokens) for x in positions):
                raise ValueError("invalid positions")
        before = llm.apply_model(lambda model: model.huncho_forward_calls)
        outputs = llm.encode([{"prompt_token_ids": item["tokens"]} for item in inputs],
            pooling_params=[PoolingParams(task="classify", use_activation=False,
                extra_kwargs={"positions": item["positions"]}) for item in inputs],
            pooling_task="classify", use_tqdm=False)
        after = llm.apply_model(lambda model: model.huncho_forward_calls)
        calls = [a - b for a, b in zip(after, before)]
        if calls != [1] or len(outputs) != len(inputs):
            raise RuntimeError("pooling scheduler did not execute exactly one complete native forward")
        logits = [result.outputs.data.tolist() for result in outputs]
        if any(len(values) != len(item["positions"]) or any(not math.isfinite(x) for x in values)
               for values, item in zip(logits, inputs)):
            raise RuntimeError("nonfinite or incomplete raw pointer readout")
        emit({"protocol": 1, "kind": "scores", "seq": request["seq"], "forward_calls": 1,
              "processed_tokens": sum(len(item["tokens"]) for item in inputs), "logits": logits})
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
