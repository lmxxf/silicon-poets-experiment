"""
诗歌虫洞实证 — V4 Flash 完整版（5 组诗，一次加载全跑）
一次 prefill 同时抽：
  - hidden states（layer 14/28/42 的 4 路 HC）→ 余弦正交 + EID
  - MoE 路由（每层 Gate 的 top-6 专家 id）→ 专家正交 + 桥接枢纽 + 哈希/分数对比
裸文本输入，不套 chat template。

【运行环境】此脚本在 deepseek-v4-experimental-platform-on-dgx-spark 双机平台内跑，
依赖容器里的 model.py / kernel_sm121.py / weight_loader.py（不在本仓库）。
跑法：把本文件放进实验平台目录，SCRIPT=probe_all.py ./run_dual_node.sh
产出 probe_all.pt（已 gitignore），再用 probe_all_analyze.py 离线分析。
"""

import os
import sys
import json
import time

import torch
import torch.distributed as dist

INFERENCE_DIR = os.environ.get("INFERENCE_DIR", "/model/inference")
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
sys.path.insert(0, WORKSPACE_DIR)
sys.path.insert(1, INFERENCE_DIR)

import kernel_sm121
sys.modules['kernel'] = kernel_sm121

from model import Transformer, ModelArgs
import model as _model_module
import torch.nn.functional as F

_original_linear = _model_module.linear
def _patched_linear(x, weight, bias=None):
    if weight.dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        return _original_linear(x, weight, bias)
    return F.linear(x.to(weight.dtype), weight, bias)
_model_module.linear = _patched_linear

_Head = _model_module.ParallelHead
def _patched_get_logits(self, x):
    return F.linear(x[:, -1].float(), self.weight.float())
_Head.get_logits = _patched_get_logits


PAIRS = [
    {
        "id": "1_personification", "type": "跨簇拟人",
        "poem": "春风知别苦，不遣柳条青。",
        "plain": "春天来了，微风吹过，让人想到离别的伤感，所以看到柳树还没变绿，心里更加难过。",
        "bridge_token": "知", "source": "李白《劳劳亭》",
    },
    {
        "id": "2_singularity", "type": "单维奇点",
        "poem": "大漠孤烟直，长河落日圆。",
        "plain": "广阔的沙漠上一缕孤零零的炊烟笔直地升起，远处黄河边上一轮落日又大又圆。",
        "bridge_token": "直", "source": "王维《使至塞上》",
    },
    {
        "id": "3_sensory_conflict", "type": "感知通道冲突",
        "poem": "空山不见人，但闻人语响。",
        "plain": "山里空荡荡的看不见人，只是偶尔听到有人说话的声音。",
        "bridge_token": "闻", "source": "王维《鹿柴》",
    },
    {
        "id": "4_temporal_collapse", "type": "时间尺度坍缩",
        "poem": "君不见高堂明镜悲白发，朝如青丝暮成雪。",
        "plain": "你看那高堂上对着镜子悲伤白发的人，早上头发还是黑的，到了晚上就白了。",
        "bridge_token": "暮", "source": "李白《将进酒》",
    },
    {
        "id": "5_multi_alignment", "type": "多维强制对齐",
        "poem": "落霞与孤鹜齐飞，秋水共长天一色。",
        "plain": "晚霞和一只孤独的野鸭一起飞翔，秋天的江水和辽阔的天空连成一片，颜色完全一样。",
        "bridge_token": "齐", "source": "王勃《滕王阁序》",
    },
]

HIDDEN_LAYERS = [14, 28, 42]  # 抽 hidden states 的层（早/中/晚）


def main():
    hf_ckpt_path = os.environ.get("HF_CKPT_PATH", "/model")
    config_path = os.environ.get("CONFIG_PATH", f"{INFERENCE_DIR}/config.json")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")
    is_main = (rank == 0)
    def log(msg):
        if is_main:
            gpu = torch.cuda.memory_allocated() / 1e9
            print(f"[rank {rank}] {msg} | GPU={gpu:.1f}GB", flush=True)

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(42)

    with open(config_path) as f:
        config = json.load(f)
    config["max_batch_size"] = 1
    config["max_seq_len"] = 4096
    args = ModelArgs(**config)
    n_hash = args.n_hash_layers
    log(f"Config: {args.n_layers} layers, {args.n_routed_experts} experts, "
        f"top-{args.n_activated_experts}, n_hash_layers={n_hash}")

    import model as model_module
    model_module.world_size = world_size
    model_module.rank = rank

    log("Creating model skeleton on CUDA...")
    with torch.device("cuda"):
        model = Transformer(args)

    from weight_loader import load_model_streaming
    log("Loading weights (streaming)...")
    t2 = time.time()
    loaded, skipped = load_model_streaming(
        model, hf_ckpt_path, rank, world_size,
        n_experts=args.n_routed_experts, device="cuda", args=args
    )
    log(f"Weights loaded in {time.time()-t2:.1f}s ({loaded} params, {skipped} skipped)")

    torch.set_default_device("cuda")
    if world_size > 1:
        dist.barrier()

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(hf_ckpt_path, "tokenizer.json"))
    log("Tokenizer loaded")

    # === Hook 1: hidden states（layer 输入，4 路 HC）===
    cap_hidden = {}
    def make_hidden_hook(lid):
        def fn(module, inp, output):
            cap_hidden[lid] = inp[0].detach().cpu().clone()
        return fn
    hidden_hooks = [model.layers[lid].register_forward_hook(make_hidden_hook(lid))
                    for lid in HIDDEN_LAYERS]

    # === Hook 2: MoE 路由（每层 Gate 输出的 indices）===
    cap_route = {}
    gate_layer_map = {id(layer.ffn.gate): lid for lid, layer in enumerate(model.layers)}
    def gate_hook(module, inp, output):
        weights, indices = output
        cap_route[gate_layer_map[id(module)]] = indices.detach().cpu().clone()
    route_hooks = [layer.ffn.gate.register_forward_hook(gate_hook) for layer in model.layers]
    log(f"Registered {len(hidden_hooks)} hidden hooks + {len(route_hooks)} gate hooks")

    results = {}
    for pair in PAIRS:
        for label in ("poem", "plain"):
            text = pair[label]
            ids = tokenizer.encode(text)
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            tokens = [tokenizer.decode([t]) for t in ids]
            log(f"[{pair['id']}/{label}] '{text}' -> {len(ids)} tokens")

            cap_hidden.clear()
            cap_route.clear()
            with torch.inference_mode():
                model.forward(input_ids, 0)

            results[f"{pair['id']}_{label}"] = {
                "text": text,
                "tokens": tokens,
                "input_ids": ids,
                "hidden": {lid: cap_hidden[lid] for lid in HIDDEN_LAYERS},
                "routing": {lid: cap_route[lid] for lid in sorted(cap_route)},
                "bridge_token": pair["bridge_token"],
                "type": pair["type"],
                "source": pair["source"],
            }

    for h in hidden_hooks + route_hooks:
        h.remove()

    if is_main:
        save_path = "/workspace/probe_all.pt"
        torch.save({
            "n_layers": args.n_layers,
            "n_routed_experts": args.n_routed_experts,
            "topk": args.n_activated_experts,
            "n_hash_layers": n_hash,
            "hidden_layers": HIDDEN_LAYERS,
            "dim": args.dim,
            "results": results,
        }, save_path)
        log(f"Saved everything to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
    log("Done.")


if __name__ == "__main__":
    main()
