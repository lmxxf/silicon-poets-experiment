"""
实验四：SAE 特征分解
用 Goodfire SAE (Layer 50) 分解 Llama-3.3-70B 的隐状态
检测诗歌桥接 token 的双域/多域特征共激活

模型：Llama-3.3-70B-Instruct-AWQ + Goodfire SAE-l50
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================

MODEL_PATH = "/workspace/models/Llama-3.3-70B-Instruct-INT8"
SAE_PATH = "/workspace/models/Llama-3.3-70B-Instruct-SAE-l50"
OUTPUT_DIR = "/workspace/ai-theorys-study/arxiv/wechat121/results"
TARGET_LAYER = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 五组对照
PAIRS = [
    {
        "id": "1_personification",
        "type": "跨簇拟人",
        "poem": "春风知别苦，不遣柳条青。",
        "plain": "春天来了，微风吹过，让人想到离别的伤感，所以看到柳树还没变绿，心里更加难过。",
        "bridge_token": "知",
        "source": "李白《劳劳亭》",
    },
    {
        "id": "2_singularity",
        "type": "单维奇点",
        "poem": "大漠孤烟直，长河落日圆。",
        "plain": "广阔的沙漠上一缕孤零零的炊烟笔直地升起，远处黄河边上一轮落日又大又圆。",
        "bridge_token": "直",
        "source": "王维《使至塞上》",
    },
    {
        "id": "3_sensory_conflict",
        "type": "感知通道冲突",
        "poem": "空山不见人，但闻人语响。",
        "plain": "山里空荡荡的看不见人，只是偶尔听到有人说话的声音。",
        "bridge_token": "闻",
        "source": "王维《鹿柴》",
    },
    {
        "id": "4_temporal_collapse",
        "type": "时间尺度坍缩",
        "poem": "君不见高堂明镜悲白发，朝如青丝暮成雪。",
        "plain": "你看那高堂上对着镜子悲伤白发的人，早上头发还是黑的，到了晚上就白了。",
        "bridge_token": "暮",
        "source": "李白《将进酒》",
    },
    {
        "id": "5_multi_alignment",
        "type": "多维强制对齐",
        "poem": "落霞与孤鹜齐飞，秋水共长天一色。",
        "plain": "晚霞和一只孤独的野鸭一起飞翔，秋天的江水和辽阔的天空连成一片，颜色完全一样。",
        "bridge_token": "齐",
        "source": "王勃《滕王阁序》",
    },
]


# ============================================================
# 加载 SAE
# ============================================================

class GoodfireSAE(nn.Module):
    """Goodfire SAE: encoder [d_model -> n_features] + decoder [n_features -> d_model]"""

    def __init__(self, path):
        super().__init__()
        state = torch.load(os.path.join(path, "model.safetensors"),
                           map_location="cpu", weights_only=False)
        # 尝试 safetensors 格式
        if not state:
            from safetensors.torch import load_file
            state = load_file(os.path.join(path, "model.safetensors"))

        self.encoder = nn.Linear(
            state["encoder_linear.weight"].shape[1],
            state["encoder_linear.weight"].shape[0],
        )
        self.decoder = nn.Linear(
            state["decoder_linear.weight"].shape[1],
            state["decoder_linear.weight"].shape[0],
        )
        self.encoder.weight.data = state["encoder_linear.weight"]
        self.encoder.bias.data = state["encoder_linear.bias"]
        self.decoder.weight.data = state["decoder_linear.weight"]
        self.decoder.bias.data = state["decoder_linear.bias"]

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def forward(self, x):
        features = self.encode(x)
        reconstruction = self.decoder(features)
        return reconstruction, features


def load_sae():
    """加载 SAE，自动检测文件格式"""
    sae_files = os.listdir(SAE_PATH)
    print(f"SAE 目录内容: {sae_files}")

    if "model.safetensors" in sae_files:
        from safetensors.torch import load_file
        state = load_file(os.path.join(SAE_PATH, "model.safetensors"))
    elif "model.pt" in sae_files:
        state = torch.load(os.path.join(SAE_PATH, "model.pt"), map_location="cpu")
    else:
        # 找任何 .safetensors 或 .pt 文件
        for f in sae_files:
            if f.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(os.path.join(SAE_PATH, f))
                break
            elif f.endswith(".pt"):
                state = torch.load(os.path.join(SAE_PATH, f), map_location="cpu")
                break
        else:
            raise FileNotFoundError(f"在 {SAE_PATH} 中没找到模型文件")

    print(f"SAE 权重 keys: {list(state.keys())}")
    for k, v in state.items():
        print(f"  {k}: {v.shape}")

    # 构建 SAE
    enc_w = state["encoder_linear.weight"]
    sae = nn.Module()
    sae.encoder_w = enc_w  # [n_features, d_model]
    sae.encoder_b = state["encoder_linear.bias"]  # [n_features]
    sae.decoder_w = state["decoder_linear.weight"]  # [d_model, n_features]
    sae.decoder_b = state["decoder_linear.bias"]  # [d_model]

    n_features, d_model = enc_w.shape
    print(f"SAE 结构: d_model={d_model}, n_features={n_features}")

    return state, d_model, n_features


print("=" * 60)
print("加载 SAE")
print("=" * 60)
sae_state, sae_d_model, sae_n_features = load_sae()


def sae_encode(hidden_vec, device="cpu"):
    """用 SAE encoder 编码一个 hidden state 向量"""
    enc_w = sae_state["encoder_linear.weight"].to(device)  # [n_features, d_model]
    enc_b = sae_state["encoder_linear.bias"].to(device)  # [n_features]
    h = hidden_vec.to(device)
    features = torch.relu(h @ enc_w.T + enc_b)
    return features


# ============================================================
# 加载 LLM
# ============================================================

print("\n" + "=" * 60)
print("加载模型:", MODEL_PATH)
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    torch_dtype=torch.float16,
    local_files_only=True,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"模型加载完成。层数: {model.config.num_hidden_layers}, d_model: {model.config.hidden_size}")


# ============================================================
# 工具函数
# ============================================================

def extract_layer50(text):
    """提取 Layer 50 的 hidden states"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states[0] = embedding, hidden_states[1] = layer 0, ...
    # 所以 hidden_states[51] = layer 50 的输出
    layer_idx = min(TARGET_LAYER + 1, len(outputs.hidden_states) - 1)
    hidden = outputs.hidden_states[layer_idx].squeeze(0).cpu().float()

    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return hidden, tokens


def find_token_idx(tokens, target):
    for i, t in enumerate(tokens):
        if target in t:
            return i
    return None


# ============================================================
# 实验四：SAE 特征分解
# ============================================================

def run_exp4(pair):
    """SAE 特征分解 + 双域共激活检测"""
    pair_id = pair["id"]
    bridge = pair["bridge_token"]
    print(f"\n{'=' * 60}")
    print(f"实验四: {pair['type']} — {pair['source']}")
    print(f"  诗: {pair['poem']}")
    print(f"  白话: {pair['plain']}")
    print(f"{'=' * 60}")

    # 提取 Layer 50 激活
    print("提取诗组 Layer 50 激活...")
    poem_hidden, poem_tokens = extract_layer50(pair["poem"])
    print(f"  tokens: {poem_tokens}")
    print(f"  hidden shape: {poem_hidden.shape}")

    print("提取白话组 Layer 50 激活...")
    plain_hidden, plain_tokens = extract_layer50(pair["plain"])
    print(f"  tokens: {plain_tokens}")

    # SAE 编码
    print("SAE 编码...")
    poem_features = sae_encode(poem_hidden)  # (n_tokens, n_features)
    plain_features = sae_encode(plain_hidden)

    # ---- 4a: 每个 token 的激活特征数 (SAF) ----
    poem_saf = (poem_features > 0).sum(dim=1).numpy()
    plain_saf = (plain_features > 0).sum(dim=1).numpy()

    print(f"\n诗组 SAF:")
    for i, t in enumerate(poem_tokens):
        print(f"  {t}: {poem_saf[i]}")
    print(f"白话组 SAF (平均): {plain_saf.mean():.1f}")

    # SAF 柱状图
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].bar(range(len(poem_tokens)), poem_saf, color="crimson", alpha=0.8)
    axes[0].set_xticks(range(len(poem_tokens)))
    axes[0].set_xticklabels(poem_tokens, fontsize=10)
    axes[0].set_title(f"诗·{pair['source']} SAF", fontsize=12)
    axes[0].set_ylabel("激活特征数")

    axes[1].bar(range(len(plain_tokens)), plain_saf, color="steelblue", alpha=0.8)
    axes[1].set_xticks(range(len(plain_tokens)))
    axes[1].set_xticklabels(plain_tokens, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("白话对照 SAF", fontsize=12)
    axes[1].set_ylabel("激活特征数")

    plt.suptitle(f"实验四·SAE 激活特征数: {pair['type']}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"exp4_{pair_id}_saf.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")

    # ---- 4b: 特征激活热力图 ----
    # 只画诗组，只画有激活的特征
    active_mask = (poem_features > 0).any(dim=0)
    active_features = poem_features[:, active_mask].numpy()
    n_active = active_features.shape[1]
    print(f"\n诗组活跃特征数: {n_active} / {sae_n_features}")

    # 如果活跃特征太多，取 top 100
    if n_active > 100:
        max_vals = active_features.max(axis=0)
        top_idx = np.argsort(max_vals)[-100:]
        active_features_plot = active_features[:, top_idx]
        n_plot = 100
    else:
        active_features_plot = active_features
        n_plot = n_active

    fig, ax = plt.subplots(figsize=(max(14, n_plot * 0.15), 6))
    im = ax.imshow(active_features_plot, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(poem_tokens)))
    ax.set_yticklabels(poem_tokens, fontsize=10)
    ax.set_xlabel(f"SAE 特征 (top {n_plot})", fontsize=11)
    ax.set_title(f"SAE 特征激活模式 — {pair['source']}", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="激活强度")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"exp4_{pair_id}_feature_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")

    # ---- 4c: 双域共激活分析 ----
    bridge_idx = find_token_idx(poem_tokens, bridge)
    stats = {
        "poem_tokens": poem_tokens,
        "poem_saf": poem_saf.tolist(),
        "plain_saf_mean": float(plain_saf.mean()),
        "n_active_features": int(n_active),
    }

    if bridge_idx is not None:
        bridge_feat = poem_features[bridge_idx].numpy()  # (n_features,)

        # 对每个 token 计算与桥接 token 的特征重叠
        overlaps = []
        for i, t in enumerate(poem_tokens):
            if i == bridge_idx:
                continue
            feat_i = poem_features[i].numpy()
            # Jaccard: 两者都激活的特征 / 至少一个激活的特征
            both_active = ((feat_i > 0) & (bridge_feat > 0)).sum()
            either_active = ((feat_i > 0) | (bridge_feat > 0)).sum()
            jaccard = both_active / either_active if either_active > 0 else 0
            # Pearson
            if feat_i.std() > 0 and bridge_feat.std() > 0:
                pearson = np.corrcoef(feat_i, bridge_feat)[0, 1]
            else:
                pearson = 0
            overlaps.append({
                "token": t,
                "idx": i,
                "jaccard": float(jaccard),
                "pearson": float(pearson),
                "shared_features": int(both_active),
            })

        # 画特征重叠柱状图
        other_tokens = [o["token"] for o in overlaps]
        jaccards = [o["jaccard"] for o in overlaps]
        pearsons = [o["pearson"] for o in overlaps]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        axes[0].bar(range(len(other_tokens)), jaccards, color="coral", alpha=0.8)
        axes[0].set_xticks(range(len(other_tokens)))
        axes[0].set_xticklabels(other_tokens, fontsize=9)
        axes[0].set_title(f"与「{bridge}」的 Jaccard 重叠", fontsize=12)
        axes[0].set_ylabel("Jaccard")

        axes[1].bar(range(len(other_tokens)), pearsons, color="teal", alpha=0.8)
        axes[1].set_xticks(range(len(other_tokens)))
        axes[1].set_xticklabels(other_tokens, fontsize=9)
        axes[1].set_title(f"与「{bridge}」的 Pearson 相关", fontsize=12)
        axes[1].set_ylabel("Pearson r")

        plt.suptitle(f"「{bridge}」与各 token 的 SAE 特征重叠 ({pair['source']})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, f"exp4_{pair_id}_feature_overlap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")

        stats["bridge_token"] = bridge
        stats["bridge_saf"] = int(poem_saf[bridge_idx])
        stats["feature_overlaps"] = overlaps

        # ---- 4d: 桥接 token 独占特征 vs 共享特征 ----
        bridge_active = set(np.where(bridge_feat > 0)[0])
        exclusive = bridge_active.copy()
        shared_with = {}

        for i, t in enumerate(poem_tokens):
            if i == bridge_idx:
                continue
            feat_i = poem_features[i].numpy()
            other_active = set(np.where(feat_i > 0)[0])
            shared = bridge_active & other_active
            if shared:
                shared_with[t] = len(shared)
            exclusive -= other_active

        stats["bridge_exclusive_features"] = len(exclusive)
        stats["bridge_shared_with"] = shared_with
        print(f"\n  「{bridge}」独占特征: {len(exclusive)}")
        print(f"  「{bridge}」共享特征:")
        for t, n in sorted(shared_with.items(), key=lambda x: -x[1]):
            print(f"    与「{t}」共享: {n}")

    # 清理
    del poem_hidden, plain_hidden, poem_features, plain_features
    torch.cuda.empty_cache()

    return stats


# ============================================================
# 主循环
# ============================================================

all_results = {}

for pair in PAIRS:
    result = run_exp4(pair)
    all_results[pair["id"]] = result

# ============================================================
# 汇总
# ============================================================

print("\n" + "=" * 60)
print("实验四汇总")
print("=" * 60)

for pair_id, result in all_results.items():
    bridge = result.get("bridge_token", "?")
    bridge_saf = result.get("bridge_saf", 0)
    exclusive = result.get("bridge_exclusive_features", 0)
    print(f"\n{pair_id}:")
    print(f"  桥接token「{bridge}」SAF={bridge_saf}, 独占特征={exclusive}")
    if "feature_overlaps" in result:
        for o in result["feature_overlaps"]:
            if o["jaccard"] > 0.1:
                print(f"  ↔「{o['token']}」Jaccard={o['jaccard']:.3f}, Pearson={o['pearson']:.3f}")

# 保存
results_path = os.path.join(OUTPUT_DIR, "exp4_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已保存: {results_path}")

print("\n实验四完成！")
