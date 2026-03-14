"""
实验一+三+EID：诗歌虫洞实证
- 实验一：逐 token 残差流轨迹追踪 + 跳跃距离
- 实验三：余弦相似度矩阵 + 跨层追踪
- EID：有效内在维度（SVD 谱熵），量化语义体积

模型：Llama-3.3-70B-Instruct-INT8，output_hidden_states=True
5 组诗/白话对照
"""

import os
import gc
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================

MODEL_PATH = "/workspace/models/Llama-3.3-70B-Instruct-INT8"
OUTPUT_DIR = "/workspace/ai-theorys-study/arxiv/wechat121/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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
# 加载模型
# ============================================================

print("=" * 60)
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

N_LAYERS = model.config.num_hidden_layers
D_MODEL = model.config.hidden_size
print(f"模型加载完成。层数: {N_LAYERS}, d_model: {D_MODEL}")


# ============================================================
# 提取 hidden states（只保留采样层，省内存）
# ============================================================

def get_sampled_layers(n_layers):
    """选 10 个均匀分布的层 + 第 0 层和最后一层"""
    indices = set([0, n_layers])  # embedding 和最后一层
    step = max(1, n_layers // 10)
    for i in range(0, n_layers + 1, step):
        indices.add(i)
    return sorted(indices)


SAMPLE_LAYERS = get_sampled_layers(N_LAYERS)
print(f"采样层: {SAMPLE_LAYERS} ({len(SAMPLE_LAYERS)} 层)")


def get_hidden_states(text):
    """提取采样层的 hidden states，其余立即丢弃"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=False)

    sampled = {}
    for i in SAMPLE_LAYERS:
        if i < len(outputs.hidden_states):
            sampled[i] = outputs.hidden_states[i].squeeze(0).cpu().float().numpy()

    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    del outputs
    torch.cuda.empty_cache()

    return sampled, tokens


def find_token_idx(tokens, target):
    for i, t in enumerate(tokens):
        if target in t:
            return i
    return None


# ============================================================
# 实验一：残差流轨迹 + 跳跃距离
# ============================================================

def run_exp1(pair, poem_sampled, poem_tokens, plain_sampled, plain_tokens):
    pair_id = pair["id"]
    print(f"\n--- 实验一: {pair['type']} ({pair['source']}) ---")

    # PCA 轨迹用 5 个关键层
    key_layers = [0, N_LAYERS // 4, N_LAYERS // 2, 3 * N_LAYERS // 4, N_LAYERS]
    # 找采样层中最接近的
    def nearest(target, available):
        return min(available, key=lambda x: abs(x - target))
    key_layers = [nearest(l, list(poem_sampled.keys())) for l in key_layers]
    key_layers = list(dict.fromkeys(key_layers))  # 去重保序

    for label, sampled, toks in [
        ("poem", poem_sampled, poem_tokens),
        ("plain", plain_sampled, plain_tokens),
    ]:
        selected = np.stack([sampled[l] for l in key_layers])
        n_sel, n_tok, d = selected.shape
        flat = selected.reshape(-1, d)

        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(flat).reshape(n_sel, n_tok, 2)

        fig, ax = plt.subplots(figsize=(14, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, n_tok))

        for t in range(n_tok):
            traj = coords_2d[:, t, :]
            ax.plot(traj[:, 0], traj[:, 1], "-o", color=colors[t],
                    linewidth=2, markersize=6, alpha=0.8, label=f'"{toks[t]}"')
            ax.annotate(toks[t], xy=(traj[-1, 0], traj[-1, 1]),
                        fontsize=11, fontweight="bold", color=colors[t],
                        textcoords="offset points", xytext=(5, 5))

        title_map = {"poem": f"诗·{pair['source']}", "plain": "白话对照"}
        ax.set_title(f"{title_map[label]} — 残差流轨迹 (PCA)", fontsize=16, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, f"exp1_{pair_id}_{label}_trajectory.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")

    # 跳跃距离（用最后一层）
    def jump_distances(sampled, toks):
        last = sampled[max(sampled.keys())]
        return [float(np.linalg.norm(last[i + 1] - last[i])) for i in range(len(toks) - 1)]

    poem_dists = jump_distances(poem_sampled, poem_tokens)
    plain_dists = jump_distances(plain_sampled, plain_tokens)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    pairs_poem = [f"{poem_tokens[i]}→{poem_tokens[i+1]}" for i in range(len(poem_tokens) - 1)]
    axes[0].bar(range(len(poem_dists)), poem_dists, color="crimson", alpha=0.8)
    axes[0].set_xticks(range(len(pairs_poem)))
    axes[0].set_xticklabels(pairs_poem, rotation=45, ha="right", fontsize=9)
    axes[0].set_title(f"诗·{pair['source']}", fontsize=13)
    axes[0].set_ylabel("欧氏距离")

    pairs_plain = [f"{plain_tokens[i]}→{plain_tokens[i+1]}" for i in range(len(plain_tokens) - 1)]
    axes[1].bar(range(len(plain_dists)), plain_dists, color="steelblue", alpha=0.8)
    axes[1].set_xticks(range(len(pairs_plain)))
    axes[1].set_xticklabels(pairs_plain, rotation=45, ha="right", fontsize=7)
    axes[1].set_title("白话对照", fontsize=13)
    axes[1].set_ylabel("欧氏距离")

    plt.suptitle(f"实验一·跳跃距离: {pair['type']}", fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"exp1_{pair_id}_jump_distances.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")

    jump_ratio = max(poem_dists) / max(plain_dists) if max(plain_dists) > 0 else float("inf")
    max_idx = int(np.argmax(poem_dists))
    return {
        "poem_max_jump": float(max(poem_dists)),
        "poem_max_jump_pair": pairs_poem[max_idx],
        "plain_max_jump": float(max(plain_dists)),
        "jump_ratio": float(jump_ratio),
        "poem_mean_jump": float(np.mean(poem_dists)),
        "plain_mean_jump": float(np.mean(plain_dists)),
        "poem_all_jumps": list(zip(pairs_poem, poem_dists)),
        "plain_all_jumps": list(zip(pairs_plain, plain_dists)),
    }


# ============================================================
# 实验三：余弦相似度矩阵
# ============================================================

def run_exp3(pair, poem_sampled, poem_tokens, plain_sampled, plain_tokens):
    pair_id = pair["id"]
    print(f"\n--- 实验三: {pair['type']} ({pair['source']}) ---")

    stats = {}

    for label, sampled, toks in [
        ("poem", poem_sampled, poem_tokens),
        ("plain", plain_sampled, plain_tokens),
    ]:
        last_layer = sampled[max(sampled.keys())]
        norms = np.linalg.norm(last_layer, axis=-1, keepdims=True)
        normalized = last_layer / (norms + 1e-10)
        cosine_sim = normalized @ normalized.T

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cosine_sim, xticklabels=toks, yticklabels=toks,
                    cmap="RdYlBu_r", vmin=0, vmax=1,
                    annot=True, fmt=".2f", ax=ax, square=True, linewidths=0.5)
        title_map = {"poem": f"诗·{pair['source']}", "plain": "白话对照"}
        ax.set_title(f"{title_map[label]} 余弦相似度（最后一层）", fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, f"exp3_{pair_id}_{label}_cosine.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")

        n = len(toks)
        off_diag_high = 0
        off_diag_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(i - j) > 2 and cosine_sim[i][j] > 0.8:
                    off_diag_high += 1
                    off_diag_pairs.append((toks[i], toks[j], float(cosine_sim[i][j])))

        # 归一化：除以 token 对总数
        total_pairs = max(1, n * (n - 1) // 2)
        off_diag_ratio = off_diag_high / total_pairs

        stats[f"{label}_off_diag_high_count"] = off_diag_high
        stats[f"{label}_off_diag_ratio"] = float(off_diag_ratio)
        stats[f"{label}_off_diag_pairs"] = off_diag_pairs
        stats[f"{label}_n_tokens"] = n

    # 跨层相似度追踪（用采样层）
    bridge = pair["bridge_token"]
    bridge_idx = find_token_idx(poem_tokens, bridge)

    if bridge_idx is not None and bridge_idx > 0:
        target_idx = 0
        layer_ids = sorted(poem_sampled.keys())
        sims = []
        for lid in layer_ids:
            v1 = poem_sampled[lid][bridge_idx]
            v2 = poem_sampled[lid][target_idx]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            sims.append(float(cos))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(layer_ids, sims, "o-", color="crimson", linewidth=2, markersize=5)
        ax.set_xlabel("层数", fontsize=12)
        ax.set_ylabel("余弦相似度", fontsize=12)
        ax.set_title(f"「{poem_tokens[target_idx]}」与「{poem_tokens[bridge_idx]}」相似度随层数变化 ({pair['source']})",
                     fontsize=13, fontweight="bold")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, f"exp3_{pair_id}_cross_layer.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")

        stats["cross_layer_sim_start"] = sims[0]
        stats["cross_layer_sim_end"] = sims[-1]
        stats["cross_layer_sim_max"] = max(sims)
        stats["cross_layer_sim_max_layer"] = layer_ids[int(np.argmax(sims))]

    return stats


# ============================================================
# EID：有效内在维度（SVD 谱熵）
# ============================================================

def compute_eid(hidden_matrix):
    """
    对 token×d_model 矩阵做 SVD，用归一化谱熵量化有效维度。
    EID = exp(H)，H = -Σ p_i log(p_i)，p_i = σ_i² / Σσ_j²
    EID 越高 = token 占据的语义维度越多 = 语义体积越大
    """
    U, S, Vt = np.linalg.svd(hidden_matrix, full_matrices=False)
    # 奇异值平方 = 方差贡献
    s2 = S ** 2
    s2 = s2[s2 > 1e-12]  # 去掉数值噪声
    p = s2 / s2.sum()
    entropy = -np.sum(p * np.log(p))
    eid = np.exp(entropy)
    return float(eid), float(entropy), p.tolist()


def run_eid(pair, poem_sampled, poem_tokens, plain_sampled, plain_tokens):
    pair_id = pair["id"]
    print(f"\n--- EID: {pair['type']} ({pair['source']}) ---")

    stats = {}

    for label, sampled, toks in [
        ("poem", poem_sampled, poem_tokens),
        ("plain", plain_sampled, plain_tokens),
    ]:
        last_layer = sampled[max(sampled.keys())]
        eid, entropy, spectrum = compute_eid(last_layer)
        n_tokens = len(toks)

        stats[f"{label}_eid"] = eid
        stats[f"{label}_entropy"] = entropy
        stats[f"{label}_n_tokens"] = n_tokens
        # 归一化 EID：除以 token 数，消除长度影响
        stats[f"{label}_eid_normalized"] = eid / n_tokens if n_tokens > 0 else 0.0
        stats[f"{label}_spectrum_top10"] = spectrum[:10]

        print(f"  {label}: EID={eid:.2f}, 归一化EID={eid/n_tokens:.3f}, "
              f"谱熵={entropy:.3f}, tokens={n_tokens}")

    # 画对比图：奇异值谱 + EID 柱状图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：奇异值谱对比
    poem_last = poem_sampled[max(poem_sampled.keys())]
    plain_last = plain_sampled[max(plain_sampled.keys())]
    _, S_poem, _ = np.linalg.svd(poem_last, full_matrices=False)
    _, S_plain, _ = np.linalg.svd(plain_last, full_matrices=False)

    # 归一化奇异值（除以最大值）
    S_poem_norm = S_poem / S_poem[0] if S_poem[0] > 0 else S_poem
    S_plain_norm = S_plain / S_plain[0] if S_plain[0] > 0 else S_plain

    n_show = min(20, len(S_poem_norm), len(S_plain_norm))
    x = np.arange(n_show)
    axes[0].bar(x - 0.2, S_poem_norm[:n_show], 0.4, color="crimson", alpha=0.8, label="诗")
    axes[0].bar(x + 0.2, S_plain_norm[:n_show], 0.4, color="steelblue", alpha=0.8, label="白话")
    axes[0].set_xlabel("奇异值序号", fontsize=12)
    axes[0].set_ylabel("归一化奇异值", fontsize=12)
    axes[0].set_title("奇异值谱对比", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].set_yscale("log")

    # 右图：EID 对比柱状图
    poem_eid = stats["poem_eid"]
    plain_eid = stats["plain_eid"]
    poem_eid_n = stats["poem_eid_normalized"]
    plain_eid_n = stats["plain_eid_normalized"]

    bar_x = [0, 1]
    bars = axes[1].bar(bar_x, [poem_eid_n, plain_eid_n],
                       color=["crimson", "steelblue"], alpha=0.8, width=0.5)
    axes[1].set_xticks(bar_x)
    axes[1].set_xticklabels(["诗", "白话"], fontsize=13)
    axes[1].set_ylabel("归一化 EID (EID/token数)", fontsize=12)
    axes[1].set_title("有效内在维度对比", fontsize=13, fontweight="bold")
    # 标数值
    for bar, eid_val, eid_n_val, n_tok in zip(
        bars,
        [poem_eid, plain_eid],
        [poem_eid_n, plain_eid_n],
        [stats["poem_n_tokens"], stats["plain_n_tokens"]],
    ):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"EID={eid_val:.1f}\n({n_tok} tokens)\n归一化={eid_n_val:.3f}",
                     ha="center", va="bottom", fontsize=10)

    plt.suptitle(f"EID·{pair['type']} ({pair['source']})", fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"eid_{pair_id}_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")

    return stats


# ============================================================
# 主循环
# ============================================================

all_results = {}

for pair in PAIRS:
    pair_id = pair["id"]
    print(f"\n{'=' * 60}")
    print(f"处理: {pair['type']} — {pair['source']}")
    print(f"  诗: {pair['poem']}")
    print(f"  白话: {pair['plain']}")
    print(f"{'=' * 60}")

    print("提取诗组 hidden states...")
    poem_sampled, poem_tokens = get_hidden_states(pair["poem"])
    print(f"  tokens: {poem_tokens}")

    print("提取白话组 hidden states...")
    plain_sampled, plain_tokens = get_hidden_states(pair["plain"])
    print(f"  tokens: {plain_tokens}")

    result = {
        "pair": pair,
        "poem_tokens": poem_tokens,
        "plain_tokens": plain_tokens,
    }

    result["exp1"] = run_exp1(pair, poem_sampled, poem_tokens, plain_sampled, plain_tokens)
    result["exp3"] = run_exp3(pair, poem_sampled, poem_tokens, plain_sampled, plain_tokens)
    result["eid"] = run_eid(pair, poem_sampled, poem_tokens, plain_sampled, plain_tokens)

    all_results[pair_id] = result

    del poem_sampled, plain_sampled
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  [完成] {pair['type']}")

# ============================================================
# 汇总
# ============================================================

print("\n" + "=" * 60)
print("汇总统计")
print("=" * 60)

summary = {}
for pair_id, result in all_results.items():
    pair = result["pair"]
    exp1 = result["exp1"]
    exp3 = result["exp3"]

    print(f"\n{pair['type']} ({pair['source']}):")
    print(f"  跳跃距离比: {exp1['jump_ratio']:.2f}x (诗 {exp1['poem_max_jump']:.1f} vs 白话 {exp1['plain_max_jump']:.1f})")
    print(f"  最大跳跃处: {exp1['poem_max_jump_pair']}")

    poem_off = exp3.get("poem_off_diag_high_count", 0)
    plain_off = exp3.get("plain_off_diag_high_count", 0)
    poem_ratio = exp3.get("poem_off_diag_ratio", 0)
    plain_ratio = exp3.get("plain_off_diag_ratio", 0)
    print(f"  非对角线高相似度对: 诗={poem_off} ({poem_ratio:.3f}), 白话={plain_off} ({plain_ratio:.3f})")

    eid = result.get("eid", {})
    poem_eid = eid.get("poem_eid", 0)
    plain_eid = eid.get("plain_eid", 0)
    poem_eid_n = eid.get("poem_eid_normalized", 0)
    plain_eid_n = eid.get("plain_eid_normalized", 0)
    eid_ratio = poem_eid_n / plain_eid_n if plain_eid_n > 0 else float("inf")
    print(f"  EID: 诗={poem_eid:.2f} (归一化{poem_eid_n:.3f}), 白话={plain_eid:.2f} (归一化{plain_eid_n:.3f}), 比值={eid_ratio:.2f}x")

    summary[pair_id] = {
        "type": pair["type"],
        "source": pair["source"],
        "jump_ratio": exp1["jump_ratio"],
        "poem_max_jump": exp1["poem_max_jump"],
        "poem_max_jump_pair": exp1["poem_max_jump_pair"],
        "plain_max_jump": exp1["plain_max_jump"],
        "poem_off_diag_high": poem_off,
        "poem_off_diag_ratio": poem_ratio,
        "plain_off_diag_high": plain_off,
        "plain_off_diag_ratio": plain_ratio,
        "poem_eid": poem_eid,
        "plain_eid": plain_eid,
        "poem_eid_normalized": poem_eid_n,
        "plain_eid_normalized": plain_eid_n,
        "eid_ratio": eid_ratio,
    }

results_path = os.path.join(OUTPUT_DIR, "exp13_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
print(f"\n完整结果已保存: {results_path}")

summary_path = os.path.join(OUTPUT_DIR, "exp13_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"汇总已保存: {summary_path}")

print("\n实验一+三+EID 完成！")
