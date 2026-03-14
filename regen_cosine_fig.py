"""
重新生成"空山不见人"余弦相似度热力图，英文标签，用于公众号配图。
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/workspace/models/Llama-3.3-70B-Instruct-INT8"
OUTPUT_DIR = "/workspace/ai-theorys-study/arxiv/wechat121/results"

POEM = "空山不见人，但闻人语响。"
PLAIN = "山里空荡荡的看不见人，只是偶尔听到有人说话的声音。"

# 英文标签映射
POEM_LABELS = {
    "<|begin_of_text|>": "<BOS>",
    "空": "kong(empty)",
    "山": "shan(mountain)",
    "不": "bu(not)",
    "见": "jian(see)",
    "人": "ren(person)",
    "，但": ", but",
    "闻": "wen(hear)",
    "语": "yu(speech)",
    "响": "xiang(sound)",
    "。": ".",
}

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map={"": 0}, torch_dtype=torch.float16,
    local_files_only=True, low_cpu_mem_usage=True,
)
model.eval()
N_LAYERS = model.config.num_hidden_layers

def get_last_layer(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last = outputs.hidden_states[-1].squeeze(0).cpu().float().numpy()
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    del outputs
    torch.cuda.empty_cache()
    return last, tokens

def make_heatmap(hidden, tokens, labels, title, filename):
    norms = np.linalg.norm(hidden, axis=-1, keepdims=True)
    normalized = hidden / (norms + 1e-10)
    cosine_sim = normalized @ normalized.T

    display = [labels.get(t, t) for t in tokens]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cosine_sim, xticklabels=display, yticklabels=display,
                cmap="RdYlBu_r", vmin=0, vmax=1,
                annot=True, fmt=".2f", ax=ax, square=True, linewidths=0.5,
                annot_kws={"size": 8})
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")

# 诗组
print("提取诗组...")
poem_hidden, poem_tokens = get_last_layer(POEM)
print(f"  tokens: {poem_tokens}")
make_heatmap(poem_hidden, poem_tokens, POEM_LABELS,
             'Poem: "kong shan bu jian ren" (Wang Wei) — Cosine Similarity',
             "fig_poem_cosine_en.png")

# 白话组 - 自动生成编号标签
print("提取白话组...")
plain_hidden, plain_tokens = get_last_layer(PLAIN)
print(f"  tokens: {plain_tokens}")
plain_labels = {}
for i, t in enumerate(plain_tokens):
    if t == "<|begin_of_text|>":
        plain_labels[t] = "<BOS>"
    else:
        plain_labels[t] = plain_labels.get(t, t)  # 保留原样，反正是方块也没关系
# 白话用纯编号，避免乱码
plain_display = [f"t{i}" if t != "<|begin_of_text|>" else "<BOS>" for i, t in enumerate(plain_tokens)]

norms = np.linalg.norm(plain_hidden, axis=-1, keepdims=True)
normalized = plain_hidden / (norms + 1e-10)
cosine_sim = normalized @ normalized.T

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cosine_sim, xticklabels=plain_display, yticklabels=plain_display,
            cmap="RdYlBu_r", vmin=0, vmax=1,
            annot=True, fmt=".2f", ax=ax, square=True, linewidths=0.3,
            annot_kws={"size": 5})
ax.set_title('Prose translation (same meaning) — Cosine Similarity', fontsize=14, fontweight="bold")
plt.tight_layout()

path = os.path.join(OUTPUT_DIR, "fig_plain_cosine_en.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"保存: {path}")

print("完成！")
