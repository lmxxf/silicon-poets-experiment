"""
让模型自己找桥接字 —— 不预设，从专家路由数据里算。
桥接字的特征：和句中各字的专家重叠都不低、谁都不偏 → 平均 Jaccard 重叠最高。
扫每个实义字对全句其他字的平均专家重叠，排序，看最高的是不是我们按《硅基诗学》指定的。
纯 CPU，宿主机直接跑，复用 probe_all.pt，无需重跑模型。
"""
import os
import json
import torch

D = torch.load("probe_all.pt", map_location="cpu", weights_only=False)
N_LAYERS = D["n_layers"]
results = D["results"]
PUNCT = {"，", "。", "、", "！", "？"}
PROBE = N_LAYERS - 1  # 末层，和正文 B 段一致

# --- 字节碎片合并解码（复用 probe_all_analyze 的逻辑） ---
TOKENIZER_JSON = os.environ.get(
    "TOKENIZER_JSON",
    "/home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash/tokenizer.json",
)
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return {chr(c): b for c, b in zip(cs, bs)}
_U2B = _bytes_to_unicode()
_ID2BYTES = None
if os.path.exists(TOKENIZER_JSON):
    _vocab = json.load(open(TOKENIZER_JSON))["model"]["vocab"]
    _id2str = {v: k for k, v in _vocab.items()}
    _ID2BYTES = {}
    for i, s in _id2str.items():
        try: _ID2BYTES[i] = bytes(_U2B[c] for c in s)
        except KeyError: _ID2BYTES[i] = None

def display_tokens(input_ids, fallback):
    if _ID2BYTES is None:
        return list(fallback)
    raw = [_ID2BYTES.get(t) for t in input_ids]
    result = [None]*len(input_ids); i = 0
    while i < len(input_ids):
        buf = raw[i] if raw[i] is not None else b""; j = i
        while True:
            try: ch = buf.decode("utf-8"); break
            except UnicodeDecodeError:
                j += 1
                if j >= len(input_ids) or raw[j] is None: ch = fallback[i]; break
                buf += raw[j]
        result[i] = ch
        for k in range(i+1, j+1): result[k] = ""
        i = j + 1
    return result

def jaccard(a, b):
    sa, sb = set(a), set(b); u = sa|sb
    return len(sa & sb)/len(u) if u else 0.0


print(f"模型自己找桥接字（末层 L{PROBE}，按平均专家重叠排序）\n")
pair_ids = sorted(set(k.rsplit("_", 1)[0] for k in results))

hit = 0
for pid in pair_ids:
    p = results[f"{pid}_poem"]
    toks = display_tokens(p["input_ids"], p["tokens"])
    idx = p["routing"][PROBE]
    designated = p["bridge_token"]

    real = [(t, tok) for t, tok in enumerate(toks) if tok.strip() and tok not in PUNCT]
    # 每个实义字对其他实义字的平均 Jaccard
    scores = []
    for ti, toki in real:
        others = [jaccard(idx[ti].tolist(), idx[tj].tolist()) for tj, _ in real if tj != ti]
        avg = sum(others)/len(others) if others else 0.0
        scores.append((toki, avg, ti))
    scores.sort(key=lambda x: -x[1])

    model_bridge = scores[0][0]
    is_hit = (designated in model_bridge) or (model_bridge in designated)
    hit += is_hit
    mark = "✅一致" if is_hit else "⚠️不一致"

    print(f"{p['source']}")
    print(f"  指定桥接字（理论）: 「{designated}」")
    print(f"  模型选出（平均重叠最高）: 「{model_bridge}」  {mark}")
    rank = "  ".join(f"{tok}={s:.2f}" for tok, s, _ in scores)
    print(f"  全字排名: {rank}\n")

print(f"=== 5 组里，模型自选 = 理论指定 的有 {hit}/5 组 ===")
