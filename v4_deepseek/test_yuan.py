"""
验证「圆」是不是诗眼 —— 用 121 期同角度（指定字 → 看它和句中各字的专家重叠）。
《使至塞上》"大漠孤烟直，长河落日圆"，《红楼梦》香菱学诗并称「直」「圆」两字。
我们当初只指定了「直」，现在补测「圆」，看 AI 同不同意它也坐中间。
对比「直」「圆」两个字各自的专家重叠分布。纯 CPU，复用 probe_all.pt。
"""
import os
import json
import torch

D = torch.load("probe_all.pt", map_location="cpu", weights_only=False)
N_LAYERS = D["n_layers"]
results = D["results"]
PUNCT = {"，", "。", "、", "！", "？"}
PROBE = N_LAYERS - 1

TOKENIZER_JSON = os.environ.get(
    "TOKENIZER_JSON",
    "/home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash/tokenizer.json",
)
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n += 1
    return {chr(c): b for c, b in zip(cs, bs)}
_U2B = _bytes_to_unicode()
_vocab = json.load(open(TOKENIZER_JSON))["model"]["vocab"]
_id2str = {v: k for k, v in _vocab.items()}
_ID2BYTES = {}
for i, s in _id2str.items():
    try: _ID2BYTES[i] = bytes(_U2B[c] for c in s)
    except KeyError: _ID2BYTES[i] = None

def display_tokens(input_ids, fallback):
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

p = results["2_singularity_poem"]
toks = display_tokens(p["input_ids"], p["tokens"])
idx = p["routing"][PROBE]
real = [(t, tok) for t, tok in enumerate(toks) if tok.strip() and tok not in PUNCT]

print(f"《使至塞上》大漠孤烟直，长河落日圆  (末层 L{PROBE})\n")
print(f"实义字: {[tok for _, tok in real]}\n")

for target in ["直", "圆"]:
    ti = next((t for t, tok in real if target in tok), None)
    if ti is None:
        print(f"「{target}」未找到\n"); continue
    overlaps = [(tok, jaccard(idx[ti].tolist(), idx[tj].tolist())) for tj, tok in real if tj != ti]
    overlaps.sort(key=lambda x: -x[1])
    avg = sum(j for _, j in overlaps)/len(overlaps)
    print(f"「{target}」平均重叠={avg:.2f}")
    print("  " + "  ".join(f"{tok}={j:.2f}" for tok, j in overlaps) + "\n")

# 全句每个字的平均重叠排名（看「直」「圆」排第几）
print("全字平均重叠排名（越高=越像诗眼，跟全句越搭）:")
scores = []
for ti, toki in real:
    o = [jaccard(idx[ti].tolist(), idx[tj].tolist()) for tj, _ in real if tj != ti]
    scores.append((toki, sum(o)/len(o)))
scores.sort(key=lambda x: -x[1])
for rank, (tok, s) in enumerate(scores, 1):
    mark = " ←" if tok in ("直", "圆") else ""
    print(f"  {rank}. {tok}={s:.2f}{mark}")
