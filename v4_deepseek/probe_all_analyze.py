"""
离线分析 probe_all.pt —— 5 组诗，三条线一起出结论。
A. 余弦正交 + EID（hidden states, layer 28 主层）
B. MoE 专家正交（桥接字 vs 两侧域的 Jaccard 重叠）
C. 哈希层 vs 分数层（语义点火）
纯 CPU，宿主机直接跑。
"""
import os
import json
import torch
import numpy as np

D = torch.load("probe_all.pt", map_location="cpu", weights_only=False)
N_LAYERS = D["n_layers"]
N_EXP = D["n_routed_experts"]
TOPK = D["topk"]
N_HASH = D["n_hash_layers"]
HID_LAYERS = D["hidden_layers"]
results = D["results"]
PUNCT = {"，", "。", "、", "！", "？"}

print(f"{N_LAYERS} 层, {N_EXP} 专家选 {TOPK}, 前 {N_HASH} 层哈希路由")
print(f"hidden 抽层: {HID_LAYERS}\n")

pair_ids = sorted(set(k.rsplit("_", 1)[0] for k in results))


# ---------- 字节级 token 正确解码 ----------
# 生僻字（如「鹜」）会被 BPE 拆成多个字节 token，单独 decode 显示为 �。
# 这里从 tokenizer.json 的词表把每个 id 还原成原始字节，相邻字节拼回完整字。
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
    with open(TOKENIZER_JSON) as f:
        _vocab = json.load(f)["model"]["vocab"]
    _id2str = {v: k for k, v in _vocab.items()}
    _ID2BYTES = {}
    for i, s in _id2str.items():
        try:
            _ID2BYTES[i] = bytes(_U2B[c] for c in s)
        except KeyError:
            _ID2BYTES[i] = None  # 含特殊符号的 token，跳过

def display_tokens(input_ids, fallback_tokens):
    """把字节碎片拼成完整字，返回 (显示字符列表, 是否多字节标记列表)。
    长度与 input_ids 一致：被拼合的碎片里，首位放完整字，其余位放''占位。"""
    if _ID2BYTES is None:
        return list(fallback_tokens), [False]*len(fallback_tokens)
    out, multibyte = [], []
    buf = b""; buf_start = 0
    raw = [_ID2BYTES.get(t) for t in input_ids]
    i = 0
    pending_positions = []
    result = [None]*len(input_ids)
    flag = [False]*len(input_ids)
    while i < len(input_ids):
        buf = raw[i] if raw[i] is not None else b""
        j = i
        # 尝试往后拼直到能 decode 成合法 utf-8
        while True:
            try:
                ch = buf.decode("utf-8")
                break
            except UnicodeDecodeError:
                j += 1
                if j >= len(input_ids) or raw[j] is None:
                    ch = fallback_tokens[i]  # 拼不出，退回原样
                    break
                buf += raw[j]
        result[i] = ch
        if j > i:
            flag[i] = True
            for k in range(i+1, j+1):
                result[k] = ""  # 占位，已被并入 i
        i = j + 1
    return result, flag


# ---------- 工具 ----------
def fold_stream0(hid):  # [1,seq,4,4096] -> [seq,4096]
    return hid[0, :, 0, :].float().numpy()

def cosine_high_pairs(h, thresh=0.8, min_dist=2):
    hn = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-10)
    cos = hn @ hn.T
    n = h.shape[0]
    return sum(1 for i in range(n) for j in range(i+1, n)
               if abs(i-j) > min_dist and cos[i][j] > thresh)

def eid(h):
    _, S, _ = np.linalg.svd(h, full_matrices=False)
    s2 = (S**2); s2 = s2[s2 > 1e-12]; p = s2/s2.sum()
    return float(np.exp(-np.sum(p*np.log(p))))

def jaccard(a, b):
    sa, sb = set(a), set(b); u = sa|sb
    return len(sa & sb)/len(u) if u else 0.0


# ============ A. 余弦正交 + EID（主层取 28）============
MAIN = 28 if 28 in HID_LAYERS else HID_LAYERS[len(HID_LAYERS)//2]
print("="*70)
print(f"A. 余弦正交 + EID  (hidden layer {MAIN}, stream0)")
print("="*70)
print(f"{'诗组':<14}{'诗·高相似对':<12}{'白话·高相似对':<14}{'诗归一EID':<11}{'白话归一EID':<13}{'EID比值':<8}")
for pid in pair_ids:
    p, pl = results[f"{pid}_poem"], results[f"{pid}_plain"]
    ph, plh = fold_stream0(p["hidden"][MAIN]), fold_stream0(pl["hidden"][MAIN])
    p_hi, pl_hi = cosine_high_pairs(ph), cosine_high_pairs(plh)
    p_eid = eid(ph)/len(p["tokens"]); pl_eid = eid(plh)/len(pl["tokens"])
    ratio = p_eid/pl_eid if pl_eid > 0 else float("inf")
    print(f"{p['source'][:12]:<14}{p_hi:<12}{pl_hi:<14}{p_eid:<11.3f}{pl_eid:<13.3f}{ratio:<8.2f}")
print()


# ============ B. MoE 专家正交：桥接字 vs 两侧域 ============
PROBE = N_LAYERS - 1
print("="*70)
print(f"B. 桥接字的专家枢纽性  (MoE layer {PROBE}, Jaccard 专家重叠)")
print("="*70)
for pid in pair_ids:
    p = results[f"{pid}_poem"]
    toks, _ = display_tokens(p["input_ids"], p["tokens"])
    idx = p["routing"][PROBE]; bridge = p["bridge_token"]
    bi = next((t for t, tok in enumerate(toks) if bridge in tok), None)
    print(f"\n{p['source']}  桥接字「{bridge}」")
    if bi is None:
        print("  (桥接字未找到)"); continue
    real = [(t, tok) for t, tok in enumerate(toks) if tok.strip() and tok not in PUNCT]
    overlaps = [(tok, jaccard(idx[bi].tolist(), idx[t].tolist()))
                for t, tok in real if t != bi]
    overlaps.sort(key=lambda x: -x[1])
    line = "  与各字专家重叠: " + "  ".join(f"{tok}={j:.2f}" for tok, j in overlaps)
    print(line)


# ============ C. 哈希层 vs 分数层：语义点火 + 专家多样性 ============
print("\n" + "="*70)
print("C. 哈希层 vs 分数层：共享专家 + 专家多样性")
print("="*70)
for pid in pair_ids:
    p = results[f"{pid}_poem"]
    toks, _ = display_tokens(p["input_ids"], p["tokens"])
    routing = p["routing"]
    real_idx = [t for t, tok in enumerate(toks) if tok.strip() and tok not in PUNCT]
    print(f"\n{p['source']} ({len(real_idx)} 实义字)")
    for lid in [0, N_HASH, PROBE]:
        idx = routing[lid]
        acts = [idx[t].tolist() for t in real_idx]
        flat = [e for a in acts for e in a]
        uniq = len(set(flat))
        # 被≥半数字共享的专家
        from collections import Counter
        c = Counter(flat)
        shared = [e for e, n in c.items() if n >= len(real_idx)/2]
        tag = "哈希" if lid < N_HASH else "分数"
        print(f"  L{lid:<2}[{tag}]: 用了 {uniq:>2} 个不同专家 / {len(real_idx)*TOPK} 次"
              f"，被半数+字共享的专家: {sorted(shared)}")
