"""
桥接字的专家重叠，在 layer 28（2/3处）vs layer 42（末层）对比。
质疑：末层可能"向输出收敛"，2/3 处语义更丰富——专家路由结论会不会随层变？
对 5 个指定桥接字，两层各算它和句中各字的平均/最高重叠 + 排名。纯 CPU，复用 probe_all.pt。
"""
import os, json, torch

D = torch.load("probe_all.pt", map_location="cpu", weights_only=False)
results = D["results"]
PUNCT = {"，", "。", "、", "！", "？"}
LAYERS = [28, D["n_layers"]-1]  # 2/3处 和 末层

# 字节碎片合并解码
TOKENIZER_JSON = os.environ.get("TOKENIZER_JSON",
    "/home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash/tokenizer.json")
def _b2u():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1))
    cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return {chr(c):b for c,b in zip(cs,bs)}
_U2B=_b2u()
_ID2B={}
for i,s in {v:k for k,v in json.load(open(TOKENIZER_JSON))["model"]["vocab"].items()}.items():
    try: _ID2B[i]=bytes(_U2B[c] for c in s)
    except KeyError: _ID2B[i]=None
def disp(ids, fb):
    raw=[_ID2B.get(t) for t in ids]; r=[None]*len(ids); i=0
    while i<len(ids):
        buf=raw[i] if raw[i] is not None else b""; j=i
        while True:
            try: ch=buf.decode("utf-8"); break
            except UnicodeDecodeError:
                j+=1
                if j>=len(ids) or raw[j] is None: ch=fb[i]; break
                buf+=raw[j]
        r[i]=ch
        for k in range(i+1,j+1): r[k]=""
        i=j+1
    return r
def jac(a,b):
    sa,sb=set(a),set(b); u=sa|sb
    return len(sa&sb)/len(u) if u else 0.0

pair_ids = sorted(set(k.rsplit("_",1)[0] for k in results))
for pid in pair_ids:
    p = results[f"{pid}_poem"]
    toks = disp(p["input_ids"], p["tokens"])
    bridge = p["bridge_token"]
    real = [(t,tok) for t,tok in enumerate(toks) if tok.strip() and tok not in PUNCT]
    bi = next((t for t,tok in real if bridge in tok), None)
    print(f"\n{p['source']}  桥接字「{bridge}」")
    if bi is None: print("  未找到"); continue
    for lid in LAYERS:
        idx = p["routing"][lid]
        ov = [(tok, jac(idx[bi].tolist(), idx[t].tolist())) for t,tok in real if t!=bi]
        avg = sum(j for _,j in ov)/len(ov)
        # 桥接字在全句平均重叠里排第几
        allscore=[]
        for ti,tk in real:
            o=[jac(idx[ti].tolist(),idx[tj].tolist()) for tj,_ in real if tj!=ti]
            allscore.append((tk,sum(o)/len(o)))
        allscore.sort(key=lambda x:-x[1])
        rank = next(r for r,(tk,_) in enumerate(allscore,1) if tk==toks[bi])
        tag = "末层" if lid==LAYERS[-1] else "2/3层"
        ovs = "  ".join(f"{tk}={j:.2f}" for tk,j in sorted(ov,key=lambda x:-x[1]))
        print(f"  L{lid}[{tag}] 平均={avg:.2f} 排名{rank}/{len(real)}: {ovs}")
