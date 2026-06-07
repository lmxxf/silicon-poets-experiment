"""
验证：余弦"高重叠字对数"在 4 路 HC 残差不同折算下会不会变？
质疑：白话组的高重叠对数偏低（使至塞上0、将进酒0），是不是 stream0 单路丢了信号？
对 5 组诗/白话，在 stream0 / 4路求和 / 4路拼接 三种折算下各数一遍。纯 CPU，复用 probe_all.pt。
"""
import torch, numpy as np

D = torch.load("probe_all.pt", map_location="cpu", weights_only=False)
HID = D["hidden_layers"]; results = D["results"]
MAIN = 28 if 28 in HID else HID[len(HID)//2]

def fold(hid, mode):
    a = hid[0].float()
    if mode=="stream0": return a[:,0,:].numpy()
    if mode=="sum":     return a.sum(dim=1).numpy()
    if mode=="concat":  return a.reshape(a.shape[0],-1).numpy()

def high_pairs(h, thr=0.8, mind=2):
    hn = h/(np.linalg.norm(h,axis=-1,keepdims=True)+1e-10)
    cos = hn@hn.T; n=h.shape[0]
    return sum(1 for i in range(n) for j in range(i+1,n) if abs(i-j)>mind and cos[i][j]>thr)

pair_ids = sorted(set(k.rsplit("_",1)[0] for k in results))
print(f"余弦高重叠对数（>0.8，非相邻），layer {MAIN}，三种折算\n")
print(f"{'诗组':<14}{'诗 s0/sum/cat':<20}{'白话 s0/sum/cat':<20}")
for pid in pair_ids:
    p, pl = results[f"{pid}_poem"], results[f"{pid}_plain"]
    pr = [high_pairs(fold(p["hidden"][MAIN],m)) for m in ("stream0","sum","concat")]
    pn = [high_pairs(fold(pl["hidden"][MAIN],m)) for m in ("stream0","sum","concat")]
    print(f"{p['source'][:12]:<14}{str(pr):<20}{str(pn):<20}")
