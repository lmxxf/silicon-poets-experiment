"""
验证：EID 打折是不是因为只用了 4 路 HC 残差里的 1 路？
对每组诗，在 stream0 / 4路求和 / 4路拼接 三种折算下都算归一化 EID 比值，
看用更多路时 EID 比值是否回升。纯 CPU，复用 probe_all.pt。
"""
import torch
import numpy as np

D = torch.load("probe_all.pt", map_location="cpu", weights_only=False)
HID_LAYERS = D["hidden_layers"]
results = D["results"]
MAIN = 28 if 28 in HID_LAYERS else HID_LAYERS[len(HID_LAYERS)//2]

def fold(hid, mode):  # hid: [1,seq,4,4096] -> [seq, d]
    a = hid[0].float()  # [seq,4,4096]
    if mode == "stream0": return a[:, 0, :].numpy()
    if mode == "sum":     return a.sum(dim=1).numpy()
    if mode == "concat":  return a.reshape(a.shape[0], -1).numpy()

def eid(h):
    _, S, _ = np.linalg.svd(h, full_matrices=False)
    s2 = (S**2); s2 = s2[s2 > 1e-12]; p = s2/s2.sum()
    return float(np.exp(-np.sum(p*np.log(p))))

pair_ids = sorted(set(k.rsplit("_", 1)[0] for k in results))
print(f"归一化 EID 比值（诗/白话），layer {MAIN}，三种折算对比\n")
print(f"{'诗组':<14}{'stream0':<10}{'4路求和':<10}{'4路拼接':<10}")
sums = {"stream0": [], "sum": [], "concat": []}
for pid in pair_ids:
    p, pl = results[f"{pid}_poem"], results[f"{pid}_plain"]
    row = f"{p['source'][:12]:<14}"
    for mode in ("stream0", "sum", "concat"):
        pe = eid(fold(p["hidden"][MAIN], mode)) / len(p["tokens"])
        ple = eid(fold(pl["hidden"][MAIN], mode)) / len(pl["tokens"])
        ratio = pe/ple if ple > 0 else 0
        sums[mode].append(ratio)
        row += f"{ratio:<10.2f}"
    print(row)
print(f"\n{'平均':<14}" + "".join(f"{np.mean(sums[m]):<10.2f}" for m in ("stream0","sum","concat")))
