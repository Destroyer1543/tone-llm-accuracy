import json, os, glob
import numpy as np
import pandas as pd


def boot_ci(d, nboot=10000, seed=0, lo=5, hi=95):
    rng = np.random.default_rng(seed)
    bt = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)])
    return d.mean(), np.percentile(bt, lo), np.percentile(bt, hi), bt


def acc_contrast(f):
    df = pd.DataFrame(json.loads(l) for l in open(f))
    piv = df.groupby(["item_id", "register"]).correct.mean().unstack("register")
    return (piv["very_polite"] - piv["very_rude"]).dropna().to_numpy()


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); adj = np.empty(n); prev = 1.0
    for rank, i in enumerate(reversed(o)):
        prev = min(prev, p[i] * n / (n - rank)); adj[i] = prev
    return adj


print("=== #2 EQUIVALENCE BOUNDS (high-power, 90% CI on vpol-vrude) ===")
print("Pooling both high-power seeds per model where available.")
for m in ["gemma2-2b", "llama31-8b", "qwen7b"]:
    ds = [acc_contrast(f) for s in [0, 1]
          for f in [f"results/hp_{m}_seed{s}.jsonl"] if os.path.exists(f) and os.path.getsize(f)]
    if not ds:
        continue
    d = np.concatenate(ds)
    mean, lo, hi, _ = boot_ci(d)
    bound = max(abs(lo), abs(hi))
    print(f"  {m:12s} dAcc={mean:+.4f}  90%CI[{lo:+.4f},{hi:+.4f}]  "
          f"-> rule out |effect| > {bound*100:.2f} pts")

print("\n=== #3 SYCOPHANCY contrasts with BH-FDR (polite-pole vs rude-pole) ===")
rng = np.random.default_rng(0)
rows = []
for f in sorted(glob.glob("results/syc_*.jsonl")):
    base = os.path.basename(f)
    if "steer" in base or "ctrl" in base or "p2" in base or "random" in base:
        continue
    df = pd.DataFrame(json.loads(l) for l in open(f))
    if "sycophantic" not in df or df.empty:
        continue
    metric = "flip" if "mild" in base else "sycophantic"
    piv = df.groupby(["item_id", "register"])[metric].mean().unstack("register")
    need = ["very_polite", "polite", "rude", "very_rude", "command"]
    if not all(c in piv for c in need):
        continue
    pol = piv[["very_polite", "polite"]].mean(1)
    rud = piv[["rude", "very_rude", "command"]].mean(1)
    d = (pol - rud).dropna().to_numpy()
    bt = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    p = 2 * min((bt <= 0).mean(), (bt >= 0).mean())
    rows.append({"file": base.replace("syc_", "").replace(".jsonl", ""),
                 "metric": metric, "d": d.mean(), "p": p})
if rows:
    res = pd.DataFrame(rows); res["p_fdr"] = bh(res["p"].values)
    res = res.sort_values("file")
    for _, r in res.iterrows():
        sig = "*" if r.p_fdr < 0.05 else ""
        print(f"  {r.file:26s} [{r.metric:11s}] d={r.d:+.4f} p={r.p:.3g} p_fdr={r.p_fdr:.3g} {sig}")
    print(f"\n  {(res.p_fdr<0.05).sum()}/{len(res)} sycophancy contrasts significant after FDR.")
