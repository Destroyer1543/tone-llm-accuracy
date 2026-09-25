import json
import glob
import os
import numpy as np
import pandas as pd


CANDIDATES = [
    ("results/phase0_qwen0.5b.jsonl", "Qwen2.5-0.5B"),
    ("results/phase0_qwen1.5b.jsonl", "Qwen2.5-1.5B"),
    ("results/phase0_qwen3b.jsonl", "Qwen2.5-3B"),
    ("results/phase0_qwen7b.jsonl", "Qwen2.5-7B"),
    ("results/phase0_gemma2-2b_full.jsonl", "Gemma-2-2B"),
    ("results/phase0_gemma2-9b_full.jsonl", "Gemma-2-9B"),
    ("results/phase0_llama31-8b_full.jsonl", "Llama-3.1-8B"),
    ("results/phase0_olmo2-7b.jsonl", "OLMo-2-7B"),
    ("results/phase0_mistral7b.jsonl", "Mistral-7B"),
    ("results/phase0_llama32-3b.jsonl", "Llama-3.2-3B"),
    ("results/phase0_llama32-1b.jsonl", "Llama-3.2-1B"),
    ("results/phase0_gptoss20b.jsonl", "gpt-oss-20B"),
]


def contrast(df, a="very_polite", b="very_rude", nboot=5000, seed=0):
    piv = df.groupby(["item_id", "register"]).correct.mean().unstack("register")
    if a not in piv or b not in piv:
        return None
    d = (piv[a] - piv[b]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    bt = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)])
    p = 2 * min((bt <= 0).mean(), (bt >= 0).mean())
    return d.mean(), np.percentile(bt, 2.5), np.percentile(bt, 97.5), p


def bh_fdr(pvals):
    p = np.asarray(pvals); n = len(p); order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank, i in enumerate(reversed(order)):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj


def main():
    rows = []
    for path, lab in CANDIDATES:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        df = pd.DataFrame(json.loads(l) for l in open(path))
        r = contrast(df)
        if r is None:
            continue
        d, lo, hi, p = r
        rows.append({"model": lab, "acc": df.correct.mean(), "d": d,
                     "lo": lo, "hi": hi, "p": p, "n_items": df.item_id.nunique()})
    if not rows:
        print("no data yet"); return
    res = pd.DataFrame(rows)
    res["p_fdr"] = bh_fdr(res["p"].values)
    res = res.sort_values("acc")
    print("=== very_polite - very_rude across models (item cluster bootstrap, BH-FDR) ===")
    print(f"{'model':16s} {'acc':>6s} {'dAcc':>8s} {'95% CI':>18s} {'p':>8s} {'p_FDR':>8s}  sig")
    for _, r in res.iterrows():
        sig = "*" if r.p_fdr < 0.05 else ""
        print(f"{r.model:16s} {r.acc:6.3f} {r.d:+8.4f}  [{r.lo:+.3f},{r.hi:+.3f}] "
              f"{r.p:8.4g} {r.p_fdr:8.4g}  {sig}")
    n_sig = (res.p_fdr < 0.05).sum()
    print(f"\n{n_sig}/{len(res)} models show a significant tone effect after FDR "
          f"correction. Signs: {sorted(np.sign(res.d).astype(int).tolist())}")


if __name__ == "__main__":
    main()
