import argparse
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - half, center + half)


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)

    return df


def accuracy_table(df):
    print("\n=== Accuracy per register (overall) ===")
    order = ["very_polite", "polite", "neutral", "rude", "very_rude", "command"]
    for reg in order:
        sub = df[df.register == reg]
        k, n = int(sub.correct.sum()), len(sub)
        lo, hi = wilson_ci(k, n)
        print(f"  {reg:12s}  acc={k/n:.4f}  [{lo:.4f},{hi:.4f}]  n={n}")

    print("\n=== Accuracy per register x dataset ===")
    for ds in sorted(df.dataset.unique()):
        print(f"  -- {ds} --")
        for reg in order:
            sub = df[(df.register == reg) & (df.dataset == ds)]
            if len(sub) == 0:
                continue
            k, n = int(sub.correct.sum()), len(sub)
            lo, hi = wilson_ci(k, n)
            print(f"    {reg:12s} acc={k/n:.4f} [{lo:.4f},{hi:.4f}] n={n}")


def mcnemar_vs_neutral(df):
    from statsmodels.stats.contingency_tables import mcnemar
    print("\n=== McNemar paired test vs. neutral (variant-averaged>=0.5) ===")

    g = (df.groupby(["item_id", "register"]).correct.mean() >= 0.5).astype(int)
    piv = g.unstack("register")
    if "neutral" not in piv.columns:
        print("  no neutral column"); return
    for reg in piv.columns:
        if reg == "neutral":
            continue
        paired = piv[["neutral", reg]].dropna()
        b = int(((paired.neutral == 1) & (paired[reg] == 0)).sum())
        c = int(((paired.neutral == 0) & (paired[reg] == 1)).sum())
        table = [[0, b], [c, 0]]
        res = mcnemar(table, exact=(b + c) < 25)
        acc_n = paired.neutral.mean()
        acc_r = paired[reg].mean()
        print(f"  neutral vs {reg:12s} dAcc={acc_r-acc_n:+.4f} "
              f"(n_right={c}, n_wrong={b}) p={res.pvalue:.4g}")


def glmm(df):
    print("\n=== Logistic GLMM: correct ~ C(register) + (1|item) ===")
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception as e:
        print("  statsmodels GLMM unavailable:", e); return
    d = df.copy()
    d["register"] = pd.Categorical(
        d["register"],
        categories=["neutral", "very_polite", "polite", "rude", "very_rude", "command"],
    )

    d["item_code"] = d["item_id"].astype("category").cat.codes
    vc = {"item": "0 + C(item_code)"}
    try:
        model = BinomialBayesMixedGLM.from_formula(
            "correct ~ C(register, Treatment('neutral'))", vc, d
        )
        res = model.fit_vb()
        print(res.summary())
    except Exception as e:
        print("  GLMM fit failed:", e)


def refusal_table(df):
    if "refusal" not in df.columns:
        return
    print("\n=== Refusal rate per register ===")
    order = ["very_polite", "polite", "neutral", "rude", "very_rude", "command"]
    for reg in order:
        sub = df[df.register == reg]
        if len(sub) == 0:
            continue
        print(f"  {reg:12s} refusal={sub.refusal.mean():.4f} n={len(sub)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    args = ap.parse_args()
    df = load(args.inp)
    print(f"loaded {len(df)} records; model(s): {df.model.unique().tolist()}")
    accuracy_table(df)
    refusal_table(df)
    mcnemar_vs_neutral(df)
    glmm(df)


if __name__ == "__main__":
    main()
