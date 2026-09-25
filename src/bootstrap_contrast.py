import argparse
import json

import numpy as np
import pandas as pd

ORDER = ["very_polite", "polite", "neutral", "rude", "very_rude", "command", "profane"]


def per_item_reg_acc(df):

    return df.groupby(["item_id", "register"]).correct.mean().unstack("register")


def bootstrap(df, n_boot=5000, seed=0):
    piv = per_item_reg_acc(df)
    items = piv.index.to_numpy()
    rng = np.random.default_rng(seed)
    cols = [c for c in ORDER if c in piv.columns]
    mat = piv[cols].to_numpy()
    boot = np.zeros((n_boot, len(cols)))
    for b in range(n_boot):
        samp = rng.integers(0, len(items), len(items))
        boot[b] = np.nanmean(mat[samp], axis=0)
    point = np.nanmean(mat, axis=0)
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    idx = {c: i for i, c in enumerate(cols)}

    print("  per-register acc [95% CI]:")
    for c in cols:
        print(f"    {c:12s} {point[idx[c]]:.4f} "
              f"[{ci[0][idx[c]]:.4f},{ci[1][idx[c]]:.4f}]")

    def contrast(a, b):
        d = boot[:, idx[a]] - boot[:, idx[b]]
        point_d = point[idx[a]] - point[idx[b]]
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        lo, hi = np.percentile(d, [2.5, 97.5])
        return point_d, lo, hi, p

    out = {}
    for a, b in [("very_polite", "very_rude")] + [(r, "neutral") for r in cols if r != "neutral"]:
        pd_, lo, hi, p = contrast(a, b)
        out[f"{a}-{b}"] = {"delta": pd_, "ci": [lo, hi], "p": p}
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {a} - {b:11s} dAcc={pd_:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4g} {star}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--exclude", nargs="*", default=[], help="datasets to drop (e.g. arc)")
    args = ap.parse_args()
    df = pd.DataFrame(json.loads(l) for l in open(args.inp))
    model = df.model.unique().tolist()
    print(f"== {model} | n={len(df)} rows, {df.item_id.nunique()} items ==")
    print("[all datasets]")
    bootstrap(df, n_boot=args.n_boot)
    if args.exclude:
        sub = df[~df.dataset.isin(args.exclude)]
        print(f"\n[excluding {args.exclude}]  ({sub.item_id.nunique()} items)")
        bootstrap(sub, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
