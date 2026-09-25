import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def load(path):
    d = np.load(path, allow_pickle=True)
    need = ["X", "polarity", "item", "variant", "plen"]
    for k in need:
        if k not in d:
            raise SystemExit(f"{path} missing '{k}' — re-extract with updated extract_activations.py")
    return d["X"], d["polarity"], d["item"], d["variant"], d["plen"]


def heldout_phrasing_auc(Xl, pol, item, var):
    m = pol != 0
    X, y, it, v = Xl[m], (pol[m] == 1).astype(int), item[m], var[m]
    tr, te = (v <= 1), (v == 2)
    if te.sum() == 0 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return np.nan
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[tr]), y[tr])
    p = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y[te], p)


def length_alone_auc(pol, plen):
    m = pol != 0
    y, L = (pol[m] == 1).astype(int), plen[m].reshape(-1, 1).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    clf = LogisticRegression(max_iter=2000).fit(L, y)
    return roc_auc_score(y, clf.predict_proba(L)[:, 1])


def residualized_auc(Xl, pol, plen):
    m = pol != 0
    X, y, L = Xl[m], (pol[m] == 1).astype(int), plen[m].reshape(-1, 1).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan

    Xr = X - LinearRegression().fit(L, X).predict(L)
    sc = StandardScaler().fit(Xr)


    clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xr), y)
    return roc_auc_score(y, clf.predict_proba(sc.transform(Xr))[:, 1])


def raw_insample_auc(Xl, pol):
    m = pol != 0
    X, y = Xl[m], (pol[m] == 1).astype(int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X), y)
    return roc_auc_score(y, clf.predict_proba(sc.transform(X))[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 10, 13, 18])
    args = ap.parse_args()
    X, pol, item, var, plen = load(args.inp)
    print(f"loaded X={X.shape}, variants={sorted(set(var.tolist()))}")

    for p in (1, -1, 0):
        mm = pol == p
        print(f"  polarity {p:+d}: mean prompt tokens = {plen[mm].mean():.1f} (n={mm.sum()})")
    print(f"  length-ALONE polite/rude AUC = {length_alone_auc(pol, plen):.3f} "
          f"(high => length is a strong confound)")
    print("\n layer  raw_AUC  heldout_phrasing_AUC  length_residualized_AUC")
    for l in args.layers:
        Xl = X[:, l, :].astype(np.float32)
        print(f"  {l:2d}    {raw_insample_auc(Xl, pol):.3f}     "
              f"{heldout_phrasing_auc(Xl, pol, item, var):.3f}                "
              f"{residualized_auc(Xl, pol, plen):.3f}")
    print("\nInterpretation: register is REAL (not lexical/length) iff held-out-phrasing"
          " AUC stays >~0.8 AND length-residualized AUC stays high while length-alone is low.")


if __name__ == "__main__":
    main()
