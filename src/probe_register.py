import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["X"], d["register"], d["polarity"], d["item"], d["dataset"]


def diff_in_means(Xl, polarity, item):
    dirs = []
    for it in np.unique(item):
        m = item == it
        pol = polarity[m]
        xpos = Xl[m][pol == 1]
        xneg = Xl[m][pol == -1]
        if len(xpos) and len(xneg):
            dirs.append(xpos.mean(0) - xneg.mean(0))
    if not dirs:
        return None
    return np.mean(dirs, axis=0)


def probe_polite_vs_rude(Xl, polarity, item):
    mask = polarity != 0
    X, y, g = Xl[mask], (polarity[mask] == 1).astype(int), item[mask]
    if len(np.unique(y)) < 2:
        return np.nan
    n_groups = len(np.unique(g))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    aucs = []
    for tr, te in gkf.split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5)
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


def probe_six_way(Xl, register, item):
    n_groups = len(np.unique(item))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    accs = []
    for tr, te in gkf.split(Xl, register, item):
        sc = StandardScaler().fit(Xl[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5)
        clf.fit(sc.transform(Xl[tr]), register[tr])
        accs.append(accuracy_score(register[te], clf.predict(sc.transform(Xl[te]))))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    X, register, polarity, item, dataset = load(args.inp)
    N, L, Dm = X.shape
    print(f"loaded X={X.shape} items={len(np.unique(item))} "
          f"chance(6-way)={1/6:.3f}")

    results = []
    for l in range(L):
        Xl = X[:, l, :].astype(np.float32)
        auc = probe_polite_vs_rude(Xl, polarity, item)
        acc6 = probe_six_way(Xl, register, item)
        d = diff_in_means(Xl, polarity, item)
        dnorm = float(np.linalg.norm(d)) if d is not None else float("nan")
        results.append({"layer": l, "auc_polite_rude": auc,
                        "acc_6way": acc6, "diffmeans_norm": dnorm})
        print(f"  layer {l:2d}  AUC(pol/rude)={auc:.3f}  "
              f"acc6={acc6:.3f}  |dmean|={dnorm:.2f}")

    best = max(results, key=lambda r: (r["auc_polite_rude"]
                                       if not np.isnan(r["auc_polite_rude"]) else -1))
    print(f"\nBEST polite/rude AUC = {best['auc_polite_rude']:.3f} "
          f"at layer {best['layer']}  (>0.8 threshold: "
          f"{'PASS' if best['auc_polite_rude'] > 0.8 else 'below'})")
    if args.out:
        json.dump({"results": results, "best": best}, open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
