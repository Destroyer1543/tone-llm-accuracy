import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = np.load(args.inp, allow_pickle=True)
    X, polarity, item = d["X"], d["polarity"], d["item"]
    N, L, D = X.shape
    dirs = np.zeros((L, D), dtype=np.float32)
    for l in range(L):
        Xl = X[:, l, :].astype(np.float32)
        per_item = []
        for it in np.unique(item):
            m = item == it
            pol = polarity[m]
            xpos = Xl[m][pol == 1]
            xneg = Xl[m][pol == -1]
            if len(xpos) and len(xneg):
                per_item.append(xpos.mean(0) - xneg.mean(0))
        dirs[l] = np.mean(per_item, axis=0)
    np.save(args.out, dirs)
    norms = np.linalg.norm(dirs, axis=1)
    meta = {"shape": [int(L), int(D)],
            "norms": [round(float(x), 3) for x in norms]}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=2)
    print(f"saved directions {dirs.shape} -> {args.out}")
    print("norms by layer:", meta["norms"])


if __name__ == "__main__":
    main()
