import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from sae import JumpReLUSAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True, help="acts npz with X,polarity,item,variant")
    ap.add_argument("--layer", type=int, required=True, help="hidden_states index (=SAE layer+1)")
    ap.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--sae-file", default="layer_9/width_16k/average_l0_73/params.npz")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="results/sae_register_feats.npz")
    args = ap.parse_args()

    d = np.load(args.acts, allow_pickle=True)
    X = d["X"][:, args.layer, :].astype(np.float32)
    pol = d["polarity"]; item = d["item"]; var = d["variant"]
    print(f"acts {X.shape}, loading SAE {args.sae_file}", flush=True)
    sae = JumpReLUSAE(args.sae_repo, args.sae_file)
    print(f"SAE d_model={sae.d_model} d_sae={sae.d_sae}", flush=True)


    xhat = sae.decode(sae.encode(X))
    cos = (X * xhat).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(xhat, axis=1) + 1e-8)
    print(f"recon cosine at hidden_states[{args.layer}] = {cos.mean():.3f} "
          f"(want >~0.9; if low, try a neighboring layer index)", flush=True)

    acts = sae.encode(X)
    m = pol != 0
    y = (pol[m] == 1).astype(int)
    A = acts[m]
    frac_active = (A > 0).mean(0)


    aucs = np.full(sae.d_sae, 0.5)
    cand = np.where(frac_active > 0.02)[0]
    for j in cand:
        col = A[:, j]
        if col.std() > 0:
            aucs[j] = roc_auc_score(y, col)

    strength = np.abs(aucs - 0.5)
    order = np.argsort(strength)[::-1][:args.topk]

    print(f"\nTop {args.topk} register-separating latents (AUC vs polite/rude):")
    print(" feat_id   AUC    frac_active   mean_polite   mean_rude")
    for j in order:
        mp = A[y == 1, j].mean(); mr = A[y == 0, j].mean()
        print(f"  {j:6d}  {aucs[j]:.3f}    {frac_active[j]:.3f}       {mp:.3f}      {mr:.3f}")


    best = order[0]
    tr = var[m] <= 1; te = var[m] == 2
    if te.sum() and len(np.unique(y[te])) > 1:
        ho = roc_auc_score(y[te], A[te, best])
        print(f"\nBest feat {best}: held-out-phrasing (train v0,v1 -> test v2) AUC = {ho:.3f}")

    np.savez(args.out, top_feats=order, aucs=aucs[order],
             all_aucs=aucs, frac_active=frac_active)
    print(f"saved -> {args.out}")
    print("\nVerdict: a CLEAN register feature exists iff top AUC>~0.9, sparse "
          "(low frac_active), and held-out-phrasing AUC stays high.")


if __name__ == "__main__":
    main()
