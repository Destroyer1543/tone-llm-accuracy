import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import gpu


def letter_token_ids(tok, letters):
    ids = {}
    for L in letters:

        for form in (f" {L}", L, f"{L}."):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) >= 1:
                ids[L] = t[0]
                break
    return ids


def ece(confs, correct, n_bins=10):
    confs = np.asarray(confs); correct = np.asarray(correct)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (confs > bins[i]) & (confs <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(confs)) * abs(correct[m].mean() - confs[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=["mmlu", "arc", "truthfulqa"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--registers", nargs="+", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-free-mb", type=int, default=20000)

    ap.add_argument("--dirs", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--coeff", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gpu.set_visible_gpu(min_free_mb=args.min_free_mb, wait=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import registers as R
    import data as D

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()


    handle = None
    if args.dirs and args.layer is not None and args.coeff != 0.0:
        dirs = np.load(args.dirs)
        dvec = torch.tensor(dirs[args.layer], dtype=torch.bfloat16, device=model.device)

        def hook(m, i, o):
            if isinstance(o, tuple):
                return (o[0] + args.coeff * dvec,) + tuple(o[1:])
            return o + args.coeff * dvec
        handle = model.model.layers[args.layer - 1].register_forward_hook(hook)
        print(f"[cal] steering L{args.layer} coeff {args.coeff}", flush=True)

    items = [it for it in D.load_many(args.datasets, n=args.n, seed=args.seed)
             if it["kind"] == "mcq"]
    regs = args.registers if args.registers else R.REGISTERS

    rows = []
    for reg in regs:
        nv = min(args.variants, R.n_variants(reg))
        for v in range(nv):
            texts, metas = [], []
            for it in items:
                prompt = R.render(reg, it["content"], variant=v)
                msgs = [{"role": "user", "content": prompt}]
                texts.append(tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
                metas.append(it)
            for start in range(0, len(texts), args.batch_size):
                chunk = list(range(start, min(start + args.batch_size, len(texts))))
                enc = tok([texts[i] for i in chunk], return_tensors="pt",
                          padding=True, add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    logits = model(**enc).logits[:, -1, :].float()
                for k, i in enumerate(chunk):
                    it = metas[i]
                    lids = letter_token_ids(tok, it["choices"])
                    letters = list(lids.keys())
                    vals = torch.tensor([logits[k, lids[L]] for L in letters])
                    probs = torch.softmax(vals, dim=0).numpy()
                    j = int(probs.argmax())
                    pred = letters[j]
                    conf = float(probs[j])
                    rows.append({"model": args.model, "register": reg, "variant": v,
                                 "coeff": args.coeff, "dataset": it["dataset"],
                                 "item_id": it["id"], "pred": pred, "gold": it["gold"],
                                 "correct": int(pred == it["gold"]), "conf": conf})
            print(f"[cal] {reg} v{v} done ({len(rows)} rows)", flush=True)

    if handle:
        handle.remove()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== calibration by register ===")
    for reg in regs:
        s = df[df.register == reg]
        if len(s) == 0:
            continue
        print(f"  {reg:12s} acc={s.correct.mean():.4f} conf={s.conf.mean():.4f} "
              f"ECE={ece(s.conf, s.correct):.4f} overconf={s.conf.mean()-s.correct.mean():+.4f}")
    print(f"[cal] DONE -> {args.out}")


if __name__ == "__main__":
    main()
