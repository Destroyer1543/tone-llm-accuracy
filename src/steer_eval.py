import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import gpu


def get_decoder_layers(model):

    return model.model.layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dirs", required=True, help=".npy [L,D] from make_directions")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--coeffs", type=float, nargs="+",
                    default=[-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4])
    ap.add_argument("--datasets", nargs="+", default=["mmlu", "arc", "truthfulqa"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1, help="≠ extraction seed(0)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-free-mb", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gpu.set_visible_gpu(min_free_mb=args.min_free_mb, wait=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import registers as R
    import data as D
    from refusal import is_refusal

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    dirs = np.load(args.dirs)
    dvec = torch.tensor(dirs[args.layer], dtype=torch.bfloat16, device=model.device)
    print(f"[steer] layer {args.layer} dir norm {float(dvec.norm()):.3f}", flush=True)


    state = {"coeff": 0.0}

    def hook(module, inp, out):
        if state["coeff"] == 0.0:
            return out
        if isinstance(out, tuple):
            h = out[0]
            h = h + state["coeff"] * dvec
            return (h,) + tuple(out[1:])
        return out + state["coeff"] * dvec


    assert args.layer >= 1, "layer 0 is embeddings; use >=1"
    layer_mod = get_decoder_layers(model)[args.layer - 1]
    handle = layer_mod.register_forward_hook(hook)


    items = D.load_many(args.datasets, n=args.n, seed=args.seed)
    texts, metas = [], []
    for it in items:
        prompt = R.render("neutral", it["content"], variant=0)
        msgs = [{"role": "user", "content": prompt}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True))
        metas.append(it)
    print(f"[steer] {len(texts)} neutral prompts; coeffs={args.coeffs}", flush=True)

    def run(mnt, idxs):
        outs = []
        for start in range(0, len(idxs), args.batch_size):
            chunk = idxs[start:start + args.batch_size]
            enc = tok([texts[i] for i in chunk], return_tensors="pt",
                      padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(tok.batch_decode(new, skip_special_tokens=True))
        return outs

    mcq_idx = [i for i, m in enumerate(metas) if m["kind"] == "mcq"]
    num_idx = [i for i, m in enumerate(metas) if m["kind"] == "numeric"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary = []
    with open(args.out, "w") as f:
        for coeff in args.coeffs:
            state["coeff"] = coeff
            per_ds = {}
            for idxs, mnt in [(mcq_idx, 8), (num_idx, 320)]:
                if not idxs:
                    continue
                outs = run(mnt, idxs)
                for j, i in enumerate(idxs):
                    it = metas[i]
                    correct, pred = D.score_item(it, outs[j])
                    rec = {"model": args.model, "layer": args.layer,
                           "coeff": coeff, "dataset": it["dataset"],
                           "item_id": it["id"], "correct": correct,
                           "refusal": int(is_refusal(outs[j])), "pred": pred,
                           "output": outs[j][:300]}
                    f.write(json.dumps(rec) + "\n")
                    per_ds.setdefault(it["dataset"], []).append(correct)
            f.flush()
            accs = {k: float(np.mean(v)) for k, v in per_ds.items()}
            overall = float(np.mean([c for v in per_ds.values() for c in v]))
            summary.append((coeff, overall, accs))
            print(f"[steer] coeff={coeff:+.2f}  acc={overall:.4f}  {accs}", flush=True)

    handle.remove()
    print("\n=== STEERING SUMMARY (neutral prompts, tokens fixed) ===")
    for coeff, overall, accs in summary:
        print(f"  coeff={coeff:+.2f}  overall_acc={overall:.4f}")
    print(f"[steer] DONE -> {args.out}")


if __name__ == "__main__":
    main()
