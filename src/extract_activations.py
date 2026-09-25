import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import gpu


POLARITY = {
    "very_polite": 1, "polite": 1, "neutral": 0,
    "rude": -1, "very_rude": -1, "command": 0, "profane": -1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=["mmlu", "arc", "truthfulqa"])
    ap.add_argument("--n", type=int, default=120, help="items per dataset")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--registers", nargs="+", default=None,
                    help="subset of registers (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-free-mb", type=int, default=24000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gpu.set_visible_gpu(min_free_mb=args.min_free_mb, wait=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import registers as R
    import data as D

    items = D.load_many(args.datasets, n=args.n, seed=args.seed)
    print(f"[extract] {len(items)} items from {args.datasets}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()


    jobs = []
    regs_to_run = args.registers if args.registers else R.REGISTERS
    for ii, it in enumerate(items):
        for reg in regs_to_run:
            nv = min(args.variants, R.n_variants(reg))
            for v in range(nv):
                content = it["content"]
                prompt = R.render(reg, content, variant=v)
                msgs = [{"role": "user", "content": prompt}]
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                jobs.append((ii, reg, v, it["dataset"], text))
    print(f"[extract] {len(jobs)} forward passes", flush=True)

    X, regs, pol, items_idx, dsets, vars, plens = [], [], [], [], [], [], []
    bs = args.batch_size
    with torch.no_grad():
        for start in range(0, len(jobs), bs):
            batch = jobs[start:start + bs]
            texts = [b[4] for b in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            out = model(**enc, output_hidden_states=True)


            hs = torch.stack(out.hidden_states, dim=1)
            last = hs[:, :, -1, :].to(torch.float16).cpu().numpy()
            X.append(last)
            for bi_, (ii, reg, v, ds, _t) in enumerate(batch):
                regs.append(R.REGISTERS.index(reg))
                pol.append(POLARITY[reg])
                items_idx.append(ii)
                dsets.append(ds)
                vars.append(v)

                plens.append(int(enc["attention_mask"][bi_].sum().item()))
            if (start // bs) % 10 == 0:
                print(f"[extract] {start+len(batch)}/{len(jobs)}", flush=True)

    X = np.concatenate(X, axis=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        register=np.array(regs, dtype=np.int32),
        polarity=np.array(pol, dtype=np.int32),
        item=np.array(items_idx, dtype=np.int32),
        dataset=np.array(dsets),
        variant=np.array(vars, dtype=np.int32),
        plen=np.array(plens, dtype=np.int32),
    )
    meta = {
        "model": args.model, "n_prompts": int(X.shape[0]),
        "n_layers": int(X.shape[1]), "d_model": int(X.shape[2]),
        "registers": R.REGISTERS, "datasets": args.datasets,
        "n_items_per_ds": args.n, "variants": args.variants,
    }
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=2)
    print(f"[extract] saved {X.shape} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
