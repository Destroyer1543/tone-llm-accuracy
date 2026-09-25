import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+",
                    default=["mmlu", "arc", "gsm8k", "truthfulqa"])
    ap.add_argument("--n", type=int, default=200, help="items per dataset")
    ap.add_argument("--variants", type=int, default=1,
                    help="paraphrase variants per register to average over")
    ap.add_argument("--registers", nargs="+", default=None,
                    help="subset of registers to run (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help=">0 enables sampling (robustness check vs greedy)")
    ap.add_argument("--gen-seed", type=int, default=0)
    ap.add_argument("--min-free-mb", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()


    gpu.set_visible_gpu(min_free_mb=args.min_free_mb, wait=True)

    import registers as R
    import data as D
    from refusal import is_refusal
    from model_runner import ModelRunner

    items = D.load_many(args.datasets, n=args.n, seed=args.seed)
    print(f"[eval] loaded {len(items)} items from {args.datasets}", flush=True)


    regs_to_run = args.registers if args.registers else R.REGISTERS
    jobs = []
    for it in items:
        for reg in regs_to_run:
            nv = min(args.variants, R.n_variants(reg))
            for v in range(nv):
                prompt = R.render(reg, it["content"], variant=v)
                jobs.append((it, reg, v, prompt))
    print(f"[eval] total prompts: {len(jobs)}", flush=True)

    runner = ModelRunner(args.model)
    print(f"[eval] model loaded: {args.model}", flush=True)


    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    n_done = 0
    with open(args.out, "w") as f:

        for kind, mnt in [("mcq", 8), ("numeric", 320)]:
            sub = [j for j in jobs if j[0]["kind"] == kind]
            if not sub:
                continue
            texts = [j[3] for j in sub]
            if args.temperature > 0:
                import torch as _torch
                _torch.manual_seed(args.gen_seed)
            outs = runner.generate(texts, batch_size=args.batch_size,
                                    max_new_tokens=mnt,
                                    do_sample=args.temperature > 0,
                                    temperature=max(args.temperature, 1e-6))
            for (it, reg, v, prompt), out in zip(sub, outs):
                correct, pred = D.score_item(it, out)
                rec = {
                    "model": args.model,
                    "dataset": it["dataset"],
                    "item_id": it["id"],
                    "register": reg,
                    "variant": v,
                    "correct": correct,
                    "pred": pred,
                    "gold": it["gold"],
                    "refusal": int(is_refusal(out)),
                    "output": out[:500],
                }
                f.write(json.dumps(rec) + "\n")
                n_done += 1
            f.flush()
            print(f"[eval] {kind}: {len(sub)} done "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"[eval] DONE {n_done} records -> {args.out} "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
