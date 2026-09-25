import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--datasets", nargs="+", default=["mmlu", "arc", "truthfulqa"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-free-mb", type=int, default=18000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gpu.set_visible_gpu(min_free_mb=args.min_free_mb, wait=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import registers as R, data as D

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                 device_map={"": 0}).eval()
    print(f"[gptoss] loaded {args.model}", flush=True)

    items = [it for it in D.load_many(args.datasets, n=args.n, seed=args.seed)
             if it["kind"] == "mcq"]
    jobs = []
    for it in items:
        for reg in R.REGISTERS:
            nv = min(args.variants, R.n_variants(reg))
            for v in range(nv):
                msg = [{"role": "user", "content": R.render(reg, it["content"], v)}]
                try:
                    text = tok.apply_chat_template(msg, tokenize=False,
                                                   add_generation_prompt=True,
                                                   reasoning_effort=args.reasoning)
                except TypeError:
                    text = tok.apply_chat_template(msg, tokenize=False,
                                                   add_generation_prompt=True)
                jobs.append((it, reg, v, text))
    print(f"[gptoss] {len(jobs)} prompts", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import time
    t0 = time.time()
    with open(args.out, "w") as f:
        for s in range(0, len(jobs), args.batch_size):
            batch = jobs[s:s + args.batch_size]
            enc = tok([b[3] for b in batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            for (it, reg, v, _), out in zip(batch, outs):
                correct, pred = D.score_mcq(out, it["gold"], it["choices"])
                f.write(json.dumps({"model": args.model, "dataset": it["dataset"],
                    "item_id": it["id"], "register": reg, "variant": v,
                    "correct": correct, "pred": pred, "gold": it["gold"],
                    "output": out[-120:]}) + "\n")
            f.flush()
            if (s // args.batch_size) % 20 == 0:
                print(f"[gptoss] {s+len(batch)}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[gptoss] DONE -> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
