import argparse, json, os, sys, time, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(__file__))
import registers as R
import data as D

KEY = os.environ["GEMINI_API_KEY"]
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=" + KEY
_THINK = {}


def _request(model, prompt, think, max_tokens):
    cfg = {"temperature": 0, "maxOutputTokens": max_tokens}
    if not think:
        cfg["thinkingConfig"] = {"thinkingBudget": 0}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": cfg}
    req = urllib.request.Request(URL.format(model=model), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    j = json.loads(urllib.request.urlopen(req, timeout=90).read())
    cand = (j.get("candidates") or [{}])[0]
    txt = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    return txt, cand.get("finishReason")


def call(model, prompt, retries=6):
    think = _THINK.get(model, False)
    for attempt in range(retries):
        try:
            txt, fr = _request(model, prompt, think, 768 if think else 16)
            if (not txt.strip()) and fr == "MAX_TOKENS" and not think:
                _THINK[model] = think = True; continue
            return txt
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace") if e.fp else ""
            if e.code in (429, 500, 503):
                time.sleep(min(60, 2 ** attempt * 2)); continue
            if e.code == 400 and ("hinking" in msg or "Budget" in msg) and not think:
                _THINK[model] = think = True; continue
            return None
        except Exception:
            time.sleep(3); continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=["mmlu", "arc", "truthfulqa"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--registers", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = [it for it in D.load_many(args.datasets, n=args.n, seed=args.seed)
             if it["kind"] == "mcq"]
    regs = args.registers if args.registers else R.REGISTERS
    jobs = [(it, reg, v) for it in items for reg in regs
            for v in range(min(args.variants, R.n_variants(reg)))]
    print(f"[gemini] {args.model}: {len(jobs)} requests", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = nnull = 0; t0 = time.time()
    with open(args.out, "w") as f:
        for it, reg, v in jobs:
            out = call(args.model, R.render(reg, it["content"], v))
            correct, pred = D.score_item(it, out) if out is not None else (0, None)
            nnull += out is None
            f.write(json.dumps({"model": args.model, "dataset": it["dataset"],
                "item_id": it["id"], "register": reg, "variant": v,
                "correct": correct, "pred": pred, "gold": it["gold"],
                "err": out is None, "output": (out or "")[:60]}) + "\n")
            done += 1
            if done % 50 == 0:
                f.flush()
                print(f"[gemini] {done}/{len(jobs)} ({done/(time.time()-t0):.1f} req/s, "
                      f"{nnull} null, think={_THINK.get(args.model, False)})", flush=True)
    print(f"[gemini] DONE {args.model} -> {args.out} "
          f"({time.time()-t0:.0f}s, {nnull} null)", flush=True)


if __name__ == "__main__":
    main()
