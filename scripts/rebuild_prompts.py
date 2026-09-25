import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import registers as R, data as D

def build(seed, n, datasets=("mmlu", "arc", "truthfulqa")):
    out = {}
    for it in D.load_many(list(datasets), n=n, seed=seed):
        for reg in R.REGISTERS:
            for v in range(R.n_variants(reg)):
                out[(it["id"], reg, v)] = {"item_id": it["id"], "dataset": it["dataset"], "register": reg,
                                           "variant": v, "gold": it["gold"], "user_message": R.render(reg, it["content"], v)}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out"); ap.add_argument("--verify")
    a = ap.parse_args()
    P = build(a.seed, a.n)
    if a.out:
        with open(a.out, "w") as f:
            for r in P.values(): f.write(json.dumps(r) + "\n")
        print("wrote", len(P), "prompts to", a.out)
    if a.verify:
        ok = bad = miss = 0
        for l in open(a.verify):
            r = json.loads(l)
            if r.get("dataset") not in ("mmlu", "arc", "truthfulqa"): continue
            k = (r["item_id"], r["register"], r.get("variant", 0))
            if k not in P: miss += 1
            elif P[k]["gold"] == r["gold"]: ok += 1
            else: bad += 1
        print(f"verify {a.verify}  matched {ok}  key mismatch {bad}  not rebuilt {miss}")
