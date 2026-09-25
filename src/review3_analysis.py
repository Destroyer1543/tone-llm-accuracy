import collections, json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "data", "responses"))
MCQ = ("mmlu", "arc", "truthfulqa")
NB = 10000


def rows(f, variant=None):
    out = []
    for l in open(f"{R}/{f}"):
        r = json.loads(l)
        if r.get("dataset") in MCQ and (variant is None or r.get("variant") == variant):
            out.append(r)
    return out


def item_diffs(rs):
    acc = collections.defaultdict(list); ds = {}
    for r in rs:
        acc[(r["item_id"], r["register"])].append(float(r["correct"])); ds[r["item_id"]] = r["dataset"]
    return {i: (ds[i], np.mean(acc[(i, "very_polite")]) - np.mean(acc[(i, "very_rude")]))
            for i in ds if (i, "very_polite") in acc and (i, "very_rude") in acc}


def pool(*dicts):
    p = collections.defaultdict(list); ds = {}
    for d in dicts:
        for i, (b, v) in d.items():
            p[i].append(v); ds[i] = b
    return {i: (ds[i], float(np.mean(v))) for i, v in p.items()}


def estimate(d, weights="natural", level=95, seed=0):
    by = {b: np.array([v for (bb, v) in d.values() if bb == b]) for b in MCQ}
    by = {b: x for b, x in by.items() if len(x)}
    rng = np.random.default_rng(seed)
    if weights == "natural":
        allv = np.concatenate(list(by.values()))
        point = allv.mean()

        bt = np.zeros(NB); n = len(allv)
        for b, x in by.items():
            bt += x[rng.integers(0, len(x), (NB, len(x)))].sum(1)
        bt /= n
    else:
        point = np.mean([x.mean() for x in by.values()])
        bt = np.mean([x[rng.integers(0, len(x), (NB, len(x)))].mean(1) for x in by.values()], axis=0)
    a = (100 - level) / 2
    p = min(1.0, 2 * min((bt <= 0).mean(), (bt >= 0).mean()))
    return dict(d=100 * point, lo=100 * np.percentile(bt, a), hi=100 * np.percentile(bt, 100 - a), p=p,
                n=sum(len(x) for x in by.values()),
                per_bench={b: 100 * x.mean() for b, x in by.items()},
                tqa_share=len(by.get("truthfulqa", [])) / sum(len(x) for x in by.values()))


def main():
    models = sys.argv[1:] or ["gemma2-2b", "llama31-8b", "qwen7b"]
    out = {}
    for m in models:
        if not os.path.exists(f"{R}/hp_{m}_seed1.jsonl"):
            continue
        s600 = {v: item_diffs(rows(f"phase0_{m}.jsonl", v)) for v in (None, 0)}
        A, B = item_diffs(rows(f"hp_{m}_seed0.jsonl")), item_diffs(rows(f"hp_{m}_seed1.jsonl"))
        P = pool(A, B)
        res = {}
        for tag, d in [("600 all variants", s600[None]), ("600 variant 0", s600[0]),
                       ("large A", A), ("large B", B), ("pooled unique", P)]:
            res[tag] = {w: estimate(d, w) for w in ("natural", "equal")}
        res["pooled unique 90"] = {w: estimate(P, w, level=90) for w in ("natural", "equal")}
        out[m] = res
    json.dump(out, open(f"{R}/review3_benchmix_variant.json", "w"), indent=1)

    for m, res in out.items():
        print(f"\n== {m}")
        print(f"   {'sample':18s} {'TQA share':>9s} | {'natural weights':>30s} | {'equal benchmark weights':>30s} | per benchmark MMLU / ARC / TQA")
        for tag in ["600 all variants", "600 variant 0", "large A", "large B", "pooled unique"]:
            n, e = res[tag]["natural"], res[tag]["equal"]
            pb = " / ".join(f"{e['per_bench'][b]:+.2f}" for b in MCQ)
            print(f"   {tag:18s} {n['tqa_share']:9.3f} | {n['d']:+.2f} [{n['lo']:+.2f},{n['hi']:+.2f}] p {n['p']:.3f} | "
                  f"{e['d']:+.2f} [{e['lo']:+.2f},{e['hi']:+.2f}] p {e['p']:.3f} | {pb}")
        for w in ("natural", "equal"):
            q = res["pooled unique 90"][w]
            eq = [t for t in (0.5, 1, 2) if -t < q["lo"] and q["hi"] < t]
            print(f"   pooled 90% CI {w:8s} [{q['lo']:+.2f},{q['hi']:+.2f}]  equivalent at {eq}")


if __name__ == "__main__":
    main()


NAMES = {"gemma2-2b": "Gemma 2 2B", "llama31-8b": "Llama 3.1 8B", "qwen7b": "Qwen2.5 7B", "qwen3b": "Qwen2.5 3B"}


def f(x):
    return f"{x:+.2f}".replace("-", "$-$")


def iv(e):
    return f"{f(e['lo'])} to {f(e['hi'])}"


def subset(d, benches):
    return {i: v for i, v in d.items() if v[0] in benches}


def write_tables(models, outdir=os.path.join(ROOT, "supp")):
    os.makedirs(outdir, exist_ok=True)
    s10, s11 = [], []
    for m in models:
        if not os.path.exists(f"{R}/hp_{m}_seed1.jsonl"):
            continue
        A, B = item_diffs(rows(f"hp_{m}_seed0.jsonl")), item_diffs(rows(f"hp_{m}_seed1.jsonl"))
        samples = [("600, 3 variants", item_diffs(rows(f"phase0_{m}.jsonl"))),
                   ("600, variant 0", item_diffs(rows(f"phase0_{m}.jsonl", 0)))]
        for s in (1, 2):
            p = f"{R}/phase0_{m}_seed{s}.jsonl"
            if os.path.exists(p) and os.path.getsize(p):
                samples.append((f"600 draw {s + 1}", item_diffs(rows(f"phase0_{m}_seed{s}.jsonl"))))
        samples += [("Large A", A), ("Large B", B), ("Pooled unique", pool(A, B))]
        for k, (tag, d) in enumerate(samples):
            nat, eq = estimate(d, "natural"), estimate(d, "equal")
            pb = [estimate(subset(d, [b]), "natural") for b in MCQ]
            s10.append((NAMES[m] if k == 0 else "", tag, nat["n"], f"{100 * nat['tqa_share']:.1f}",
                        f(nat["d"]), f(eq["d"]), *[f(x["d"]) for x in pb]))
        P = pool(A, B)

        for k, (tag, d, w) in enumerate([("Equal weights", P, "equal"),
                                         ("MMLU and ARC", subset(P, ["mmlu", "arc"]), "natural"),
                                         ("TruthfulQA", subset(P, ["truthfulqa"]), "natural")]):
            e = estimate(d, w, level=90)
            yes = ["yes" if -t < e["lo"] and e["hi"] < t else "no" for t in (0.5, 1, 2)]
            s11.append((NAMES[m] if k == 0 else "", tag, e["n"], f(e["d"]), iv(e), *yes))

    def table(rows_, label, cap, cols, head):
        body = "\n".join(" & ".join(str(c) for c in r) + " \\\\" for r in rows_)
        return (f"\\begin{{table}}[h]\n\\caption{{{cap}}}\\label{{{label}}}\n\\resizebox{{\\textwidth}}{{!}}{{%\n"
                f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}\n\\botrule\n"
                f"\\end{{tabular}}}}\n\\end{{table}}\n")
    open(f"{outdir}/tabS10.tex", "w").write(table(
        s10, "tabS10", "Very polite minus very rude by sample, benchmark weighting and benchmark, in accuracy points.",
        "llrrrrrrrr", "Model & Sample & Items & TQA \\% & Natural & Equal & MMLU & ARC & TQA"))
    open(f"{outdir}/tabS11.tex", "w").write(table(
        s11, "tabS11", "Equivalence of the pooled estimates under alternative benchmark weightings, 90 percent intervals.",
        "llrrlccc", "Model & Weighting & Items & Difference & 90\\% interval & 0.5 & 1 & 2"))
    print("wrote tabS10.tex and tabS11.tex with", len(s10), "and", len(s11), "rows")


if __name__ == "__main__" and os.environ.get("WRITE_TABLES"):
    write_tables(["gemma2-2b", "llama31-8b", "qwen7b", "qwen3b"])
