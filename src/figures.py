import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "data", "responses"))
OUT = os.environ.get("FIGURES_DIR", os.path.join(ROOT, "figures"))
os.makedirs(OUT, exist_ok=True)


BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
MCQ = ["mmlu", "arc", "truthfulqa"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def style(ax, xgrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.8, colors=INK2)


def load(path, mcq_only=True):
    df = pd.DataFrame(json.loads(l) for l in open(path))
    if mcq_only and "dataset" in df:
        df = df[df.dataset.isin(MCQ)]
    return df


def contrast(df, a="very_polite", b="very_rude", col="correct", nboot=5000, seed=0):
    piv = df.groupby(["item_id", "register"])[col].mean().unstack("register")
    if a not in piv or b not in piv:
        return None
    d = (piv[a] - piv[b]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    bt = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)])
    p = 2 * min((bt <= 0).mean(), (bt >= 0).mean())
    return d.mean(), np.percentile(bt, 2.5), np.percentile(bt, 97.5), p, len(d)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank, i in enumerate(reversed(o)):
        prev = min(prev, p[i] * n / (n - rank)); adj[i] = prev
    return adj


def save(fig, name):

    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.eps")
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .eps / .png")


MODELS = [
    ("phase0_qwen0.5b.jsonl", "Qwen2.5-0.5B"),
    ("phase0_qwen1.5b.jsonl", "Qwen2.5-1.5B"),
    ("phase0_qwen3b.jsonl", "Qwen2.5-3B"),
    ("phase0_qwen7b.jsonl", "Qwen2.5-7B"),
    ("phase0_gemma2-2b_full.jsonl", "Gemma-2-2B"),
    ("phase0_gemma2-9b_full.jsonl", "Gemma-2-9B"),
    ("phase0_llama31-8b_full.jsonl", "Llama-3.1-8B"),
    ("phase0_olmo2-7b.jsonl", "OLMo-2-7B"),
    ("phase0_mistral7b.jsonl", "Mistral-7B"),
    ("phase0_gptoss20b.jsonl", "gpt-oss-20B"),
    ("gem_flash35_seed0.jsonl", "Gemini-3.5-Flash"),
    ("gem_flash37_seed0.jsonl", "Gemini-3.7-Flash"),
]


def fig1():
    rows = []
    for f, lab in MODELS:
        p = f"{R}/{f}"
        if not (os.path.exists(p) and os.path.getsize(p)):
            continue
        df = load(p)
        c = contrast(df)
        if c is None:
            continue
        d, lo, hi, pv, n = c
        rows.append(dict(model=lab, acc=df.correct.mean(), d=d, lo=lo, hi=hi, p=pv))
    t = pd.DataFrame(rows)
    t["p_fdr"] = bh(t.p.values)
    t = t.sort_values("acc").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    style(ax)

    ax.axvspan(-1, 1, color="#f0efec", lw=0, zorder=0)
    ax.axvline(0, color=BASE, lw=1.0, zorder=1)
    y = np.arange(len(t))
    for i, r in t.iterrows():
        sig = r.p_fdr < 0.05
        col = ORANGE if sig else BLUE
        ax.plot([r.lo * 100, r.hi * 100], [i, i], color=col, lw=1.6, solid_capstyle="round", zorder=3)
        ax.plot(r.d * 100, i, marker="o", ms=4.5, color=col,
                mfc=col if sig else "white", mew=1.2, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(t.model)

    ax.set_ylim(-0.6, len(t) + 0.9)
    ax.set_xlabel("Very polite minus very rude, accuracy points")
    handles = [Line2D([], [], marker="o", ls="none", ms=4.5, mfc=ORANGE, color=ORANGE,
                      mew=1.2, label="significant after FDR"),
               Line2D([], [], marker="o", ls="none", ms=4.5, mfc="white", color=BLUE,
                      mew=1.2, label="not significant")]
    ax.legend(handles=handles, loc="upper left", frameon=False, handletextpad=0.4,
              ncol=2, columnspacing=1.2, borderaxespad=0.2)
    save(fig, "Fig1")
    return t


def fig2():
    spec = [("gemma2-2b", "Gemma-2-2B"), ("gemma2-9b", "Gemma-2-9B"),
            ("llama31-8b", "Llama-3.1-8B"), ("qwen3b", "Qwen2.5-3B")]
    spec = [s for s in spec if os.path.exists(f"{R}/phase0_{s[0]}_seed2.jsonl")
            and os.path.getsize(f"{R}/phase0_{s[0]}_seed2.jsonl")]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    style(ax, xgrid=False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.axhline(0, color=BASE, lw=1.0)
    cols = [BLUE, MUTED, ORANGE, AQUA]
    marks = ["o", "s", "^", "D"]
    for k, (m, lab) in enumerate(spec):
        xs, ys, sig = [], [], []
        for s, suf in enumerate(["", "_seed1", "_seed2"]):
            p = f"{R}/phase0_{m}{suf}.jsonl"
            if not (os.path.exists(p) and os.path.getsize(p)):
                continue
            c = contrast(load(p))
            if c is None:
                continue
            xs.append(s); ys.append(c[0] * 100); sig.append(c[3] < 0.05)

        off = (k - (len(spec) - 1) / 2) * 0.035
        xs = [x + off for x in xs]
        ax.plot(xs, ys, color=cols[k], lw=1.4, zorder=2, label=lab)
        for x, yv, sg in zip(xs, ys, sig):
            ax.plot(x, yv, marker=marks[k], ms=5, color=cols[k],
                    mfc=cols[k] if sg else "white", mew=1.2, zorder=3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Sample 1", "Sample 2", "Sample 3"])
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylabel("Accuracy difference, points")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.5)
    save(fig, "Fig3")


def fig3():
    spec = [("gemma2-2b", "Gemma-2-2B"), ("llama31-8b", "Llama-3.1-8B"),
            ("qwen3b", "Qwen2.5-3B"), ("qwen7b", "Qwen2.5-7B")]
    spec = [s for s in spec if os.path.exists(f"{R}/hp_{s[0]}_seed1.jsonl")
            and os.path.getsize(f"{R}/hp_{s[0]}_seed1.jsonl")]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    style(ax, xgrid=False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.axhline(0, color=BASE, lw=1.0)
    for i, (m, lab) in enumerate(spec):
        lowf = f"{R}/phase0_{m}.jsonl"

        low0 = load(lowf)
        low = contrast(low0[low0.variant == 0])[0] if os.path.exists(lowf) else np.nan


        per, bench = [], {}
        for s in (0, 1):
            dfs = load(f"{R}/hp_{m}_seed{s}.jsonl")
            bench.update(dict(zip(dfs.item_id, dfs.dataset)))
            pv = dfs.groupby(["item_id", "register"]).correct.mean().unstack("register")
            per.append(pv["very_polite"] - pv["very_rude"])
        pooled = pd.concat(per).groupby(level=0).mean()
        high = float(pooled.groupby(pooled.index.map(bench)).mean().mean())
        low, high = low * 100, high * 100
        ax.annotate("", xy=(i, high), xytext=(i, low),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2,
                                    shrinkA=2.5, shrinkB=2.5))
        ax.plot(i, low, marker="o", ms=4.6, color=ORANGE, mfc="white", mew=1.3, zorder=3)
        ax.plot(i, high, marker="o", ms=4.6, color=BLUE, mfc=BLUE, mew=1.1, zorder=3)
    ax.set_xticks(range(len(spec))); ax.set_xticklabels([s[1] for s in spec])
    ax.set_xlim(-0.5, len(spec) - 0.5)
    ax.set_ylabel("Accuracy difference, points")
    handles = [Line2D([], [], marker="o", ls="none", ms=4.6, mfc="white", color=ORANGE,
                      mew=1.3, label="600 items, one wording"),
               Line2D([], [], marker="o", ls="none", ms=5.5, mfc=BLUE, color=BLUE,
                      mew=1.2, label="6838 unique items, equal weights")]

    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, handletextpad=0.4, columnspacing=1.6)
    save(fig, "Fig4")


REG = ["very_polite", "polite", "neutral", "command", "rude", "very_rude", "profane"]
RLAB = ["very\npolite", "polite", "neutral", "command", "rude", "very\nrude", "profane"]


def fig4():
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    style(ax, xgrid=False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.axhline(0, color=BASE, lw=1.0, zorder=1)
    rng = np.random.default_rng(3)
    per_reg = {r: [] for r in REG}
    for f, lab in MODELS:
        p = f"{R}/{f}"
        if not (os.path.exists(p) and os.path.getsize(p)):
            continue
        m = load(p).groupby("register").correct.mean()
        if not all(r in m.index for r in REG):
            continue
        centre = np.mean([m[r] for r in REG])
        for r in REG:
            per_reg[r].append((m[r] - centre) * 100)
    for i, r in enumerate(REG):
        vals = per_reg[r]
        jit = rng.uniform(-0.13, 0.13, len(vals))
        ax.plot(i + jit, vals, ls="none", marker="o", ms=3.4, mfc="white",
                color=BLUE, mew=0.9, zorder=3)
        ax.plot([i - 0.26, i + 0.26], [np.mean(vals)] * 2, color=ORANGE, lw=2.0,
                solid_capstyle="round", zorder=4)
    ax.set_xticks(np.arange(len(REG))); ax.set_xticklabels(RLAB)
    ax.set_xlim(-0.5, len(REG) - 0.5)
    ax.set_ylabel("Accuracy minus model mean, points")
    handles = [Line2D([], [], marker="o", ls="none", ms=3.4, mfc="white", color=BLUE,
                      mew=0.9, label="one model"),
               Line2D([], [], color=ORANGE, lw=2.0, label="mean over models")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              ncol=2, frameon=False, handletextpad=0.5, columnspacing=1.6)
    save(fig, "Fig2")


def fig5():
    spec = [("gemma2-9b", "Gemma-2-9B", BLUE, "o"),
            ("gemma2-2b", "Gemma-2-2B", ORANGE, "s"),
            ("llama31-8b", "Llama-3.1-8B", AQUA, "^")]
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    style(ax, xgrid=False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    x = np.arange(len(REG))
    for m, lab, c, mk in spec:
        p = f"{R}/syc_{m}.jsonl"
        if not os.path.exists(p):
            continue
        df = pd.DataFrame(json.loads(l) for l in open(p))
        s = df.groupby("register").sycophantic.mean()
        if not all(r in s.index for r in REG):
            continue
        ax.plot(x, [100 * s[r] for r in REG], color=c, lw=1.3, marker=mk, ms=4,
                label=lab, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(RLAB)
    ax.set_xlim(-0.3, len(REG) - 0.7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Switching to the wrong answer, percent")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3,
              frameon=False, handletextpad=0.5, columnspacing=1.6)
    save(fig, "Fig5")


def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))

    ax = axes[0]; style(ax, xgrid=False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    j = json.load(open(f"{R}/probe_gemma2-2b.json"))
    res = j["results"]
    ax.plot([r["layer"] for r in res], [r["auc_polite_rude"] for r in res],
            color=BLUE, lw=1.6, zorder=3)
    ax.axhline(0.5, color=BASE, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlabel("Layer"); ax.set_ylabel("Probe AUC")
    ax.set_ylim(0.4, 1.04)
    ax.text(-0.22, 1.06, "A", transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK, va="bottom", ha="left")

    ax = axes[1]; style(ax, xgrid=False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    names = [("gemma2-2b", "register"), ("shuffle_2b", "shuffled"), ("topic_2b", "topic")]
    cols = [BLUE, MUTED, ORANGE]; marks = ["o", "s", "^"]
    for k, (tag, lab) in enumerate(names):
        xs, ys = [], []
        for c in (1, 2, 4):
            p = f"{R}/syc_p2_{tag}_L9_c{c}.jsonl"
            if not os.path.exists(p):
                continue
            d = pd.DataFrame(json.loads(l) for l in open(p))
            xs.append(c); ys.append(100 * d.sycophantic.mean())
        if xs:
            ax.plot(xs, ys, color=cols[k], lw=1.4, marker=marks[k], ms=4.5,
                    label=lab, zorder=3)
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel("Steering coefficient")
    ax.set_ylabel("Switching, percent")
    ax.legend(loc="lower left", frameon=False, handletextpad=0.5)
    ax.text(-0.24, 1.06, "B", transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK, va="bottom", ha="left")
    fig.subplots_adjust(wspace=0.40)
    save(fig, "Fig6")


if __name__ == "__main__":
    print("generating figures...")
    t = fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    t.to_csv(f"{OUT}/fig1_table.csv", index=False)
    print("\nFig1 underlying numbers:")
    print(t.to_string(index=False,
                      formatters={"acc": "{:.3f}".format, "d": "{:+.4f}".format,
                                  "lo": "{:+.4f}".format, "hi": "{:+.4f}".format,
                                  "p": "{:.3g}".format, "p_fdr": "{:.3g}".format}))
