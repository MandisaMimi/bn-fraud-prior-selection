"""
prior_by_imbalance.py
=====================
Which parameter-estimation method (prior) works best at which level of class
imbalance (true fraud rate)?

DESIGN
------
1. For every target fraud rate in FRAUD_RATES we build a ground-truth Bayesian
   network with that fraud rate (ground_truth.build_ground_truth(seed, rate)).
2. The SAME seeds (42..) and the SAME sample sizes are used for every fraud
   rate, so differences between rates are not caused by different seeds or
   different n.  (Seed s gives the same graph at every rate; only the isFraud
   table, and therefore the sampled data, changes with the rate.)
3. Estimators fitted on the TRUE structure (isolates parameter learning):
      MLE, Jeffreys (0.5), Laplace (1), and the fraud-informed prior at several
      strengths, in TWO versions:
        FPfix_s*  prior centred on a FIXED assumed rate (config.TARGET_FRAUD_RATE
                  = 0.5%). At rates other than 0.5% this is a WRONG guess.
        FPtrue_s* prior centred on the world's TRUE target rate. This is the
                  "you know the base rate" best case.
4. Exact evaluation by enumerating all 781,250 states (see learning_curves.py):
   KL(P_true || P_model), best-threshold F1, AUROC.
5. Results -> outputs/prior_by_imbalance_results.csv and three figures:
      pbi_kl_vs_n.png     KL vs n, one panel per fraud rate
      pbi_f1_vs_n.png     F1 vs n, one panel per fraud rate
      pbi_best_prior.png  best estimator for each (fraud rate, n)

Run:  python prior_by_imbalance.py [n_seeds] [--plots-only]
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config
import ground_truth
import data_generation
import learning_curves as lc

FRAUD_RATES = [0.001, 0.005, 0.02, 0.05, 0.10]
N_GRID = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
STRENGTHS = [2, 10, 50, 200]
N_SEEDS = 20
BASE_SEED = config.RANDOM_SEED_BASE
OUT_CSV = "outputs/prior_by_imbalance_results.csv"
FIG_DIR = "outputs/figures"


def estimators_for(rate):
    reg = {"MLE": lc.alpha_mle, "Jeffreys": lc.alpha_uniform(0.5), "Laplace": lc.alpha_uniform(1.0)}
    for s in STRENGTHS:
        reg[f"FPfix_s{s}"] = lc.alpha_fraud_prior(s, rate=config.TARGET_FRAUD_RATE)
    for s in STRENGTHS:
        reg[f"FPtrue_s{s}"] = lc.alpha_fraud_prior(s, rate=rate)
    return reg


def run_one(args):
    seed, rate = args
    gt = ground_truth.build_ground_truth(seed, target_fraud_rate=rate)
    ctx = lc.TruthContext(gt)
    world_rate = float(ctx.p_true[ctx.labels == 1].sum())      # realised marginal P(fraud) of this world
    rng = np.random.RandomState(seed + 10_000)
    X_all = data_generation.generate_dataset(gt, max(N_GRID), rng)[lc.NODES].to_numpy()
    pm = gt["parents"]
    rows = []
    for n in N_GRID:
        X = X_all[:n]
        n_fraud_in_sample = int(X[:, lc.FRAUD].sum())
        for name, afn in estimators_for(rate).items():
            kl, f1, auroc = ctx.score(lc.fit_tables(X, pm, afn), pm, ctx.cache)
            rows.append({"seed": seed, "target_rate": rate, "world_rate": world_rate, "n": n,
                         "n_fraud_in_sample": n_fraud_in_sample, "estimator": name,
                         "kl": kl, "f1": f1, "auroc": auroc,
                         "oracle_f1": ctx.oracle_f1, "oracle_auroc": ctx.oracle_auroc})
    return rows


def run_all(n_seeds=N_SEEDS):
    tasks = [(BASE_SEED + i, r) for r in FRAUD_RATES for i in range(n_seeds)]
    rows, t0 = [], time.time()
    with Pool(2) as pool:
        for k, res in enumerate(pool.imap_unordered(run_one, tasks), 1):
            rows.extend(res)
            if k % 5 == 0 or k == len(tasks):
                print(f"{k}/{len(tasks)} worlds done ({time.time() - t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
STYLE = {
    "MLE":       dict(color="#444444", ls="-",  marker="o", label="MLE (no prior)"),
    "Jeffreys":  dict(color="#2a6fb0", ls="-",  marker="s", label="Jeffreys (α=0.5)"),
    "Laplace":   dict(color="#7b4fb0", ls="-",  marker="^", label="Laplace (α=1)"),
    "FPfix_s50": dict(color="#e0701a", ls="--", marker="D", label="Fraud prior s=50, centred on 0.5% (fixed guess)"),
    "FPtrue_s50": dict(color="#1a9850", ls="-", marker="P", label="Fraud prior s=50, centred on the true rate"),
}


def ci_half(g):
    k = g.count()
    return g.std(ddof=1) / np.sqrt(k) * st.t.ppf(0.975, k - 1)


def curve_figure(df, value, ylabel, fname, logy):
    rates = sorted(df["target_rate"].unique())
    fig, axes = plt.subplots(1, len(rates), figsize=(4.0 * len(rates), 4.2), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), rates):
        d = df[df["target_rate"] == r]
        for name, s in STYLE.items():
            e = d[d["estimator"] == name].groupby("n")[value]
            m, h = e.mean(), ci_half(e)
            ax.plot(m.index, m.values, color=s["color"], ls=s["ls"], marker=s["marker"], ms=4, lw=1.6, label=s["label"])
            ax.fill_between(m.index, m - h, m + h, color=s["color"], alpha=0.10)
        if value == "f1":
            ax.plot(sorted(d["n"].unique()), d.groupby("n")["oracle_f1"].mean(), color="k", lw=1, ls=":", label="Best possible (true model)")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"True fraud rate = {r:.1%}")
        ax.set_xlabel("Training samples (n)")
        ax.grid(alpha=0.25, which="both")
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close(fig)


def family(name):
    return name.split("_")[0] if name.startswith("FP") else name


FAM_ORDER = ["MLE", "Jeffreys", "Laplace", "FPfix", "FPtrue"]
FAM_COLOR = ["#999999", "#2a6fb0", "#7b4fb0", "#e0701a", "#1a9850"]
FAM_LABEL = {"MLE": "MLE", "Jeffreys": "Jeffreys", "Laplace": "Laplace",
             "FPfix": "Fraud prior (fixed 0.5% guess)", "FPtrue": "Fraud prior (true rate)"}


def best_table(df, value, lower_is_better):
    """Winner per (rate, n) by mean of `value` across seeds + gain over Jeffreys."""
    m = df.groupby(["target_rate", "n", "estimator"])[value].mean().reset_index()
    out = []
    for (r, n), g in m.groupby(["target_rate", "n"]):
        g = g.set_index("estimator")[value]
        best = g.idxmin() if lower_is_better else g.idxmax()
        ref = g["Jeffreys"]
        gain = (ref - g[best]) / ref if lower_is_better else (g[best] - ref) / ref
        out.append({"target_rate": r, "n": n, "best": best, "family": family(best), "gain_vs_jeffreys": gain})
    return pd.DataFrame(out)


def best_prior_figure(df, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    rates = sorted(df["target_rate"].unique())
    ns = sorted(df["n"].unique())
    cmap = ListedColormap(FAM_COLOR)
    tabs = {}
    for ax, (value, lower, title) in zip(axes, [("kl", True, "Closest to the true distribution (lowest KL)"),
                                                ("f1", False, "Best fraud detection (highest F1)")]):
        t = best_table(df, value, lower)
        tabs[value] = t
        grid = np.full((len(ns), len(rates)), np.nan)
        for _, row in t.iterrows():
            grid[ns.index(row["n"]), rates.index(row["target_rate"])] = FAM_ORDER.index(row["family"])
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(FAM_ORDER) - 0.5, aspect="auto", origin="lower", alpha=0.75)
        for _, row in t.iterrows():
            i, j = ns.index(row["n"]), rates.index(row["target_rate"])
            lab = row["best"].replace("FPfix_", "fix ").replace("FPtrue_", "true ")
            ax.text(j, i, f"{lab}\n{row['gain_vs_jeffreys']:+.0%}", ha="center", va="center", fontsize=7.5)
        ax.set_xticks(range(len(rates)), [f"{r:.1%}" for r in rates])
        ax.set_yticks(range(len(ns)), [f"{n:,}" for n in ns])
        ax.set_xlabel("True fraud rate")
        ax.set_ylabel("Training samples (n)")
        ax.set_title(title, fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.75) for c in FAM_COLOR]
    fig.legend(handles, [FAM_LABEL[f] for f in FAM_ORDER], loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Best estimator in each condition (cell text: winning estimator, and its improvement over Jeffreys)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close(fig)
    for k, t in tabs.items():
        t.to_csv(f"outputs/prior_by_imbalance_best_{k}.csv", index=False)


def make_plots(df):
    os.makedirs(FIG_DIR, exist_ok=True)
    curve_figure(df, "kl", "KL(P_true ‖ P_model)  (lower = better)", "pbi_kl_vs_n.png", logy=True)
    curve_figure(df, "f1", "Best-threshold F1  (higher = better)", "pbi_f1_vs_n.png", logy=False)
    best_prior_figure(df, "pbi_best_prior.png")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_seeds = int(args[0]) if args else N_SEEDS
    if "--plots-only" in sys.argv:
        data = pd.read_csv(OUT_CSV)
    else:
        os.makedirs("outputs", exist_ok=True)
        data = run_all(n_seeds)
    make_plots(data)
    print("done")
