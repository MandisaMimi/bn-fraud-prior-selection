"""
plot_learning_curves.py
=======================
Draws the learning-curve figures from outputs/learning_curves_results.csv.

Every curve is the MEAN over independent ground truths (seeds); the shaded band
is the 95% confidence interval across seeds (t-interval).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "outputs/learning_curves_results.csv"
FIG_DIR = "outputs/figures"

STYLE = {
    "MLE":      dict(color="#444444", ls="-",  marker="o", label="MLE (no prior)"),
    "Jeffreys": dict(color="#2a6fb0", ls="-",  marker="s", label="Jeffreys (α = 0.5)"),
    "Laplace":  dict(color="#7b4fb0", ls="-",  marker="^", label="Laplace (α = 1)"),
    "FP_s1":    dict(color="#f0a030", ls="--", marker="v", label="Fraud prior, strength 1"),
    "FP_s10":   dict(color="#e0701a", ls="--", marker="D", label="Fraud prior, strength 10"),
    "FP_s50":   dict(color="#b0401f", ls="-",  marker="P", label="Fraud prior, strength 50"),
    "FP_s200":  dict(color="#7a1f10", ls="-",  marker="X", label="Fraud prior, strength 200"),
}
MAIN = ["MLE", "Jeffreys", "Laplace", "FP_s1", "FP_s10", "FP_s50", "FP_s200"]


from scipy import stats as _st


def mean_se(df, value):
    """Mean and HALF-WIDTH of the 95% t-confidence interval across ground truths (column kept as 'se')."""
    g = df.groupby(["n"])[value]
    k = g.count()
    half = g.std(ddof=1) / np.sqrt(k) * _st.t.ppf(0.975, k - 1)
    return pd.DataFrame({"mean": g.mean(), "se": half}).reset_index()


def draw(ax, df, structure, estimators, value, logy=False):
    d = df[df["structure"] == structure]
    for name in estimators:
        s = d[d["estimator"] == name]
        if s.empty:
            continue
        m = mean_se(s, value)
        st = STYLE[name]
        ax.plot(m["n"], m["mean"], color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=5, lw=1.8, label=st["label"])
        ax.fill_between(m["n"], m["mean"] - m["se"], m["mean"] + m["se"], color=st["color"], alpha=0.12)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Number of training samples (n)")
    ax.grid(alpha=0.25, which="both")


def fig_kl_vs_n(df):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    draw(ax, df, "true", MAIN, "kl", logy=True)
    ax.set_ylabel("KL( P_true || P_estimated )   [lower = closer to the truth]")
    ax.set_title("Estimating the distribution: KL divergence vs number of samples\n"
                 "(true structure, parameters estimated from data)", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/lc_kl_vs_n.png", dpi=160)
    plt.close(fig)


def fig_detection_vs_n(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, value, label in [(axes[0], "f1", "Best-threshold F1   [higher = better]"),
                             (axes[1], "auroc", "AUROC   [higher = better]")]:
        draw(ax, df, "true", MAIN, value)
        ora = df[df["structure"] == "true"].groupby("seed")[f"oracle_{value}"].first().mean()
        ax.axhline(ora, color="black", ls=":", lw=1.5, label="Best possible (true model)")
        ax.set_ylabel(label)
    axes[0].set_title("Fraud detection: F1 vs number of samples", fontsize=12)
    axes[1].set_title("Fraud detection: AUROC vs number of samples", fontsize=12)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("True structure; exact population metrics, mean over ground truths", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/lc_detection_vs_n.png", dpi=160)
    plt.close(fig)


def fig_strength(df):
    d = df[(df["structure"] == "true") & df["estimator"].str.startswith("FP_s")].copy()
    d["s"] = d["estimator"].str.replace("FP_s", "").astype(int)
    mle = df[(df["structure"] == "true") & (df["estimator"] == "MLE")][["seed", "n", "kl"]].rename(columns={"kl": "kl_mle"})
    d = d.merge(mle, on=["seed", "n"])
    d["kl_ratio"] = d["kl"] / d["kl_mle"]
    ns = [200, 1000, 5000, 20000, 50000]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(ns)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for c, n in zip(cmap, ns):
        s_ = d[d["n"] == n].groupby("s")
        r = s_["kl_ratio"].mean()
        se = s_["kl_ratio"].std(ddof=1) / np.sqrt(s_["kl_ratio"].count()) * _st.t.ppf(0.975, s_["kl_ratio"].count() - 1)
        axes[0].plot(r.index, r.values, marker="o", color=c, label=f"n = {n:,}")
        axes[0].fill_between(r.index, r.values - se.values, r.values + se.values, color=c, alpha=0.12)
        f = s_["f1"].mean()
        axes[1].plot(f.index, f.values, marker="o", color=c, label=f"n = {n:,}")
    axes[0].axhline(1.0, color="black", ls=":", lw=1.5)
    axes[0].annotate("same as MLE", xy=(0.02, 0.965), xycoords="axes fraction", fontsize=9)
    axes[0].set_ylabel("KL of fraud prior ÷ KL of MLE   [< 1 means better than MLE]")
    axes[0].set_title("Prior strength vs KL divergence", fontsize=12)
    axes[1].set_ylabel("Best-threshold F1   [higher = better]")
    axes[1].set_title("Prior strength vs F1", fontsize=12)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Strength s of the fraud-informed prior (pseudo-samples)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=9)
    fig.suptitle("Changing the strength of the prior instead of the number of samples (true structure)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/lc_prior_strength.png", dpi=160)
    plt.close(fig)


def median_iqr(df, value):
    g = df.groupby("n")[value]
    return pd.DataFrame({"med": g.median(), "lo": g.quantile(0.25), "hi": g.quantile(0.75)}).reset_index()


def fig_rq1_surrogate(df):
    """Median and inter-quartile range are used (not mean +/- SE) because a few ground truths
    end in a poor local optimum of greedy Hill Climbing, which would dominate a mean."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.3))
    rows = [("true", "Laplace", "#2e7d4f", "True structure (best case)"),
            ("learned", "Laplace", "#b0401f", "Learned by Hill Climbing + BIC"),
            ("empty", "Laplace", "#888888", "No edges (all variables independent)")]
    for structure, est, color, label in rows:
        s = df[(df["structure"] == structure) & (df["estimator"] == est)]
        for ax, value in [(axes[0], "kl"), (axes[1], "auroc")]:
            m = median_iqr(s, value)
            ax.plot(m["n"], m["med"], color=color, marker="o", ms=5, lw=1.8, label=label)
            ax.fill_between(m["n"], m["lo"], m["hi"], color=color, alpha=0.12)
    l = df[(df["structure"] == "learned") & (df["estimator"] == "Laplace")]
    frac = l.groupby("n")["has_fraud_edge"].mean()
    axes[2].plot(frac.index, frac.values, color="#b0401f", marker="o", lw=1.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("KL( P_true || P_model )   [lower = better surrogate]")
    axes[0].set_title("How good a surrogate for the distribution?", fontsize=12)
    axes[1].set_ylabel("AUROC   [higher = better]")
    axes[1].set_title("Fraud detection with each structure", fontsize=12)
    axes[2].set_ylabel("Share of ground truths where an edge into isFraud was learned")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title("Does Hill Climbing connect isFraud to anything?", fontsize=12)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Number of training samples (n)")
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=9)
    fig.suptitle("RQ1 reframed: the learned network as a surrogate for the true distribution "
                 "(Laplace parameters; line = median over 20 ground truths, band = inter-quartile range)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/lc_rq1_surrogate.png", dpi=160)
    plt.close(fig)


def fig_prior_centre():
    path = "outputs/learning_curves_prior_centre.csv"
    if not os.path.exists(path):
        return
    c = pd.read_csv(path)
    ref = pd.read_csv(CSV)
    ref = ref[(ref["structure"] == "true")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.3))
    cols = plt.cm.plasma(np.linspace(0.05, 0.85, c["believed_rate"].nunique()))
    for color, (r, d) in zip(cols, c.groupby("believed_rate")):
        lab = f"believed rate {r:g}" + ("  (true rate)" if abs(r - 0.005) < 1e-9 else "")
        for ax, v in zip(axes, ["kl", "f1", "auroc"]):
            m = d.groupby("n")[v].mean()
            ax.plot(m.index, m.values, color=color, marker="o", ms=4, lw=2.2 if abs(r - 0.005) < 1e-9 else 1.4, label=lab)
    for ax, v in zip(axes, ["kl", "f1", "auroc"]):
        for est, col in [("MLE", "#444444"), ("Jeffreys", "#2a6fb0")]:
            m = ref[ref["estimator"] == est].groupby("n")[v].mean()
            ax.plot(m.index, m.values, color=col, ls=":", lw=1.8, label=est)
        ax.set_xscale("log")
        ax.set_xlabel("Number of training samples (n)")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("KL( P_true || P_estimated )  [lower = better]")
    axes[1].set_ylabel("Best-threshold F1  [higher = better]")
    axes[2].set_ylabel("AUROC  [higher = better]")
    axes[0].legend(fontsize=8.5)
    fig.suptitle("What if the assumed base rate is wrong? Fraud prior of strength 50 centred on different believed fraud rates (true rate = 0.005)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/lc_prior_centre.png", dpi=160)
    plt.close(fig)


def fig_learned_estimators(df):
    est = ["MLE", "Jeffreys", "Laplace", "FP_s50"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    draw(axes[0], df, "learned", est, "kl", logy=True)
    draw(axes[1], df, "learned", est, "f1")
    axes[0].set_ylabel("KL( P_true || P_estimated )")
    axes[1].set_ylabel("Best-threshold F1")
    axes[0].set_title("KL vs samples (structure learned by Hill Climbing)", fontsize=12)
    axes[1].set_title("F1 vs samples (structure learned by Hill Climbing)", fontsize=12)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/lc_learned_structure_estimators.png", dpi=160)
    plt.close(fig)


def fig_strength_by_belief():
    """Is a stronger prior always better?  strength x believed base rate (true rate 0.005)."""
    path = "outputs/learning_curves_strength_centre_grid.csv"
    if not os.path.exists(path):
        return
    g = pd.read_csv(path)
    ref = pd.read_csv(CSV)
    ref = ref[ref["structure"] == "true"]
    ns = [500, 5000, 50000]
    rates = sorted(g["believed_rate"].unique())
    cols = plt.cm.plasma(np.linspace(0.05, 0.85, len(rates)))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for j, n in enumerate(ns):
        for i, (v, lab) in enumerate([("auroc", "AUROC  [higher = better]"), ("kl", "KL from the truth  [lower = better]")]):
            ax = axes[i, j]
            for c, r in zip(cols, rates):
                d = g[(g["n"] == n) & (g["believed_rate"] == r)].groupby("strength")[v].mean()
                tag = "  (true rate)" if abs(r - 0.005) < 1e-9 else ""
                ax.plot(d.index, d.values, marker="o", ms=4, color=c, lw=2.4 if tag else 1.4, label=f"believed rate {r:g}{tag}")
            for est, col, ls in [("MLE", "#444444", ":"), ("Jeffreys", "#2a6fb0", "--")]:
                ax.axhline(ref[(ref["n"] == n) & (ref["estimator"] == est)][v].mean(), color=col, ls=ls, lw=1.6, label=est)
            ax.set_xscale("log")
            if v == "kl":
                ax.set_yscale("log")
            ax.set_xlabel("Prior strength s (imaginary samples)")
            ax.set_ylabel(lab)
            ax.set_title(f"n = {n:,} training samples", fontsize=11)
            ax.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Is a stronger prior always better? Prior strength vs how wrong the believed base rate is "
                 "(mean over 20 ground truths; lines below the MLE line = worse than using no prior)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/lc_strength_by_belief.png", dpi=160)
    plt.close(fig)


def summary_tables(df):
    t = df[df["structure"] == "true"]
    for value in ["kl", "f1", "auroc"]:
        piv = t.pivot_table(index="n", columns="estimator", values=value, aggfunc="mean")
        cols = [c for c in ["MLE", "Jeffreys", "Laplace", "FP_s1", "FP_s10", "FP_s50", "FP_s200"] if c in piv.columns]
        piv[cols].to_csv(f"outputs/learning_curves_summary_{value}_true_structure.csv")
    return t


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else CSV
    os.makedirs(FIG_DIR, exist_ok=True)
    data = pd.read_csv(csv)
    print(f"{data['seed'].nunique()} ground truths, {len(data)} rows")
    fig_kl_vs_n(data)
    fig_detection_vs_n(data)
    fig_strength(data)
    fig_rq1_surrogate(data)
    fig_learned_estimators(data)
    fig_prior_centre()
    fig_strength_by_belief()
    summary_tables(data)
    print("figures written to", FIG_DIR)
