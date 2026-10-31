"""
learning_curve_analysis.py
==========================
Turns the learning-curve results into statements that can be defended:
  1. convergence speed and "when does the prior stop mattering?"
  2. paired statistical tests + effect sizes (Holm-corrected) across ground truths
  3. does the benefit of the priors survive when the structure is LEARNED (not true)?
"""
import numpy as np, pandas as pd
from scipy import stats

pd.set_option("display.width", 220)
df = pd.read_csv("outputs/learning_curves_results.csv")
T = df[df.structure == "true"]
NS = [100, 500, 2000, 5000, 20000, 50000]

# ---------------------------------------------------------------- 1. convergence
mean_kl = T.pivot_table(index="n", columns="estimator", values="kl", aggfunc="mean")
rows = []
for est in ["MLE", "Jeffreys", "Laplace", "FP_s50"]:
    big = mean_kl.loc[1000:, est]
    slope = np.polyfit(np.log(big.index.values), np.log(big.values), 1)[0]
    rows.append({"estimator": est, "KL at n=100": mean_kl.loc[100, est], "KL at n=5,000": mean_kl.loc[5000, est],
                 "KL at n=50,000": mean_kl.loc[50000, est], "log-log slope (n>=1000)": slope})
conv = pd.DataFrame(rows).set_index("estimator")
gap = (mean_kl["MLE"] / mean_kl[["Jeffreys", "Laplace", "FP_s50"]].min(axis=1)).rename("MLE KL / best-prior KL")
conv.to_csv("outputs/learning_curves_convergence.csv")
print("== CONVERGENCE (true structure) ==")
print(conv.round(4).to_string())
print("\nHow many times larger is MLE's KL than the best prior's?")
print(gap.round(2).to_string())

# ---------------------------------------------------------------- 2. paired tests
def pivot(est, metric):
    return T[T.estimator == est].pivot_table(index="seed", columns="n", values=metric)

pairs = [("FP_s50", "MLE"), ("FP_s50", "Jeffreys"), ("FP_s50", "Laplace"), ("Jeffreys", "MLE"), ("Laplace", "MLE"), ("Laplace", "Jeffreys")]
recs = []
for a, b in pairs:
    for metric in ["kl", "f1", "auroc"]:
        A, B = pivot(a, metric), pivot(b, metric)
        for n in NS:
            d = (A[n] - B[n]).values
            se = d.std(ddof=1) / np.sqrt(len(d))
            tcrit = stats.t.ppf(0.975, len(d) - 1)
            try:
                p = stats.wilcoxon(d).pvalue
            except ValueError:
                p = 1.0
            recs.append({"A": a, "B": b, "metric": metric, "n": n, "mean_diff_A_minus_B": d.mean(),
                         "ci95_low": d.mean() - tcrit * se, "ci95_high": d.mean() + tcrit * se,
                         "effect_size_dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
                         "A_better_in": int((d < 0).sum() if metric == "kl" else (d > 0).sum()), "of": len(d), "wilcoxon_p": p})
S = pd.DataFrame(recs)
order = S["wilcoxon_p"].rank(method="first").astype(int).values - 1       # Holm step-down
m = len(S); idx = np.argsort(S["wilcoxon_p"].values); adj = np.empty(m); run = 0
for rank, i in enumerate(idx):
    run = max(run, (m - rank) * S["wilcoxon_p"].values[i]); adj[i] = min(1.0, run)
S["p_holm"] = adj
S["significant_holm_0.05"] = S["p_holm"] < 0.05
S.to_csv("outputs/learning_curves_stats_true_structure.csv", index=False)
print(f"\n== PAIRED TESTS: {m} comparisons, Holm-corrected; wrote outputs/learning_curves_stats_true_structure.csv ==")
for (a, b, metric) in [("FP_s50", "MLE", "auroc"), ("FP_s50", "Laplace", "auroc"), ("FP_s50", "Laplace", "f1"), ("FP_s50", "Jeffreys", "kl"), ("Laplace", "FP_s50", "kl")]:
    s = S[(S.A == a) & (S.B == b) & (S.metric == metric)] if (a, b) in pairs else S[(S.A == b) & (S.B == a) & (S.metric == metric)]
    print(f"\n{a} vs {b}, {metric}:")
    print(s[["n", "mean_diff_A_minus_B", "ci95_low", "ci95_high", "effect_size_dz", "A_better_in", "p_holm", "significant_holm_0.05"]].round(4).to_string(index=False))

# "when does the prior matter?" -> direction-aware verdict vs MLE at each n (Holm-corrected, alpha=0.05)
print("\n== PRIOR vs MLE, direction-aware verdict at each n  (B = better than MLE, W = worse than MLE, . = no significant difference) ==")
verdicts = []
for a in ["Jeffreys", "Laplace", "FP_s50"]:
    for metric in ["kl", "f1", "auroc"]:
        row = {"estimator": a, "metric": metric}
        for n in NS:
            r = S[(S.A == a) & (S.B == "MLE") & (S.metric == metric) & (S.n == n)].iloc[0]
            better = r.mean_diff_A_minus_B < 0 if metric == "kl" else r.mean_diff_A_minus_B > 0
            row[n] = ("B" if better else "W") if r["significant_holm_0.05"] else "."
        verdicts.append(row)
V = pd.DataFrame(verdicts)
V.to_csv("outputs/learning_curves_prior_vs_mle_verdicts.csv", index=False)
print(V.to_string(index=False))

# ---------------------------------------------------------------- 3. learned structure
L = df[df.structure == "learned"]
print("\n== LEARNED STRUCTURE (Hill Climbing + BIC): do the priors still help? (medians over 20 ground truths) ==")
for metric in ["kl", "f1", "auroc"]:
    print(f"\nmedian {metric}")
    print(L.pivot_table(index="n", columns="estimator", values=metric, aggfunc="median").round(4).to_string())
print("\nMean KL, only worlds where an edge into isFraud WAS learned vs NOT (n >= 10,000):")
big = L[(L.n >= 10000)]
print(big.groupby(["has_fraud_edge", "estimator"])[["kl", "f1", "auroc"]].median().round(4).to_string())
mle = L[L.estimator == "MLE"].pivot_table(index="seed", columns="n", values="kl")
for est in ["Jeffreys", "Laplace", "FP_s50"]:
    e = L[L.estimator == est].pivot_table(index="seed", columns="n", values="kl")
    print(f"{est}: KL lower than MLE in (of 20 worlds) " + ", ".join(f"n={n}: {(e[n] < mle[n]).sum()}" for n in [100, 500, 2000, 10000, 50000]))
L.groupby(["n", "estimator"])[["kl", "f1", "auroc"]].median().to_csv("outputs/learning_curves_learned_structure_summary.csv")
