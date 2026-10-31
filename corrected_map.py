"""Corrected map: FP(true-rate, s=50) vs EACH of MLE/Jeffreys/Laplace (paired Wilcoxon over 20 worlds);
Holm correction over every test in the experiment (45 cells x 3 comparisons x 2 metrics = 270).
Cell label: FP  = FP significantly better than all three standard methods (intersection test, no comparator selection)
            STD = at least one standard method significantly better than FP
            =   otherwise."""
import sys, numpy as np, pandas as pd
from scipy import stats
src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src)
FPN = "FPtrue_s50"; STD = ["MLE", "Jeffreys", "Laplace"]
recs = []
for (r, n), g in d.groupby(["target_rate", "n"]):
    for metric, lower in [("kl", True), ("f1", False)]:
        w = g.pivot(index="seed", columns="estimator", values=metric)
        for s in STD:
            diff = (w[s] - w[FPN]) if lower else (w[FPN] - w[s])   # positive = FP better
            diff = diff.dropna()
            p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            recs.append(dict(rate=r, n=n, metric=metric, std=s, mean_diff=diff.mean(), p=p))
t = pd.DataFrame(recs)
# Holm
order = np.argsort(t.p.values); m = len(t); adj = np.empty(m); run = 0
for rank, i in enumerate(order):
    run = max(run, (m - rank) * t.p.values[i]); adj[i] = min(1, run)
t["p_holm"] = adj
def label(g):
    sig = g[g.p_holm < 0.05]
    if len(sig[sig.mean_diff > 0]) == 3: return "FP"
    if len(sig[sig.mean_diff < 0]) >= 1:
        return "STD:" + ",".join(sig[sig.mean_diff < 0]["std"].str[:3])
    return "="
lab = t.groupby(["metric", "rate", "n"]).apply(label).rename("label").reset_index()
lab.to_csv(out, index=False); t.to_csv(out.replace(".csv", "_tests.csv"), index=False)
for metric in ["kl", "f1"]:
    print(metric); print(lab[lab.metric == metric].pivot(index="n", columns="rate", values="label").to_string())
print("raw-p cells (uncorrected) FP-all-sig:", sum(1 for (_, _, _), g in t.groupby(["metric", "rate", "n"]) if ((g.p < .05) & (g.mean_diff > 0)).sum() == 3))
