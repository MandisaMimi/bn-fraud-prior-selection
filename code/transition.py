"""transition.py -- builds the mechanism table (tab_mech.tex), outputs/transition_table.csv and figs/mechanism.png from
outputs/bic_mechanism.csv (run `python mechanism.py bic outputs/bic_mechanism.csv` first) and outputs/learning_curves_results.csv."""
import pandas as pd, numpy as np, matplotlib, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
O = os.environ.get("REPORT_DIR", "../report/")
b = pd.read_csv("outputs/bic_mechanism.csv")
s = b[b.kind == "single"].groupby("n").agg(nf=("n_fraud", "mean"), dLL=("dLL", "mean"), pen=("penalty", "mean"), dB=("dBIC", "mean"), pos=("dBIC", lambda x: (x > 0).mean()))
d = pd.read_csv("outputs/learning_curves_results.csv"); l = d[(d.structure == "learned") & (d.estimator == "Laplace")].copy(); l["fe"] = l.has_fraud_edge.astype(bool)
t = l.groupby("n").agg(fe=("fe", "mean"), ne=("n_edges", "mean"), kl=("kl", "median"), au=("auroc", "mean"))
e = d[d.structure == "empty"].groupby("n").kl.median().rename("kl0")
a1 = l[l.fe].groupby("n").auroc.agg(["mean", "size"]).rename(columns={"mean": "au_fe", "size": "nfe"}); a0 = l[~l.fe].groupby("n").auroc.mean().rename("au_nofe")
tt = pd.read_csv("outputs/trace_synth_runs.csv")
tg = tt.groupby("n").agg(steps=("n_steps", "mean"), linked_hc=("fraud_linked", "mean"), first=("first_fraud_step", "median"))
tg["stop"] = tt[~tt.fraud_linked].groupby("n").stop_best_fraud_delta.median()
tr = pd.concat([s, t, e, a1, a0, tg], axis=1); tr.to_csv("outputs/transition_table.csv"); n = tr.index.values
fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9))
ax[0].plot(n, tr.dLL, "o-", label="likelihood gain"); ax[0].plot(n, tr.pen, "s-", label="BIC penalty"); ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("training rows $n$"); ax[0].set_ylabel("per true fraud edge"); ax[0].legend(fontsize=7); ax[0].set_title("(a) edge gain vs penalty", fontsize=9)
ax[1].plot(n, tr.pos, "o-", label="true fraud edges with net BIC gain"); ax[1].plot(n, tr.linked_hc, "s-", label="worlds where HC links fraud"); ax[1].set_xscale("log"); ax[1].set_xlabel("training rows $n$"); ax[1].set_ylabel("share"); ax[1].legend(fontsize=7); ax[1].set_title("(b) when fraud edges appear", fontsize=9)
ax[2].plot(n, tr.au, "o-", label="AUROC (learned, all worlds)"); ax[2].plot(n, tr.au_nofe, "s--", label="AUROC, no fraud edge"); ax[2].set_xscale("log"); ax[2].set_xlabel("training rows $n$"); ax[2].set_ylabel("AUROC")
ax2 = ax[2].twinx(); ax2.plot(n, tr.kl / tr.kl0, "^:", color="C3", label="KL / KL(empty)"); ax2.set_ylabel("KL relative to empty graph")
h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[2].legend(h1 + h2, l1 + l2, fontsize=6.5, loc="center left"); ax[2].set_title("(c) detection vs distribution", fontsize=9)
for a in ax: a.tick_params(labelsize=7)
plt.tight_layout(); os.makedirs(O + "figs", exist_ok=True); plt.savefig(O + "figs/mechanism.png", dpi=170)
L = [r"\begin{tabular}{rrrrrrrrrr}\toprule" + "\n" + r"$n$ & Fraud rows & $\Delta$LL & Penalty & $\Delta$BIC & Edges paying & Search stop $\Delta$BIC & Worlds linked & First fraud step & AUROC\\\midrule"]
for i, r in tr.iterrows():
    fs = "--" if pd.isna(r["first"]) else f"{r['first']:.0f}"; st = "--" if pd.isna(r["stop"]) else f"{r['stop']:.1f}"
    L.append(f"{int(i):,} & {r['nf']:.1f} & {r['dLL']:.1f} & {r['pen']:.1f} & {r['dB']:+.1f} & {100*r['pos']:.0f}\\% & {st} & {100*r['linked_hc']:.0f}\\% & {fs} & {r['au']:.2f}" + r"\\")
L.append(r"\bottomrule\end{tabular}"); open(O + "tab_mech.tex", "w").write("\n".join(L)); print("ok")
