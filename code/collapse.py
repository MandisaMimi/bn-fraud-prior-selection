import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config, ground_truth, learning_curves as lc
q={}
for s in range(42,62):
    gt=ground_truth.build_ground_truth(s); q[s]=lc.n_cfg(gt["parents"]["isFraud"])
print("q range",min(q.values()),max(q.values()))
def prep(path):
    d=pd.read_csv(path)
    if "world_rate" not in d:
        pass
    return d
res={}
for name,path in [("orig","outputs/prior_by_imbalance_results.csv"),("ctrl","outputs/ctrl_results.csv")]:
    d=pd.read_csv(path)
    d["q"]=d.seed.map(q)
    d["k_total"]=d.n*d.target_rate
    d["k_cfg"]=d.k_total/d.q
    keep=["MLE","Laplace","Jeffreys","FPtrue_s50"]
    w=d[d.estimator.isin(keep)].pivot_table(index=["seed","target_rate","n","k_total","k_cfg","n_fraud_in_sample"],columns="estimator",values=["kl","f1","auroc"]).reset_index()
    w.columns=["_".join([c for c in col if c]) for col in w.columns]
    for m in ["auroc","f1"]:
        w[f"{m}_gain_vs_lap"]=w[f"{m}_FPtrue_s50"]-w[f"{m}_Laplace"]
        w[f"{m}_gain_vs_mle"]=w[f"{m}_FPtrue_s50"]-w[f"{m}_MLE"]
    w["kl_gain_vs_lap"]=w["kl_Laplace"]-w["kl_FPtrue_s50"]
    w["kl_gain_vs_mle"]=w["kl_MLE"]-w["kl_FPtrue_s50"]
    res[name]=w
    w.to_csv(f"outputs/collapse_{name}.csv",index=False)
# quantify collapse: bin by x (log), compute fraction of variance in gain explained by x-bins, versus by (rate,n) grouping baseline
def r2_bins(w,xcol,ycol,nb=10):
    x=np.log10(w[xcol].replace(0,np.nan)); ok=x.notna()&w[ycol].notna()
    b=pd.qcut(x[ok],nb,duplicates="drop"); y=w.loc[ok,ycol]
    return 1-((y-y.groupby(b).transform("mean"))**2).sum()/((y-y.mean())**2).sum()
for name,w in res.items():
    for y in ["auroc_gain_vs_lap","auroc_gain_vs_mle","f1_gain_vs_lap","f1_gain_vs_mle"]:
        print(name,y,{x:round(r2_bins(w,x,y),3) for x in ["n","k_total","k_cfg"]})
# plots
fig,axs=plt.subplots(2,2,figsize=(9,6.4),sharex=False)
cols=plt.cm.viridis(np.linspace(0,.9,5))
for ax,(name,y,t) in zip(axs.ravel(),[("orig","auroc_gain_vs_lap","AUROC gain vs Laplace"),("orig","auroc_gain_vs_mle","AUROC gain vs MLE"),("ctrl","auroc_gain_vs_lap","AUROC gain vs Laplace"),("ctrl","auroc_gain_vs_mle","AUROC gain vs MLE")]):
    w=res[name]
    for c,(r,g) in zip(cols,w.groupby("target_rate")):
        m=g.groupby("n").agg(x=("k_cfg","mean"),y=(y,"mean"))
        ax.plot(m.x,m.y,"-o",ms=3,color=c,label=f"{r:.1%}")
    ax.set_xscale("log"); ax.axhline(0,color="k",lw=.6); ax.set_title(f"{'Original' if name=='orig' else 'Controlled'} worlds: {t}",fontsize=8)
    ax.set_xlabel("expected fraud rows per fraud-node configuration",fontsize=8)
axs[0,0].legend(title="fraud rate",fontsize=7)
fig.tight_layout(); fig.savefig("outputs/collapse_auroc.png",dpi=160)
