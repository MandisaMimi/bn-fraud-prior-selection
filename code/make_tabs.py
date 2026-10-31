import pandas as pd,numpy as np
from scipy.stats import wilcoxon
import os
O=os.environ.get("REPORT_DIR","../report/")
bins=[0,0.3,1,3,10,1e9];labs=["$<0.3$","0.3--1","1--3","3--10","$>10$"]
rng=np.random.RandomState(0)
def boot(g,col,B=2000):
    """cluster bootstrap over worlds (seed, rate): mean of col and 95% CI"""
    cl=[v[col].values for _,v in g.groupby(["seed","rate"])]
    m=np.mean(np.concatenate(cl)) if cl else np.nan
    if len(cl)<3: return m,np.nan,np.nan
    bs=[np.mean(np.concatenate([cl[i] for i in rng.randint(0,len(cl),len(cl))])) for _ in range(B)]
    return m,np.percentile(bs,2.5),np.percentile(bs,97.5)
rows=[]
for nm,f in [("Original","collapse_orig.csv"),("Controlled","collapse_ctrl.csv")]:
    w=pd.read_csv("outputs/"+f);w["kb"]=pd.cut(w.k_cfg,bins,labels=labs);w["gm"]=w.auroc_FPtrue_s50-w.auroc_Jeffreys;w["rate"]=w.target_rate
    rows.append((nm,{l:boot(w[w.kb==l],"gm") for l in labs},{l:(w.kb==l).sum() for l in labs},{l:(w[w.kb==l].gm>0).mean() for l in labs}))
d=pd.read_csv("outputs/latent_results.csv");d["k"]=d.n*d.rate/d.q
p=d[d.structure=="FParents"].pivot_table(index=['seed','rate','n','k'],columns='estimator',values='auroc').reset_index()
p["gm"]=p.FP50-p.Jeffreys;p["kb"]=pd.cut(p.k,bins,labels=labs)
rows.append(("Latent-class",{l:boot(p[p.kb==l],"gm") for l in labs},{l:(p.kb==l).sum() for l in labs},{l:(p[p.kb==l].gm>0).mean() for l in labs}))
L=[r"\begin{tabular}{l"+"c"*5+r"}\toprule","World & "+" & ".join(r"$k=$ "+l for l in labs)+r"\\\midrule"]
for nm,bt,cn,ps in rows:
    L.append(nm+" & "+" & ".join(f"{bt[l][0]:+.3f} ({bt[l][1]:+.3f}, {bt[l][2]:+.3f})" for l in labs)+r"\\")
L.append(r"\bottomrule\end{tabular}");open(O+"tab_guide.tex","w").write("\n".join(L))
for nm,bt,cn,ps in rows: print(nm,[round(ps[l],2) for l in labs],[cn[l] for l in labs])

# strength
s=pd.read_csv("outputs/strength_collapse.csv")
pj=pd.read_csv("outputs/prior_by_imbalance_results.csv");pj=pj[(pj.estimator=="Jeffreys")&(pj.target_rate==0.005)][["seed","n","auroc"]].rename(columns={"auroc":"au_jef"})
s=s.merge(pj,on=["seed","n"],how="left");s["dAU"]=s.auroc-s.au_jef
s["pb"]=pd.cut(s.phi,[0,1,3,10,1e9],labels=["$\\le1$","1--3","3--10",">10"])
s["rb"]=pd.cut(s.ratio,[0,0.2,1.5,5,1e9],labels=["$0.1\\times$","$1\\times$","$4\\times$","$\\ge10\\times$"])
m=s.pivot_table(index="pb",columns="rb",values="dAU",aggfunc="mean",observed=False)
L=[r"\begin{tabular}{lcccc}\toprule"+"\n"+r"$\varphi=sr_b/k$ & "+" & ".join(m.columns)+r"\\\midrule"]
for i,r in m.iterrows(): L.append(f"{i} & "+" & ".join(f"{v:+.3f}" for v in r)+r"\\")
L.append(r"\bottomrule\end{tabular}");open(O+"tab_strength.tex","w").write("\n".join(L))
# structures
def blk(df,gen,struct,L):
    g=df[df.structure==struct]
    q=g.groupby(['seed','rate','n']).agg(q=('q','first'),fp=('fraud_parents','first'),k=('k','first')).reset_index()
    pv=g.pivot_table(index=['seed','rate','n'],columns='estimator',values='auroc').reset_index()
    x=pv.FP50-pv.Jeffreys
    pw=wilcoxon(x).pvalue
    niche=((q.k<3)&(q.fp>0)).mean()
    L.append(f"{gen} & {struct} & {q.q.mean():.0f} & {q.fp.mean():.1f} & {niche:.2f} & {pv.MLE.mean():.3f} & {pv.Jeffreys.mean():.3f} & {pv.FP50.mean():.3f} & {x.mean():+.4f}"+r"\\")
o=pd.read_csv("outputs/structalt_orig.csv");o["k"]=o.n*o.rate/o.q
d["k"]=d.n*d.rate/d.q
L=[r"\begin{tabular}{llrcccccc}\toprule"+"\n"+r"World & Structure & $q$ & Parents & Niche & MLE & Jeff. & FP50 & FP50$-$Jeff.\\\midrule"]
for s_ in ["HC","NB","TAN"]: blk(o,"Original",s_,L)
for s_ in ["FParents","HC","NB"]: blk(d,"Latent",s_,L)
L.append(r"\bottomrule\end{tabular}");open(O+"tab_struct.tex","w").write("\n".join(L))
print("\n".join(L))

# ---- matched-smoothing table (95% CI: bootstrap over the 20 worlds)
d=pd.read_csv("outputs/fair_smoothing.csv")
w=d.pivot_table(index=['seed','rate','n'],columns='estimator',values=['kl','f1','auroc'])
def ci(x,B=2000):
    x=np.asarray(x);bs=[x[rng.randint(0,len(x),len(x))].mean() for _ in range(B)];return x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
L=[r"\begin{tabular}{rrcccc}\toprule"+"\n"+r"Fraud rate & $n$ & KL gain (95\% CI) & Closer in & AUROC gain (95\% CI) & F1 gain\\\midrule"]
for rate in (0.005,0.02):
    for n in (100,500,2000,10000,50000):
        kl=(w['kl']['Laplace']-w['kl']['FPlap50']).xs((rate,n),level=(1,2)); au=(w['auroc']['FPlap50']-w['auroc']['Laplace']).xs((rate,n),level=(1,2)); f1=(w['f1']['FPlap50']-w['f1']['Laplace']).xs((rate,n),level=(1,2))
        k=ci(kl);a_=ci(au)
        L.append(f"{100*rate:g}\\% & {n:,} & {k[0]:.3f} ({k[1]:.3f}, {k[2]:.3f}) & {int(round((kl>0).sum()))}/20 & {a_[0]:+.3f} ({a_[1]:+.3f}, {a_[2]:+.3f}) & {f1.mean():+.3f}"+r"\\")
L.append(r"\bottomrule\end{tabular}");open(O+"tab_fair.tex","w").write("\n".join(L))
# ---- where the prior's rate comes from
R=pd.read_csv("outputs/rate_source.csv"); W=R.pivot_table(index=['seed','rate','n'],columns='centre',values='auroc')
cols=[("true","True"),("hist10k","Hist.\\,10k"),("hist1k","Hist.\\,1k"),("own","Training"),("d10",r"$0.1\times$"),("x10",r"$10\times$")]
L=[r"\begin{tabular}{rr"+"c"*len(cols)+r"}\toprule"+"\n"+r"Fraud rate & $n$ & "+" & ".join(c[1] for c in cols)+r"\\\midrule"]
for rate in (0.005,0.02):
    for n in (100,500,2000,10000,50000):
        L.append(f"{100*rate:g}\\% & {n:,} & "+" & ".join(f"{(W[c]-W['Jeffreys']).xs((rate,n),level=(1,2)).mean():+.3f}" for c,_ in cols)+r"\\")
L.append(r"\bottomrule\end{tabular}");open(O+"tab_rate.tex","w").write("\n".join(L))
# ---- do other measures of fraud counts predict the gain better than k?  (5-fold CV R2 of a binned fit)
from sklearn.model_selection import KFold
K=pd.read_csv("outputs/kmeasures.csv");K['lk']=np.log10(K.k+1e-4);K['lnf']=np.log10(K.n_fraud+1)
def cv(x,y,nb=6):
    pred=np.zeros(len(y))
    for tr,te in KFold(5,shuffle=True,random_state=1).split(x):
        e=np.unique(np.quantile(x[tr],np.linspace(0,1,nb+1)[1:-1]));bb=np.searchsorted(e,x[tr],side='right');mu={i:y[tr][bb==i].mean() for i in np.unique(bb)}
        pred[te]=[mu.get(i,y[tr].mean()) for i in np.searchsorted(e,x[te],side='right')]
    return 1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
names=[("lnf","Total fraud rows"),("lk","$k=n\\pi/q$ (mean per configuration)"),("kmed","Median fraud rows per configuration"),("z","Share of configurations with no fraud row"),("m0","Probability mass of those configurations"),("kmass","Mass-weighted fraud rows (capped at 10)")]
L=[r"\begin{tabular}{lcc}\toprule"+"\n"+r"Measure & KL gain & AUROC gain\\\midrule"]
for m,nm in names: L.append(f"{nm} & {cv(K[m].values,K['d_kl'].values):.2f} & {cv(K[m].values,K['d_auroc'].values):.2f}"+r"\\")
L.append(r"\bottomrule\end{tabular}");open(O+"tab_kmeas.tex","w").write("\n".join(L)); print("\n".join(L))
r=pd.read_csv("outputs/real_results.csv"); g=r.groupby(['structure','n','estimator']).auroc.mean().unstack()
nf=r[r.estimator=='Laplace'].groupby('n').n_fraud_train.mean()
L=[r"\begin{tabular}{rrcccccccccc}\toprule"+"\n"+r" & & \multicolumn{3}{c}{Hill Climbing} & \multicolumn{3}{c}{Naive Bayes} & \multicolumn{3}{c}{TAN}\\\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}"+"\n"+r"$n$ & Fraud rows & MLE & Jeff. & FP50 & MLE & Jeff. & FP50 & MLE & Jeff. & FP50\\\midrule"]
for n in (1000,5000,20000,60000):
    L.append(f"{n:,} & {nf[n]:.0f} & "+" & ".join(f"{g.loc[(s,n),e]:.3f}" for s in ('HC','NB','TAN') for e in ('MLE','Jeffreys','FP50'))+r"\\")
L.append(r"\bottomrule\end{tabular}");open(O+"tab_real.tex","w").write("\n".join(L))
print("\n".join(L))
