"""
realdata.py -- real-data check on the ULB credit-card fraud data (284,807 transactions, 492 fraud).
No ground truth exists, so models are scored on held-out data:
  AUROC, average precision (PR-AUC) and log loss of P(fraud | x).
Per repeat: stratified 70/30 split; 8 features chosen by mutual information with the label on the TRAINING pool only,
discretised into (up to) 5 quantile bins fitted on the pool; training subsamples of size n drawn at random (unstratified).
Structures: HC+BIC, naive Bayes, TAN, and HC with Class parents restricted to the 3 most informative features.
Estimators: MLE, Jeffreys, Laplace, fraud prior (assumed rate = pool rate; s=10,50,200) and a mis-centred one (r=0.5%).
Usage: python realdata.py path_to_csv out_csv [n_repeats]
"""
import sys, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score, average_precision_score, mutual_info_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
from pgmpy.estimators import HillClimbSearch, BIC, TreeSearch
from pgmpy.causal_discovery import ExpertKnowledge

CSV, OUT = sys.argv[1], sys.argv[2]
NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 20
N_GRID = [1000, 5000, 20000, 60000, 0]          # 0 = whole pool
NFEAT = 8
D = pd.read_csv(CSV)
COLS = [c for c in D.columns if c != "Class"]

def discretise(pool, other, cols):
    out_p, out_o, K = {}, {}, []
    for c in cols:
        edges = np.unique(np.quantile(pool[c].values, [.2, .4, .6, .8]))
        out_p[c] = np.searchsorted(edges, pool[c].values, side="right")
        out_o[c] = np.searchsorted(edges, other[c].values, side="right")
        K.append(len(edges) + 1)
    return pd.DataFrame(out_p, index=pool.index), pd.DataFrame(out_o, index=other.index), K

def cfg_index(X, parents, K):
    idx = np.zeros(len(X), dtype=np.int64)
    for p in parents: idx = idx * K[p] + X[:, p]
    return idx

def fit_tables(X, pm, K, alpha_fn):
    tabs = {}
    for j in range(len(K)):
        pl = pm[j]; nc = int(np.prod([K[p] for p in pl])) if pl else 1
        ci = cfg_index(X, pl, K)
        cnt = np.bincount(ci * K[j] + X[:, j], minlength=nc * K[j]).reshape(nc, K[j]).astype(float)
        a = np.asarray(alpha_fn(j), float); tot = cnt.sum(1, keepdims=True) + a.sum()
        tabs[j] = np.where(tot > 0, (cnt + a) / np.where(tot > 0, tot, 1), 1.0 / K[j])
    return tabs

def joint_lp(X, tabs, pm, K):
    lp = np.zeros(len(X))
    for j in range(len(K)):
        lp += np.log(np.maximum(tabs[j][cfg_index(X, pm[j], K), X[:, j]], 1e-300))
    return lp

def score(Xte, yte, tabs, pm, K):
    X1 = Xte.copy(); X1[:, -1] = 1; X0 = Xte.copy(); X0[:, -1] = 0
    l1, l0 = joint_lp(X1, tabs, pm, K), joint_lp(X0, tabs, pm, K)
    m = np.maximum(l1, l0); p = np.exp(l1 - m) / (np.exp(l1 - m) + np.exp(l0 - m))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -np.mean(yte * np.log(pc) + (1 - yte) * np.log(1 - pc))
    return roc_auc_score(yte, p), average_precision_score(yte, p), ll

def estimators(rate):
    def a_const(v): return lambda j: None
    E = {"MLE": lambda K: (lambda j: np.zeros(K[j])),
         "Jeffreys": lambda K: (lambda j: np.full(K[j], .5)),
         "Laplace": lambda K: (lambda j: np.full(K[j], 1.))}
    def fp(s, r):
        def mk(K):
            def f(j):
                if j == len(K) - 1: return np.array([s * (1 - r), s * r])
                return np.full(K[j], .5)
            return f
        return mk
    for s in (10, 50, 200):
        E[f"FP{s}"] = fp(s, rate)
        E[f"FPmis{s}"] = fp(s, 0.005)
    return E

def structures(df, names, K, top3):
    out = {}
    cls = names[-1]
    feats = names[:-1]
    hc = HillClimbSearch(df)
    dag = hc.estimate(scoring_method=BIC(df), max_indegree=4, show_progress=False)
    out["HC"] = list(dag.edges())
    forb = [(f, cls) for f in feats if f not in top3]
    dag = HillClimbSearch(df).estimate(scoring_method=BIC(df), max_indegree=4, show_progress=False,
                                       expert_knowledge=ExpertKnowledge(forbidden_edges=forb))
    out["HCtop3"] = list(dag.edges())
    out["NB"] = [(cls, f) for f in feats]
    try:
        out["TAN"] = list(TreeSearch(df).estimate(estimator_type="tan", class_node=cls, show_progress=False).edges())
    except Exception as e:
        out["TAN"] = out["NB"]
    return out

def run_rep(rep):
    rng_seed = 1000 + rep
    tr, te = train_test_split(D, test_size=0.3, stratify=D.Class, random_state=rng_seed)
    mi = {c: mutual_info_score(np.searchsorted(np.unique(np.quantile(tr[c], [.2, .4, .6, .8])), tr[c], side="right"), tr.Class) for c in COLS}
    feats = sorted(mi, key=mi.get, reverse=True)[:NFEAT]
    dtr, dte, K = discretise(tr, te, feats)
    names = feats + ["Class"]; K = K + [2]
    dtr["Class"] = tr.Class.values; dte["Class"] = te.Class.values
    Xte = dte[names].to_numpy(); yte = dte.Class.values
    rate = float(tr.Class.mean())
    top3 = feats[:3]
    E = estimators(rate)
    rng = np.random.RandomState(rep)
    rows = []
    for n in N_GRID:
        sub = dtr.sample(n=len(dtr) if n == 0 else n, random_state=rng.randint(1 << 30))
        Xtr = sub[names].to_numpy(); nf = int(sub.Class.sum())
        sts = structures(sub[names], names, K, top3)
        for sname, edges in sts.items():
            pm = {j: [] for j in range(len(K))}
            for a, b in edges: pm[names.index(b)].append(names.index(a))
            nconf = int(np.prod([K[p] for p in pm[len(K) - 1]])) if pm[len(K) - 1] else 1
            for ename, mk in E.items():
                tabs = fit_tables(Xtr, pm, K, mk(K))
                au, ap, ll = score(Xte, yte, tabs, pm, K)
                rows.append(dict(rep=rep, n=len(sub), n_fraud_train=nf, structure=sname, estimator=ename, auroc=au, ap=ap, logloss=ll,
                                 n_class_parents=len(pm[len(K) - 1]), class_configs=nconf, pool_rate=rate, n_test_fraud=int(yte.sum())))
    return rows

if __name__ == "__main__":
    t0, allr = time.time(), []
    with Pool(2) as p:
        for k, r in enumerate(p.imap_unordered(run_rep, range(NREP)), 1):
            allr.extend(r); print(k, NREP, f"{time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(allr).to_csv(OUT, index=False)
