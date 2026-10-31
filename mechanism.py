"""
mechanism.py -- two direct tests.
 bic : does a TRUE fraud edge pay for its BIC penalty?  For every true parent p of isFraud, the change in log-likelihood,
       the BIC penalty and the net BIC change when p->isFraud is added to the graph in which isFraud has no parents
       (this is the single-edge move that Hill Climbing considers), also for the full true parent set.  n grid as in the learning curves.
 fair: identical smoothing on every non-fraud node.  Estimators with the true structure:
       Laplace (all nodes 1), FPlap50 (non-fraud nodes 1, fraud node Beta prior s=50 centred on the true rate),
       Jeffreys (all 0.5), FP50 (non-fraud nodes 0.5, same fraud prior).  Pairs differ ONLY in the fraud node.
Usage: python mechanism.py bic out.csv | fair out.csv
"""
import sys, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import config, ground_truth, data_generation
import learning_curves as lc
NODES = lc.NODES; FR = "isFraud"
SEEDS = [config.RANDOM_SEED_BASE + i for i in range(20)]
NS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

def node_ll(X, node, parents):
    k = int(lc.K[lc.IDX[node]]); nc = lc.n_cfg(parents); ci = lc.cfg_index(X, parents)
    c = np.bincount(ci * k + X[:, lc.IDX[node]], minlength=nc * k).reshape(nc, k).astype(float)
    t = c.sum(1, keepdims=True); m = c > 0
    return float((c[m] * np.log((c / np.where(t > 0, t, 1))[m])).sum()), nc * (k - 1)

def bic_one(seed):
    gt = ground_truth.build_ground_truth(seed)
    Xall = data_generation.generate_dataset(gt, max(NS), np.random.RandomState(seed + 10_000))[NODES].to_numpy()
    fpar = gt["parents"][FR]; rows = []
    for n in NS:
        X = Xall[:n]; ll0, p0 = node_ll(X, FR, [])
        for kind, ps in [("single", [[p] for p in fpar]), ("all", [list(fpar)])]:
            for pset in ps:
                ll1, p1 = node_ll(X, FR, pset); dll = ll1 - ll0; pen = 0.5 * np.log(n) * (p1 - p0)
                rows.append(dict(seed=seed, n=n, kind=kind, parent="+".join(pset), n_fraud=int(X[:, lc.FRAUD].sum()), dLL=dll, penalty=pen, dBIC=dll - pen))
    return rows

def fair_one(args):
    seed, rate = args
    gt = ground_truth.build_ground_truth(seed, target_fraud_rate=rate); ctx = lc.TruthContext(gt)
    Xall = data_generation.generate_dataset(gt, max(NS), np.random.RandomState(seed + 10_000))[NODES].to_numpy()
    pm = gt["parents"]; r = float(ctx.p_true[ctx.labels == 1].sum()); rows = []
    def fp(s, base):
        def f(node):
            if node == FR: return np.array([s * (1 - r), s * r])
            return np.full(int(lc.K[lc.IDX[node]]), base)
        return f
    E = {"Laplace": lc.alpha_uniform(1.0), "Jeffreys": lc.alpha_uniform(0.5), "FP50": fp(50, 0.5), "FPlap50": fp(50, 1.0), "FPlap10": fp(10, 1.0), "FPlap200": fp(200, 1.0)}
    for n in NS:
        X = Xall[:n]
        for e, a in E.items():
            kl, f1, au = ctx.score(lc.fit_tables(X, pm, a), pm, ctx.cache)
            rows.append(dict(seed=seed, rate=rate, n=n, estimator=e, kl=kl, f1=f1, auroc=au, n_fraud=int(X[:, lc.FRAUD].sum())))
    return rows

if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]; t0 = time.time(); rows = []
    if mode == "bic":
        with Pool(2) as p:
            for r in p.imap_unordered(bic_one, SEEDS): rows += r
    else:
        tasks = [(s, r) for r in (0.005, 0.02) for s in SEEDS]
        with Pool(2) as p:
            for k, r in enumerate(p.imap_unordered(fair_one, tasks), 1):
                rows += r; print(k, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False); print("done", time.time() - t0)
