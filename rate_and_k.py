"""
rate_and_k.py -- (1) where does the prior's rate come from?  (2) does k predict the prior's benefit better than other measures?
 rate: true structure, original worlds at 0.5% and 2% fraud (20 worlds, nine sizes). Fraud prior of strength 50 (Jeffreys on the other
       nodes, so it differs from Jeffreys smoothing only in the fraud node) centred on: the TRUE world rate; a rate ESTIMATED on a separate
       historical sample of m=1,000 or m=10,000 rows (Jeffreys-smoothed fraction); the rate estimated on the TRAINING sample itself
       (Jeffreys-smoothed fraction); a rate 10 times too high; a rate 10 times too low.
 kmeas: for the same runs, k = n*pi/q and measures computed from the actual fraud counts per parent configuration of the fraud node:
       z = share of configurations with no fraud row; m0 = true probability mass of those configurations; kmed = median fraud rows per configuration;
       kmass = mass-weighted mean of min(fraud rows, 10).  Usage: python rate_and_k.py out_rate.csv out_kmeas.csv
"""
import sys, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import config, ground_truth, data_generation, learning_curves as lc
FR = "isFraud"; NS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]; SEEDS = [42 + i for i in range(20)]

def fp(s, r):
    def f(node):
        if node == FR: return np.array([s * (1 - r), s * r])
        return np.full(int(lc.K[lc.IDX[node]]), 0.5)
    return f

def one(args):
    seed, rate = args
    gt = ground_truth.build_ground_truth(seed, target_fraud_rate=rate); ctx = lc.TruthContext(gt); pm = gt["parents"]
    Xall = data_generation.generate_dataset(gt, max(NS), np.random.RandomState(seed + 10_000))[lc.NODES].to_numpy()
    r_true = float(ctx.p_true[ctx.labels == 1].sum())
    hist = data_generation.generate_dataset(gt, 10_000, np.random.RandomState(seed + 20_000))[lc.NODES].to_numpy()
    h1k = (hist[:1000, lc.FRAUD].sum() + .5) / 1001; h10k = (hist[:, lc.FRAUD].sum() + .5) / 10001
    # true mass of each parent configuration of the fraud node
    fp_par = pm[FR]; q = lc.n_cfg(fp_par); S = lc.STATES
    ci_all = lc.cfg_index(S, fp_par); mass = np.bincount(ci_all, weights=ctx.p_true, minlength=q)
    rows, krows = [], []
    for n in NS:
        X = Xall[:n]; est = (X[:, lc.FRAUD].sum() + .5) / (n + 1)
        E = {"Jeffreys": lc.alpha_uniform(.5), "true": fp(50, r_true), "hist1k": fp(50, h1k), "hist10k": fp(50, h10k), "own": fp(50, est), "x10": fp(50, min(.5, 10 * r_true)), "d10": fp(50, r_true / 10)}
        res = {}
        for e, a in E.items(): res[e] = ctx.score(lc.fit_tables(X, pm, a), pm, ctx.cache)
        for e, (kl, f1, au) in res.items(): rows.append(dict(seed=seed, rate=rate, n=n, centre=e, kl=kl, f1=f1, auroc=au, rate_used=0 if e == "Jeffreys" else {"true": r_true, "hist1k": h1k, "hist10k": h10k, "own": est, "x10": min(.5, 10 * r_true), "d10": r_true / 10}[e]))
        cnt = np.bincount(lc.cfg_index(X, fp_par)[X[:, lc.FRAUD] == 1], minlength=q).astype(float)
        krows.append(dict(seed=seed, rate=rate, n=n, q=q, n_fraud=int(cnt.sum()), k=cnt.sum() / q, z=float((cnt == 0).mean()), m0=float(mass[cnt == 0].sum()), kmed=float(np.median(cnt)),
                          kmass=float((mass * np.minimum(cnt, 10)).sum()), d_auroc=res["true"][2] - res["Jeffreys"][2], d_kl=res["Jeffreys"][0] - res["true"][0]))
    return rows, krows

if __name__ == "__main__":
    t0 = time.time(); R, K = [], []; tasks = [(s, r) for r in (0.005, 0.02) for s in SEEDS]
    with Pool(2) as p:
        for i, (r, k) in enumerate(p.imap_unordered(one, tasks), 1): R += r; K += k; print(i, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(R).to_csv(sys.argv[1], index=False); pd.DataFrame(K).to_csv(sys.argv[2], index=False); print("done")
