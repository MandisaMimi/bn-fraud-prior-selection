"""
robustness.py -- checks requested after review of the report.
  floor   : sensitivity of KL to the probability floor, and effect of the MLE
            unseen-configuration convention (uniform vs node-marginal).
  thresh  : F1 with the decision threshold chosen on SEPARATE validation data
            (a validation sample of the same size as the training sample),
            compared with the oracle (best-threshold) F1.
  calib   : find Beta concentration per fraud rate so that mean oracle AUROC is
            about equal across rates (controls predictability).
  ctrl    : rerun the prior-by-imbalance comparison with those concentrations.
Usage: python robustness.py floor|thresh|calib|ctrl
"""
import sys, time, json
from multiprocessing import Pool
import numpy as np, pandas as pd
import config, ground_truth, data_generation
import learning_curves as lc

SEEDS = [config.RANDOM_SEED_BASE + i for i in range(20)]
FLOORS = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
RATE = 0.005

def mle_marg_tables(X, pm):
    """MLE where an unseen parent configuration gets the node's overall sample frequencies."""
    tabs = lc.fit_tables(X, pm, lc.alpha_mle)
    for node in lc.NODES:
        k = int(lc.K[lc.IDX[node]]); plist = pm[node]; nc = lc.n_cfg(plist)
        ci = lc.cfg_index(X, plist)
        cnt = np.bincount(ci, minlength=nc)
        marg = np.bincount(X[:, lc.IDX[node]], minlength=k) / len(X)
        un = cnt == 0
        tabs[node][un] = marg
    return tabs

def floor_one(seed):
    gt = ground_truth.build_ground_truth(seed, target_fraud_rate=RATE)
    ctx = lc.TruthContext(gt); pm = gt["parents"]
    rng = np.random.RandomState(seed + 10_000)
    Xall = data_generation.generate_dataset(gt, 50000, rng)[lc.NODES].to_numpy()
    rows = []
    ests = {"MLE": lc.alpha_mle, "Jeffreys": lc.alpha_uniform(.5), "Laplace": lc.alpha_uniform(1.), "FP50": lc.alpha_fraud_prior(50)}
    for n in [100, 500, 2000, 10000, 50000]:
        X = Xall[:n]
        tabs = {k: lc.fit_tables(X, pm, a) for k, a in ests.items()}
        tabs["MLE-marginal"] = mle_marg_tables(X, pm)
        for name, t in tabs.items():
            for fl in FLOORS:
                lc.EPS = fl
                lp = lc.joint_logprob(t, pm, floor=True, cache=ctx.cache)
                kl = float(np.sum(ctx.p_true * (ctx.lp_true - lp)))
                f1, au = ctx._metrics_from_logp(lp)
                rows.append(dict(seed=seed, n=n, estimator=name, floor=fl, kl=kl, f1=f1, auroc=au))
            lc.EPS = 1e-4
            lp0 = lc.joint_logprob(t, pm, floor=False, cache=ctx.cache)
            # true-probability mass on states the model gives probability < 1e-4 (i.e. affected by the default floor)
            rows.append(dict(seed=seed, n=n, estimator=name, floor=-1, kl=float(ctx.p_true[lp0 < np.log(1e-4)].sum()), f1=np.nan, auroc=np.nan))
    return rows

def thresh_one(seed):
    gt = ground_truth.build_ground_truth(seed, target_fraud_rate=RATE)
    ctx = lc.TruthContext(gt); pm = gt["parents"]
    Xall = data_generation.generate_dataset(gt, 50000, np.random.RandomState(seed + 10_000))[lc.NODES].to_numpy()
    Vall = data_generation.generate_dataset(gt, 50000, np.random.RandomState(seed + 20_000))[lc.NODES].to_numpy()
    shape = tuple(int(k) for k in lc.K)
    ests = {"MLE": lc.alpha_mle, "Jeffreys": lc.alpha_uniform(.5), "Laplace": lc.alpha_uniform(1.), "FP50": lc.alpha_fraud_prior(50)}
    rows = []
    for n in [500, 2000, 5000, 20000, 50000]:
        X, V = Xall[:n], Vall[:n]
        vidx = np.ravel_multi_index(V.T, shape); vy = V[:, lc.FRAUD]
        for name, a in ests.items():
            t = lc.fit_tables(X, pm, a)
            lp = lc.joint_logprob(t, pm, floor=True, cache=ctx.cache)
            logit = np.repeat(lp.reshape(-1, 2)[:, 1] - lp.reshape(-1, 2)[:, 0], 2)
            f1_or, au = ctx._metrics_from_logp(lp)
            vs = logit[vidx]
            f1_val = np.nan
            if vy.sum() > 0:
                best, thr = -1, None
                for th in np.unique(vs):
                    pred = vs >= th
                    tp = (pred & (vy == 1)).sum(); fp = (pred & (vy == 0)).sum(); fn = vy.sum() - tp
                    f = 2 * tp / (2 * tp + fp + fn)
                    if f > best: best, thr = f, th
                pred = logit >= thr
                w = ctx.p_true
                tp = w[pred & (ctx.labels == 1)].sum(); fp = w[pred & (ctx.labels == 0)].sum(); P = w[ctx.labels == 1].sum()
                f1_val = float(2 * tp / (tp + fp + P))
            rows.append(dict(seed=seed, n=n, estimator=name, f1_oracle=f1_or, f1_val=f1_val, auroc=au, n_fraud_val=int(vy.sum())))
    return rows

def conc_auc(args):
    rate, conc, seeds = args
    config.FRAUD_BETA_CONCENTRATION = conc
    out = []
    for s in seeds:
        gt = ground_truth.build_ground_truth(s, target_fraud_rate=rate)
        out.append(lc.TruthContext(gt).oracle_auroc)
    return float(np.mean(out))

def calib():
    target = 0.80
    res = {}
    rates = [0.001, 0.005, 0.02, 0.05, 0.10]
    seeds = SEEDS[:8]
    with Pool(2) as pool:
        for r in rates:
            lo, hi = np.log(1.0), np.log(2000.0)   # log concentration; AUROC decreases with concentration
            for _ in range(9):
                mid = (lo + hi) / 2
                au = pool.apply(conc_auc, ((r, float(np.exp(mid)), seeds),))
                if au > target: lo = mid
                else: hi = mid
            res[r] = float(np.exp((lo + hi) / 2))
            print(r, res[r], flush=True)
    json.dump(res, open("outputs/ctrl_concentration.json", "w"))

def ctrl_one(args):
    seed, rate, conc = args
    config.FRAUD_BETA_CONCENTRATION = conc
    gt = ground_truth.build_ground_truth(seed, target_fraud_rate=rate)
    ctx = lc.TruthContext(gt); pm = gt["parents"]
    Xall = data_generation.generate_dataset(gt, 50000, np.random.RandomState(seed + 10_000))[lc.NODES].to_numpy()
    ests = {"MLE": lc.alpha_mle, "Jeffreys": lc.alpha_uniform(.5), "Laplace": lc.alpha_uniform(1.)}
    for s in [2, 10, 50, 200]:
        ests[f"FPtrue_s{s}"] = lc.alpha_fraud_prior(s, rate=rate)
    rows = []
    for n in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        X = Xall[:n]
        for name, a in ests.items():
            kl, f1, au = ctx.score(lc.fit_tables(X, pm, a), pm, ctx.cache)
            rows.append(dict(seed=seed, target_rate=rate, conc=conc, n=n, estimator=name, kl=kl, f1=f1, auroc=au,
                             oracle_f1=ctx.oracle_f1, oracle_auroc=ctx.oracle_auroc, n_fraud_in_sample=int(X[:, lc.FRAUD].sum())))
    return rows

def run_pool(fn, tasks, out):
    rows, t0 = [], time.time()
    with Pool(2) as p:
        for k, r in enumerate(p.imap_unordered(fn, tasks), 1):
            rows.extend(r); print(k, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)

if __name__ == "__main__":
    m = sys.argv[1]
    if m == "floor": run_pool(floor_one, SEEDS, "outputs/robust_floor.csv")
    elif m == "thresh": run_pool(thresh_one, SEEDS, "outputs/robust_thresh.csv")
    elif m == "calib": calib()
    elif m == "ctrl":
        conc = json.load(open("outputs/ctrl_concentration.json"))
        run_pool(ctrl_one, [(s, float(r), c) for r, c in conc.items() for s in SEEDS], "outputs/ctrl_results.csv")
