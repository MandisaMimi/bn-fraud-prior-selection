"""
structalt.py -- (a) alternative structure learners with and without a fraud prior, on the original synthetic worlds;
                (b) a SECOND generator that is not a Bayesian network built from Dirichlet/Beta tables.

Second generator ("latent"): 3 latent classes; each of the 8 features has a skewed (geometric-like) pmf per class, so the
features are dependent through a hidden variable and the observed joint has NO sparse DAG representation.  Fraud follows a
logistic model of three of the features plus the latent class; the intercept is solved so P(fraud) equals the target rate,
and the effect size is solved so that the best achievable AUROC is about 0.80.  Exact evaluation by enumeration as before.

Structures: HC (BIC), NB (fraud -> all), TAN, HCtop3 (HC with fraud parents limited to the 3 features most informative
about fraud in the sample), and for the latent generator FParents (fraud's parents = the 3 features the generator uses,
features otherwise independent).  Estimators: MLE, Jeffreys, Laplace, FP s=10/50/200 centred on the true rate.
Usage: python structalt.py orig|latent out_csv
"""
import sys, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import brentq
from sklearn.metrics import mutual_info_score
warnings.filterwarnings("ignore")
import config, ground_truth, data_generation, structure_learning
import learning_curves as lc
from pgmpy.estimators import HillClimbSearch, BIC, TreeSearch
from pgmpy.causal_discovery import ExpertKnowledge

NODES = lc.NODES; FR = NODES[-1]; FEATS = NODES[:-1]
SEEDS = [config.RANDOM_SEED_BASE + i for i in range(20)]
STATES = lc.STATES
L = 3

class LatentCtx(lc.TruthContext):
    def __init__(self, seed, rate, target_auc=0.80):
        rng = np.random.RandomState(seed)
        pi = rng.dirichlet(np.ones(L) * 2)
        pm = np.zeros((L, 8, 5))
        for z in range(L):
            for j in range(8):
                g = rng.uniform(0.7, 2.2)
                p = np.exp(-g * np.arange(5)); p = p[rng.permutation(5)]; pm[z, j] = p / p.sum()
        self.S = sorted(rng.choice(8, 3, replace=False).tolist())
        util = rng.normal(size=(8, 5)); zeff = rng.normal(size=L)
        Xf = STATES[::2, :8]                       # feature combos (390,625); fraud is the fastest-varying column
        px = np.zeros(len(Xf))
        comp = np.zeros((L, len(Xf)))
        for z in range(L):
            lp = np.zeros(len(Xf))
            for j in range(8): lp += np.log(pm[z, j][Xf[:, j]])
            comp[z] = pi[z] * np.exp(lp)
        px = comp.sum(0); post = comp / px           # P(z | x)
        score = sum(util[j][Xf[:, j]] for j in self.S)
        # class-dependence enters through P(z|x): expected latent effect
        zterm = (post * zeff[:, None]).sum(0)
        base = score + zterm
        def p_fraud(b, scale): return 0.5 * (1 + np.tanh(0.5 * (b + scale * base)))
        def calib(scale):
            R = scale * float(np.abs(base).max()) + 60
            return brentq(lambda b: float((px * p_fraud(b, scale)).sum()) - rate, -R, R)
        def auc_for(scale):
            b = calib(scale); pf = p_fraud(b, scale)
            w1 = px * pf; w0 = px * (1 - pf); o = np.argsort(-pf, kind="stable")
            tp = np.cumsum(w1[o]); fp = np.cumsum(w0[o]); P = tp[-1]; N = fp[-1]
            return float(np.trapezoid(np.r_[0, tp / P], np.r_[0, fp / N]))
        scale = brentq(lambda s_: auc_for(s_) - target_auc, 0.02, 40.0)
        b = calib(scale); pf = p_fraud(b, scale)
        pjoint = np.empty(len(Xf) * 2); pjoint[0::2] = px * (1 - pf); pjoint[1::2] = px * pf
        self.p_true = pjoint; self.lp_true = np.log(np.maximum(pjoint, 1e-300))
        self.labels = STATES[:, lc.FRAUD].astype(float)
        self.cache = None
        self.oracle_f1, self.oracle_auroc = self._metrics_from_logp(self.lp_true)
        self.px = px; self.cum = np.cumsum(pjoint); self.world_rate = float(pjoint[1::2].sum())
        self.fparents = [FEATS[j] for j in self.S]
    def sample(self, n, rng):
        idx = np.searchsorted(self.cum, rng.uniform(size=n) * self.cum[-1])
        idx = np.minimum(idx, len(self.cum) - 1)
        return STATES[idx].astype(int)

def top3(df):
    mi = {f: mutual_info_score(df[f], df[FR]) for f in FEATS}
    return sorted(mi, key=mi.get, reverse=True)[:3]

def edges_for(name, df, fparents=None):
    if name == "NB": return [(FR, f) for f in FEATS]
    if name == "FParents": return [(p, FR) for p in fparents]
    if name == "HC": return structure_learning.learn_structure(df)
    if name == "HCtop3":
        t3 = top3(df); forb = [(f, FR) for f in FEATS if f not in t3]
        dag = HillClimbSearch(df).estimate(scoring_method=BIC(df), max_indegree=config.HC_MAX_INDEGREE, show_progress=False,
                                           expert_knowledge=ExpertKnowledge(forbidden_edges=forb))
        return list(dag.edges())
    if name == "TAN":
        try: return list(TreeSearch(df).estimate(estimator_type="tan", class_node=FR, show_progress=False).edges())
        except Exception: return [(FR, f) for f in FEATS]
    raise ValueError(name)

def estimators(rate):
    E = {"MLE": lc.alpha_mle, "Jeffreys": lc.alpha_uniform(.5), "Laplace": lc.alpha_uniform(1.)}
    for s in (10, 50, 200): E[f"FP{s}"] = lc.alpha_fraud_prior(s, rate=rate)
    return E

def run_one(args):
    mode, seed, rate, structs, ns = args
    if mode == "orig":
        gt = ground_truth.build_ground_truth(seed, target_fraud_rate=rate)
        ctx = lc.TruthContext(gt); Xall = data_generation.generate_dataset(gt, max(ns), np.random.RandomState(seed + 10_000))[NODES].to_numpy()
        fpar = None; wr = float(ctx.p_true[ctx.labels == 1].sum())
    else:
        ctx = LatentCtx(seed, rate); Xall = ctx.sample(max(ns), np.random.RandomState(seed + 10_000)); fpar = ctx.fparents; wr = ctx.world_rate
    E = estimators(rate); rows = []
    for n in ns:
        X = Xall[:n]; df = pd.DataFrame(X, columns=NODES)
        for sname in structs:
            edges = edges_for(sname, df, fpar)
            pm = lc.parents_map_from_edges(edges); cache = lc.make_cache(pm)
            q = lc.n_cfg(pm[FR]); fedge = len(pm[FR]) > 0
            for ename, a in E.items():
                kl, f1, au = ctx.score(lc.fit_tables(X, pm, a), pm, cache)
                rows.append(dict(mode=mode, seed=seed, rate=rate, world_rate=wr, n=n, structure=sname, estimator=ename, kl=kl, f1=f1, auroc=au,
                                 oracle_f1=ctx.oracle_f1, oracle_auroc=ctx.oracle_auroc, q=q, fraud_parents=len(pm[FR]), n_fraud=int(X[:, lc.FRAUD].sum())))
    return rows

if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "orig":
        rates, structs, ns = [0.005, 0.02], ["HC", "HCtop3", "NB", "TAN"], [500, 2000, 10000, 50000]
    else:
        rates, structs, ns = [0.001, 0.005, 0.02, 0.05, 0.10], ["FParents", "NB", "HC"], [100, 500, 2000, 10000, 50000]
    tasks = [(mode, s, r, structs, ns) for r in rates for s in SEEDS]
    rows, t0 = [], time.time()
    with Pool(2) as p:
        for k, r in enumerate(p.imap_unordered(run_one, tasks), 1):
            rows.extend(r); print(k, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
            if k % 5 == 0 or k == len(tasks): pd.DataFrame(rows).to_csv(out, index=False)
