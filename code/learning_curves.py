"""
learning_curves.py
==================
Learning-curve experiments requested for RQ1 / RQ2.

WHAT IS PLOTTED
---------------
x-axis : number of training samples (n), or prior strength (s)
y-axis : KL(P_true || P_model)   -> "how close is the estimated distribution?"
         best-threshold F1, AUROC -> "how well does it detect fraud?"
Each curve is one estimator: MLE, Jeffreys, Laplace, or the fraud-informed
prior with a given strength s.

WHY THIS FILE IS EXACT (NO SAMPLING NOISE)
------------------------------------------
The ground-truth network has only 5^8 * 2 = 781,250 joint states, so every
quantity can be computed by enumerating ALL states:

    KL(P_true || P_model) = sum_x P_true(x) * [log P_true(x) - log P_model(x)]

and, because we know P_true(x) for every state, the *population* F1 and AUROC
of a fitted model (what it would score on an infinite test set) are also exact:
each state is a "row" weighted by P_true(x). So the curves are smooth and the
only randomness left is which training sample / which ground truth was drawn.

CONVENTIONS KEPT FROM THE EARLIER EXPERIMENTS (so numbers are comparable)
-------------------------------------------------------------------------
* A model probability below 1e-4 is floored at 1e-4 before taking logs
  (evaluation.row_log_prob does the same). This mainly affects MLE, whose
  zero counts would otherwise give log(0).
* A parent configuration never seen in training gets a uniform distribution.
* Jeffreys: alpha = 0.5 on every cell. Laplace: alpha = 1 on every cell.
* Fraud-informed prior with strength s on the isFraud node only:
      alpha_fraud = s * r,  alpha_legit = s * (1 - r),  r = config.TARGET_FRAUD_RATE
  (a Beta(s*r, s*(1-r)) prior centred on the fixed base rate r, NOT on the
  realised rate of the training sample). All other nodes use Jeffreys.
* F1 is the best F1 over all decision thresholds (as in priors.classification_metrics).
"""

import sys
import time
import numpy as np
import pandas as pd

import config
import ground_truth
import data_generation
import structure_learning

NODES = config.NODES
K = np.array([config.STATE_COUNTS[n] for n in NODES])
IDX = {n: i for i, n in enumerate(NODES)}
FRAUD = IDX["isFraud"]
EPS = 1e-4

N_GRID = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
STRENGTHS = [1, 2, 3, 4, 10, 25, 50, 100, 200, 500]   # fraud-prior strength s
N_SEEDS_LC = 20
BASE_SEED = config.RANDOM_SEED_BASE


# ---------------------------------------------------------------------------
# Enumeration of every joint state (last column = isFraud, fastest-varying)
# ---------------------------------------------------------------------------
def all_states():
    return np.indices(tuple(int(k) for k in K)).reshape(len(NODES), -1).T.astype(np.int16)


STATES = all_states()  # (781250, 9)


def cfg_index(X, parents):
    """Mixed-radix index of each row's parent configuration (same order as `parents`)."""
    idx = np.zeros(len(X), dtype=np.int64)
    for p in parents:
        idx = idx * int(K[IDX[p]]) + X[:, IDX[p]]
    return idx


def n_cfg(parents):
    return int(np.prod([K[IDX[p]] for p in parents])) if parents else 1


# ---------------------------------------------------------------------------
# Models are stored as {node: table of shape (n_parent_configs, n_states)}
# ---------------------------------------------------------------------------
def truth_tables(gt):
    tables = {}
    for node in NODES:
        plist = gt["parents"][node]
        k = int(K[IDX[node]])
        if not plist:
            tables[node] = np.asarray(gt["marginals"][node], dtype=float).reshape(1, k).copy()
            continue
        T = np.empty((n_cfg(plist), k))
        for cfg, vec in gt["cpts"][node].items():
            i = 0
            for p, s in zip(plist, cfg):
                i = i * int(K[IDX[p]]) + s
            T[i] = vec
        tables[node] = T
    return tables


def parents_map_from_edges(edges):
    pm = {n: [] for n in NODES}
    for parent, child in edges:
        pm[child].append(parent)
    return pm


def fit_tables(X, parents_map, alpha_for_node):
    """Posterior-mean CPTs.  alpha_for_node(node) -> length-k vector (0s for MLE)."""
    tables = {}
    for node in NODES:
        plist = parents_map[node]
        k = int(K[IDX[node]])
        nc = n_cfg(plist)
        ci = cfg_index(X, plist)
        counts = np.bincount(ci * k + X[:, IDX[node]], minlength=nc * k).reshape(nc, k).astype(float)
        a = np.asarray(alpha_for_node(node), dtype=float)
        tot = counts.sum(axis=1, keepdims=True) + a.sum()
        probs = np.where(tot > 0, (counts + a) / np.where(tot > 0, tot, 1.0), 1.0 / k)
        tables[node] = probs
    return tables


def alpha_mle(node):
    return np.zeros(int(K[IDX[node]]))


def alpha_uniform(value):
    return lambda node: np.full(int(K[IDX[node]]), float(value))


def alpha_fraud_prior(strength, rate=None):
    r = config.TARGET_FRAUD_RATE if rate is None else rate

    def f(node):
        if node == "isFraud":
            return np.array([strength * (1.0 - r), strength * r])
        return np.full(int(K[IDX[node]]), 0.5)

    return f


def estimator_registry():
    reg = {"MLE": alpha_mle, "Jeffreys": alpha_uniform(0.5), "Laplace": alpha_uniform(1.0)}
    for s in STRENGTHS:
        reg[f"FP_s{s}"] = alpha_fraud_prior(s)
    return reg


# ---------------------------------------------------------------------------
# Exact evaluation by enumeration
# ---------------------------------------------------------------------------
def joint_logprob(tables, parents_map, floor=True, cache=None):
    lp = np.zeros(len(STATES))
    for node in NODES:
        if cache is not None and node in cache:
            ci = cache[node]
        else:
            ci = cfg_index(STATES, parents_map[node])
        p = tables[node][ci, STATES[:, IDX[node]]]
        lp += np.log(np.maximum(p, EPS if floor else 1e-300))
    return lp


def make_cache(parents_map):
    return {node: cfg_index(STATES, parents_map[node]) for node in NODES}


def population_metrics(weights, labels, score):
    """Exact best-threshold F1 and AUROC when each state is a row weighted by `weights`."""
    order = np.argsort(-score, kind="stable")
    s = score[order]
    w_pos = (weights * labels)[order]
    w_neg = (weights * (1 - labels))[order]
    cum_tp = np.cumsum(w_pos)
    cum_fp = np.cumsum(w_neg)
    P = cum_tp[-1]
    N = cum_fp[-1]
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]  # end of each tie group
    tp, fp = cum_tp[last], cum_fp[last]
    f1 = 2 * tp / (tp + fp + P)
    tpr = np.r_[0.0, tp / P]
    fpr = np.r_[0.0, fp / N]
    auroc = float(np.trapezoid(tpr, fpr))
    return float(f1.max()), auroc


class TruthContext:
    """Everything about one ground truth that does not depend on the model being scored."""

    def __init__(self, gt):
        self.gt = gt
        self.tables = truth_tables(gt)
        self.cache = make_cache(gt["parents"])
        self.lp_true = joint_logprob(self.tables, gt["parents"], floor=False, cache=self.cache)
        self.p_true = np.exp(self.lp_true)
        self.labels = STATES[:, FRAUD].astype(float)
        self.oracle_f1, self.oracle_auroc = self._metrics_from_logp(self.lp_true)

    def _metrics_from_logp(self, lp):
        pair = lp.reshape(-1, 2)
        logit = np.repeat(pair[:, 1] - pair[:, 0], 2)
        return population_metrics(self.p_true, self.labels, logit)

    def score(self, tables, parents_map, cache=None):
        lp = joint_logprob(tables, parents_map, floor=True, cache=cache)
        kl = float(np.sum(self.p_true * (self.lp_true - lp)))
        f1, auroc = self._metrics_from_logp(lp)
        return kl, f1, auroc


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_seed(seed, n_grid=N_GRID, with_learned=True):
    gt = ground_truth.build_ground_truth(seed)
    ctx = TruthContext(gt)
    rng = np.random.RandomState(seed + 10_000)
    df_all = data_generation.generate_dataset(gt, max(n_grid), rng)
    X_all = df_all[NODES].to_numpy()
    reg = estimator_registry()
    true_pm = gt["parents"]
    empty_pm = {n: [] for n in NODES}
    rows = []

    def add(structure, estimator, n, kl, f1, auroc, extra=None):
        rows.append({"seed": seed, "n": n, "structure": structure, "estimator": estimator,
                     "kl": kl, "f1": f1, "auroc": auroc,
                     "oracle_f1": ctx.oracle_f1, "oracle_auroc": ctx.oracle_auroc, **(extra or {})})

    for n in n_grid:
        X = X_all[:n]

        # (A) TRUE structure, every estimator  -> isolates RQ2 (parameter estimation)
        for name, afn in reg.items():
            kl, f1, auroc = ctx.score(fit_tables(X, true_pm, afn), true_pm, ctx.cache)
            add("true", name, n, kl, f1, auroc)

        if not with_learned:
            continue

        # (B) LEARNED structure (Hill Climbing + BIC) and the empty graph -> RQ1 surrogate quality
        edges = structure_learning.learn_structure(pd.DataFrame(X, columns=NODES))
        learned_pm = parents_map_from_edges(edges)
        learned_cache = make_cache(learned_pm)
        has_fraud_edge = any(c == "isFraud" for _, c in edges)
        for name in ["MLE", "Jeffreys", "Laplace", "FP_s50"]:
            kl, f1, auroc = ctx.score(fit_tables(X, learned_pm, reg[name]), learned_pm, learned_cache)
            add("learned", name, n, kl, f1, auroc,
                {"n_edges": len(edges), "has_fraud_edge": has_fraud_edge})
        empty_cache = make_cache(empty_pm)
        kl, f1, auroc = ctx.score(fit_tables(X, empty_pm, reg["Laplace"]), empty_pm, empty_cache)
        add("empty", "Laplace", n, kl, f1, auroc)
    return rows


def run_all(n_seeds=N_SEEDS_LC, seed_offset=0, out_csv="outputs/learning_curves_results.csv", verbose=True):
    all_rows = []
    t0 = time.time()
    for i in range(seed_offset, seed_offset + n_seeds):
        seed = BASE_SEED + i
        t1 = time.time()
        all_rows.extend(run_seed(seed))
        if verbose:
            print(f"seed {seed} done in {time.time() - t1:.1f}s (total {time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else N_SEEDS_LC
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else "outputs/learning_curves_results.csv"
    run_all(n_seeds, offset, out)
