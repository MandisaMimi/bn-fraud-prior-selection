"""
baselines_exact.py
==================
Benchmark of the Bayesian network against one model from each major paradigm of
machine learning, evaluated EXACTLY on the same 20 ground-truth worlds
(seeds 42-61, fraud rate 0.5%) and the same nested training samples as the
learning-curve experiments.

Models (all default-style settings, NOT tuned; categorical variables one-hot encoded):
    Logistic regression      linear, probabilistic
    Linear SVM (LinearSVC)   margin-based / linearly separable paradigm
    Decision tree            tree-based
    Random forest            tree ensemble
    Neural network (MLP)     one hidden layer of 32 units
    Naive Bayes (CategoricalNB, alpha=1)   simple Bayesian classifier

Exact evaluation: every one of the 5^8 = 390,625 feature combinations is scored
by the model and weighted by its true probability, so the F1 and AUROC are the
population values (what an infinite test set would give), exactly as for the
Bayesian network in learning_curves.py.  Best-threshold F1 (optimistic, same
convention as before) and AUROC.

Bayesian-network rows are read from outputs/learning_curves_results.csv
(true structure + fraud prior FP_s50 = best case; learned structure + Laplace).

Run: python baselines_exact.py [n_seeds]
"""

import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import CategoricalNB

import config
import ground_truth
import data_generation
import learning_curves as lc

warnings.filterwarnings("ignore")
N_GRID = [500, 2000, 5000, 20000, 50000]
N_SEEDS = 20
OUT = "outputs/baselines_exact_results.csv"
FEAT = [i for i in range(len(lc.NODES)) if i != lc.FRAUD]
KF = lc.K[FEAT]
ALL_X = np.indices(tuple(int(k) for k in KF)).reshape(len(KF), -1).T.astype(np.int16)  # (390625, 8)


def onehot(X):
    cols = []
    for j, k in enumerate(KF):
        cols.append(np.eye(int(k), dtype=np.float32)[X[:, j]])
    return np.hstack(cols)


ALL_OH = onehot(ALL_X)


def make_models(seed):
    return {
        "Logistic regression": (LogisticRegression(max_iter=2000), "oh"),
        "Linear SVM": (LinearSVC(max_iter=5000), "oh"),
        "Decision tree": (DecisionTreeClassifier(min_samples_leaf=10, random_state=seed), "raw"),
        "Random forest": (RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1), "raw"),
        "Neural network": (MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=seed), "oh"),
        "Naive Bayes": (CategoricalNB(alpha=1.0, min_categories=[int(k) for k in KF]), "raw"),
    }


def run_seed(seed):
    gt = ground_truth.build_ground_truth(seed)
    ctx = lc.TruthContext(gt)
    rng = np.random.RandomState(seed + 10_000)
    X_all = data_generation.generate_dataset(gt, max(N_GRID), rng)[lc.NODES].to_numpy()
    rows = []
    for n in N_GRID:
        X = X_all[:n]
        Xf, y = X[:, FEAT], X[:, lc.FRAUD]
        for name, (mdl, kind) in make_models(seed).items():
            t0 = time.time()
            if y.sum() == 0 or y.sum() == len(y):
                f1, auroc = float("nan"), float("nan")
            else:
                Xtr = onehot(Xf) if kind == "oh" else Xf
                Xte = ALL_OH if kind == "oh" else ALL_X
                mdl.fit(Xtr, y)
                s = mdl.predict_proba(Xte)[:, 1] if hasattr(mdl, "predict_proba") else mdl.decision_function(Xte)
                s = np.repeat(np.asarray(s, dtype=float), 2)   # fraud is the fastest-varying column
                f1, auroc = lc.population_metrics(ctx.p_true, ctx.labels, s)
            rows.append({"seed": seed, "n": n, "model": name, "f1": f1, "auroc": auroc,
                         "oracle_f1": ctx.oracle_f1, "oracle_auroc": ctx.oracle_auroc,
                         "n_fraud_train": int(y.sum()), "secs": time.time() - t0})
    return rows


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else N_SEEDS
    seeds = [config.RANDOM_SEED_BASE + i for i in range(n_seeds)]
    allrows, t0 = [], time.time()
    with Pool(2) as p:
        for k, r in enumerate(p.imap_unordered(run_seed, seeds), 1):
            allrows.extend(r)
            print(f"{k}/{len(seeds)} worlds ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(allrows).to_csv(OUT, index=False)
    print("done")
