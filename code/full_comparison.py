"""
full_comparison.py
===================
The consolidated results table for the dissertation's Results chapter:
for each of N seeds, learns G* (Hill Climbing + BIC), fits it under all
three Dirichlet priors, fits the three competitor baselines (Logistic
Regression, Random Forest, Naive Bayes -- Section 3.4.2), and evaluates
every one of the six models on the SAME held-out data with the SAME
metrics (precision, recall, F1, AUROC -- all at the F1-optimal threshold,
via priors.classification_metrics, for a fair comparison against the BN
models, which are evaluated the same way).

This directly answers "is a Bayesian Network with a sensible prior
competitive with standard discriminative baselines, despite the added
interpretability" -- the central practical question motivating the whole
dissertation (see Chapter 2's discussion of the accuracy/interpretability
trade-off in the literature).
"""

import time
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

import config
import ground_truth
import data_generation
import structure_learning
import priors
import baselines

BN_MODEL_NAMES = ["jeffreys", "laplace", "expert"]
BASELINE_MODEL_NAMES = ["logistic_regression", "random_forest", "naive_bayes", "majority_class"]
ALL_MODEL_NAMES = BN_MODEL_NAMES + BASELINE_MODEL_NAMES


def run_single_seed(seed: int, rows=None, verbose=True):
    rows = rows or config.ROWS_PER_DATASET
    t0 = time.time()

    random.seed(seed)
    np.random.seed(seed)

    gt = ground_truth.build_ground_truth(seed)
    data_rng = np.random.RandomState(seed + 10_000)
    train_df = data_generation.generate_dataset(gt, rows, data_rng)
    holdout_df = data_generation.generate_dataset(gt, config.HOLDOUT_ROWS, data_rng)

    learned_edges = structure_learning.learn_structure(train_df)
    y_true = holdout_df[priors.TARGET_NODE].to_numpy()

    rows_out = []

    # --- Bayesian Network under each of the three priors -------------------
    for prior_name in BN_MODEL_NAMES:
        model = priors.fit_with_prior(learned_edges, train_df, prior_name)
        y_prob = priors.predict_fraud_probability(model, holdout_df)
        metrics = priors.classification_metrics(y_true, y_prob)
        rows_out.append({"seed": seed, "model": f"BN_{prior_name}", "model_family": "Bayesian Network", **metrics})

    # --- Competitor baselines ------------------------------------------------
    fitted_baselines = baselines.fit_all_baselines(train_df, seed=seed)
    baseline_probs = baselines.predict_all_baselines(fitted_baselines, holdout_df)
    for name, y_prob in baseline_probs.items():
        metrics = priors.classification_metrics(y_true, y_prob)
        rows_out.append({"seed": seed, "model": name, "model_family": "Discriminative baseline", **metrics})

    elapsed = time.time() - t0
    if verbose:
        summary = "  ".join(f"{r['model']}: F1={r['f1']:.3f} AUROC={r['auroc']:.3f}" for r in rows_out)
        print(f"  seed={seed:3d}  {summary}  ({elapsed:.1f}s)")

    return rows_out


def run_full_comparison(n_seeds=15, base_seed=600, rows=None, verbose=True):
    all_rows = []
    for i in range(n_seeds):
        seed = base_seed + i
        all_rows.extend(run_single_seed(seed, rows=rows, verbose=verbose))
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    print("Running full comparison: BN (3 priors) vs Logistic Regression vs Random Forest vs Naive Bayes")
    print("-" * 100)
    df = run_full_comparison(n_seeds=15)

    print("\nAggregate results by model (mean +/- std across seeds):")
    agg = df.groupby(["model_family", "model"])[["precision", "recall", "f1", "auroc"]].agg(["mean", "std"])
    print(agg)

    df.to_csv("outputs/full_comparison_results.csv", index=False)
    print("\nSaved -> outputs/full_comparison_results.csv")
