"""
rq2_experiment.py
==================
RQ2: "How does the choice of parameter-estimation method and Dirichlet
prior affect recovery of the true data-generating probability
distribution, and downstream fraud-detection performance, under different
levels of fraud imbalance?"

WORDING CORRECTION: earlier drafts described this as recovering "the true
CPT probabilities". That overstates what is actually measured. The KL
estimate computed here, KL(P_true || P_method), is evaluated over
COMPLETE observations using each model's full factorisation (see
evaluation.row_log_prob) -- it is a measure of distribution-level
recovery, not a direct cell-by-cell comparison of individual CPT table
entries. A direct CPT comparison is not always even well-defined here,
since the LEARNED structure can have different parents for a given node
than the TRUE structure does (e.g. if Hill Climbing finds `amount` as a
parent of `isFraud` when the true DAG uses `delta_dest` instead, the two
CPTs are indexed by different parent configurations entirely and cannot
be compared entry-by-entry). Distribution-level KL divergence sidesteps
that problem entirely by scoring whole rows, which is why it is the
metric used throughout this file.

REVISION NOTE: an earlier version of this experiment only compared the
three Dirichlet priors AGAINST EACH OTHER (specifically, against a
Jeffreys baseline) rather than directly against the known TRUE
distribution. That was flagged as the central gap: the research question
is about recovering the true distribution, so the evaluation needs to
measure distance from truth directly, not just relative differences
between priors. This version fixes that, and also adds MLE (no prior at
all) as a fourth comparison point, since "choice of parameter-estimation
method" in the research question implies MLE should be one of the options
being compared, not an implicit assumption baked into the main experiment
only.

For each of N seeds: build a ground truth, generate train/holdout data,
learn a structure (Hill Climbing + BIC, same as main.py), then fit that
SAME learned structure using FOUR parameter-estimation methods:

  - MLE            (no prior -- structure_learning.fit_learned_model)
  - Jeffreys       (alpha_v = 0.5, uniform)
  - Laplace        (alpha_v = 1.0, uniform)
  - Expert-informed (asymmetric prior on isFraud, centred on the
                     realised fraud rate)

and evaluate, for each of the four:

  1. KL(P_true || P_method) -- DIRECT comparison against the known ground
     truth, using the exact same evaluation.estimate_kl_divergence
     machinery as the main RQ1 experiment. This is the primary answer to
     "which method recovers the true probabilities most accurately?"
  2. Classification performance (precision, recall, F1, AUROC) on held-out
     data -- "does better parameter recovery translate into better fraud
     detection?"

Results are also reported split by `has_fraud_edge` (whether Hill
Climbing found any edge into isFraud at all for that seed), because when
no edge exists, ALL FOUR methods necessarily produce near-identical,
uninformative predictions (AUROC=0.5) -- conflating those seeds with
seeds that DO have structure would understate how much the parameter-
estimation method matters when there is something for it to act on.
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
import evaluation

METHOD_NAMES = ["mle", "jeffreys", "laplace", "expert"]


def _fit_method(method: str, train_df: pd.DataFrame, learned_edges: list):
    if method == "mle":
        return structure_learning.fit_learned_model(train_df, learned_edges, config.NODES)
    return priors.fit_with_prior(learned_edges, train_df, method)


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
    has_fraud_edge = any(e[1] == priors.TARGET_NODE for e in learned_edges)

    y_true = holdout_df[priors.TARGET_NODE].to_numpy()

    rows_out = []
    for method in METHOD_NAMES:
        model = _fit_method(method, train_df, learned_edges)

        # --- (1) Direct comparison against the TRUE CPTs ---------------
        m_parents, m_marg, m_cpts = evaluation.pgmpy_model_to_dict(model)
        kl_mean, kl_std = evaluation.estimate_kl_divergence(
            holdout_df,
            gt["parents"], gt["marginals"], gt["cpts"],
            m_parents, m_marg, m_cpts,
        )

        # --- (2) Downstream classification performance -----------------
        y_prob = priors.predict_fraud_probability(model, holdout_df)
        metrics = priors.classification_metrics(y_true, y_prob)

        rows_out.append({
            "seed": seed,
            "rows": rows,
            "method": method,
            "realised_fraud_rate": train_df[priors.TARGET_NODE].mean(),
            "has_fraud_edge": has_fraud_edge,
            "kl_vs_true": kl_mean,       # KL(P_true || P_method) -- DIRECT, absolute
            "kl_vs_true_std": kl_std,
            **metrics,
        })

    elapsed = time.time() - t0
    if verbose:
        summary = "  ".join(f"{r['method']}: KL={r['kl_vs_true']:.4f} F1={r['f1']:.3f} "
                             f"AUROC={r['auroc']:.3f}" for r in rows_out)
        print(f"  seed={seed:3d} rows={rows:6d} edge_to_fraud={has_fraud_edge}  {summary}  ({elapsed:.1f}s)")

    return rows_out


def run_rq2_experiment(n_seeds=15, base_seed=300, rows=None, verbose=True):
    all_rows = []
    for i in range(n_seeds):
        seed = base_seed + i
        all_rows.extend(run_single_seed(seed, rows=rows, verbose=verbose))
    return pd.DataFrame(all_rows)


def run_rows_sweep(row_sizes, seeds_per_size=5, base_seed=400, verbose=True):
    """Secondary experiment: how does KL(true||method) change as training
    set size shrinks? Directly tests the data-scarcity hypothesis."""
    all_rows = []
    seed_counter = base_seed
    for n_rows in row_sizes:
        for _ in range(seeds_per_size):
            seed = seed_counter
            seed_counter += 1
            all_rows.extend(run_single_seed(seed, rows=n_rows, verbose=verbose))
    return pd.DataFrame(all_rows)


def summarise(df: pd.DataFrame):
    """Prints the KL-vs-true and classification metrics by method, split by
    whether the seed had any learned edge into isFraud -- the split matters
    because seeds with no edge necessarily make all four methods look
    identical (AUROC=0.5), which would otherwise understate how much the
    method matters when there IS structure to parameterise."""
    print("\n--- All seeds ---")
    print(df.groupby("method")[["kl_vs_true", "precision", "recall", "f1", "auroc"]].agg(["mean", "std"]))

    for has_edge in [True, False]:
        subset = df[df["has_fraud_edge"] == has_edge]
        if len(subset) == 0:
            continue
        label = "seeds WITH a learned edge into isFraud" if has_edge else "seeds WITH NO learned edge into isFraud"
        print(f"\n--- Split: {label} (n={subset['seed'].nunique()} seeds) ---")
        print(subset.groupby("method")[["kl_vs_true", "precision", "recall", "f1", "auroc"]].agg(["mean", "std"]))


if __name__ == "__main__":
    print("Running RQ2 experiment (a): default row count, MLE vs Jeffreys vs Laplace vs Expert, direct KL-vs-truth")
    print("-" * 110)
    df_main = run_rq2_experiment(n_seeds=15)
    summarise(df_main)

    df_main.to_csv("outputs/rq2_prior_comparison_results.csv", index=False)
    print("\nSaved -> outputs/rq2_prior_comparison_results.csv")

    print("\n" + "-" * 110)
    print("Running RQ2 experiment (b): row-count sweep, testing the data-scarcity hypothesis")
    print("-" * 110)
    df_sweep = run_rows_sweep(row_sizes=[500, 2000, 8000, 20000], seeds_per_size=5)

    print("\nMean KL(true||method) by row count:")
    print(df_sweep.groupby(["rows", "method"])["kl_vs_true"].agg(["mean", "std"]))

    df_sweep.to_csv("outputs/rq2_rows_sweep_results.csv", index=False)
    print("\nSaved -> outputs/rq2_rows_sweep_results.csv")
