"""
edge_recovery_analysis.py
==========================
Implements two of the reviewer's concrete recommendations:

1. EDGE-LEVEL RECOVERY: rather than only reporting aggregate precision/
   recall, this tracks, for every distinct directed (parent, child) pair
   that appears as a TRUE edge in at least one of the 15 main-experiment
   seeds, how many seeds had it as a true edge and how many of those Hill
   Climbing successfully recovered. This turns "69.8% precision" into
   "which specific relationships does Hill Climbing find reliably, and
   which does it struggle with" -- a genuine "why", not just a number.

2. isFraud-SPECIFIC STRUCTURE ANALYSIS: for each seed, records whether
   Hill Climbing recovered at least one of isFraud's TRUE parents, and
   separately whether it learned ANY edge into isFraud at all (which can
   include false-positive edges that don't match the true parents). Then
   re-fits all four RQ2 parameter-estimation methods and reports downstream
   KL/precision/recall/F1/AUROC split by whether a true fraud-relevant
   dependency was actually recovered. This directly tests the reviewer's
   proposed finding: "downstream predictive performance depends strongly
   on whether structure learning successfully identifies dependencies
   involving the fraud node."
"""

import time
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from collections import defaultdict

import config
import ground_truth
import data_generation
import structure_learning
import evaluation
import priors

TARGET = "isFraud"


def run_single_seed_detailed(seed: int, rows=None):
    rows = rows or config.ROWS_PER_DATASET

    random.seed(seed)
    np.random.seed(seed)

    gt = ground_truth.build_ground_truth(seed)
    true_edges = [(p, n) for n, plist in gt["parents"].items() for p in plist]
    true_fraud_parents = set(gt["parents"].get(TARGET, []))

    data_rng = np.random.RandomState(seed + 10_000)
    train_df = data_generation.generate_dataset(gt, rows, data_rng)
    holdout_df = data_generation.generate_dataset(gt, config.HOLDOUT_ROWS, data_rng)

    learned_edges = structure_learning.learn_structure(train_df)
    learned_edges_set = set(learned_edges)
    learned_fraud_parents = {p for (p, c) in learned_edges if c == TARGET}

    recovered_true_fraud_parent = len(true_fraud_parents & learned_fraud_parents) > 0
    any_learned_fraud_edge = len(learned_fraud_parents) > 0

    return {
        "seed": seed,
        "true_edges": true_edges,
        "learned_edges_set": learned_edges_set,
        "true_fraud_parents": true_fraud_parents,
        "learned_fraud_parents": learned_fraud_parents,
        "recovered_true_fraud_parent": recovered_true_fraud_parent,
        "any_learned_fraud_edge": any_learned_fraud_edge,
        "train_df": train_df,
        "holdout_df": holdout_df,
        "learned_edges_list": learned_edges,
    }


def edge_type_recovery_table(seed_results: list) -> pd.DataFrame:
    """For every distinct true (parent, child) pair seen across all seeds,
    reports how many seeds had it as a true edge and how many were
    recovered."""
    true_count = defaultdict(int)
    recovered_count = defaultdict(int)

    for res in seed_results:
        for edge in res["true_edges"]:
            true_count[edge] += 1
            if edge in res["learned_edges_set"]:
                recovered_count[edge] += 1

    rows = []
    for edge, n_true in true_count.items():
        rows.append({
            "parent": edge[0],
            "child": edge[1],
            "n_seeds_true": n_true,
            "n_seeds_recovered": recovered_count[edge],
            "recovery_rate": recovered_count[edge] / n_true,
        })

    df = pd.DataFrame(rows).sort_values(["child", "recovery_rate"], ascending=[True, False])
    return df.reset_index(drop=True)


def fraud_edge_downstream_comparison(seed_results: list) -> pd.DataFrame:
    """Re-fits all four parameter-estimation methods for every seed, and
    reports downstream metrics split by whether a TRUE fraud parent was
    recovered for that seed."""
    rows_out = []
    for res in seed_results:
        train_df = res["train_df"]
        holdout_df = res["holdout_df"]
        learned_edges = res["learned_edges_list"]

        y_true = holdout_df[TARGET].to_numpy()

        for method in ["mle", "jeffreys", "laplace", "expert"]:
            if method == "mle":
                model = structure_learning.fit_learned_model(train_df, learned_edges, config.NODES)
            else:
                model = priors.fit_with_prior(learned_edges, train_df, method)

            y_prob = priors.predict_fraud_probability(model, holdout_df)
            metrics = priors.classification_metrics(y_true, y_prob)

            rows_out.append({
                "seed": res["seed"],
                "method": method,
                "recovered_true_fraud_parent": res["recovered_true_fraud_parent"],
                "any_learned_fraud_edge": res["any_learned_fraud_edge"],
                **metrics,
            })

    return pd.DataFrame(rows_out)


def run_full_analysis(n_seeds=15, base_seed=42, verbose=True):
    """Reuses the exact same seeds as the main RQ1 experiment (42..56)."""
    seed_results = []
    for i in range(n_seeds):
        seed = base_seed + i
        t0 = time.time()
        res = run_single_seed_detailed(seed)
        if verbose:
            print(f"  seed={seed:3d}  true_fraud_parents={sorted(res['true_fraud_parents'])}  "
                  f"learned_fraud_parents={sorted(res['learned_fraud_parents'])}  "
                  f"recovered_true={res['recovered_true_fraud_parent']}  ({time.time()-t0:.1f}s)")
        seed_results.append(res)

    edge_table = edge_type_recovery_table(seed_results)
    fraud_split_df = fraud_edge_downstream_comparison(seed_results)

    return edge_table, fraud_split_df, seed_results


if __name__ == "__main__":
    print("Running edge-recovery and isFraud-specific structure analysis (15 seeds)")
    print("-" * 100)
    edge_table, fraud_split_df, seed_results = run_full_analysis(n_seeds=15)

    print("\n--- Edge-type recovery table ---")
    print(edge_table.to_string(index=False))
    edge_table.to_csv("outputs/edge_recovery_table.csv", index=False)
    print("\nSaved -> outputs/edge_recovery_table.csv")

    n_recovered = sum(1 for r in seed_results if r["recovered_true_fraud_parent"])
    n_any = sum(1 for r in seed_results if r["any_learned_fraud_edge"])
    print(f"\nSeeds where a TRUE fraud parent was recovered: {n_recovered}/{len(seed_results)}")
    print(f"Seeds where ANY edge into isFraud was learned (true or not): {n_any}/{len(seed_results)}")

    print("\n--- Downstream metrics split by whether a true fraud parent was recovered ---")
    summary = fraud_split_df.groupby(["recovered_true_fraud_parent", "method"])[
        ["precision", "recall", "f1", "auroc"]
    ].mean()
    print(summary)

    fraud_split_df.to_csv("outputs/fraud_edge_downstream_comparison.csv", index=False)
    print("\nSaved -> outputs/fraud_edge_downstream_comparison.csv")
