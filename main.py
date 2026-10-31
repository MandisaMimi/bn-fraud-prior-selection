"""
main.py
=======
Entry point. Runs the full pipeline described in the design discussion:

  Stage 1  Ground-truth structure generation      (ground_truth.py)
  Stage 2  Ground-truth parameter generation       (ground_truth.py)
  Stage 3  Dataset generation (ancestral sampling) (data_generation.py)
  Stage 4  Structure learning (Hill Climbing/BIC)  (structure_learning.py)
  Stage 5  Evaluation (SHD, precision/recall, KL)  (evaluation.py)
  Stage 6  Secondary fraud-rate sweep experiment
  Stage 7  Save example figures + aggregate results (visualize.py)

Run with:  python main.py
All tunable settings live in config.py -- nothing here should need editing
for a normal re-run.
"""

# ---------------------------------------------------------------------------
# REPRODUCIBILITY GUARD -- must run before anything else in this file.
#
# Python randomises its internal string-hash seed by default, which changes
# set/dict iteration order between runs. pgmpy's Hill Climbing search uses
# hash-ordered collections internally when there are BIC-score TIES between
# candidate edge changes, so without a fixed hash seed, two runs of the
# EXACT same code on the EXACT same data can silently learn different
# structures. Setting PYTHONHASHSEED must happen before the interpreter
# starts, so if it isn't already fixed, this block re-launches the script
# once with it fixed, so `python main.py` always gives you bit-identical
# results without you having to remember a flag.
#
# This uses subprocess (NOT os.exec*) specifically because os.exec* is a
# POSIX process-replacement trick that behaves unreliably on Windows (it
# can detach the child's console output, making the script look like it
# silently did nothing). subprocess.run() with inherited stdio works the
# same way on Windows, macOS, and Linux.
# ---------------------------------------------------------------------------
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    import subprocess
    print("Re-launching with a fixed hash seed (PYTHONHASHSEED=0) for reproducible results...")
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    result = subprocess.run([sys.executable] + sys.argv, env=env)
    sys.exit(result.returncode)

import time
import random
import numpy as np
import pandas as pd

import config
import ground_truth
import data_generation
import structure_learning
import evaluation
import visualize
import reporting
import cross_distribution
import priors
import rq2_experiment
import full_comparison
import edge_recovery_analysis


def run_single_seed(seed: int, fraud_rate=None, rows=None, verbose=True):
    """
    Executes Stages 1-5 for ONE seed and returns a dict of results plus the
    raw true/learned edge lists (needed later for the example figure).
    """
    rows = rows or config.ROWS_PER_DATASET
    t0 = time.time()

    # Reset BOTH the global numpy and python random states for this seed.
    # Our own sampling only ever uses the explicit np.random.RandomState
    # objects created below, but resetting the global state too is a cheap
    # extra guarantee of bit-identical reruns given the pinned library
    # versions in requirements.txt.
    random.seed(seed)
    np.random.seed(seed)

    # --- Stage 1 & 2: ground truth structure + parameters -----------------
    gt = ground_truth.build_ground_truth(seed, target_fraud_rate=fraud_rate)
    true_edges = [(p, n) for n, plist in gt["parents"].items() for p in plist]

    # --- Stage 3: generate training data + a separate held-out set --------
    data_rng = np.random.RandomState(seed + 10_000)  # distinct stream from structure/CPT rng
    train_df = data_generation.generate_dataset(gt, rows, data_rng)
    holdout_df = data_generation.generate_dataset(gt, config.HOLDOUT_ROWS, data_rng)

    realized_fraud_rate = train_df["isFraud"].mean()

    # --- Stage 4: learn structure + fit parameters -------------------------
    learned_edges = structure_learning.learn_structure(train_df)
    learned_model = structure_learning.fit_learned_model(train_df, learned_edges, config.NODES)

    # --- Stage 5: evaluation -------------------------------------------------
    shd = evaluation.structural_hamming_distance(true_edges, learned_edges)
    precision, recall = evaluation.edge_precision_recall(true_edges, learned_edges)

    learned_parents, learned_marginals, learned_cpts = evaluation.pgmpy_model_to_dict(learned_model)
    kl_mean, kl_std = evaluation.estimate_kl_divergence(
        holdout_df,
        gt["parents"], gt["marginals"], gt["cpts"],
        learned_parents, learned_marginals, learned_cpts,
    )

    elapsed = time.time() - t0
    if verbose:
        print(f"  seed={seed:3d}  fraud_rate(target/realised)={fraud_rate or config.TARGET_FRAUD_RATE:.4f}/{realized_fraud_rate:.4f}  "
              f"SHD={shd:2d}  P={precision:.2f}  R={recall:.2f}  KL~={kl_mean:.3f}  ({elapsed:.1f}s)")

    return {
        "seed": seed,
        "target_fraud_rate": fraud_rate or config.TARGET_FRAUD_RATE,
        "realized_fraud_rate": realized_fraud_rate,
        "n_true_edges": len(true_edges),
        "n_learned_edges": len(learned_edges),
        "shd": shd,
        "precision": precision,
        "recall": recall,
        "kl_estimate": kl_mean,
        "kl_estimate_std": kl_std,
        "elapsed_seconds": elapsed,
        "_true_edges": true_edges,
        "_learned_edges": learned_edges,
        "_ground_truth": gt,
        "_learned_model": learned_model,
        "_train_df": train_df,
        "_holdout_df": holdout_df,
    }


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # =========================================================================
    # STAGES 1-5: main multi-seed experiment
    # =========================================================================
    print(f"Running main experiment: N_SEEDS={config.N_SEEDS}, "
          f"ROWS_PER_DATASET={config.ROWS_PER_DATASET}, target fraud rate={config.TARGET_FRAUD_RATE}")
    print("-" * 100)

    results = []
    for i in range(config.N_SEEDS):
        seed = config.RANDOM_SEED_BASE + i
        results.append(run_single_seed(seed))

    results_df = pd.DataFrame(results).drop(
        columns=["_true_edges", "_learned_edges", "_ground_truth", "_learned_model", "_train_df", "_holdout_df"]
    )
    results_df.to_csv(config.RESULTS_CSV, index=False)
    print(f"\nSaved per-seed results -> {config.RESULTS_CSV}")

    print("\nAggregate results across all seeds (mean +/- std):")
    for col in ["shd", "precision", "recall", "kl_estimate"]:
        print(f"  {col:12s}: {results_df[col].mean():.3f} +/- {results_df[col].std():.3f}")

    # -------------------------------------------------------------------
    # Save the ACTUAL generated dataset for every main-experiment seed,
    # so the synthetic data itself (not just summary metrics) is available.
    # -------------------------------------------------------------------
    os.makedirs(config.DATASETS_DIR, exist_ok=True)
    for r in results:
        seed = r["seed"]
        r["_train_df"].to_csv(os.path.join(config.DATASETS_DIR, f"seed_{seed}_train.csv"), index=False)
    # Also keep the held-out set for the example seed (used for the CPT/KL demo below)
    example = results[0]
    example["_holdout_df"].to_csv(
        os.path.join(config.DATASETS_DIR, f"seed_{example['seed']}_holdout.csv"), index=False
    )
    print(f"Saved {len(results)} generated datasets -> {config.DATASETS_DIR}/")

    # -------------------------------------------------------------------
    # Example seed: two large, separate DAG pictures + full CPTs for both
    # networks, so each picture has a matching "complete" (structure +
    # parameters) text file next to it.
    # -------------------------------------------------------------------
    os.makedirs(config.CPTS_DIR, exist_ok=True)
    true_fig_path = os.path.join(config.FIGURES_DIR, "ground_truth_network.png")
    learned_fig_path = os.path.join(config.FIGURES_DIR, "learned_network.png")
    visualize.plot_example_comparison(
        example["_true_edges"], example["_learned_edges"], example["seed"],
        true_fig_path, learned_fig_path,
    )
    print(f"Saved ground-truth network picture -> {true_fig_path}")
    print(f"Saved learned network picture -> {learned_fig_path}")

    pipeline_diagram_path = os.path.join(config.FIGURES_DIR, "ground_truth_pipeline_diagram.png")
    visualize.plot_ground_truth_pipeline_diagram(pipeline_diagram_path)
    print(f"Saved ground-truth pipeline diagram -> {pipeline_diagram_path}")

    true_cpt_path = os.path.join(config.CPTS_DIR, f"seed_{example['seed']}_ground_truth_cpts.txt")
    learned_cpt_path = os.path.join(config.CPTS_DIR, f"seed_{example['seed']}_learned_cpts.txt")
    reporting.write_true_cpts(example["_ground_truth"], true_cpt_path)
    reporting.write_learned_cpts(example["_learned_model"], learned_cpt_path, example["seed"])
    print(f"Saved ground-truth CPTs -> {true_cpt_path}")
    print(f"Saved learned CPTs -> {learned_cpt_path}")

    # -------------------------------------------------------------------
    # THE "complete network" files: diagram + full CPTs in ONE PDF each,
    # so there's no separate file to miss. This is what to look at if you
    # want the graph and its probability tables together.
    # -------------------------------------------------------------------
    true_complete_path = os.path.join(config.FIGURES_DIR, f"seed_{example['seed']}_ground_truth_COMPLETE.pdf")
    learned_complete_path = os.path.join(config.FIGURES_DIR, f"seed_{example['seed']}_learned_COMPLETE.pdf")
    visualize.plot_network_complete(
        example["_true_edges"], config.NODES,
        f"Ground-truth Bayesian Network structure (seed {example['seed']})",
        reporting._true_cpt_lines(example["_ground_truth"]),
        true_complete_path,
    )
    visualize.plot_network_complete(
        example["_learned_edges"], config.NODES,
        f"Learned Bayesian Network structure via Hill Climbing + BIC (seed {example['seed']})",
        reporting._learned_cpt_lines(example["_learned_model"], example["seed"]),
        learned_complete_path,
    )
    print(f"Saved COMPLETE ground-truth network (diagram + full CPTs) -> {true_complete_path}")
    print(f"Saved COMPLETE learned network (diagram + full CPTs) -> {learned_complete_path}")

    agg_fig_path = os.path.join(config.FIGURES_DIR, "aggregate_recovery_metrics.png")
    visualize.plot_aggregate_results(results_df, agg_fig_path)
    print(f"Saved aggregate metrics figure -> {agg_fig_path}")

    # =========================================================================
    # STAGE 6: secondary experiment -- fraud-rate sweep
    # =========================================================================
    print("\n" + "-" * 100)
    print(f"Running fraud-rate sweep: rates={config.SWEEP_FRAUD_RATES}, "
          f"{config.SWEEP_SEEDS_PER_RATE} seeds each")
    print("-" * 100)

    sweep_results = []
    seed_counter = 90_000  # separate seed space so it never collides with the main experiment
    for rate in config.SWEEP_FRAUD_RATES:
        for j in range(config.SWEEP_SEEDS_PER_RATE):
            seed = seed_counter
            seed_counter += 1
            r = run_single_seed(seed, fraud_rate=rate)
            r["fraud_rate"] = rate
            sweep_results.append(r)

    sweep_df = pd.DataFrame(sweep_results).drop(
        columns=["_true_edges", "_learned_edges", "_ground_truth", "_learned_model", "_train_df", "_holdout_df"]
    )
    sweep_df.to_csv(config.SWEEP_CSV, index=False)
    print(f"\nSaved fraud-rate sweep results -> {config.SWEEP_CSV}")

    sweep_fig_path = os.path.join(config.FIGURES_DIR, "fraud_rate_sweep.png")
    visualize.plot_fraud_rate_sweep(sweep_df, sweep_fig_path)
    print(f"Saved fraud-rate sweep figure -> {sweep_fig_path}")

    # =========================================================================
    # STAGE 7: cross-distribution validation
    # =========================================================================
    # Addresses supervisor feedback directly: shows KL divergence between two
    # GENUINELY DIFFERENT distributions, not just P compared against its own
    # (imperfect) recovery. See cross_distribution.py's module docstring and
    # the "Cross-distribution validation" section of README.md.
    print("\n" + "-" * 100)
    print("Running cross-distribution validation (KL(P||P_hat) vs KL(P||Q))")
    print("-" * 100)

    cross_df = cross_distribution.run_cross_distribution_experiment(n_pairs=10)
    cross_csv_path = os.path.join(config.OUTPUT_DIR, "cross_distribution_results.csv")
    cross_df.to_csv(cross_csv_path, index=False)
    print(f"\nSaved cross-distribution results -> {cross_csv_path}")

    print("\nSummary (mean +/- std across all pairs):")
    print(f"  KL(P||P_hat) [same distribution]:      "
          f"{cross_df['kl_same_distribution'].mean():.3f} +/- {cross_df['kl_same_distribution'].std():.3f}")
    print(f"  KL(P||Q)     [different distribution]: "
          f"{cross_df['kl_different_distribution'].mean():.3f} +/- {cross_df['kl_different_distribution'].std():.3f}")
    print(f"  Mean ratio (different / same): {cross_df['ratio'].mean():.1f}x")

    cross_fig_path = os.path.join(config.FIGURES_DIR, "cross_distribution_comparison.png")
    visualize.plot_cross_distribution_comparison(cross_df, cross_fig_path)
    print(f"Saved cross-distribution comparison figure -> {cross_fig_path}")

    # =========================================================================
    # STAGE 8: RQ2 -- Dirichlet prior comparison
    # =========================================================================
    print("\n" + "-" * 100)
    print("Running RQ2: Jeffreys / Laplace / Expert-informed prior comparison")
    print("-" * 100)

    rq2_df = rq2_experiment.run_rq2_experiment(n_seeds=15)
    rq2_csv_path = os.path.join(config.OUTPUT_DIR, "rq2_prior_comparison_results.csv")
    rq2_df.to_csv(rq2_csv_path, index=False)
    print(f"\nSaved RQ2 results -> {rq2_csv_path}")

    rq2_sweep_df = rq2_experiment.run_rows_sweep(row_sizes=[500, 2000, 8000, 20000], seeds_per_size=5)
    rq2_sweep_csv_path = os.path.join(config.OUTPUT_DIR, "rq2_rows_sweep_results.csv")
    rq2_sweep_df.to_csv(rq2_sweep_csv_path, index=False)
    print(f"Saved RQ2 data-scarcity sweep -> {rq2_sweep_csv_path}")

    rq2_fig_path = os.path.join(config.FIGURES_DIR, "rq2_prior_comparison.png")
    visualize.plot_rq2_comparison(rq2_df, rq2_sweep_df, rq2_fig_path)
    print(f"Saved RQ2 comparison figure -> {rq2_fig_path}")

    # =========================================================================
    # STAGE 9: full comparison -- BN (3 priors) vs discriminative baselines
    # =========================================================================
    print("\n" + "-" * 100)
    print("Running full comparison: BN vs Logistic Regression / Random Forest / Naive Bayes")
    print("-" * 100)

    full_df = full_comparison.run_full_comparison(n_seeds=15)
    full_csv_path = os.path.join(config.OUTPUT_DIR, "full_comparison_results.csv")
    full_df.to_csv(full_csv_path, index=False)
    print(f"\nSaved full comparison results -> {full_csv_path}")

    full_fig_path = os.path.join(config.FIGURES_DIR, "full_comparison.png")
    visualize.plot_full_comparison(full_df, full_fig_path)
    print(f"Saved full comparison figure -> {full_fig_path}")

    # =========================================================================
    # STAGE 10: edge-level recovery and isFraud-specific structure analysis
    # =========================================================================
    print("\n" + "-" * 100)
    print("Running edge-recovery and isFraud-specific structure analysis")
    print("-" * 100)

    edge_table, fraud_split_df, seed_results = edge_recovery_analysis.run_full_analysis(n_seeds=15, base_seed=42)

    edge_table_path = os.path.join(config.OUTPUT_DIR, "edge_recovery_table.csv")
    edge_table.to_csv(edge_table_path, index=False)
    print(f"\nSaved edge recovery table -> {edge_table_path}")

    fraud_split_path = os.path.join(config.OUTPUT_DIR, "fraud_edge_downstream_comparison.csv")
    fraud_split_df.to_csv(fraud_split_path, index=False)
    print(f"Saved fraud-edge downstream comparison -> {fraud_split_path}")

    n_recovered = sum(1 for r in seed_results if r["recovered_true_fraud_parent"])
    print(f"\nSeeds where a TRUE fraud parent was recovered: {n_recovered}/{len(seed_results)}")

    print("\nDone. All outputs are in the 'outputs/' folder.")


if __name__ == "__main__":
    main()
