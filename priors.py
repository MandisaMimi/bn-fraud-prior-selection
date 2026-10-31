"""
priors.py
=========
Implements RQ2: given the structure G* learned in the main experiment,
parameterise it under three Dirichlet prior configurations and evaluate
fraud-detection performance under each (precision, recall, F1, AUROC),
per Section 3.3 / Table 3 of the proposal:

    (i)   Jeffreys prior         alpha_v = 0.5   (every node)
    (ii)  Laplace prior          alpha_v = 1.0   (every node)
    (iii) Expert-informed prior  alpha_v = 0.5   (every node except isFraud)
                                  isFraud centred on config.TARGET_FRAUD_RATE,
                                  a FIXED, externally-known rate (analogous
                                  to citing a published fraud base rate),
                                  NOT the realised rate of whatever specific
                                  training sample is being fit on -- see the
                                  correction note in fit_with_prior below.

All data stays as plain integers throughout (never cast to str) -- this is
what evaluation.py already does and is what avoids the int/str state-space
bug that silently doubles a variable's state space if the two get mixed
(each state becomes two "different" states, e.g. int 0 and str '0').
"""

import itertools
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.inference import VariableElimination

import config

TARGET_NODE = "isFraud"

# Total Dirichlet concentration for the expert-informed prior on isFraud,
# analogous to config.FRAUD_BETA_CONCENTRATION used to build the ground
# truth itself. "50" matches the "alpha_fraud = 50 x base rate" language
# in Table 3 of the proposal, interpreted as the prior's total strength
# (a + b in Beta terms), NOT as a symmetric per-state alpha -- see the
# bug note in _pseudo_counts_for below for why that distinction matters.
EXPERT_CONCENTRATION = 50.0

PRIOR_CONFIGS = {
    "jeffreys": {"alpha": 0.5},
    "laplace": {"alpha": 1.0},
    # "expert" is handled specially below -- alpha_fraud depends on the
    # realised fraud rate of the specific dataset being fit on.
}


def _pseudo_counts_for(model: DiscreteBayesianNetwork, alpha_by_node: dict,
                        expert_fraud_rate: float = None) -> dict:
    """
    Builds the pseudo_counts dict pgmpy's DiscreteBayesianEstimator expects:
    {node: array of shape (n_states, n_parent_configs)}, using the FULL
    state space from config.STATE_COUNTS (not just what a given training
    sample happens to contain), so every possible state gets a non-zero
    prior mass and inference can never crash on a genuinely unseen state.

    IMPORTANT CORRECTION: for the expert-informed prior, isFraud's pseudo
    counts must be ASYMMETRIC across its two states (legitimate vs fraud),
    not a single alpha value applied uniformly to both. A uniform alpha
    only changes the overall smoothing STRENGTH -- it does not pull the
    posterior mean toward the target fraud rate at all, so it fails to
    encode "expert believes fraud is rare" in any meaningful way (this
    was the actual reason the three priors were producing near-identical
    predictions in early testing). The fix mirrors exactly how
    ground_truth.py itself builds isFraud's true CPT: a Beta(a, b) with
    mean = target fraud rate and concentration = a + b. Here:
        alpha_legit = EXPERT_CONCENTRATION * (1 - expert_fraud_rate)
        alpha_fraud = EXPERT_CONCENTRATION * expert_fraud_rate
    which gives a Dirichlet whose MEAN is the target fraud rate, not 0.5.
    """
    pseudo_counts = {}
    for node in model.nodes():
        n_states = config.STATE_COUNTS[node]
        parents = list(model.get_parents(node))
        n_parent_configs = 1
        for p in parents:
            n_parent_configs *= config.STATE_COUNTS[p]

        if node == TARGET_NODE and expert_fraud_rate is not None:
            rate = min(max(expert_fraud_rate, 1e-4), 1 - 1e-4)
            alpha_legit = EXPERT_CONCENTRATION * (1 - rate)
            alpha_fraud = EXPERT_CONCENTRATION * rate
            col = np.array([alpha_legit, alpha_fraud])  # shape (2,)
            pseudo_counts[node] = np.tile(col.reshape(-1, 1), (1, n_parent_configs))
            continue

        alpha = alpha_by_node.get(node, 0.5)
        pseudo_counts[node] = np.full((n_states, n_parent_configs), alpha)
    return pseudo_counts


def _full_state_names(nodes) -> dict:
    """Explicit state space for every node, as plain ints -- matches
    evaluation.pgmpy_model_to_dict's convention of using config.STATE_COUNTS
    as the ground truth for how many states a node has."""
    return {n: list(range(config.STATE_COUNTS[n])) for n in nodes}


def fit_with_prior(learned_edges: list, train_df: pd.DataFrame, prior_name: str,
                    all_nodes=None) -> DiscreteBayesianNetwork:
    """
    Fits the structure `learned_edges` (from structure_learning.learn_structure)
    with one of the three Dirichlet prior configurations. Returns a fitted
    DiscreteBayesianNetwork ready for inference.
    """
    all_nodes = all_nodes or config.NODES
    model = DiscreteBayesianNetwork(learned_edges)
    for n in all_nodes:
        if n not in model.nodes():
            model.add_node(n)

    expert_fraud_rate = None
    if prior_name in PRIOR_CONFIGS:
        alpha = PRIOR_CONFIGS[prior_name]["alpha"]
        alpha_by_node = {n: alpha for n in all_nodes}
    elif prior_name == "expert":
        # IMPORTANT CORRECTION: this must be a FIXED, externally-known rate
        # (analogous to citing a published fraud base rate, e.g. Nassiri et
        # al. 2024 -- exactly what Section 3.3 of the proposal describes:
        # "alpha_v values specified according to published fraud base
        # rates"), NOT the realised rate of the specific training sample
        # being fit on. Using train_df's own mean here would be circular --
        # the "expert belief" would just be restating whatever the current
        # data happens to show, which isn't an independent prior at all.
        # config.TARGET_FRAUD_RATE is the same fixed constant used to build
        # the ground truth itself, so it plays the role of "the external,
        # known true base rate an expert would cite" without ever looking
        # at the specific training draw.
        expert_fraud_rate = config.TARGET_FRAUD_RATE
        alpha_by_node = {n: 0.5 for n in all_nodes}  # non-target nodes: same as Jeffreys
    else:
        raise ValueError(f"Unknown prior_name: {prior_name}")

    pseudo_counts = _pseudo_counts_for(model, alpha_by_node, expert_fraud_rate)
    state_names = _full_state_names(all_nodes)
    estimator = DiscreteBayesianEstimator(
        state_names=state_names, prior_type="dirichlet", pseudo_counts=pseudo_counts
    )
    model.fit(train_df, estimator=estimator)
    return model


def predict_fraud_probability(model: DiscreteBayesianNetwork, evidence_df: pd.DataFrame,
                                target=TARGET_NODE) -> np.ndarray:
    """
    Computes P(target=1 | evidence) for every row in evidence_df, using
    exact inference (Variable Elimination). Deduplicates identical evidence
    rows first (discretised data has heavy repetition, so this is exact,
    not approximate -- duplicate rows genuinely get the same prediction).
    """
    feature_cols = [c for c in evidence_df.columns if c != target]
    infer = VariableElimination(model)

    unique_evidence = evidence_df[feature_cols].drop_duplicates()
    prob_lookup = {}
    for _, row in unique_evidence.iterrows():
        evidence = {c: int(row[c]) for c in feature_cols}
        result = infer.query(variables=[target], evidence=evidence, show_progress=False)
        p1 = float(result.values[1])  # P(target = 1 | evidence)
        prob_lookup[tuple(row.values)] = p1

    return evidence_df[feature_cols].apply(
        lambda r: prob_lookup[tuple(r.values)], axis=1
    ).to_numpy()


def excess_kl_vs_baseline(model_a: DiscreteBayesianNetwork, model_b: DiscreteBayesianNetwork,
                           holdout_df: pd.DataFrame) -> tuple:
    """
    Estimates KL(P_true || model_b) - KL(P_true || model_a), using held-out
    data sampled from the TRUE ground-truth distribution (NOT from model_a
    or model_b themselves).

    IMPORTANT: because the holdout rows come from P_true rather than from
    model_a, the quantity mean(log P_a(x) - log P_b(x)) over these rows is
    NOT a raw pairwise KL(model_a || model_b) -- that would require samples
    drawn from model_a. What it actually estimates, by the standard
    cross-entropy identity E_{x~P}[log Q(x)] = -H(P) - KL(P||Q), is:

        KL(P_true || model_b) - KL(P_true || model_a)

    i.e. "how much MORE divergent from the true distribution is model_b
    than model_a" -- a positive value means model_b fits the true
    generating distribution WORSE than model_a; negative means model_b
    fits BETTER. This is exactly the comparison RQ2 needs (which prior
    configuration's posterior is closer to the true data-generating
    distribution), and it is a more informative quantity here than a raw
    KL(model_a||model_b) would have been, since it grounds both priors
    against the same known ground truth rather than against each other.
    """
    import evaluation

    parents_a, marg_a, cpts_a = evaluation.pgmpy_model_to_dict(model_a)
    parents_b, marg_b, cpts_b = evaluation.pgmpy_model_to_dict(model_b)

    diffs = []
    for _, row in holdout_df.iterrows():
        row_d = row.to_dict()
        lp_a = evaluation.row_log_prob(row_d, parents_a, marg_a, cpts_a)
        lp_b = evaluation.row_log_prob(row_d, parents_b, marg_b, cpts_b)
        diffs.append(lp_a - lp_b)
    diffs = np.array(diffs)
    return float(diffs.mean()), float(diffs.std())


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold=0.5) -> dict:
    """
    Precision, recall, F1, AUROC given true labels and predicted P(fraud=1).

    Reports metrics at BOTH the fixed 0.5 threshold AND the F1-optimal
    threshold (found via the precision-recall curve). This matters here
    specifically: the ground-truth data generator (config.py) draws
    isFraud's probability from a Beta distribution centred on the target
    rate for EVERY parent configuration, with fairly low variance across
    configurations -- so predicted probabilities can legitimately never
    cross 0.5 for any row even when the model has learned real, useful
    signal (visible in AUROC). This is a property of the synthetic ground
    truth, not a bug, and it's exactly why the proposal specifies picking
    the prior that "maximises F1-score" rather than fixing a threshold.
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        precision_recall_curve, accuracy_score,
    )

    y_pred_default = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy_at_0.5": accuracy_score(y_true, y_pred_default),
        "precision_at_0.5": precision_score(y_true, y_pred_default, zero_division=0),
        "recall_at_0.5": recall_score(y_true, y_pred_default, zero_division=0),
        "f1_at_0.5": f1_score(y_true, y_pred_default, zero_division=0),
    }

    # F1-optimal threshold via the precision-recall curve
    if len(np.unique(y_true)) > 1:
        prec_curve, rec_curve, thresholds = precision_recall_curve(y_true, y_prob)
        f1_curve = np.where(
            (prec_curve + rec_curve) > 0,
            2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-12),
            0.0,
        )
        best_idx = int(np.argmax(f1_curve[:-1])) if len(f1_curve) > 1 else 0
        best_threshold = thresholds[best_idx] if len(thresholds) > 0 else threshold
        y_pred_best = (y_prob >= best_threshold).astype(int)
        metrics["best_threshold"] = float(best_threshold)
        metrics["accuracy"] = accuracy_score(y_true, y_pred_best)
        metrics["precision"] = precision_score(y_true, y_pred_best, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred_best, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred_best, zero_division=0)
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["best_threshold"] = float("nan")
        metrics["accuracy"] = accuracy_score(y_true, y_pred_default)
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0
        metrics["auroc"] = float("nan")

    return metrics
