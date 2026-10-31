"""
evaluation.py
=============
Everything needed to compare a LEARNED Bayesian Network against the TRUE
ground-truth Bayesian Network it was learned from:

  1. Structural metrics: Structural Hamming Distance, edge precision/recall.
  2. Distributional metric: an unbiased KL-divergence ESTIMATE,

         KL(P_true || P_learned) ~= E_true[ log P_true(x) - log P_learned(x) ]

     estimated using a held-out sample x ~ P_true that was NEVER used for
     structure learning or parameter fitting. This avoids having to force
     the learned CPDs into the same node/parent shape as the true CPDs --
     both models are simply asked "how likely is this held-out row?" and we
     compare the answers on a genuinely common footing (the row itself).
"""

import itertools
import numpy as np

import config


# ---------------------------------------------------------------------------
# 1. Structural metrics
# ---------------------------------------------------------------------------
def structural_hamming_distance(true_edges, learned_edges):
    """
    Standard SHD over directed edges: counts additions, deletions, and
    reversals needed to turn `learned_edges` into `true_edges`.
    A reversal (A->B in one, B->A in the other) counts as ONE operation,
    not two.
    """
    true_set = set(true_edges)
    learned_set = set(learned_edges)

    missing = true_set - learned_set       # edges that should exist but don't
    extra = learned_set - true_set          # edges that exist but shouldn't

    reversals = set()
    for (a, b) in list(missing):
        if (b, a) in extra:
            reversals.add((a, b))

    missing -= reversals
    extra -= {(b, a) for (a, b) in reversals}

    return len(missing) + len(extra) + len(reversals)


def edge_precision_recall(true_edges, learned_edges):
    """Exact directed-edge precision/recall (a reversed edge counts as wrong)."""
    true_set = set(true_edges)
    learned_set = set(learned_edges)

    if len(learned_set) == 0:
        precision = 0.0
    else:
        precision = len(true_set & learned_set) / len(learned_set)

    if len(true_set) == 0:
        recall = 0.0
    else:
        recall = len(true_set & learned_set) / len(true_set)

    return precision, recall


# ---------------------------------------------------------------------------
# 2. Convert a learned pgmpy model into the same plain dict format used for
#    the ground-truth model, so ONE log-probability function can score both.
# ---------------------------------------------------------------------------
def pgmpy_model_to_dict(model, epsilon=1e-4):
    """
    Returns (parents, marginals, cpts) in exactly the format produced by
    ground_truth.build_ground_truth(), so evaluate_row_logprob() below can
    be reused unchanged for both the true and the learned model.

    `epsilon` provides Laplace-style smoothing for any state/parent-config
    that never appeared in the training data (so a rare held-out state
    never produces -inf log-probability).
    """
    parents = {}
    marginals = {}
    cpts = {}

    for node in config.NODES:
        k = config.STATE_COUNTS[node]
        cpd = model.get_cpds(node)
        evidence = cpd.variables[1:]  # parents, in the order pgmpy stores them
        parents[node] = list(evidence)

        values = cpd.get_values()  # shape (own_card_seen, n_parent_configs_seen)
        own_states_seen = cpd.state_names[node]
        state_index = {s: i for i, s in enumerate(own_states_seen)}

        if not evidence:
            # Root/isolated node -> build a full-length marginal vector,
            # filling any state that pgmpy never observed with epsilon.
            vec = np.full(k, epsilon)
            for s in range(k):
                if s in state_index:
                    vec[s] = values[state_index[s], 0]
            vec = vec / vec.sum()
            marginals[node] = vec
            continue

        # Non-root node: enumerate the FULL state space (config.STATE_COUNTS),
        # not just the configs pgmpy happened to see, and fill in epsilon
        # for anything unseen.
        parent_cards = [config.STATE_COUNTS[p] for p in evidence]
        seen_parent_states = [cpd.state_names[p] for p in evidence]
        seen_parent_index = {
            cfg: i for i, cfg in enumerate(itertools.product(*seen_parent_states))
        }

        node_cpt = {}
        for cfg in itertools.product(*[range(c) for c in parent_cards]):
            vec = np.full(k, epsilon)
            if cfg in seen_parent_index:
                col = seen_parent_index[cfg]
                for s in range(k):
                    if s in state_index:
                        vec[s] = values[state_index[s], col]
            vec = vec / vec.sum()
            node_cpt[cfg] = vec

        cpts[node] = node_cpt

    return parents, marginals, cpts


def row_log_prob(row: dict, parents: dict, marginals: dict, cpts: dict, epsilon=1e-4):
    """
    Compute log P(row) under a BN specified in the plain-dict format, by
    multiplying each node's local conditional probability (product rule
    factorisation -- same formula as Eq. 1 in the proposal).
    """
    log_p = 0.0
    for node in config.NODES:
        plist = parents[node]
        val = row[node]
        if not plist:
            probs = marginals[node]
        else:
            cfg = tuple(row[p] for p in plist)
            probs = cpts[node].get(cfg)
            if probs is None:
                k = config.STATE_COUNTS[node]
                probs = np.ones(k) / k
        p = max(probs[val], epsilon)
        log_p += np.log(p)
    return log_p


def estimate_kl_divergence(holdout_df, true_parents, true_marginals, true_cpts,
                            learned_parents, learned_marginals, learned_cpts):
    """
    KL(P_true || P_learned) ~= mean over holdout rows of
        [ log P_true(row) - log P_learned(row) ]

    This is a standard Monte-Carlo estimator: since the holdout rows are
    genuinely drawn from P_true, the sample average of the log-ratio
    converges to the true KL divergence. A value near 0 means the learned
    model explains held-out ground-truth data almost as well as the true
    model itself; larger values mean bigger loss of accuracy from imperfect
    structure/parameter recovery.
    """
    diffs = []
    for _, row in holdout_df.iterrows():
        row_d = row.to_dict()
        lp_true = row_log_prob(row_d, true_parents, true_marginals, true_cpts)
        lp_learned = row_log_prob(row_d, learned_parents, learned_marginals, learned_cpts)
        diffs.append(lp_true - lp_learned)
    diffs = np.array(diffs)
    return float(diffs.mean()), float(diffs.std())
