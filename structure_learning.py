"""
structure_learning.py
======================
Wraps pgmpy's Hill Climbing + BIC score search, and MLE parameter fitting
on top of the learned structure. This is the ONLY file that talks to pgmpy's
structure-learning API, so if the library's interface changes, this is the
only file that needs updating.
"""

import pandas as pd
import warnings

# pgmpy is mid-way through reorganising its module layout (HillClimbSearch
# is moving to pgmpy.causal_discovery in a future release) and prints a
# FutureWarning both at import time and on every call as a result. It
# doesn't affect correctness -- we pin pgmpy==1.1.2 in requirements.txt
# specifically so this API keeps working -- so it's silenced here, BEFORE
# the pgmpy import below, to keep the console output readable (the
# import-time warning fires the instant `from pgmpy.estimators import ...`
# runs, so the filter has to be set before that line, not after it).
warnings.filterwarnings("ignore", category=FutureWarning)

from pgmpy.estimators import HillClimbSearch, BIC
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE

import config


def learn_structure(df: pd.DataFrame):
    """
    Run Hill Climbing with BIC scoring on df.

    Returns
    -------
    learned_edges : list[tuple[str, str]]
        Directed edges (parent, child) of the learned DAG.
    """
    hc = HillClimbSearch(df)
    scorer = BIC(df)
    best_dag = hc.estimate(
        scoring_method=scorer,
        max_indegree=config.HC_MAX_INDEGREE,
        max_iter=config.HC_MAX_ITER,
        show_progress=False,
    )
    return list(best_dag.edges())


def fit_learned_model(df: pd.DataFrame, learned_edges: list, all_nodes: list):
    """
    Fit CPD parameters (via MLE) on top of a learned structure.

    Nodes that ended up with NO edges at all (isolated in the learned graph)
    are added explicitly so every node in config.NODES is represented.
    """
    model = DiscreteBayesianNetwork(learned_edges)
    for n in all_nodes:
        if n not in model.nodes():
            model.add_node(n)
    model.fit(df, estimator=DiscreteMLE())
    return model
