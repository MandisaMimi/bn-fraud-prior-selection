"""
baselines.py
============
Implements the three competitor models named in Section 3.4.2 of the
proposal, to contextualise Bayesian Network performance:

  - Logistic Regression: standard interpretable discriminative baseline
    (ell_2 regularisation, C=1.0)
  - Random Forest: ensemble tree method, strong empirical performance on
    imbalanced financial data (100 trees, max depth 10, balanced class
    weights)
  - Naive Bayes: equivalent to a Bayesian Network with NO edges, serving
    as a structural lower bound -- any improvement of G* over Naive Bayes
    quantifies the value of the dependency structure Hill Climbing found

All three are fit on the exact same train/holdout split used for the
Bayesian Network experiments (ground_truth.py / data_generation.py), so
comparisons are apples-to-apples: same data, same features, same target.

Categorical (not one-hot) features are used directly where the estimator
supports it (CategoricalNB, and tree-based Random Forest, which handles
integer-coded categoricals natively). Logistic Regression requires one-hot
encoding since it assumes a linear relationship in the feature space and
would otherwise treat bin index 4 as "more" than bin index 0, which is not
meaningful for a nominal variable like `type`.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder

import config

TARGET = "isFraud"


def _feature_cols():
    return [c for c in config.NODES if c != TARGET]


def fit_logistic_regression(train_df: pd.DataFrame):
    """One-hot encodes all features (since LR assumes a linear/ordinal
    relationship that discretised categorical bins do not actually have),
    then fits ell_2-regularised Logistic Regression."""
    feature_cols = _feature_cols()
    encoder = OneHotEncoder(
        categories=[list(range(config.STATE_COUNTS[c])) for c in feature_cols],
        handle_unknown="ignore",
        sparse_output=False,
    )
    X = encoder.fit_transform(train_df[feature_cols])
    y = train_df[TARGET].to_numpy()

    model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
    model.fit(X, y)
    return {"model": model, "encoder": encoder, "feature_cols": feature_cols}


def fit_random_forest(train_df: pd.DataFrame, seed=42):
    """Random Forest works directly on integer-coded categoricals (tree
    splits don't assume ordinality the way linear models do)."""
    feature_cols = _feature_cols()
    X = train_df[feature_cols].to_numpy()
    y = train_df[TARGET].to_numpy()

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced", random_state=seed
    )
    model.fit(X, y)
    return {"model": model, "feature_cols": feature_cols}


def fit_naive_bayes(train_df: pd.DataFrame):
    """CategoricalNB: the correct Naive Bayes variant for discrete/binned
    features (GaussianNB would wrongly assume continuous, normally-
    distributed features). This is the "Bayesian Network with no edges"
    structural lower bound named in Section 3.4.2."""
    feature_cols = _feature_cols()
    n_categories = [config.STATE_COUNTS[c] for c in feature_cols]
    X = train_df[feature_cols].to_numpy()
    y = train_df[TARGET].to_numpy()

    model = CategoricalNB(min_categories=n_categories)
    model.fit(X, y)
    return {"model": model, "feature_cols": feature_cols}


def predict_proba_logistic_regression(fitted, eval_df: pd.DataFrame) -> np.ndarray:
    X = fitted["encoder"].transform(eval_df[fitted["feature_cols"]])
    return fitted["model"].predict_proba(X)[:, 1]


def predict_proba_random_forest(fitted, eval_df: pd.DataFrame) -> np.ndarray:
    X = eval_df[fitted["feature_cols"]].to_numpy()
    return fitted["model"].predict_proba(X)[:, 1]


def predict_proba_naive_bayes(fitted, eval_df: pd.DataFrame) -> np.ndarray:
    X = eval_df[fitted["feature_cols"]].to_numpy()
    return fitted["model"].predict_proba(X)[:, 1]


def fit_majority_class(train_df: pd.DataFrame):
    """The simplest possible baseline: always predict the majority class
    (legitimate). Included specifically to demonstrate concretely why
    accuracy is an inappropriate metric under this level of imbalance --
    this baseline will show very high accuracy while detecting zero fraud,
    which the proposal argues conceptually but did not, before this,
    actually demonstrate with a fitted model."""
    majority_class = int(train_df[TARGET].mode()[0])
    return {"majority_class": majority_class}


def predict_proba_majority_class(fitted, eval_df: pd.DataFrame) -> np.ndarray:
    # Constant probability equal to 0.0 or 1.0 for every row, matching
    # "always predict legitimate" (or, in the degenerate case where fraud
    # is somehow the majority class in a given sample, "always predict
    # fraud") -- always the majority class, never a real decision.
    return np.full(len(eval_df), float(fitted["majority_class"]))


BASELINE_FITTERS = {
    "logistic_regression": fit_logistic_regression,
    "random_forest": fit_random_forest,
    "naive_bayes": fit_naive_bayes,
    "majority_class": fit_majority_class,
}

BASELINE_PREDICTORS = {
    "logistic_regression": predict_proba_logistic_regression,
    "random_forest": predict_proba_random_forest,
    "naive_bayes": predict_proba_naive_bayes,
    "majority_class": predict_proba_majority_class,
}


def fit_all_baselines(train_df: pd.DataFrame, seed=42) -> dict:
    return {
        "logistic_regression": fit_logistic_regression(train_df),
        "random_forest": fit_random_forest(train_df, seed=seed),
        "naive_bayes": fit_naive_bayes(train_df),
        "majority_class": fit_majority_class(train_df),
    }


def predict_all_baselines(fitted_baselines: dict, eval_df: pd.DataFrame) -> dict:
    return {
        name: BASELINE_PREDICTORS[name](fitted, eval_df)
        for name, fitted in fitted_baselines.items()
    }
