import numpy as np

# =========================================================
# EXPLAINABILITY SCORE
# Supports: RandomForest, GradientBoosting, VotingClassifier,
#           VotingRegressor, and any model with feature_importances_
# =========================================================

# Voting ensemble weights (must match train_model.py weights=[2, 3])
_VOTING_WEIGHTS = [2, 3]


def _extract_feature_importances(model):
    """
    Extract feature importances using 3 fallback strategies.

    Strategy 1 — Direct attribute  : RF, GB, XGBoost, etc.
    Strategy 2 — Voting ensemble   : iterate model.estimators_ (list of
                                     fitted objects, NOT tuples) and
                                     compute weighted average.
    Strategy 3 — Wrapper models    : CalibratedClassifierCV, Pipeline, etc.

    Returns
    -------
    np.ndarray with shape (n_features,) or None
    """

    # ── Strategy 1: model directly exposes importances ────────────────
    if hasattr(model, "feature_importances_"):
        return np.array(model.feature_importances_)

    # ── Strategy 2: VotingClassifier / VotingRegressor ────────────────
    # After fitting, model.estimators_ is a plain LIST of fitted estimators
    # (not a list of (name, estimator) tuples — that is model.estimators).
    if hasattr(model, "estimators_"):

        estimators_fitted = model.estimators_   # list of fitted objects

        importances_list = []

        for estimator in estimators_fitted:
            if hasattr(estimator, "feature_importances_"):
                importances_list.append(
                    np.array(estimator.feature_importances_)
                )

        if importances_list:

            n = len(importances_list)
            weights = _VOTING_WEIGHTS[:n]       # match however many we got

            # Pad with 1.0 if more estimators than defined weights
            if len(weights) < n:
                weights = weights + [1.0] * (n - len(weights))

            weighted_avg = np.average(
                importances_list,
                axis=0,
                weights=weights,
            )

            return weighted_avg

    # ── Strategy 3: wrapper / pipeline models ─────────────────────────
    for attr in ("base_estimator", "estimator", "best_estimator_"):
        inner = getattr(model, attr, None)
        if inner is not None:
            result = _extract_feature_importances(inner)
            if result is not None:
                return result

    return None


def _gini_spread(importances: np.ndarray) -> float:
    """
    Measure how evenly spread the feature importances are.

    Uses Gini impurity:  G = 1 - sum(p_i^2)

    G near 0 → one feature dominates  → low explainability
    G near 1 → importances spread evenly → high explainability

    Returns float in [0, 1].
    """

    if importances is None or len(importances) == 0:
        return 0.5

    total = importances.sum()
    if total == 0:
        return 0.5

    p    = importances / total
    gini = 1.0 - float(np.sum(p ** 2))

    return float(np.clip(gini, 0.0, 1.0))


def calculate_explainability_score(model) -> float:
    """
    Compute an explainability score in [0, 1].

    Score bands
    -----------
    0.90 – 0.95 : Very High  — importances well spread across many features
    0.80 – 0.89 : High       — good spread, tree-based model
    0.60 – 0.79 : Moderate   — some feature concentration
    0.00 – 0.59 : Low        — black-box / no importances available

    Formula
    -------
    base_score = 0.80   (any tree model with importances)
               = 0.50   (black-box, no importances)

    spread_bonus = (0.95 - 0.80) × gini_spread
                 → 0.00 when all importance on one feature
                 → 0.15 when perfectly spread

    final_score = clamp(base_score + spread_bonus, 0.0, 1.0)
    """

    print("\n========== EXPLAINABILITY SCORE ==========")

    importances = _extract_feature_importances(model)

    # ── Black-box fallback ─────────────────────────────────────────────
    if importances is None:
        print(
            "Feature Importance : Not Available (black-box model)\n"
            "Explainability Score : 0.50"
        )
        return 0.50

    print("Feature Importance : Available")

    gini        = _gini_spread(importances)
    base_score  = 0.80
    max_score   = 0.95
    spread_bonus = (max_score - base_score) * gini
    score        = round(float(np.clip(base_score + spread_bonus, 0.0, 1.0)), 4)

    print(
        f"Importance Spread (Gini) : {gini:.4f}\n"
        f"Explainability Score     : {score}"
    )

    return score
