import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    VotingClassifier,
    VotingRegressor,
)


# =====================================================================
# INTERNAL HELPERS
# =====================================================================

def _safe_max_depth_rf(max_depth, n_samples):
    """
    Cap RF max_depth to avoid complete memorisation on small data.
    Rule: never deeper than log2(n_samples) * 2, hard cap at 20.
    """
    if max_depth is None:
        cap = min(20, max(6, int(np.log2(max(n_samples, 2)) * 2)))
        return cap
    return max_depth


def _safe_max_depth_gb(max_depth):
    """
    GB is prone to overfitting with deep trees.
    Hard cap at 6; default to 4 if None.
    """
    if max_depth is None:
        return 4
    return min(max_depth, 6)


def _safe_n_estimators_gb(n_estimators):
    """GB rarely benefits beyond 200 trees and gets slow."""
    return min(n_estimators, 200)


# =====================================================================
# MODEL BUILDERS — CLASSIFICATION
# =====================================================================

def _build_rf_clf(n_estimators, max_depth, min_samples_split,
                  min_samples_leaf, class_weight, n_samples):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=_safe_max_depth_rf(max_depth, n_samples),
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,              # RF supports parallel jobs safely
        verbose=0,
    )


def _build_gb_clf(n_estimators, max_depth, min_samples_split,
                  min_samples_leaf):
    # GB does NOT accept class_weight or n_jobs — omitted intentionally
    return GradientBoostingClassifier(
        n_estimators=_safe_n_estimators_gb(n_estimators),
        learning_rate=0.05,
        max_depth=_safe_max_depth_gb(max_depth),
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=0.8,          # row sampling per tree — reduces variance
        random_state=42,
        verbose=0,
    )


# =====================================================================
# MODEL BUILDERS — REGRESSION
# =====================================================================

def _build_rf_reg(n_estimators, max_depth, min_samples_split,
                  min_samples_leaf, n_samples):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=_safe_max_depth_rf(max_depth, n_samples),
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )


def _build_gb_reg(n_estimators, max_depth, min_samples_split,
                  min_samples_leaf):
    return GradientBoostingRegressor(
        n_estimators=_safe_n_estimators_gb(n_estimators),
        learning_rate=0.05,
        max_depth=_safe_max_depth_gb(max_depth),
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )


# =====================================================================
# FEATURE IMPORTANCE INJECTOR
# =====================================================================

def _inject_feature_importances(model):
    """
    Safely adds feature importance support without creating
    dynamic classes (which break joblib/pickle).
    """

    from sklearn.ensemble import VotingClassifier, VotingRegressor

    # Voting models
    if isinstance(model, (VotingClassifier, VotingRegressor)):

        importances = []

        for est in model.estimators_:

            if hasattr(est, "feature_importances_"):

                fi = est.feature_importances_

                if fi is not None:
                    importances.append(fi)

        if len(importances) > 0:

            model._fi_cache = np.mean(
                importances,
                axis=0
            )
            model.feature_importances_ = (model._fi_cache)

        else:

            model._fi_cache = None

    # RF / GB already have native feature_importances_
    elif hasattr(model, "feature_importances_"):

        model._fi_cache = model.feature_importances_

    # Other models
    else:

        model._fi_cache = None

    return model

def get_feature_importance(model):

    if hasattr(model, "feature_importances_"):

        try:
            return model.feature_importances_

        except Exception:
            pass

    return getattr(
        model,
        "_fi_cache",
        None
    )


# =====================================================================
# STRATEGY SELECTOR
# =====================================================================

def _select_strategy(n_samples, n_classes, task_type):
    """
    Return one of: 'ensemble' | 'random_forest' | 'gradient_boosting'

    Decision rules (apply in order):
      1. Many classes (>10)          → gradient_boosting
         RF splits become too shallow per class with many classes.
      2. Very small data (<500)      → gradient_boosting
         RF needs many rows to build diverse trees.
      3. Small data (500-2000)       → random_forest
         Fast, stable, good on medium-sized data.
      4. Large data (>2000)          → ensemble
         Enough data for both RF and GB to be reliable; ensemble wins.
    """
    if task_type == "classification" and n_classes > 10:
        return "gradient_boosting"
    if n_samples < 500:
        return "gradient_boosting"
    if n_samples < 2000:
        return "random_forest"
    return "ensemble"


# =====================================================================
# ENSEMBLE BUILDERS
# =====================================================================

def _build_ensemble_clf(n_estimators, max_depth, min_samples_split,
                        min_samples_leaf, class_weight, n_samples):
    rf = _build_rf_clf(
        n_estimators, max_depth, min_samples_split,
        min_samples_leaf, class_weight, n_samples
    )
    gb = _build_gb_clf(
        n_estimators, max_depth, min_samples_split, min_samples_leaf
    )
    # n_jobs on VotingClassifier itself is safe — it parallelises .predict
    # across estimators. Individual estimators manage their own threading.
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",      # uses predict_proba — smoother decisions
        weights=[2, 3],     # GB slightly preferred (usually more accurate)
        n_jobs=-1,
    )
    return ensemble


def _build_ensemble_reg(n_estimators, max_depth, min_samples_split,
                        min_samples_leaf, n_samples):
    rf = _build_rf_reg(
        n_estimators, max_depth, min_samples_split,
        min_samples_leaf, n_samples
    )
    gb = _build_gb_reg(
        n_estimators, max_depth, min_samples_split, min_samples_leaf
    )
    ensemble = VotingRegressor(
        estimators=[("rf", rf), ("gb", gb)],
        weights=[2, 3],
        n_jobs=-1,
    )
    return ensemble


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def train_model(
    X_train,
    y_train,
    task_type: str,
    n_estimators: int = 200,
    max_depth=None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight=None,
    use_ensemble: bool = True,
):
    """
    Train the best possible model for any dataset.

    Parameters
    ----------
    X_train           : pd.DataFrame or np.ndarray — feature matrix
    y_train           : pd.Series or np.ndarray   — target vector
    task_type         : "classification" | "regression"
    n_estimators      : number of trees (auto-capped for GB)
    max_depth         : max tree depth  (auto-capped per model type)
    min_samples_split : min samples needed to split a node
    min_samples_leaf  : min samples required at a leaf node
    class_weight      : None | "balanced" (classification only)
    use_ensemble      : True  → use auto-selected strategy
                        False → force single best model

    Returns
    -------
    Fitted model with .feature_importances_ always available.
    """

    n_samples = len(X_train)
    n_classes = int(y_train.nunique()) if task_type == "classification" else 0

    # Auto-select strategy based on data characteristics
    strategy = _select_strategy(n_samples, n_classes, task_type) if use_ensemble else (
        "gradient_boosting" if n_samples < 500 else "random_forest"
    )

    print(
        f"\n========== MODEL SELECTION ==========\n"
        f"Samples    : {n_samples}\n"
        f"Task       : {task_type.upper()}\n"
        f"Classes    : {n_classes if task_type == 'classification' else 'N/A'}\n"
        f"Strategy   : {strategy.upper().replace('_', ' ')}\n"
        f"Trees      : {n_estimators}\n"
        f"Max Depth  : {max_depth if max_depth else 'Auto-capped per model'}\n"
    )

    # ------------------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------------------
    if task_type == "classification":

        if strategy == "ensemble":
            model = _build_ensemble_clf(
                n_estimators, max_depth, min_samples_split,
                min_samples_leaf, class_weight, n_samples
            )

        elif strategy == "gradient_boosting":
            model = _build_gb_clf(
                n_estimators, max_depth, min_samples_split, min_samples_leaf
            )

        else:  # random_forest
            model = _build_rf_clf(
                n_estimators, max_depth, min_samples_split,
                min_samples_leaf, class_weight, n_samples
            )

    # ------------------------------------------------------------------
    # REGRESSION
    # ------------------------------------------------------------------
    else:

        if strategy == "ensemble":
            model = _build_ensemble_reg(
                n_estimators, max_depth, min_samples_split,
                min_samples_leaf, n_samples
            )

        elif strategy == "gradient_boosting":
            model = _build_gb_reg(
                n_estimators, max_depth, min_samples_split, min_samples_leaf
            )

        else:  # random_forest
            model = _build_rf_reg(
                n_estimators, max_depth, min_samples_split,
                min_samples_leaf, n_samples
            )

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    model.fit(X_train, y_train)

    # Inject feature_importances_ so downstream visualization always works
    model = _inject_feature_importances(model)

    print(
        f"Model Trained Successfully\n"
        f"Strategy Used : {strategy.upper().replace('_', ' ')}\n"
        f"Feature Importance : {'Available' if get_feature_importance(model) is not None else 'Not available'}\n"
        f"======================================"
    )

    return model
