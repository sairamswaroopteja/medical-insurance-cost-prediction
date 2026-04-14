"""
train.py
--------
Production-level training pipeline with:
  - Default vs tuned model comparison
  - GridSearchCV / RandomizedSearchCV per model
  - 5-fold cross-validation scores
  - Full results tracking → DataFrame
  - Individual model persistence via joblib
"""

import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge":            Ridge(),
    "Lasso":            Lasso(max_iter=10_000),
    "KNN":              KNeighborsRegressor(n_jobs=-1),
    "RandomForest":     RandomForestRegressor(random_state=42, n_jobs=-1),
    "AdaBoost":         AdaBoostRegressor(random_state=42),
}

# ── Hyperparameter grids ──────────────────────────────────────────────────────
PARAM_GRIDS = {
    "Ridge": {
        "alpha": [0.01, 0.1, 1, 10, 100],
    },
    "Lasso": {
        "alpha": [0.001, 0.01, 0.1, 1, 10],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 10, 15, 20],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan"],
    },
    "RandomForest": {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "AdaBoost": {
        "n_estimators":  [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "estimator":     [
            DecisionTreeRegressor(max_depth=1),
            DecisionTreeRegressor(max_depth=2),
            DecisionTreeRegressor(max_depth=3),
        ],
    },
}

# Large grids → RandomizedSearchCV
RANDOMIZED_SEARCH_MODELS = {"RandomForest", "AdaBoost"}
N_ITER_RANDOM = 20
CV_FOLDS      = 5
SCORING       = "neg_root_mean_squared_error"


# ── Metric helpers ────────────────────────────────────────────────────────────
def inverse_log(arr):
    return np.exp(np.array(arr))


def compute_metrics(y_true, y_pred, log_target: bool = True):
    if log_target:
        y_true = inverse_log(y_true)
        y_pred = inverse_log(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2


def _fresh_model(name):
    """Return a fresh (unfitted) instance for search."""
    defaults = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(),
        "Lasso":            Lasso(max_iter=10_000),
        "KNN":              KNeighborsRegressor(n_jobs=-1),
        "RandomForest":     RandomForestRegressor(random_state=42, n_jobs=-1),
        "AdaBoost":         AdaBoostRegressor(random_state=42),
    }
    return defaults[name]


# ── Core training function ────────────────────────────────────────────────────
def train_all(
    X_train, y_train,
    X_test,  y_test,
    models_dir: str = "models",
    log_target: bool = True,
):
    """
    Train every model with default + tuned settings.

    Returns
    -------
    results_df : pd.DataFrame   full comparison table
    trained    : dict           name → best fitted estimator
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    results = []
    trained = {}

    for name, base_model in MODELS.items():
        print(f"\n{'─'*60}")
        print(f"  Training  ▶  {name}")
        print(f"{'─'*60}")

        # 1. Fit default model ─────────────────────────────────────────────
        t0 = time.time()
        base_model.fit(X_train, y_train)
        default_time = time.time() - t0

        default_preds = base_model.predict(X_test)
        default_rmse, default_mae, default_r2 = compute_metrics(
            y_test, default_preds, log_target
        )
        print(f"  [Default]   RMSE=${default_rmse:>10,.2f}  "
              f"MAE=${default_mae:>9,.2f}  R²={default_r2:.4f}  ({default_time:.1f}s)")

        # 2. Cross-validation on default ───────────────────────────────────
        cv_scores    = cross_val_score(
            base_model, X_train, y_train,
            cv=CV_FOLDS, scoring=SCORING, n_jobs=-1
        )
        cv_rmse_mean = -cv_scores.mean()
        cv_rmse_std  =  cv_scores.std()
        print(f"  [CV-{CV_FOLDS}]      RMSE=${cv_rmse_mean:>10,.2f} ± ${cv_rmse_std:,.2f}")

        # 3. Hyperparameter tuning ─────────────────────────────────────────
        best_params  = {}
        tuning_time  = 0.0
        tuned_model  = base_model
        tuned_rmse   = default_rmse
        tuned_mae    = default_mae
        tuned_r2     = default_r2

        if name in PARAM_GRIDS:
            grid   = PARAM_GRIDS[name]
            fresh  = _fresh_model(name)
            t0     = time.time()

            if name in RANDOMIZED_SEARCH_MODELS:
                searcher = RandomizedSearchCV(
                    fresh,
                    param_distributions=grid,
                    n_iter=N_ITER_RANDOM,
                    cv=CV_FOLDS,
                    scoring=SCORING,
                    random_state=42,
                    n_jobs=-1,
                    refit=True,
                )
                search_label = f"RandomizedSearchCV (n_iter={N_ITER_RANDOM})"
            else:
                searcher = GridSearchCV(
                    fresh,
                    param_grid=grid,
                    cv=CV_FOLDS,
                    scoring=SCORING,
                    n_jobs=-1,
                    refit=True,
                )
                search_label = "GridSearchCV"

            searcher.fit(X_train, y_train)
            tuning_time = time.time() - t0

            tuned_model  = searcher.best_estimator_
            best_params  = searcher.best_params_
            tuned_preds  = tuned_model.predict(X_test)
            tuned_rmse, tuned_mae, tuned_r2 = compute_metrics(
                y_test, tuned_preds, log_target
            )
            delta = default_rmse - tuned_rmse

            print(f"  [{search_label}]  ({tuning_time:.1f}s)")
            print(f"  Best params : {best_params}")
            print(f"  [Tuned]     RMSE=${tuned_rmse:>10,.2f}  "
                  f"MAE=${tuned_mae:>9,.2f}  R²={tuned_r2:.4f}  "
                  f"(Δ RMSE {delta:+,.2f})")
        else:
            print(f"  No param grid — default model is final.")

        # 4. Save individual model ─────────────────────────────────────────
        model_path = f"{models_dir}/{name}.pkl"
        joblib.dump(tuned_model, model_path)
        print(f"  Saved  →  {model_path}")

        trained[name] = tuned_model
        results.append({
            "Model":          name,
            "Default_RMSE":   round(default_rmse, 2),
            "Tuned_RMSE":     round(tuned_rmse, 2),
            "Improvement":    round(default_rmse - tuned_rmse, 2),
            "MAE":            round(tuned_mae, 2),
            "R2":             round(tuned_r2, 4),
            "CV_RMSE_Mean":   round(cv_rmse_mean, 2),
            "CV_RMSE_Std":    round(cv_rmse_std, 2),
            "Best_Params":    str(best_params) if best_params else "default",
            "Tuning_Time_s":  round(tuning_time, 1),
            "Default_Time_s": round(default_time, 1),
        })

    results_df = (
        pd.DataFrame(results)
        .sort_values("Tuned_RMSE")
        .reset_index(drop=True)
    )
    return results_df, trained
