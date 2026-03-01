"""
src.models.classical.regressors
===============================
Classical ML regression models for SOH and RUL prediction.

All models follow a unified interface:
    train_*(X_train, y_train, **kwargs) → fitted model
    evaluate_*(model, X_test, y_test) → metrics dict

Hyperparameter optimization is done with Optuna where applicable.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.evaluation.metrics import regression_metrics, tolerance_accuracy
from src.utils.config import CV_FOLDS, MODELS_DIR, N_OPTUNA_TRIALS, RANDOM_STATE


def _save_model(model: Any, name: str) -> None:
    path = MODELS_DIR / "classical" / f"{name}.joblib"
    joblib.dump(model, path)


def _load_model(name: str) -> Any:
    path = MODELS_DIR / "classical" / f"{name}.joblib"
    return joblib.load(path)


# ── Ridge Regression ─────────────────────────────────────────────────────────
def train_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    model.fit(X, y)
    _save_model(model, "ridge")
    return model


# ── Lasso Regression ─────────────────────────────────────────────────────────
def train_lasso(X: np.ndarray, y: np.ndarray, alpha: float = 0.01) -> Lasso:
    model = Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=10000)
    model.fit(X, y)
    _save_model(model, "lasso")
    return model


# ── ElasticNet ───────────────────────────────────────────────────────────────
def train_elasticnet(X: np.ndarray, y: np.ndarray, alpha: float = 0.01, l1_ratio: float = 0.5) -> ElasticNet:
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE, max_iter=10000)
    model.fit(X, y)
    _save_model(model, "elasticnet")
    return model


# ── KNN Regressor ────────────────────────────────────────────────────────────
def train_knn(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> KNeighborsRegressor:
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)
    model.fit(X, y)
    _save_model(model, f"knn_k{n_neighbors}")
    return model


# ── SVR ──────────────────────────────────────────────────────────────────────
def train_svr(X: np.ndarray, y: np.ndarray, C: float = 10.0, gamma: str = "scale") -> SVR:
    model = SVR(kernel="rbf", C=C, gamma=gamma)
    model.fit(X, y)
    _save_model(model, "svr")
    return model


# ── Random Forest ────────────────────────────────────────────────────────────
def train_random_forest(
    X: np.ndarray, y: np.ndarray,
    n_estimators: int = 500,
    max_depth: int | None = None,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X, y)
    _save_model(model, "random_forest")
    return model


# ── XGBoost with Optuna HPO ─────────────────────────────────────────────────
def train_xgboost(
    X: np.ndarray, y: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
    cv_folds: int = CV_FOLDS,
) -> Any:
    """Train XGBoost regressor with Optuna hyperparameter optimization."""
    import optuna
    from xgboost import XGBRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = XGBRegressor(
            **params, tree_method="hist", random_state=RANDOM_STATE,
            verbosity=0, n_jobs=-1,
        )
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring="neg_mean_absolute_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_model = XGBRegressor(
        **best_params, tree_method="hist", random_state=RANDOM_STATE,
        verbosity=0, n_jobs=-1,
    )
    best_model.fit(X, y)
    _save_model(best_model, "xgboost")
    _save_model(study.best_params, "xgboost_best_params")
    return best_model


# ── LightGBM with Optuna HPO ────────────────────────────────────────────────
def train_lightgbm(
    X: np.ndarray, y: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
    cv_folds: int = CV_FOLDS,
) -> Any:
    """Train LightGBM regressor with Optuna hyperparameter optimization."""
    import optuna
    from lightgbm import LGBMRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
        model = LGBMRegressor(**params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring="neg_mean_absolute_error")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_model = LGBMRegressor(**best_params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
    best_model.fit(X, y)
    _save_model(best_model, "lightgbm")
    _save_model(study.best_params, "lightgbm_best_params")
    return best_model


# ── Unified evaluation ───────────────────────────────────────────────────────
def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "",
    soh_tolerance: float = 2.0,
    rul_tolerance: float = 5.0,
    target_type: str = "soh",
) -> dict[str, float]:
    """Evaluate any sklearn-compatible model and return metrics dict."""
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred, prefix=model_name)
    tol = soh_tolerance if target_type == "soh" else rul_tolerance
    metrics[f"{model_name}_tolerance_acc"] = tolerance_accuracy(y_test, y_pred, tol)
    return metrics
