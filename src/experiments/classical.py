"""Leakage-aware grouped evaluation for eight classical and two ensembles."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import regression_metrics
from src.evaluation.protocol import grouped_train_val_test_folds
from src.utils.config import FEATURE_COLS_V3


def _tabular_pipeline(model: RegressorMixin, *, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def classical_factories(random_state: int = 42) -> dict[str, Callable[[], RegressorMixin]]:
    """Return fresh estimator factories with deterministic CPU settings."""
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "extra_trees": lambda: _tabular_pipeline(ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        )),
        "gradient_boosting": lambda: _tabular_pipeline(GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.04, max_depth=3, loss="huber",
            random_state=random_state,
        )),
        "random_forest": lambda: _tabular_pipeline(RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.8,
            random_state=random_state, n_jobs=-1,
        )),
        "xgboost": lambda: _tabular_pipeline(XGBRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=5, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=1.0, objective="reg:squarederror",
            tree_method="hist", random_state=random_state, n_jobs=-1,
        )),
        "lightgbm": lambda: _tabular_pipeline(LGBMRegressor(
            n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=1.0, random_state=random_state,
            verbosity=-1, n_jobs=-1,
        )),
        "svr": lambda: _tabular_pipeline(SVR(C=10.0, epsilon=0.1), scale=True),
        "ridge": lambda: _tabular_pipeline(Ridge(alpha=1.0), scale=True),
        "knn": lambda: _tabular_pipeline(KNeighborsRegressor(
            n_neighbors=7, weights="distance", p=2, n_jobs=-1
        ), scale=True),
    }


def _ensemble_predictions(
    val_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    y_val: np.ndarray,
    train_predictions: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    names = list(val_predictions)
    val_matrix = np.column_stack([val_predictions[name] for name in names])
    test_matrix = np.column_stack([test_predictions[name] for name in names])
    train_matrix = np.column_stack([train_predictions[name] for name in names])
    errors = np.mean(np.abs(val_matrix - y_val[:, None]), axis=0)
    inverse = 1.0 / np.maximum(errors, 1e-8)
    weights = inverse / inverse.sum()
    meta = TransformedTargetRegressor(regressor=Ridge(alpha=1.0))
    meta.fit(val_matrix, y_val)
    test_ensembles = {
        "stacking_ensemble": meta.predict(test_matrix),
        "weighted_ensemble": test_matrix @ weights,
    }
    # These are in-sample base-model diagnostics, not a second model-selection
    # criterion. The ensemble weights and meta-model still use validation data.
    train_ensembles = {
        "stacking_ensemble": meta.predict(train_matrix),
        "weighted_ensemble": train_matrix @ weights,
    }
    return test_ensembles, train_ensembles


def run_grouped_tabular_benchmark(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    n_splits: int = 5,
    seeds: tuple[int, ...] = (17, 42, 2026),
    feature_cols: list[str] = FEATURE_COLS_V3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate tabular models and return metric and prediction tables."""
    missing = set(feature_cols + ["SoH", "battery_id"]).difference(frame.columns)
    if missing:
        raise KeyError(f"Benchmark frame is missing: {sorted(missing)}")
    X = frame[feature_cols].to_numpy(dtype=float)
    y = frame["SoH"].to_numpy(dtype=float)
    groups = frame["battery_id"].astype(str).to_numpy()
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for seed in seeds:
        folds = grouped_train_val_test_folds(
            groups, n_splits=n_splits, validation_fraction=0.2, random_state=seed
        )
        for fold, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
            factories = classical_factories(seed)
            val_predictions: dict[str, np.ndarray] = {}
            test_predictions: dict[str, np.ndarray] = {}
            train_predictions: dict[str, np.ndarray] = {}
            for model_id, factory in factories.items():
                model = factory()
                model.fit(X[train_idx], y[train_idx])
                val_predictions[model_id] = model.predict(X[val_idx])
                test_pred = model.predict(X[test_idx])
                test_predictions[model_id] = test_pred
                train_pred = model.predict(X[train_idx])
                train_predictions[model_id] = train_pred
                metrics = regression_metrics(y[test_idx], test_pred, n_predictors=len(feature_cols))
                train_metrics = regression_metrics(y[train_idx], train_pred, n_predictors=len(feature_cols))
                metric_rows.append({
                    "dataset": dataset_name, "seed": seed, "fold": fold,
                    "model": model_id, **metrics,
                    "train_mae": train_metrics["mae"],
                    "generalization_gap_mae": metrics["mae"] - train_metrics["mae"],
                })
            ensemble_test, ensemble_train = _ensemble_predictions(
                val_predictions, test_predictions, y[val_idx], train_predictions
            )
            test_predictions.update(ensemble_test)
            for model_id in ("stacking_ensemble", "weighted_ensemble"):
                pred = test_predictions[model_id]
                metrics = regression_metrics(y[test_idx], pred, n_predictors=len(feature_cols))
                train_metrics = regression_metrics(
                    y[train_idx], ensemble_train[model_id], n_predictors=len(feature_cols)
                )
                metric_rows.append({
                    "dataset": dataset_name, "seed": seed, "fold": fold,
                    "model": model_id, **metrics,
                    "train_mae": train_metrics["mae"],
                    "generalization_gap_mae": metrics["mae"] - train_metrics["mae"],
                })
            for model_id, pred in test_predictions.items():
                for local, row_idx in enumerate(test_idx):
                    prediction_rows.append({
                        "dataset": dataset_name, "seed": seed, "fold": fold,
                        "model": model_id, "row_index": int(row_idx),
                        "battery_id": groups[row_idx],
                        "cycle_number": int(frame.iloc[row_idx]["cycle_number"]),
                        "y_true": y[row_idx], "y_pred": float(pred[local]),
                        "residual": float(y[row_idx] - pred[local]),
                    })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def run_zero_shot_tabular(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    source_name: str = "NASA",
    target_name: str,
    random_state: int = 42,
    feature_cols: list[str] = FEATURE_COLS_V3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the complete source domain and score an untouched target domain.

    Ensemble weights and the stacking meta-model are learned from source-domain
    out-of-fold predictions; target labels never influence fitting.
    """
    X_source = source[feature_cols].to_numpy(dtype=float)
    y_source = source["SoH"].to_numpy(dtype=float)
    groups = source["battery_id"].astype(str).to_numpy()
    X_target = target[feature_cols].to_numpy(dtype=float)
    y_target = target["SoH"].to_numpy(dtype=float)
    factories = classical_factories(random_state)
    names = list(factories)
    oof = np.zeros((len(source), len(names)), dtype=float)
    cv = GroupKFold(n_splits=min(5, np.unique(groups).size), shuffle=True, random_state=random_state)
    for train_idx, val_idx in cv.split(X_source, y_source, groups):
        for column, name in enumerate(names):
            model = factories[name]()
            model.fit(X_source[train_idx], y_source[train_idx])
            oof[val_idx, column] = model.predict(X_source[val_idx])
    errors = np.mean(np.abs(oof - y_source[:, None]), axis=0)
    weights = (1.0 / np.maximum(errors, 1e-8))
    weights /= weights.sum()
    meta = Ridge(alpha=1.0).fit(oof, y_source)
    target_matrix = np.zeros((len(target), len(names)), dtype=float)
    predictions: dict[str, np.ndarray] = {}
    for column, name in enumerate(names):
        model = factories[name]()
        model.fit(X_source, y_source)
        pred = model.predict(X_target)
        predictions[name] = pred
        target_matrix[:, column] = pred
    predictions["stacking_ensemble"] = meta.predict(target_matrix)
    predictions["weighted_ensemble"] = target_matrix @ weights
    metric_rows, prediction_rows = [], []
    for name, pred in predictions.items():
        metric_rows.append({
            "source_dataset": source_name,
            "target_dataset": target_name,
            "model": name,
            **regression_metrics(y_target, pred, n_predictors=len(feature_cols)),
        })
        for index, value in enumerate(pred):
            prediction_rows.append({
                "source_dataset": source_name,
                "target_dataset": target_name,
                "model": name,
                "battery_id": str(target.iloc[index]["battery_id"]),
                "cycle_number": int(target.iloc[index]["cycle_number"]),
                "y_true": y_target[index],
                "y_pred": float(value),
                "residual": float(y_target[index] - value),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)
