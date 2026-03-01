"""
src.evaluation.metrics
======================
Comprehensive evaluation metrics for battery lifecycle prediction.

Provides:
- Regression metrics: MAE, MSE, RMSE, R², MAPE, tolerance accuracy
- Classification metrics: accuracy, F1-macro, confusion matrix
- Per-battery evaluation for cross-entity analysis
- Summary table builder
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Compute full regression metric suite.

    Returns dict with keys: MAE, MSE, RMSE, R2, MAPE
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # MAPE — avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}MAE": mae,
        f"{p}MSE": mse,
        f"{p}RMSE": rmse,
        f"{p}R2": r2,
        f"{p}MAPE": mape,
    }


def tolerance_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float = 2.0,
) -> float:
    """Fraction of predictions within ±tolerance of true values.

    Parameters
    ----------
    tolerance : float
        Absolute tolerance (e.g., 2.0 for ±2% SOH or ±2 cycles RUL).
    """
    return float(np.mean(np.abs(y_true - y_pred) <= tolerance))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list | None = None,
) -> dict[str, Any]:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def per_battery_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    battery_ids: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Compute regression metrics for each battery separately.

    Returns
    -------
    pd.DataFrame
        One row per battery, columns = metrics.
    """
    results = []
    for bid in np.unique(battery_ids):
        mask = np.asarray(battery_ids) == bid
        if mask.sum() < 2:
            continue
        m = regression_metrics(y_true[mask], y_pred[mask])
        m["battery_id"] = bid
        m["n_samples"] = int(mask.sum())
        results.append(m)
    return pd.DataFrame(results)


def build_summary_table(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Merge per-model results into one summary table.

    Parameters
    ----------
    results : dict
        ``{model_name: {metric: value, ...}, ...}``

    Returns
    -------
    pd.DataFrame
        One row per model.
    """
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
