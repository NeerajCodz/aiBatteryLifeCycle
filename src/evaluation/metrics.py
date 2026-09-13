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
    n_predictors: int | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """Compute full regression metric suite.

    Values are expressed in SOH percentage points where applicable.  The
    tolerance metric is absolute (percentage points), not relative percentage.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = y_true.size
    adjusted_r2 = (
        1.0 - (1.0 - r2) * (n - 1) / (n - n_predictors - 1)
        if n_predictors is not None and n > n_predictors + 1
        else np.nan
    )
    # MAPE — avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}mae": float(mae),
        f"{p}mse": float(mse),
        f"{p}rmse": float(rmse),
        f"{p}r2": float(r2),
        f"{p}adjusted_r2": float(adjusted_r2),
        f"{p}mape": float(mape),
        f"{p}median_absolute_error": float(np.median(np.abs(y_true - y_pred))),
        f"{p}p95_absolute_error": float(np.percentile(np.abs(y_true - y_pred), 95)),
        f"{p}bias": float(np.mean(y_pred - y_true)),
        f"{p}within_5pp": tolerance_accuracy(y_true, y_pred, tolerance=5.0),
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
    table = pd.DataFrame(rows).set_index("model")
    aliases = {"MAE": "mae", "MSE": "mse", "RMSE": "rmse", "R2": "r2", "MAPE": "mape"}
    table = table.rename(columns={old: new for old, new in aliases.items() if old in table.columns and new not in table.columns})
    return table
