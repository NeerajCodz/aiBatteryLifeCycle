"""Battery-grouped validation, clustered uncertainty, and paired testing."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.evaluation.metrics import regression_metrics


def grouped_train_val_test_folds(
    groups: np.ndarray | pd.Series,
    *,
    n_splits: int = 5,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield disjoint battery-grouped train, validation, and test indices."""
    groups = np.asarray(groups)
    indices = np.arange(groups.size)
    outer = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_val_idx, test_idx in outer.split(indices, groups=groups):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
        rel_train, rel_val = next(
            splitter.split(train_val_idx, groups=groups[train_val_idx])
        )
        train_idx = train_val_idx[rel_train]
        val_idx = train_val_idx[rel_val]
        assert not (
            set(groups[train_idx]) & set(groups[val_idx])
            or set(groups[train_idx]) & set(groups[test_idx])
            or set(groups[val_idx]) & set(groups[test_idx])
        )
        yield train_idx, val_idx, test_idx


def battery_cluster_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    battery_ids: np.ndarray | pd.Series,
    *,
    n_predictors: int | None = None,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return battery-cluster bootstrap confidence intervals for all metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    battery_ids = np.asarray(battery_ids)
    unique = np.unique(battery_ids)
    if unique.size < 2:
        raise ValueError("At least two batteries are required for clustered bootstrap")
    rng = np.random.default_rng(random_state)
    draws: list[dict[str, float]] = []
    group_indices = {group: np.flatnonzero(battery_ids == group) for group in unique}
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([group_indices[group] for group in sampled])
        draws.append(regression_metrics(y_true[idx], y_pred[idx], n_predictors=n_predictors))
    frame = pd.DataFrame(draws)
    alpha = (1.0 - confidence) / 2.0
    point = regression_metrics(y_true, y_pred, n_predictors=n_predictors)
    return pd.DataFrame({
        "metric": list(point),
        "estimate": [point[key] for key in point],
        "ci_low": [frame[key].quantile(alpha) for key in point],
        "ci_high": [frame[key].quantile(1.0 - alpha) for key in point],
    })


def holm_adjust(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Holm family-wise error correction in original hypothesis order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((p.size - np.arange(p.size)) * p[order])
    adjusted = np.empty_like(p)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return adjusted


def paired_battery_wilcoxon(
    scores: pd.DataFrame,
    *,
    battery_col: str = "battery_id",
    model_col: str = "model",
    score_col: str = "mae",
    reference_model: str,
) -> pd.DataFrame:
    """Compare each model with a reference using paired per-battery scores."""
    pivot = scores.pivot(index=battery_col, columns=model_col, values=score_col)
    if reference_model not in pivot:
        raise KeyError(f"Reference model not found: {reference_model}")
    rows = []
    for model in pivot.columns:
        if model == reference_model:
            continue
        pair = pivot[[reference_model, model]].dropna()
        if len(pair) < 2:
            continue
        statistic, p_value = wilcoxon(
            pair[model], pair[reference_model], zero_method="zsplit"
        )
        rows.append({
            "reference_model": reference_model,
            "model": model,
            "n_batteries": len(pair),
            "median_mae_difference": float(np.median(pair[model] - pair[reference_model])),
            "statistic": float(statistic),
            "p_value": float(p_value),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm"] = holm_adjust(result["p_value"].to_numpy())
    return result
