from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    build_summary_table,
    classification_metrics,
    per_battery_evaluation,
    regression_metrics,
    tolerance_accuracy,
)


def test_regression_metrics_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    m = regression_metrics(y_true, y_pred, prefix="m")
    assert {"m_MAE", "m_MSE", "m_RMSE", "m_R2", "m_MAPE"} <= set(m.keys())


def test_tolerance_accuracy():
    y_true = np.array([10.0, 10.0, 10.0])
    y_pred = np.array([10.5, 12.5, 9.8])
    acc = tolerance_accuracy(y_true, y_pred, tolerance=1.0)
    assert np.isclose(acc, 2 / 3)


def test_classification_metrics_contains_confusion_matrix():
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 1, 0, 2])
    m = classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert "confusion_matrix" in m
    assert m["confusion_matrix"].shape == (3, 3)


def test_per_battery_evaluation_skips_singletons():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.1, 2.8])
    bids = np.array(["B1", "B1", "B2"])  # B2 has one sample only
    out = per_battery_evaluation(y_true, y_pred, bids)
    assert isinstance(out, pd.DataFrame)
    assert set(out["battery_id"]) == {"B1"}


def test_build_summary_table():
    tbl = build_summary_table({"rf": {"MAE": 1.2}, "xgb": {"MAE": 1.0}})
    assert list(tbl.index) == ["rf", "xgb"]
    assert "MAE" in tbl.columns
