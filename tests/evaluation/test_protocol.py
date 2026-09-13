from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.protocol import (
    battery_cluster_bootstrap,
    grouped_train_val_test_folds,
    holm_adjust,
    paired_battery_wilcoxon,
)


def test_grouped_folds_keep_batteries_disjoint():
    groups = np.repeat([f"B{i}" for i in range(10)], 3)
    folds = list(grouped_train_val_test_folds(groups, n_splits=5))
    assert len(folds) == 5
    for train, val, test in folds:
        assert not set(groups[train]) & set(groups[val])
        assert not set(groups[train]) & set(groups[test])
        assert not set(groups[val]) & set(groups[test])


def test_cluster_bootstrap_and_holm_outputs():
    y = np.arange(12, dtype=float)
    pred = y + 0.2
    groups = np.repeat(["A", "B", "C"], 4)
    ci = battery_cluster_bootstrap(y, pred, groups, n_bootstrap=50)
    assert {"metric", "estimate", "ci_low", "ci_high"} <= set(ci)
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.all((adjusted >= 0) & (adjusted <= 1))


def test_paired_wilcoxon_uses_per_battery_scores():
    rows = []
    for battery, a, b in zip(["A", "B", "C", "D"], [1, 1, 1, 1], [2, 3, 2, 4]):
        rows.extend([
            {"battery_id": battery, "model": "reference", "mae": a},
            {"battery_id": battery, "model": "candidate", "mae": b},
        ])
    result = paired_battery_wilcoxon(pd.DataFrame(rows), reference_model="reference")
    assert result.loc[0, "n_batteries"] == 4
    assert "p_value_holm" in result
