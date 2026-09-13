from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.classical import run_grouped_tabular_benchmark
from src.utils.config import FEATURE_COLS_V3


def test_grouped_tabular_benchmark_has_ten_models(monkeypatch):
    rows = []
    for battery_index in range(5):
        for cycle in range(4):
            row = {
                "battery_id": f"B{battery_index}",
                "cycle_number": cycle,
                "SoH": 100 - 2 * cycle - battery_index,
            }
            row.update({name: float(cycle + battery_index) for name in FEATURE_COLS_V3})
            rows.append(row)
    frame = pd.DataFrame(rows)
    from src.experiments import classical
    from sklearn.linear_model import Ridge
    monkeypatch.setattr(classical, "classical_factories", lambda seed: {
        name: (lambda: Ridge()) for name in [
            "extra_trees", "gradient_boosting", "random_forest", "xgboost",
            "lightgbm", "svr", "ridge", "knn",
        ]
    })
    metrics, predictions = run_grouped_tabular_benchmark(
        frame, dataset_name="fixture", n_splits=5, seeds=(42,)
    )
    assert metrics["model"].nunique() == 10
    assert predictions["model"].nunique() == 10
    assert not predictions["residual"].isna().any()
