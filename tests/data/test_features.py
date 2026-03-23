from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import features as feat


def test_compute_soc_and_soh():
    df = pd.DataFrame({"Current_measured": [-1.0, -1.0], "Time": [0.0, 3600.0]})
    soc = feat.compute_soc(df, nominal_capacity_ah=2.0)
    assert soc.iloc[0] == 100.0
    assert soc.iloc[-1] <= 100.0
    assert feat.compute_soh(1.0, nominal_capacity_ah=2.0) == 50.0


def test_compute_rul_and_degradation_state():
    cap = pd.Series([2.0, 1.8, 1.5, 1.3])
    rul = feat.compute_rul_series(cap, eol_threshold=1.4)
    assert rul.iloc[-1] == 0
    labels = feat.classify_degradation_state(np.array([95.0, 85.0, 75.0, 60.0]))
    assert labels.tolist() == [0, 1, 2, 3]


def test_extract_cycle_features():
    df = pd.DataFrame(
        {
            "Voltage_measured": [4.2, 3.7],
            "Current_measured": [-2.0, -1.0],
            "Temperature_measured": [25.0, 30.0],
            "Time": [0.0, 100.0],
        }
    )
    out = feat.extract_cycle_features(df)
    assert out["peak_voltage"] == 4.2
    assert out["cycle_duration"] == 100.0


def test_add_v3_features_and_impute():
    df = pd.DataFrame(
        {
            "battery_id": ["B1", "B1", "B2"],
            "Capacity": [2.0, 1.9, 1.8],
            "Re": [0.05, np.nan, 0.06],
            "Rct": [0.08, 0.09, np.nan],
            "SoH": [100.0, 95.0, 90.0],
            "peak_voltage": [4.2, 4.1, 4.0],
            "min_voltage": [2.5, 2.5, 2.5],
        }
    )
    out = feat.add_v3_features(df)
    assert "capacity_retention" in out.columns
    fixed = feat.impute_features(out)
    assert fixed["Re"].isna().sum() == 0
    assert fixed["Rct"].isna().sum() == 0
