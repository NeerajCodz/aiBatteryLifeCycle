from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.adapters import DischargeCycle
from src.data.benchmark import build_benchmark_bundle
from src.utils.config import FEATURE_COLS_V3


def _record(cycle_number: int, capacity: float) -> DischargeCycle:
    return DischargeCycle(
        dataset="fixture",
        battery_id="B1",
        cycle_number=cycle_number,
        capacity_ah=capacity,
        ambient_temperature_c=25.0,
        source="fixture",
        measurements=pd.DataFrame({
            "Time": np.arange(7, dtype=float),
            "Voltage_measured": [4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5],
            "Current_measured": [-2.0] * 7,
            "Temperature_measured": np.linspace(25, 26, 7),
        }),
    )


def test_bundle_aligns_features_sequences_and_targets():
    bundle = build_benchmark_bundle(
        [_record(i, 2.0 - i * 0.05) for i in range(4)],
        rated_capacity_by_dataset={"fixture": 2.0},
    )
    assert bundle.features.shape[0] == bundle.sequences.shape[0] == 4
    assert bundle.sequences.shape[1:] == (64, 5)
    assert all(column in bundle.features for column in FEATURE_COLS_V3)
    assert np.isclose(bundle.features.loc[0, "prior_equivalent_full_cycles"], 0.0)
    assert bundle.features.loc[3, "prior_equivalent_full_cycles"] > 0
