from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.partial_cycle import (
    PartialCycleConfig,
    add_soh_labels,
    extract_partial_cycle_features,
    make_partial_cycle_sequence,
    prior_equivalent_full_cycles,
    select_partial_discharge,
)
from src.utils.config import FEATURE_COLS_V3, SEQUENCE_FEATURE_COLS_V3


def _cycle() -> pd.DataFrame:
    return pd.DataFrame({
        "Time": np.arange(8, dtype=float) * 10.0,
        "Voltage_measured": [4.2, 4.05, 3.99, 3.90, 3.80, 3.70, 3.59, 3.40],
        "Current_measured": [-2.0] * 8,
        "Temperature_measured": np.linspace(25.0, 27.0, 8),
    })


def test_soh_reference_is_median_of_first_three_valid_cycles():
    cycles = pd.DataFrame({
        "battery_id": ["B1"] * 4,
        "cycle_number": [0, 1, 2, 3],
        "Capacity": [2.0, 1.98, 1.96, 1.80],
    })
    labelled = add_soh_labels(cycles)
    assert np.isclose(labelled["reference_capacity_ah"].iloc[0], 1.98)
    assert np.isclose(labelled["SoH"].iloc[-1], 100 * 1.80 / 1.98)


def test_prior_throughput_excludes_current_cycle():
    efc = prior_equivalent_full_cycles(pd.Series([2.0, 1.8, 1.6]), 2.0)
    assert np.allclose(efc, [0.0, 1.0, 1.9])


def test_partial_feature_and_sequence_contracts():
    segment = select_partial_discharge(_cycle())
    assert segment["Voltage_measured"].iloc[0] <= 4.0
    assert segment["Voltage_measured"].iloc[-1] <= 3.6
    features = extract_partial_cycle_features(
        _cycle(), cycle_index=7, prior_efc=6.2, ambient_temperature=24.0,
        rated_capacity_ah=2.0,
    )
    assert list(features) == FEATURE_COLS_V3
    assert "Capacity" not in features and "SoH" not in features
    sequence = make_partial_cycle_sequence(
        _cycle(), ambient_temperature=24.0, rated_capacity_ah=2.0,
        config=PartialCycleConfig(sequence_bins=64),
    )
    assert sequence.shape == (64, 5)
    assert list(sequence) == SEQUENCE_FEATURE_COLS_V3
