from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.adapters import (
    _calce_cycles_from_frame,
    _matlab_datenum_to_elapsed_seconds,
    _sanitize_temperature_channel,
    cycle_inventory,
)


def test_calce_frame_normalizes_discharge_cycle():
    frame = pd.DataFrame({
        "Test_Time(s)": [0, 1, 2, 3, 4],
        "Date_Time": pd.date_range("2026-01-01", periods=5, freq="s"),
        "Cycle_Index": [1] * 5,
        "Current(A)": [0, -1, -1, -1, 0],
        "Voltage(V)": [4.2, 4.0, 3.8, 3.5, 3.4],
        "Discharge_Capacity(Ah)": [0, 0.1, 0.5, 1.0, 1.0],
    })
    candidates = _calce_cycles_from_frame(frame, battery_id="CS2_35", source="fixture")
    assert len(candidates) == 1
    record = candidates[0][1]
    assert record.capacity_ah == 1.0
    assert list(record.measurements) == [
        "Time", "Voltage_measured", "Current_measured", "Temperature_measured"
    ]
    inventory = cycle_inventory([record])
    assert inventory.loc[0, "dataset"] == "CALCE"


def test_matlab_datenum_is_converted_from_days_to_seconds():
    observed = _matlab_datenum_to_elapsed_seconds([735954.0, 735954.5, 735955.0])
    np.testing.assert_allclose(observed, [0.0, 43_200.0, 86_400.0])


def test_grossly_inconsistent_temperature_channel_is_marked_missing():
    observed = _sanitize_temperature_channel([0.51, 0.52, 0.53], ambient_temperature_c=40.0)
    assert np.isnan(observed).all()
    valid = _sanitize_temperature_channel([39.8, 40.1, 41.0], ambient_temperature_c=40.0)
    assert np.isfinite(valid).all()
