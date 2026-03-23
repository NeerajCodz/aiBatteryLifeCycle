from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import preprocessing as prep


def _base_df():
    rows = []
    for bid in ["B1", "B2", "B3", "B4"]:
        for c in range(5):
            rows.append(
                {
                    "battery_id": bid,
                    "cycle_number": c,
                    "value": c + 1,
                    "Capacity": 2.0 - 0.01 * c,
                }
            )
    return pd.DataFrame(rows)


def test_group_battery_split_no_leakage():
    df = _base_df()
    train, test = prep.group_battery_split(df, train_ratio=0.5, random_state=1)
    assert not set(train["battery_id"]).intersection(set(test["battery_id"]))
    assert len(train) + len(test) == len(df)


def test_leave_one_battery_out():
    df = _base_df()
    train, test = prep.leave_one_battery_out(df, test_battery="B3")
    assert set(test["battery_id"].unique()) == {"B3"}
    assert "B3" not in set(train["battery_id"].unique())


def test_make_sliding_windows_1d():
    arr = np.arange(10)
    X, y = prep.make_sliding_windows(arr, window_size=3, stride=1)
    assert X.shape == (7, 3, 1)
    assert y.shape == (7,)
    assert y[0] == 3


def test_make_multistep_windows_2d():
    arr = np.arange(40).reshape(20, 2)
    X, y = prep.make_multistep_windows(arr, input_window=4, output_window=2, stride=2)
    assert X.shape[1:] == (4, 2)
    assert y.shape[1:] == (2, 2)


def test_downsample_to_bins():
    df = pd.DataFrame({"a": np.arange(100), "b": np.arange(100) * 2})
    out = prep.downsample_to_bins(df, n_bins=10)
    assert out.shape == (10, 2)


def test_fit_and_load_scaler_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "SCALER_DIR", tmp_path)
    data = np.array([[1.0], [2.0], [3.0]])
    prep.fit_and_save_scaler(data, scaler_type="standard", name="demo")
    loaded = prep.load_scaler("demo", scaler_type="standard")
    transformed = loaded.transform(np.array([[2.0]]))
    assert transformed.shape == (1, 1)


def test_load_scaler_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "SCALER_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        prep.load_scaler("missing", scaler_type="standard")
