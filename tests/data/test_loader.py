from __future__ import annotations

import pandas as pd
import pytest

from src.data import loader


def test_parse_matlab_datevec():
    dt = loader._parse_matlab_datevec("[2010. 7. 21. 15. 0. 35.093]")
    assert dt.year == 2010 and dt.month == 7 and dt.day == 21
    assert loader._parse_matlab_datevec("[]") is None


def test_load_cycle_csv_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_cycle_csv(1)


def test_load_discharge_capacities_and_battery_ids(monkeypatch):
    fake_meta = pd.DataFrame(
        {
            "battery_id": ["B1", "B1", "B2", "B2"],
            "type": ["discharge", "impedance", "discharge", "discharge"],
            "Capacity": [2.0, 0.0, 1.9, -1.0],
            "ambient_temperature": [24, 24, 43, 43],
            "test_id": [1, 2, 1, 2],
            "datetime": [None, None, None, None],
            "Re": [0.05, 0.06, 0.07, 0.08],
            "Rct": [0.08, 0.09, 0.1, 0.11],
        }
    )
    monkeypatch.setattr(loader, "load_metadata", lambda **kwargs: fake_meta.copy())

    dis = loader.load_discharge_capacities(drop_zero=True)
    assert set(dis["battery_id"]) == {"B1", "B2"}
    assert all(dis["Capacity"] > 0)

    ids = loader.get_battery_ids()
    assert ids == ["B1", "B2"]


def test_load_impedance_scalars(monkeypatch):
    fake_meta = pd.DataFrame(
        {
            "battery_id": ["B1", "B1", "B2"],
            "type": ["impedance", "impedance", "discharge"],
            "Capacity": [2.0, 1.9, 1.8],
            "ambient_temperature": [24, 24, 43],
            "test_id": [1, 2, 1],
            "datetime": [None, None, None],
            "Re": [0.05, 0.06, 0.07],
            "Rct": [0.08, 0.09, 0.1],
        }
    )
    monkeypatch.setattr(loader, "load_metadata", lambda **kwargs: fake_meta.copy())
    out = loader.load_impedance_scalars()
    assert list(out["battery_id"].unique()) == ["B1"]


def test_get_eol_threshold_default():
    assert loader.get_eol_threshold("B0005") in (loader.EOL_20PCT, loader.EOL_30PCT)
    assert loader.get_eol_threshold("UNKNOWN") == loader.EOL_30PCT
