from __future__ import annotations

import numpy as np
import pytest

from api.routers import simulate
from api.routers.simulate import BatterySimConfig, SimulateRequest


class StubSimRegistry:
    def __init__(self):
        self.default_model = "xgboost"
        self.feature_cols = [
            "cycle_number",
            "ambient_temperature",
            "peak_voltage",
            "min_voltage",
            "voltage_range",
            "avg_current",
            "avg_temp",
            "temp_rise",
            "cycle_duration",
            "Re",
            "Rct",
            "delta_capacity",
        ]
        self.model_meta = {
            "xgboost": {"family": "classical", "requires_scaling": False},
            "best_ensemble": {"components": ["xgboost"]},
        }
        self.models = {"xgboost": object(), "best_ensemble": object()}

    def predict_array(self, X, model_name=None):
        n = X.shape[0]
        return np.clip(np.linspace(95.0, 85.0, n), 0, 100), (model_name or "xgboost")


@pytest.fixture
def stubbed_registry(monkeypatch):
    reg = StubSimRegistry()
    monkeypatch.setattr(simulate, "registry_v2", reg)
    return reg


def test_simulation_helper_vec_classify():
    out = simulate._vec_classify(np.array([95.0, 85.0, 75.0, 60.0]))
    assert out == ["Healthy", "Moderate", "Degraded", "End-of-Life"]


def test_simulation_helper_vec_color():
    out = simulate._vec_color(np.array([95.0, 85.0, 75.0, 60.0]))
    assert out == ["#22c55e", "#eab308", "#f97316", "#ef4444"]


def test_simulation_compute_rul_and_eol_cycle_unit():
    soh = np.array([95.0, 90.0, 80.0, 69.0], dtype=float)
    cycles = np.array([1, 2, 3, 4], dtype=np.int64)
    rul, rul_time, eol_cycle, eol_time = simulate._compute_rul_and_eol(
        soh_arr=soh,
        initial_soh=100.0,
        eol_thr=70.0,
        cycle_start=1,
        cycle_arr=cycles,
        elapsed_cycles=np.array([0, 1, 2, 3], dtype=float),
        cycle_dur=3600.0,
        tu_sec=None,
    )
    assert eol_cycle == 4
    assert eol_time == 4.0
    assert rul.shape == (4,)
    assert rul_time.shape == (4,)


def test_simulate_batteries_basic(stubbed_registry, run_async):
    req = SimulateRequest(
        batteries=[
            BatterySimConfig(
                battery_id="B0005",
                initial_soh=95.0,
                delta_capacity=-0.005,
            )
        ],
        steps=5,
        time_unit="day",
        use_ml=True,
    )
    resp = run_async(simulate.simulate_batteries(req))
    assert resp.steps == 5
    assert len(resp.results) == 1
    assert len(resp.results[0].soh_history) == 5
    assert resp.model_used in ("xgboost", "best_ensemble", "linear_fallback")


def test_simulate_invalid_time_unit_falls_back(stubbed_registry, run_async):
    req = SimulateRequest(
        batteries=[BatterySimConfig(battery_id="B0006")],
        steps=2,
        time_unit="invalid",
        use_ml=False,
    )
    resp = run_async(simulate.simulate_batteries(req))
    assert resp.time_unit == "day"
