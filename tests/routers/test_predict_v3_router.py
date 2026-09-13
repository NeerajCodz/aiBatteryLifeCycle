from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api.routers import predict_v3
from api.schemas import V3BatchPredictRequest, V3PredictRequest, RecommendationRequest


def v3_request() -> V3PredictRequest:
    return V3PredictRequest(
        battery_id="B0005", cycle_number=10, prior_equivalent_full_cycles=9,
        ambient_temperature=24, segment_duration_s=800, segment_energy_wh=0.9,
        voltage_start_v=4.0, voltage_end_v=3.6, voltage_mean_v=3.8,
        voltage_std_v=0.1, voltage_linear_slope_v_per_s=-0.0005,
        voltage_curvature=0.0, current_mean_c_rate=1.0,
        current_std_c_rate=0.02, temperature_mean_c=30,
        temperature_std_c=2, temperature_rise_c=6,
        temperature_slope_c_per_s=0.007, temperature_observed=1,
    )


class StubV3Registry:
    def __init__(self):
        self.default_model = "xgboost"
        self.last_features = None

    def predict(self, features, model_name=None):
        self.last_features = dict(features)
        if features.get("cycle_number") == -1:
            raise RuntimeError("boom")
        return {
            "soh_pct": 93.0,
            "rul_cycles": 140.0,
            "degradation_state": "Healthy",
            "confidence_lower": 90.0,
            "confidence_upper": 95.0,
            "model_used": model_name or "xgboost",
            "model_version": "3.0.0",
        }

    def predict_batch(self, battery_id, cycles):
        return [
            {
                "cycle_number": c.get("cycle_number", i + 1),
                "soh_pct": 93.0,
                "rul_cycles": 130.0,
                "degradation_state": "Healthy",
                "model_used": "xgboost",
                "model_version": "3.0.0",
            }
            for i, c in enumerate(cycles)
        ]

    def list_models(self):
        return [{"name": "xgboost"}]

    def predict_array(self, X, model_name=None):
        n = X.shape[0]
        return np.clip(np.linspace(95.0, 85.0, n), 0, 100), (model_name or "xgboost")


@pytest.fixture
def stubbed_v3_registry(monkeypatch):
    reg = StubV3Registry()
    monkeypatch.setattr(predict_v3, "registry_v3", reg)
    return reg


def test_predict_v3(stubbed_v3_registry, run_async):
    req = v3_request()
    resp = run_async(predict_v3.predict_v3(req))
    assert resp.soh_pct == 93.0
    assert resp.model_version == "3.0.0"
    assert stubbed_v3_registry.last_features["cycle_index"] == 9
    assert "capacity_ah" not in stubbed_v3_registry.last_features


def test_v3_schema_rejects_target_and_unknown_fields():
    payload = v3_request().model_dump()
    payload["capacity_ah"] = 1.8
    with pytest.raises(ValidationError):
        V3PredictRequest(**payload)


def test_predict_batch_v3(stubbed_v3_registry, run_async):
    row = v3_request().model_dump(exclude={"battery_id", "model_name"})
    req = V3BatchPredictRequest(
        battery_id="B0005",
        cycles=[{**row, "cycle_number": 1}, {**row, "cycle_number": 2}],
    )
    resp = run_async(predict_v3.predict_batch_v3(req))
    assert len(resp.predictions) == 2


def test_recommend_v3(stubbed_v3_registry, run_async):
    req = RecommendationRequest(
        battery_id="B0005",
        current_cycle=100,
        current_soh=88.0,
        ambient_temperature=24.0,
        top_k=3,
    )
    resp = run_async(predict_v3.recommend_v3(req))
    assert len(resp.recommendations) == 3
    assert resp.recommendations[0].rank == 1


def test_predict_v3_wraps_error(monkeypatch, run_async):
    class ErrorRegistry(StubV3Registry):
        def predict(self, features, model_name=None):
            raise RuntimeError("bad")

    monkeypatch.setattr(predict_v3, "registry_v3", ErrorRegistry())
    req = v3_request()
    with pytest.raises(HTTPException):
        run_async(predict_v3.predict_v3(req))


def test_list_models_v3(stubbed_v3_registry, run_async):
    rows = run_async(predict_v3.list_models_v3())
    assert rows[0]["name"] == "xgboost"
