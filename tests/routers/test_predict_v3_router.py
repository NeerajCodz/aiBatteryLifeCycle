from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from api.routers import predict_v3
from api.schemas import BatchPredictRequest, PredictRequest, RecommendationRequest


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
    req = PredictRequest(battery_id="B0005", cycle_number=10)
    resp = run_async(predict_v3.predict_v3(req))
    assert resp.soh_pct == 93.0
    assert resp.model_version == "3.0.0"
    assert stubbed_v3_registry.last_features["voltage_range"] == pytest.approx(1.58)


def test_predict_batch_v3(stubbed_v3_registry, run_async):
    req = BatchPredictRequest(
        battery_id="B0005",
        cycles=[{"cycle_number": 1}, {"cycle_number": 2}],
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
    req = PredictRequest(battery_id="B0005", cycle_number=10)
    with pytest.raises(HTTPException):
        run_async(predict_v3.predict_v3(req))


def test_list_models_v3(stubbed_v3_registry, run_async):
    rows = run_async(predict_v3.list_models_v3())
    assert rows[0]["name"] == "xgboost"
