from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.routers import predict_v2
from api.schemas import BatchPredictRequest, PredictRequest, RecommendationRequest


class StubV2Registry:
    def __init__(self):
        self.default_model = "extra_trees"
        self.last_features = None

    def predict(self, features, model_name=None):
        self.last_features = dict(features)
        if features.get("cycle_number") == -1:
            raise RuntimeError("broken")
        return {
            "soh_pct": 88.8,
            "rul_cycles": 95.0,
            "degradation_state": "Moderate",
            "confidence_lower": 86.0,
            "confidence_upper": 91.0,
            "model_used": model_name or "extra_trees",
            "model_version": "2.0.0",
        }

    def predict_batch(self, battery_id, cycles):
        return [
            {
                "cycle_number": c.get("cycle_number", i + 1),
                "soh_pct": 88.0,
                "rul_cycles": 90.0,
                "degradation_state": "Moderate",
                "model_used": "extra_trees",
                "model_version": "2.0.0",
            }
            for i, c in enumerate(cycles)
        ]

    def list_models(self):
        return [{"name": "extra_trees", "family": "classical"}]


@pytest.fixture
def stubbed_registry(monkeypatch):
    reg = StubV2Registry()
    monkeypatch.setattr(predict_v2, "registry_v2", reg)
    return reg


def test_predict_v2_uses_input_avg_temp(stubbed_registry, run_async):
    req = PredictRequest(
        battery_id="B0018",
        cycle_number=10,
        ambient_temperature=24.0,
        avg_temp=24.0,
    )
    resp = run_async(predict_v2.predict_v2(req))
    assert resp.soh_pct == 88.8
    assert resp.model_version == "2.0.0"
    assert stubbed_registry.last_features["avg_temp"] == 24.0


def test_predict_batch_v2(stubbed_registry, run_async):
    req = BatchPredictRequest(
        battery_id="B0018",
        cycles=[{"cycle_number": 1}, {"cycle_number": 2}],
    )
    resp = run_async(predict_v2.predict_batch_v2(req))
    assert len(resp.predictions) == 2
    assert resp.predictions[0].model_version == "2.0.0"


def test_recommend_v2_returns_ranked_rows(stubbed_registry, run_async):
    req = RecommendationRequest(
        battery_id="B0018",
        current_cycle=110,
        current_soh=82.0,
        ambient_temperature=24.0,
        top_k=4,
    )
    resp = run_async(predict_v2.recommend_v2(req))
    assert len(resp.recommendations) == 4
    assert [r.rank for r in resp.recommendations] == [1, 2, 3, 4]


def test_list_models_v2(stubbed_registry, run_async):
    rows = run_async(predict_v2.list_models_v2())
    assert rows[0]["name"] == "extra_trees"


def test_predict_v2_wraps_errors(monkeypatch, run_async):
    class ErrorRegistry(StubV2Registry):
        def predict(self, features, model_name=None):
            raise RuntimeError("bad")

    monkeypatch.setattr(predict_v2, "registry_v2", ErrorRegistry())
    req = PredictRequest(battery_id="B0018", cycle_number=10)
    with pytest.raises(HTTPException) as ex:
        run_async(predict_v2.predict_v2(req))
    assert ex.value.status_code == 500
