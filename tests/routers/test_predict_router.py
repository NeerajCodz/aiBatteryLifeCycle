from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.routers import predict as predict_router
from api.schemas import V3BatchPredictRequest, V3PredictRequest, PredictRequest, RecommendationRequest


def v3_request() -> V3PredictRequest:
    return V3PredictRequest(
        battery_id="B0005", cycle_number=10, prior_equivalent_full_cycles=8.7,
        ambient_temperature=24, segment_duration_s=800, segment_energy_wh=0.9,
        voltage_start_v=4.0, voltage_end_v=3.6, voltage_mean_v=3.8,
        voltage_std_v=0.1, voltage_linear_slope_v_per_s=-0.0005,
        voltage_curvature=0.0, current_mean_c_rate=1.0,
        current_std_c_rate=0.02, temperature_mean_c=30,
        temperature_std_c=2, temperature_rise_c=6,
        temperature_slope_c_per_s=0.007, temperature_observed=1,
    )


class StubRegistry:
    def __init__(self):
        self.default_model = "random_forest"
        self.last_features = None

    def predict(self, features, model_name=None):
        self.last_features = dict(features)
        if features.get("cycle_number") == -1:
            raise RuntimeError("boom")
        return {
            "soh_pct": 91.2,
            "rul_cycles": 123.4,
            "degradation_state": "Healthy",
            "confidence_lower": 89.0,
            "confidence_upper": 93.0,
            "model_used": model_name or "random_forest",
            "model_version": "3.0.0",
        }

    def predict_batch(self, battery_id, cycles):
        return [
            {
                "cycle_number": c.get("cycle_number", i + 1),
                "soh_pct": 90.0,
                "rul_cycles": 100.0,
                "degradation_state": "Healthy",
                "model_used": "random_forest",
                "model_version": "3.0.0",
            }
            for i, c in enumerate(cycles)
        ]

    def list_models(self):
        return [{"name": "random_forest", "version": "3.0.0"}]


@pytest.fixture
def stubbed_predict_registry(monkeypatch):
    reg = StubRegistry()
    monkeypatch.setattr(predict_router, "registry", reg)
    monkeypatch.setattr(predict_router, "registry_v1", reg)
    return reg


def test_predict_uses_v3_contract(stubbed_predict_registry, run_async):
    req = v3_request()
    resp = run_async(predict_router.predict(req))
    assert resp.soh_pct == 91.2
    assert resp.model_used == "random_forest"
    assert stubbed_predict_registry.last_features["cycle_index"] == 9
    assert "capacity_ah" not in stubbed_predict_registry.last_features


def test_predict_batch_returns_predictions(stubbed_predict_registry, run_async):
    row = v3_request().model_dump(exclude={"battery_id", "model_name"})
    req = V3BatchPredictRequest(
        battery_id="B0006",
        cycles=[{**row, "cycle_number": 1}, {**row, "cycle_number": 2}],
    )
    resp = run_async(predict_router.predict_batch(req))
    assert resp.battery_id == "B0006"
    assert len(resp.predictions) == 2


def test_recommend_returns_top_k(stubbed_predict_registry, run_async):
    req = RecommendationRequest(
        battery_id="B0007",
        current_cycle=100,
        current_soh=85.0,
        ambient_temperature=24.0,
        top_k=3,
    )
    resp = run_async(predict_router.recommend(req))
    assert len(resp.recommendations) == 3
    assert resp.recommendations[0].rank == 1


def test_list_models(stubbed_predict_registry, run_async):
    rows = run_async(predict_router.list_models())
    assert rows[0]["name"] == "random_forest"


def test_list_models_v1(stubbed_predict_registry, run_async):
    rows = run_async(predict_router.list_models_v1())
    assert rows[0]["name"] == "random_forest"


def test_list_model_versions_grouping(monkeypatch, run_async):
    class GroupRegistry(StubRegistry):
        def list_models(self):
            return [
                {"name": "ridge", "version": "1.2.0"},
                {"name": "bilstm", "version": "2.0.0"},
                {"name": "best_ensemble", "version": "3.0.0"},
                {"name": "mystery", "version": "?"},
            ]

    reg = GroupRegistry()
    monkeypatch.setattr(predict_router, "registry", reg)
    out = run_async(predict_router.list_model_versions())
    assert len(out["v1_classical"]) == 1
    assert len(out["v2_deep"]) == 1
    assert len(out["v2_ensemble"]) == 1
    assert len(out["other"]) == 1
    assert out["default_model"] == "random_forest"


def test_predict_v1(stubbed_predict_registry, run_async):
    req = PredictRequest(battery_id="B0005", cycle_number=12)
    resp = run_async(predict_router.predict_v1(req))
    assert resp.model_version == "3.0.0"


def test_predict_wraps_exceptions(monkeypatch, run_async):
    class ErrorRegistry(StubRegistry):
        def predict(self, features, model_name=None):
            raise RuntimeError("bad")

    monkeypatch.setattr(predict_router, "registry", ErrorRegistry())
    req = v3_request()
    with pytest.raises(HTTPException) as ex:
        run_async(predict_router.predict(req))
    assert ex.value.status_code == 500
