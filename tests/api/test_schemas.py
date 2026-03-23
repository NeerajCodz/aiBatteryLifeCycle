from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import (
    BatchPredictRequest,
    PredictRequest,
    PredictResponse,
    RecommendationRequest,
)


def test_predict_request_defaults_and_bounds():
    req = PredictRequest(battery_id="B0005", cycle_number=1)
    assert req.ambient_temperature == 24.0
    assert req.min_voltage == 2.61

    with pytest.raises(ValidationError):
        PredictRequest(battery_id="B0005", cycle_number=0)


def test_recommendation_request_topk_range():
    ok = RecommendationRequest(
        battery_id="B0006",
        current_cycle=20,
        current_soh=90.0,
        top_k=5,
    )
    assert ok.top_k == 5

    with pytest.raises(ValidationError):
        RecommendationRequest(
            battery_id="B0006",
            current_cycle=20,
            current_soh=90.0,
            top_k=0,
        )


def test_batch_predict_request_accepts_cycles():
    req = BatchPredictRequest(
        battery_id="B0007",
        cycles=[{"cycle_number": 1}, {"cycle_number": 2}],
    )
    assert len(req.cycles) == 2


def test_predict_response_optional_fields():
    resp = PredictResponse(
        battery_id="B0005",
        cycle_number=10,
        soh_pct=92.5,
        rul_cycles=120.0,
        degradation_state="Healthy",
        model_used="random_forest",
    )
    assert resp.model_version is None
