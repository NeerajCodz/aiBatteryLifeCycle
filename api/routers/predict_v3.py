"""
api.routers.predict_v3
======================
v3 prediction & recommendation endpoints.

v3 improvements over v2:
- Higher accuracy classical models (XGBoost R²=0.9866, GradientBoosting R²=0.9860)
- Updated ensemble weights proportional to v3 R² values
- Version-aware model loading from artifacts/v3/
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.model_registry import registry_v3, classify_degradation, soh_to_color
from api.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    RecommendationRequest, RecommendationResponse, SingleRecommendation,
)

router = APIRouter(prefix="/api/v3", tags=["v3-prediction"])


# ── Single prediction ────────────────────────────────────────────────────────
@router.post("/predict", response_model=PredictResponse)
async def predict_v3(req: PredictRequest):
    """Predict SOH for a single cycle using v3 models."""
    features = req.model_dump(exclude={"battery_id"})
    features["voltage_range"] = features["peak_voltage"] - features["min_voltage"]

    try:
        result = registry_v3.predict(features, req.model_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictResponse(
        battery_id=req.battery_id,
        cycle_number=req.cycle_number,
        soh_pct=result["soh_pct"],
        rul_cycles=result["rul_cycles"],
        degradation_state=result["degradation_state"],
        confidence_lower=result["confidence_lower"],
        confidence_upper=result["confidence_upper"],
        model_used=result["model_used"],
        model_version=result.get("model_version", "3.0"),
    )


# ── Batch prediction ─────────────────────────────────────────────────────────
@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_v3(req: BatchPredictRequest):
    """Predict SOH for multiple cycles using v3 models."""
    results = registry_v3.predict_batch(req.battery_id, req.cycles)
    predictions = [
        PredictResponse(
            battery_id=req.battery_id,
            cycle_number=r["cycle_number"],
            soh_pct=r["soh_pct"],
            rul_cycles=r["rul_cycles"],
            degradation_state=r["degradation_state"],
            confidence_lower=r.get("confidence_lower"),
            confidence_upper=r.get("confidence_upper"),
            model_used=r["model_used"],
            model_version=r.get("model_version", "3.0"),
        )
        for r in results
    ]
    return BatchPredictResponse(battery_id=req.battery_id, predictions=predictions)


# ── Recommendations (v3) ─────────────────────────────────────────────────────
@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_v3(req: RecommendationRequest):
    """Get operational recommendations using v3 models.

    Ranking is based on net RUL improvement versus a model-derived baseline,
    with small guardrail penalties for clearly harsher operating conditions.
    """
    import itertools

    temps = [4.0, 24.0, 43.0]
    currents = [0.5, 1.0, 2.0, 4.0]
    cutoffs = [2.0, 2.2, 2.5, 2.7]

    base_features = {
        "cycle_number": req.current_cycle,
        "ambient_temperature": req.ambient_temperature,
        "peak_voltage": 4.19,
        "min_voltage": 2.61,
        "voltage_range": 4.19 - 2.61,
        "avg_current": 1.82,
        "avg_temp": req.ambient_temperature + 8.0,
        "temp_rise": 15.0,
        "cycle_duration": 3690.0,
        "Re": 0.045,
        "Rct": 0.069,
        "delta_capacity": -0.005,
    }

    def guardrail_penalty(temp: float, current: float, cutoff: float) -> float:
        """Penalty in cycle-units for stress-heavy operating points.

        This keeps recommendations aligned with battery-care intent even when
        model outputs are noisy for out-of-distribution combinations.
        """
        temp_penalty = max(0.0, temp - 30.0) * 3.0 + max(0.0, 12.0 - temp) * 1.5
        current_penalty = max(0.0, current - 1.5) * 12.0
        cutoff_penalty = max(0.0, 2.4 - cutoff) * 8.0
        return temp_penalty + current_penalty + cutoff_penalty

    baseline_features = {
        **base_features,
        "ambient_temperature": req.ambient_temperature,
        "avg_current": 1.82,
        "min_voltage": 2.61,
        "voltage_range": 4.19 - 2.61,
        "avg_temp": req.ambient_temperature + 8.0,
    }
    baseline_pred = registry_v3.predict(baseline_features, req.model_name)
    baseline_rul = max(0.0, float(baseline_pred.get("rul_cycles", 0) or 0))
    baseline_adjusted_rul = baseline_rul - guardrail_penalty(
        req.ambient_temperature,
        1.82,
        2.61,
    )

    candidates = []
    for t, c, v in itertools.product(temps, currents, cutoffs):
        feat = {**base_features, "ambient_temperature": t, "avg_current": c,
                "min_voltage": v, "voltage_range": 4.19 - v,
                "avg_temp": t + 8.0}
        result = registry_v3.predict(feat, req.model_name)
        rul = max(0.0, float(result.get("rul_cycles", 0) or 0))
        adjusted_rul = rul - guardrail_penalty(t, c, v)
        improvement = adjusted_rul - baseline_adjusted_rul
        candidates.append({
            "raw_rul": rul,
            "adjusted_rul": adjusted_rul,
            "improvement": improvement,
            "temp": t,
            "current": c,
            "cutoff": v,
        })

    candidates.sort(
        key=lambda x: (
            x["improvement"] > 0,
            x["improvement"],
            x["adjusted_rul"],
            -abs(x["temp"] - 24.0),
            -x["current"],
        ),
        reverse=True,
    )
    top = candidates[: req.top_k]

    recs = []
    for rank, rec in enumerate(top, 1):
        rul = rec["raw_rul"]
        t = rec["temp"]
        c = rec["current"]
        v = rec["cutoff"]
        improvement = rec["improvement"]
        pct = (improvement / baseline_rul * 100) if baseline_rul > 0 else 0
        impact = "improves" if improvement > 0 else "does not improve"
        recs.append(SingleRecommendation(
            rank=rank,
            ambient_temperature=t,
            discharge_current=c,
            cutoff_voltage=v,
            predicted_rul=rul,
            rul_improvement=improvement,
            rul_improvement_pct=round(pct, 1),
            explanation=(
                f"Operate at {t}°C, {c}A, cutoff {v}V for ~{rul:.0f} cycles RUL; "
                f"this {impact} lifespan by {improvement:+.0f} cycles vs your baseline."
            ),
        ))

    return RecommendationResponse(
        battery_id=req.battery_id,
        current_soh=req.current_soh,
        recommendations=recs,
    )


# ── Model listing ─────────────────────────────────────────────────────────────
@router.get("/models")
async def list_models_v3():
    """List all v3 registered models."""
    return registry_v3.list_models()
