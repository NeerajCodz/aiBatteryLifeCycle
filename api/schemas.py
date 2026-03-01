"""
api.schemas
===========
Pydantic models for request / response validation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ── Prediction ───────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Request body for single-cycle prediction."""
    battery_id: str = Field(..., description="Battery identifier, e.g. 'B0005'")
    cycle_number: int = Field(..., ge=1, description="Current cycle number")
    ambient_temperature: float = Field(default=24.0, description="Ambient temperature (°C)")
    peak_voltage: float = Field(default=4.19, description="Peak charge voltage (V)")
    min_voltage: float = Field(default=2.61, description="Discharge cutoff voltage (V)")
    avg_current: float = Field(default=1.82, description="Average discharge current (A)")
    avg_temp: float = Field(default=32.6, description="Average cell temperature (°C) — typically higher than ambient")
    temp_rise: float = Field(default=14.7, description="Temperature rise during cycle (°C) — NASA dataset mean ≈ 15°C")
    cycle_duration: float = Field(default=3690.0, description="Cycle duration (seconds)")
    Re: float = Field(default=0.045, description="Electrolyte resistance (Ω) — training range 0.027–0.156")
    Rct: float = Field(default=0.069, description="Charge transfer resistance (Ω) — training range 0.04–0.27")
    delta_capacity: float = Field(default=-0.005, description="Capacity change from last cycle (Ah)")


class PredictResponse(BaseModel):
    """Response body for a prediction."""
    battery_id: str
    cycle_number: int
    soh_pct: float = Field(..., description="Predicted State of Health (%)")
    rul_cycles: Optional[float] = Field(None, description="Predicted Remaining Useful Life (cycles)")
    degradation_state: str = Field(..., description="Degradation state label")
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    model_used: str = Field(..., description="Name of the model that produced the prediction")
    model_version: Optional[str] = Field(None, description="Semantic version of the model used")


# ── Batch Prediction ─────────────────────────────────────────────────────────
class BatchPredictRequest(BaseModel):
    """Batch prediction for multiple cycles of one battery."""
    battery_id: str
    cycles: list[dict] = Field(..., description="List of cycle feature dicts")


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    battery_id: str
    predictions: list[PredictResponse]


# ── Recommendation ───────────────────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    """Request for operational recommendations."""
    battery_id: str
    current_cycle: int = Field(..., ge=1)
    current_soh: float = Field(..., ge=0, le=100)
    ambient_temperature: float = Field(default=24.0)
    top_k: int = Field(default=5, ge=1, le=20)


class SingleRecommendation(BaseModel):
    """One recommendation entry."""
    rank: int
    ambient_temperature: float
    discharge_current: float
    cutoff_voltage: float
    predicted_rul: float
    rul_improvement: float
    rul_improvement_pct: float
    explanation: str


class RecommendationResponse(BaseModel):
    """Response with ranked recommendations."""
    battery_id: str
    current_soh: float
    recommendations: list[SingleRecommendation]


# ── Model info ───────────────────────────────────────────────────────────────
class ModelInfo(BaseModel):
    """Info about a registered model."""
    name: str
    version: Optional[str] = None
    display_name: Optional[str] = None
    family: str
    algorithm: Optional[str] = None
    target: str
    r2: Optional[float] = None
    metrics: dict[str, float]
    is_default: bool = False
    loaded: bool = True
    load_error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str
    models_loaded: int
    device: str


# ── Visualization ────────────────────────────────────────────────────────────
class BatteryVizData(BaseModel):
    """Data for 3D battery visualization."""
    battery_id: str
    soh_pct: float
    temperature: float
    cycle_number: int
    degradation_state: str
    color_hex: str = Field(..., description="Color hex code for SOH heatmap")


class DashboardData(BaseModel):
    """Full dashboard payload."""
    batteries: list[BatteryVizData]
    capacity_fade: dict[str, list[float]]
    model_metrics: dict[str, dict[str, float]]
    best_model: str
