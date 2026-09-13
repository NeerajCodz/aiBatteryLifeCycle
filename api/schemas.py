"""
api.schemas
===========
Pydantic models for request / response validation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
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
    model_name: Optional[str] = Field(default=None, description="Optional model key override")


class V3CycleFeatures(BaseModel):
    """Leakage-safe v3 features for one partial-discharge observation.

    All measurements describe the observable 4.0--3.6 V portion of the current
    discharge. Full-cycle capacity and current-cycle SOH are deliberately absent.
    """
    model_config = ConfigDict(extra="forbid")

    cycle_number: int = Field(..., ge=1, description="Human-facing 1-based cycle number")
    prior_equivalent_full_cycles: float = Field(
        ..., ge=0,
        description="Cumulative prior discharged Ah divided by public rated capacity",
    )
    ambient_temperature: float
    segment_duration_s: float = Field(..., gt=0)
    segment_energy_wh: float = Field(..., ge=0)
    voltage_start_v: float
    voltage_end_v: float
    voltage_mean_v: float
    voltage_std_v: float = Field(..., ge=0)
    voltage_linear_slope_v_per_s: float
    voltage_curvature: float
    current_mean_c_rate: float = Field(
        ..., ge=0,
        description="Mean absolute segment current divided by public rated capacity"
    )
    current_std_c_rate: float = Field(..., ge=0)
    temperature_mean_c: float
    temperature_std_c: float = Field(..., ge=0)
    temperature_rise_c: float
    temperature_slope_c_per_s: float
    temperature_observed: float = Field(..., ge=0, le=1)


class V3PredictRequest(V3CycleFeatures):
    """Single-cycle v3 request with battery and optional model identifiers."""
    battery_id: str = Field(..., description="Battery identifier")
    model_name: Optional[str] = Field(default=None, description="Optional deployable v3 model key")


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


class V3BatchPredictRequest(BaseModel):
    """Validated batch of leakage-safe partial-discharge observations."""
    battery_id: str
    cycles: list[V3CycleFeatures]


# ── Recommendation ───────────────────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    """Request for operational recommendations."""
    battery_id: str
    current_cycle: int = Field(..., ge=1)
    current_soh: float = Field(..., ge=0, le=100)
    ambient_temperature: float = Field(default=24.0)
    top_k: int = Field(default=5, ge=1, le=20)
    model_name: Optional[str] = Field(default=None, description="Optional model key override")


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
    validation_status: str = "exploratory_prototype"
    disclaimer: str = (
        "Illustrative scenario analysis only; recommendations and RUL values "
        "have not been prospectively validated for operational control."
    )


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
