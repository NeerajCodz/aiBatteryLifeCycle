"""
api.routers.simulate
====================
Bulk battery lifecycle simulation endpoint - vectorized ML-driven.

Performance design (O(1) Python overhead per battery regardless of step count):
    1. SEI impedance growth  - numpy cumsum (no Python loop)
    2. Feature matrix build  - numpy column_stack -> (N_steps, n_features)
    3. ML prediction         - single model.predict() call via predict_array()
    4. RUL / EOL             - numpy diff / cumsum / searchsorted
    5. Classify / colorize   - numpy searchsorted on pre-built label arrays

Scaler dispatch mirrors training exactly:
    Tree models (RF / ET / XGB / LGB / GB)  -> raw numpy   (no scaler)
    Linear / SVR / KNN                       -> standard_scaler.joblib.transform(X)
    best_ensemble                            -> per-component dispatch (same rules)
    Deep sequence models (PyTorch / Keras)   -> not batchable, falls back to physics
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.model_registry import (
    FEATURE_COLS_SCALAR, V3_FEATURE_COLS, classify_degradation, soh_to_color,
    registry_v3 as registry_v2,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["simulation"])

# -- Physics constants --------------------------------------------------------
_EA_OVER_R = 6200.0   # Ea/R in Kelvin
_Q_NOM     = 2.0      # NASA PCoE nominal capacity (Ah)
_T_REF     = 24.0     # Reference ambient temperature (deg C)
_I_REF     = 1.82     # Reference discharge current (A)
_V_REF     = 4.19     # Reference peak voltage (V)

_TIME_UNIT_SECONDS: dict[str, float | None] = {
    "cycle":  None,        "second": 1.0,        "minute": 60.0,
    "hour":   3_600.0,     "day":    86_400.0,   "week":   604_800.0,
    "month":  2_592_000.0, "year":   31_536_000.0,
}
_TIME_UNIT_LABELS: dict[str, str] = {
    "cycle":  "Cycles",  "second": "Seconds", "minute": "Minutes",
    "hour":   "Hours",   "day":    "Days",    "week":   "Weeks",
    "month":  "Months",  "year":   "Years",
}

# Column index map - must stay in sync with FEATURE_COLS_SCALAR (12 features)
_F = {col: idx for idx, col in enumerate(FEATURE_COLS_SCALAR)}
# Column index map for V3_FEATURE_COLS (18 features)
_F3 = {col: idx for idx, col in enumerate(V3_FEATURE_COLS)}

# Pre-built label/color arrays for O(1) numpy-vectorized classification
_SOH_BINS   = np.array([70.0, 80.0, 90.0])                       # searchsorted thresholds
_DEG_LABELS = np.array(["End-of-Life", "Degraded", "Moderate", "Healthy"], dtype=object)
_COLOR_HEX  = np.array(["#ef4444",     "#f97316",  "#eab308",  "#22c55e"], dtype=object)


def _vec_classify(soh: np.ndarray) -> list[str]:
    """Vectorized classify_degradation - single numpy call, no Python for-loop."""
    return _DEG_LABELS[np.searchsorted(_SOH_BINS, soh, side="left")].tolist()


def _vec_color(soh: np.ndarray) -> list[str]:
    """Vectorized soh_to_color - single numpy call, no Python for-loop."""
    return _COLOR_HEX[np.searchsorted(_SOH_BINS, soh, side="left")].tolist()


# -- Schemas ------------------------------------------------------------------
class BatterySimConfig(BaseModel):
    battery_id:          str
    label:               Optional[str] = None
    initial_soh:         float = Field(default=100.0, ge=0.0, le=100.0)
    start_cycle:         int   = Field(default=1, ge=1)
    ambient_temperature: float = Field(default=24.0)
    peak_voltage:        float = Field(default=4.19)
    min_voltage:         float = Field(default=2.61)
    avg_current:         float = Field(default=1.82)
    avg_temp:            float = Field(default=32.6)
    temp_rise:           float = Field(default=14.7)
    cycle_duration:      float = Field(default=3690.0)
    Re:                  float = Field(default=0.045)
    Rct:                 float = Field(default=0.069)
    delta_capacity:      float = Field(default=-0.005)


class SimulateRequest(BaseModel):
    batteries:     List[BatterySimConfig]
    steps:         int           = Field(default=200, ge=1, le=10_000)
    time_unit:     str           = Field(default="day")
    eol_threshold: float         = Field(default=70.0, ge=0.0, le=100.0)
    model_name:    Optional[str] = Field(default=None)
    use_ml:        bool          = Field(default=True)


class BatterySimResult(BaseModel):
    battery_id:          str
    label:               Optional[str]
    soh_history:         List[float]
    rul_history:         List[float]
    rul_time_history:    List[float]
    re_history:          List[float]
    rct_history:         List[float]
    cycle_history:       List[int]
    time_history:        List[float]
    degradation_history: List[str]
    color_history:       List[str]
    eol_cycle:           Optional[int]
    eol_time:            Optional[float]
    final_soh:           float
    final_rul:           float
    deg_rate_avg:        float
    model_used:          str = "physics"


class SimulateResponse(BaseModel):
    results:         List[BatterySimResult]
    time_unit:       str
    time_unit_label: str
    steps:           int
    model_used:      str = "physics"


# -- Helpers ------------------------------------------------------------------
def _sei_growth(
    re0: float, rct0: float, steps: int, temp_f: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized SEI impedance growth over `steps` cycles.

    Returns (re_arr, rct_arr) each shaped (steps,) using cumsum - no Python loop.
    Matches the incremental SEI model used during feature engineering (NB02).
    """
    s         = np.arange(steps, dtype=np.float64)
    delta_re  = 0.00012 * temp_f * (1.0 + s * 5e-5)
    delta_rct = 0.00018 * temp_f * (1.0 + s * 8e-5)
    re_arr    = np.minimum(re0  + np.cumsum(delta_re),  2.0)
    rct_arr   = np.minimum(rct0 + np.cumsum(delta_rct), 3.0)
    return re_arr, rct_arr


def _build_feature_matrix(
    b: BatterySimConfig, steps: int,
    re_arr: np.ndarray, rct_arr: np.ndarray,
) -> np.ndarray:
    """Build (steps, n_features) feature matrix in registry.feature_cols order.

    For v3 (18 features) the extra 6 engineered columns are estimated from physics:
      - capacity_retention: current capacity / initial capacity
      - cumulative_energy:  cumsumed capacity per cycle (Ah)
      - dRe_dn / dRct_dn:  per-cycle derivative of SEI impedances
      - soh_rolling_mean:   rolling mean of physics-estimated SOH trajectory
      - voltage_slope:      assumed constant (0) in simulation

    Column ordering uses registry_v2.feature_cols so predictions are correct
    regardless of whether the registry is v1/v2 (12 cols) or v3 (18 cols).
    """
    N      = steps
    step_a = np.arange(N, dtype=np.float64)
    cycles = step_a + b.start_cycle

    # ---- 12 base features (always present) ---------------------------------
    feat_dict: dict[str, np.ndarray] = {
        "cycle_number":        cycles,
        "ambient_temperature": np.full(N, b.ambient_temperature),
        "peak_voltage":        np.full(N, b.peak_voltage),
        "min_voltage":         np.full(N, b.min_voltage),
        "voltage_range":       np.full(N, b.peak_voltage - b.min_voltage),
        "avg_current":         np.full(N, b.avg_current),
        "avg_temp":            np.full(N, b.avg_temp),
        "temp_rise":           np.full(N, b.temp_rise),
        "cycle_duration":      np.full(N, b.cycle_duration),
        "Re":                  re_arr,
        "Rct":                 rct_arr,
        "delta_capacity":      np.full(N, b.delta_capacity),
    }

    # ---- 6 extra v3 features (estimated from physics) ----------------------
    initial_cap = max(b.initial_soh / 100.0 * _Q_NOM, 1e-6)  # Ah
    cap_per_step = np.maximum(initial_cap + b.delta_capacity * step_a, 0.0)

    # capacity_retention = current_capacity / initial_capacity (ratio ~0-1)
    cap_retention = np.clip(cap_per_step / initial_cap, 0.0, None)

    # cumulative energy delivered (Ah)
    cum_energy = np.cumsum(cap_per_step)

    # per-cycle SEI impedance derivatives
    dRe_dn  = np.diff(re_arr,  prepend=b.Re)
    dRct_dn = np.diff(rct_arr, prepend=b.Rct)

    # physics-estimated SOH rolling mean (window=10, min_periods=1) ---
    # used as a proxy since soh_rolling_mean is a v3 training feature
    deg_pct_per_cycle = abs(b.delta_capacity) / _Q_NOM * 100.0
    soh_est = np.maximum(b.initial_soh - deg_pct_per_cycle * step_a, 0.0)
    # rolling mean via cumsum (O(N), no Python loop)
    window = 10
    csoh  = np.cumsum(np.concatenate([[0.0], soh_est]))
    cnt   = np.minimum(np.arange(1, N + 1), window)
    start = np.maximum(np.arange(N + 1)[1:] - window, 0)
    soh_rolling = (csoh[np.arange(1, N + 1)] - csoh[start]) / cnt

    feat_dict.update({
        "capacity_retention": cap_retention,
        "cumulative_energy":  cum_energy,
        "dRe_dn":             dRe_dn,
        "dRct_dn":            dRct_dn,
        "soh_rolling_mean":   soh_rolling,
        "voltage_slope":      np.zeros(N),
        "coulombic_efficiency": np.zeros(N),  # always 0 in training data
    })

    # Build matrix in registry's feature_cols order; unknown cols default to 0
    feat_cols = registry_v2.feature_cols  # 12 for v1/v2, 18 for v3
    return np.column_stack([feat_dict.get(col, np.zeros(N)) for col in feat_cols])


def _physics_soh(b: BatterySimConfig, steps: int, temp_f: float) -> np.ndarray:
    """Pure Arrhenius physics fallback - fully vectorized, returns (steps,) SOH."""
    rate_base = float(np.clip(abs(b.delta_capacity) / _Q_NOM * 100.0, 0.005, 1.5))
    curr_f    = 1.0 + max(0.0, (b.avg_current - _I_REF) * 0.18)
    volt_f    = 1.0 + max(0.0, (b.peak_voltage - _V_REF) * 0.55)
    age_f     = 1.0 + (0.08 if b.initial_soh < 85.0 else 0.0) + (0.12 if b.initial_soh < 75.0 else 0.0)
    deg_rate  = float(np.clip(rate_base * temp_f * curr_f * volt_f * age_f, 0.0, 2.0))
    soh_arr   = b.initial_soh - deg_rate * np.arange(1, steps + 1, dtype=np.float64)
    return np.clip(soh_arr, 0.0, 100.0)


def _compute_rul_and_eol(
    soh_arr:     np.ndarray,
    initial_soh: float,
    eol_thr:     float,
    cycle_start: int,
    cycle_dur:   float,
    tu_sec:      float | None,
) -> tuple[np.ndarray, np.ndarray, Optional[int], Optional[float]]:
    """Vectorized RUL and EOL from SOH trajectory.

    Returns (rul_cycles, rul_time, eol_cycle, eol_time).
    Uses rolling-average degradation rate for smooth RUL estimate.
    """
    N      = len(soh_arr)
    steps  = np.arange(N, dtype=np.float64)
    cycles = (cycle_start + steps).astype(np.int64)

    # Rolling average degradation rate (smoothed, avoids division-by-zero)
    soh_prev = np.concatenate([[initial_soh], soh_arr[:-1]])
    step_deg = np.maximum(0.0, soh_prev - soh_arr)
    cum_deg  = np.cumsum(step_deg)
    avg_rate = np.maximum(cum_deg / (steps + 1), 1e-6)

    rul_cycles = np.where(soh_arr > eol_thr, (soh_arr - eol_thr) / avg_rate, 0.0)
    rul_time   = (rul_cycles * cycle_dur / tu_sec) if tu_sec is not None else rul_cycles.copy()

    # EOL: first step where SOH <= threshold
    below     = soh_arr <= eol_thr
    eol_cycle: Optional[int]   = None
    eol_time:  Optional[float] = None
    if below.any():
        idx       = int(np.argmax(below))
        eol_cycle = int(cycles[idx])
        elapsed_s = eol_cycle * cycle_dur
        eol_time  = round((elapsed_s / tu_sec) if tu_sec else float(eol_cycle), 3)

    return rul_cycles, rul_time, eol_cycle, eol_time


# -- Endpoint -----------------------------------------------------------------
@router.post(
    "/simulate",
    response_model=SimulateResponse,
    summary="Bulk battery lifecycle simulation (vectorized, ML-driven)",
)
async def simulate_batteries(req: SimulateRequest):
    """
    Vectorized simulation: builds all N feature rows at once per battery,
    dispatches to the ML model as a single batch predict() call, then
    post-processes entirely with numpy (no Python for-loops).

    Scaler usage mirrors NB03 training exactly:
      - Tree models (RF/ET/XGB/LGB/GB): raw numpy X, no scaler
      - Linear/SVR/KNN:                 standard_scaler.joblib.transform(X)
      - best_ensemble:                  per-component family dispatch
    """
    time_unit = req.time_unit.lower()
    if time_unit not in _TIME_UNIT_SECONDS:
        time_unit = "day"

    tu_sec   = _TIME_UNIT_SECONDS[time_unit]
    tu_label = _TIME_UNIT_LABELS[time_unit]
    eol_thr  = req.eol_threshold
    N        = req.steps

    model_name = req.model_name or registry_v2.default_model or "best_ensemble"

    # Deep sequence models need per-sample tensors — cannot batch vectorise
    # Tree / linear / ensemble models support predict_array() batch calls.
    # We do NOT gate on model_count here: predict_array() has a try/except
    # fallback to physics, so a partial load still works.
    family = registry_v2.model_meta.get(model_name, {}).get("family", "classical")
    is_deep = family in ("deep_pytorch", "deep_keras")
    ml_batchable = (
        req.use_ml
        and not is_deep
        and (model_name == "best_ensemble" or model_name in registry_v2.models)
    )

    # Determine scaler note for logging (mirrors training decision exactly)
    if model_name in registry_v2._LINEAR_FAMILIES:
        scaler_note = "standard_scaler"
    elif model_name == "best_ensemble":
        scaler_note = "per-component (tree=none / linear=standard_scaler)"
    else:
        scaler_note = "none (tree)"

    effective_model = "physics"
    log.info(
        "simulate: %d batteries x %d steps | model=%s | batchable=%s | scaler=%s | unit=%s",
        len(req.batteries), N, model_name, ml_batchable, scaler_note, time_unit,
    )

    results: list[BatterySimResult] = []

    for b in req.batteries:
        # 1. SEI impedance growth - vectorized cumsum (no Python loop)
        T_K     = 273.15 + b.ambient_temperature
        T_REF_K = 273.15 + _T_REF
        temp_f  = float(np.clip(math.exp(_EA_OVER_R * (1.0 / T_REF_K - 1.0 / T_K)), 0.15, 25.0))
        re_arr, rct_arr = _sei_growth(b.Re, b.Rct, N, temp_f)

        # 2. SOH prediction - single batch call regardless of N
        #    predict_array() applies the correct scaler per model family,
        #    exactly matching the preprocessing done during NB03 training:
        #      * standard_scaler.transform(X)  for Ridge / SVR / KNN / Lasso / ElasticNet
        #      * raw numpy                      for RF / ET / XGB / LGB / GB
        #      * per-component dispatch         for best_ensemble
        if ml_batchable:
            X = _build_feature_matrix(b, N, re_arr, rct_arr)
            try:
                soh_arr, effective_model = registry_v2.predict_array(X, model_name)
            except Exception as exc:
                log.warning(
                    "predict_array failed for %s (%s) - falling back to physics",
                    b.battery_id, exc,
                )
                soh_arr = _physics_soh(b, N, temp_f)
                effective_model = "physics"
        else:
            soh_arr = _physics_soh(b, N, temp_f)
            effective_model = "physics"

        soh_arr = np.clip(soh_arr, 0.0, 100.0)

        # 3. RUL + EOL - vectorized
        rul_cycles, rul_time, eol_cycle, eol_time = _compute_rul_and_eol(
            soh_arr, b.initial_soh, eol_thr, b.start_cycle, b.cycle_duration, tu_sec,
        )

        # 4. Time axis - vectorized
        cycle_arr = np.arange(b.start_cycle, b.start_cycle + N, dtype=np.int64)
        time_arr  = (
            (cycle_arr * b.cycle_duration / tu_sec).astype(np.float64)
            if tu_sec is not None
            else cycle_arr.astype(np.float64)
        )

        # 5. Labels + colors - fully vectorized via numpy searchsorted
        #    Replaces O(N) Python for-loop with a single C-level call
        deg_h   = _vec_classify(soh_arr)
        color_h = _vec_color(soh_arr)

        avg_dr = float(np.mean(np.maximum(0.0, -np.diff(soh_arr, prepend=b.initial_soh))))

        # 6. Build result - numpy round + .tolist() (no per-element Python conversion)
        results.append(BatterySimResult(
            battery_id          = b.battery_id,
            label               = b.label or b.battery_id,
            soh_history         = np.round(soh_arr,    3).tolist(),
            rul_history         = np.round(rul_cycles, 1).tolist(),
            rul_time_history    = np.round(rul_time,   2).tolist(),
            re_history          = np.round(re_arr,     6).tolist(),
            rct_history         = np.round(rct_arr,    6).tolist(),
            cycle_history       = cycle_arr.tolist(),
            time_history        = np.round(time_arr,   3).tolist(),
            degradation_history = deg_h,
            color_history       = color_h,
            eol_cycle           = eol_cycle,
            eol_time            = eol_time,
            final_soh           = round(float(soh_arr[-1]),    3),
            final_rul           = round(float(rul_cycles[-1]), 1),
            deg_rate_avg        = round(avg_dr, 6),
            model_used          = effective_model,
        ))

    return SimulateResponse(
        results         = results,
        time_unit       = time_unit,
        time_unit_label = tu_label,
        steps           = N,
        model_used      = effective_model,
    )

