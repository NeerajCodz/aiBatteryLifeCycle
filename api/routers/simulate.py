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
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.model_registry import (
    FEATURE_COLS_SCALAR, V3_FEATURE_COLS, classify_degradation, soh_to_color,
    registry_v3 as registry_v2,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["simulation"])

_Q_NOM     = 2.0      # NASA PCoE nominal capacity (Ah)

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

# Optional learned Re/Rct progression model (trained in NB03).
_RE_RCT_MODEL_PATH = Path("artifacts") / "v3" / "models" / "classical" / "re_rct_progression.joblib"
try:
    _re_rct_model = joblib.load(_RE_RCT_MODEL_PATH)
except Exception:
    _re_rct_model = None


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
def _build_feature_matrix(
    b: BatterySimConfig,
    cycle_arr: np.ndarray,
    re_arr: np.ndarray,
    rct_arr: np.ndarray,
    soh_rolling_override: np.ndarray | None = None,
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
    cycles = np.asarray(cycle_arr, dtype=np.float64)
    N = len(cycles)
    steps_elapsed = np.maximum(cycles - cycles[0], 0.0)

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
    cap_per_step = np.maximum(initial_cap + b.delta_capacity * steps_elapsed, 0.0)

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
    soh_est = np.maximum(b.initial_soh - deg_pct_per_cycle * steps_elapsed, 0.0)
    # rolling mean via cumsum (O(N), no Python loop)
    window = 10
    csoh  = np.cumsum(np.concatenate([[0.0], soh_est]))
    cnt   = np.minimum(np.arange(1, N + 1), window)
    start = np.maximum(np.arange(N + 1)[1:] - window, 0)
    soh_rolling = (csoh[np.arange(1, N + 1)] - csoh[start]) / cnt

    if soh_rolling_override is not None:
        soh_rolling = np.asarray(soh_rolling_override, dtype=np.float64)

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


def _rolling_mean(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Fast rolling mean with min_periods=1 semantics."""
    x = np.asarray(arr, dtype=np.float64)
    n = len(x)
    csum = np.cumsum(np.concatenate([[0.0], x]))
    idx = np.arange(1, n + 1)
    start = np.maximum(idx - window, 0)
    count = np.minimum(idx, window)
    return (csum[idx] - csum[start]) / count


def _ml_re_rct(
    re0: float,
    rct0: float,
    cycle_arr: np.ndarray,
    elapsed_cycles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict Re/Rct from learned regressors and anchor to user initial state."""
    if _re_rct_model is None:
        # Minimal linear fallback if helper model is unavailable.
        n = np.asarray(elapsed_cycles, dtype=np.float64)
        re_arr = np.minimum(re0 + 0.00012 * n, 2.0)
        rct_arr = np.minimum(rct0 + 0.00018 * n, 3.0)
        return re_arr, rct_arr

    x = np.asarray(cycle_arr, dtype=np.float64).reshape(-1, 1)
    re_pred = _re_rct_model["re_model"].predict(x)
    rct_pred = _re_rct_model["rct_model"].predict(x)
    re_arr = np.clip(re_pred + (re0 - float(re_pred[0])), 0.0, 2.0)
    rct_arr = np.clip(rct_pred + (rct0 - float(rct_pred[0])), 0.0, 3.0)
    return re_arr, rct_arr


def _compute_rul_and_eol(
    soh_arr:     np.ndarray,
    initial_soh: float,
    eol_thr:     float,
    cycle_start: int,
    cycle_arr:   np.ndarray,
    elapsed_cycles: np.ndarray,
    cycle_dur:   float,
    tu_sec:      float | None,
) -> tuple[np.ndarray, np.ndarray, Optional[int], Optional[float]]:
    """Vectorized RUL and EOL from SOH trajectory.

    Returns (rul_cycles, rul_time, eol_cycle, eol_time).
    Uses rolling-average degradation rate for smooth RUL estimate.
    """
    cycles = np.asarray(cycle_arr, dtype=np.int64)
    elapsed = np.asarray(elapsed_cycles, dtype=np.float64)

    # Rolling average degradation rate (smoothed, avoids division-by-zero)
    soh_prev = np.concatenate([[initial_soh], soh_arr[:-1]])
    step_deg = np.maximum(0.0, soh_prev - soh_arr)
    cum_deg  = np.cumsum(step_deg)
    avg_rate = np.maximum(cum_deg / np.maximum(elapsed, 1.0), 1e-6)

    rul_cycles = np.where(soh_arr > eol_thr, (soh_arr - eol_thr) / avg_rate, 0.0)
    rul_time = (rul_cycles * cycle_dur / tu_sec) if tu_sec is not None else rul_cycles.copy()

    # EOL: first step where SOH <= threshold
    below     = soh_arr <= eol_thr
    eol_cycle: Optional[int]   = None
    eol_time:  Optional[float] = None
    if below.any():
        idx       = int(np.argmax(below))
        eol_cycle = int(cycles[idx])
        elapsed_s = max(0.0, float(eol_cycle - cycle_start) * cycle_dur)
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

    # Deep sequence models need per-sample tensors and are not used in this endpoint.
    # Classical + ensemble models use batch predict_array().
    family = registry_v2.model_meta.get(model_name, {}).get("family", "classical")
    is_deep = family in ("deep_pytorch", "deep_keras")
    ml_batchable = (
        req.use_ml
        and not is_deep
        and (model_name == "best_ensemble" or model_name in registry_v2.models)
    )

    # Determine scaler note for logging (mirrors training decision exactly)
    if registry_v2.model_meta.get(model_name, {}).get("requires_scaling", False):
        scaler_note = "standard_scaler"
    elif model_name == "best_ensemble":
        scaler_note = "per-component (tree=none / linear=standard_scaler)"
    else:
        scaler_note = "none (tree)"

    effective_model = "linear_fallback"
    log.info(
        "simulate: %d batteries x %d steps | model=%s | batchable=%s | scaler=%s | unit=%s",
        len(req.batteries), N, model_name, ml_batchable, scaler_note, time_unit,
    )

    results: list[BatterySimResult] = []

    for b in req.batteries:
        cycles_per_step = 1.0 if tu_sec is None else (tu_sec / max(b.cycle_duration, 1e-6))
        elapsed_cycles = np.arange(N, dtype=np.float64) * cycles_per_step
        cycle_arr_float = b.start_cycle + elapsed_cycles
        cycle_arr = np.maximum(b.start_cycle, np.floor(cycle_arr_float).astype(np.int64))

        # 1. Re/Rct progression - prefer learned model, fallback to simple linear slopes.
        re_arr, rct_arr = _ml_re_rct(b.Re, b.Rct, cycle_arr, elapsed_cycles)

        # 2. SOH prediction with two-pass soh_rolling_mean correction.
        if ml_batchable:
            X_pass1 = _build_feature_matrix(b, cycle_arr, re_arr, rct_arr)
            try:
                soh_pass1, effective_model = registry_v2.predict_array(X_pass1, model_name)
                soh_roll = _rolling_mean(soh_pass1, window=5)
                X_pass2 = _build_feature_matrix(
                    b,
                    cycle_arr,
                    re_arr,
                    rct_arr,
                    soh_rolling_override=soh_roll,
                )
                soh_arr, effective_model = registry_v2.predict_array(X_pass2, model_name)
            except Exception as exc:
                log.warning(
                    "predict_array failed for %s (%s) - falling back to linear",
                    b.battery_id, exc,
                )
                deg_pct_per_cycle = abs(b.delta_capacity) / _Q_NOM * 100.0
                soh_arr = np.clip(b.initial_soh - deg_pct_per_cycle * (elapsed_cycles + 1.0), 0.0, 100.0)
                effective_model = "linear_fallback"
        else:
            deg_pct_per_cycle = abs(b.delta_capacity) / _Q_NOM * 100.0
            soh_arr = np.clip(b.initial_soh - deg_pct_per_cycle * (elapsed_cycles + 1.0), 0.0, 100.0)
            effective_model = "linear_fallback"

        soh_arr = np.clip(soh_arr, 0.0, 100.0)

        # 3. RUL + EOL - vectorized
        rul_cycles, rul_time, eol_cycle, eol_time = _compute_rul_and_eol(
            soh_arr,
            b.initial_soh,
            eol_thr,
            b.start_cycle,
            cycle_arr,
            elapsed_cycles,
            b.cycle_duration,
            tu_sec,
        )

        # 4. Time axis - vectorized
        time_arr = np.arange(N, dtype=np.float64) if tu_sec is not None else cycle_arr.astype(np.float64)

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

