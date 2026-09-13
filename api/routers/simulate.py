"""
api.routers.simulate
====================
Exploratory, vectorized battery lifecycle scenario endpoint.

This route intentionally does not call the v3 SOH estimators. Those estimators
were validated only for current-cycle observations, not counterfactual future
trajectories. The returned curves are deterministic illustrations driven by the
user-provided capacity-loss assumption and must not be treated as forecasts.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

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
    use_ml:        bool          = Field(
        default=False,
        description="Reserved for compatibility; v3 simulation remains an exploratory physics prototype.",
    )


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
    validation_status: str = "exploratory_prototype"
    disclaimer: str = (
        "Illustrative trajectory only; not a validated forecast or control model."
    )


# -- Helpers ------------------------------------------------------------------
def _ml_re_rct(
    re0: float,
    rct0: float,
    cycle_arr: np.ndarray,
    elapsed_cycles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return illustrative impedance-growth curves anchored to user inputs."""
    del cycle_arr  # retained in the signature for compatibility with older callers
    n = np.asarray(elapsed_cycles, dtype=np.float64)
    return np.minimum(re0 + 0.00012 * n, 2.0), np.minimum(rct0 + 0.00018 * n, 3.0)


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
    summary="Exploratory battery lifecycle scenario (not a forecast)",
)
async def simulate_batteries(req: SimulateRequest):
    """
    Vectorized deterministic scenario based on the user-supplied capacity-loss
    rate. ``model_name`` and ``use_ml`` are accepted only for compatibility and
    do not select a fitted model.
    """
    time_unit = req.time_unit.lower()
    if time_unit not in _TIME_UNIT_SECONDS:
        time_unit = "day"

    tu_sec   = _TIME_UNIT_SECONDS[time_unit]
    tu_label = _TIME_UNIT_LABELS[time_unit]
    eol_thr  = req.eol_threshold
    N        = req.steps

    effective_model = "exploratory_physics_prototype"
    log.info(
        "simulate: %d batteries x %d steps | requested=%s | effective=%s | unit=%s",
        len(req.batteries), N, req.model_name, effective_model, time_unit,
    )

    results: list[BatterySimResult] = []

    for b in req.batteries:
        cycles_per_step = 1.0 if tu_sec is None else (tu_sec / max(b.cycle_duration, 1e-6))
        elapsed_cycles = np.arange(N, dtype=np.float64) * cycles_per_step
        cycle_arr_float = b.start_cycle + elapsed_cycles
        cycle_arr = np.maximum(b.start_cycle, np.floor(cycle_arr_float).astype(np.int64))

        # 1. Illustrative Re/Rct progression.
        re_arr, rct_arr = _ml_re_rct(b.Re, b.Rct, cycle_arr, elapsed_cycles)

        # 2. Deterministic SOH scenario; this is not a fitted-model prediction.
        deg_pct_per_cycle = abs(b.delta_capacity) / _Q_NOM * 100.0
        soh_arr = np.clip(b.initial_soh - deg_pct_per_cycle * (elapsed_cycles + 1.0), 0.0, 100.0)

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
        validation_status = "exploratory_prototype",
        disclaimer = (
            "Illustrative trajectory only; the v3 SOH estimator was not trained "
            "or validated for future-life simulation or operational control."
        ),
    )

