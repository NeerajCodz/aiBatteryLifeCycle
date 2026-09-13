"""Leakage-safe current-cycle SOH labels and partial-discharge inputs.

The current cycle's full discharge capacity is used only to define the target.
Predictors are computed from the early 4.0 V to 3.6 V discharge segment and
from usage accumulated strictly before the current cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import FEATURE_COLS_V3, SEQUENCE_FEATURE_COLS_V3


@dataclass(frozen=True)
class PartialCycleConfig:
    upper_voltage_v: float = 4.0
    lower_voltage_v: float = 3.6
    sequence_bins: int = 64
    reference_cycles: int = 3


def add_soh_labels(
    cycles: pd.DataFrame,
    *,
    battery_col: str = "battery_id",
    capacity_col: str = "Capacity",
    cycle_col: str = "cycle_number",
    reference_cycles: int = 3,
) -> pd.DataFrame:
    """Add reference capacity and SOH without exposing them as predictors."""
    required = {battery_col, capacity_col, cycle_col}
    missing = required.difference(cycles.columns)
    if missing:
        raise KeyError(f"Missing columns for SOH labels: {sorted(missing)}")
    out = cycles.sort_values([battery_col, cycle_col]).copy()
    valid = out[np.isfinite(out[capacity_col]) & (out[capacity_col] > 0)]
    refs = (
        valid.groupby(battery_col, sort=False)[capacity_col]
        .apply(lambda values: reference_capacity_from_characterization(
            values, reference_cycles=reference_cycles
        ))
        .rename("reference_capacity_ah")
    )
    out = out.join(refs, on=battery_col)
    out["SoH"] = 100.0 * out[capacity_col] / out["reference_capacity_ah"]
    return out.sort_index()


def reference_capacity_from_characterization(
    capacities_ah: pd.Series | list[float] | np.ndarray,
    *,
    reference_cycles: int = 3,
    characterization_fraction: float = 0.85,
) -> float:
    """Estimate Q_ref from the first three full-capacity characterizations.

    NASA archives sometimes begin with intentionally partial diagnostic
    discharges. Full-capacity candidates are therefore defined relative to the
    battery's robust 95th-percentile capacity, and the earliest candidates are
    used. This affects target calibration only; capacity is never a predictor.
    """
    values = pd.to_numeric(pd.Series(capacities_ah), errors="coerce")
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < reference_cycles:
        raise ValueError("Too few valid capacity characterizations")
    threshold = characterization_fraction * float(values.quantile(0.95))
    candidates = values[values >= threshold]
    if len(candidates) < reference_cycles:
        raise ValueError("Too few full-capacity characterization cycles")
    return float(candidates.iloc[:reference_cycles].median())


def prior_equivalent_full_cycles(
    capacities_ah: pd.Series,
    rated_capacity_ah: float,
) -> pd.Series:
    """Cumulative discharged throughput strictly before each cycle."""
    if not np.isfinite(rated_capacity_ah) or rated_capacity_ah <= 0:
        raise ValueError("rated_capacity_ah must be positive")
    cap = pd.to_numeric(capacities_ah, errors="coerce").fillna(0.0)
    return cap.shift(1, fill_value=0.0).cumsum() / rated_capacity_ah


def select_partial_discharge(
    cycle_df: pd.DataFrame,
    *,
    upper_voltage_v: float = 4.0,
    lower_voltage_v: float = 3.6,
) -> pd.DataFrame:
    """Return the contiguous descending-voltage segment from 4.0 V to 3.6 V."""
    if upper_voltage_v <= lower_voltage_v:
        raise ValueError("upper_voltage_v must exceed lower_voltage_v")
    if "Voltage_measured" not in cycle_df:
        raise KeyError("Voltage_measured is required")
    frame = cycle_df.reset_index(drop=True).copy()
    voltage = pd.to_numeric(frame["Voltage_measured"], errors="coerce").to_numpy()
    start_candidates = np.flatnonzero(voltage <= upper_voltage_v)
    if start_candidates.size == 0:
        raise ValueError("Cycle never reaches the upper partial-voltage boundary")
    start = int(start_candidates[0])
    end_candidates = np.flatnonzero(voltage[start:] <= lower_voltage_v)
    if end_candidates.size == 0:
        raise ValueError("Cycle never reaches the lower partial-voltage boundary")
    end = start + int(end_candidates[0])
    segment = frame.iloc[start : end + 1].copy()
    if len(segment) < 3:
        raise ValueError("Partial discharge segment contains fewer than three samples")
    return segment.reset_index(drop=True)


def _linear_slope(time_s: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(time_s) & np.isfinite(values)
    if mask.sum() < 2 or np.ptp(time_s[mask]) == 0:
        return 0.0
    return float(np.polyfit(time_s[mask], values[mask], 1)[0])


def extract_partial_cycle_features(
    cycle_df: pd.DataFrame,
    *,
    cycle_index: int,
    prior_efc: float,
    ambient_temperature: float,
    rated_capacity_ah: float,
    config: PartialCycleConfig = PartialCycleConfig(),
) -> dict[str, float]:
    """Extract the 18-feature v3 vector from an observable partial segment."""
    seg = select_partial_discharge(
        cycle_df,
        upper_voltage_v=config.upper_voltage_v,
        lower_voltage_v=config.lower_voltage_v,
    )
    time_s = pd.to_numeric(seg.get("Time"), errors="coerce").to_numpy(dtype=float)
    time_s = time_s - time_s[0]
    voltage = pd.to_numeric(seg["Voltage_measured"], errors="coerce").to_numpy(dtype=float)
    current = pd.to_numeric(seg.get("Current_measured"), errors="coerce").to_numpy(dtype=float)
    current_c = np.abs(current) / rated_capacity_ah
    observed_temp = "Temperature_measured" in seg and seg["Temperature_measured"].notna().any()
    if observed_temp:
        temperature = pd.to_numeric(seg["Temperature_measured"], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    else:
        temperature = np.full(len(seg), float(ambient_temperature))
    dt = np.diff(time_s, prepend=time_s[0])
    energy_wh = float(np.sum(np.abs(current) * voltage * np.maximum(dt, 0.0)) / 3600.0)
    v_slope = _linear_slope(time_s, voltage)
    if len(voltage) >= 3 and np.ptp(time_s) > 0:
        x = time_s / np.ptp(time_s)
        curvature = float(np.polyfit(x, voltage, 2)[0])
    else:
        curvature = 0.0
    features = {
        "cycle_index": float(cycle_index),
        "prior_equivalent_full_cycles": float(prior_efc),
        "ambient_temperature": float(ambient_temperature),
        "segment_duration_s": float(time_s[-1]),
        "segment_energy_wh": energy_wh,
        "voltage_start_v": float(voltage[0]),
        "voltage_end_v": float(voltage[-1]),
        "voltage_mean_v": float(np.nanmean(voltage)),
        "voltage_std_v": float(np.nanstd(voltage)),
        "voltage_linear_slope_v_per_s": v_slope,
        "voltage_curvature": curvature,
        "current_mean_c_rate": float(np.nanmean(current_c)),
        "current_std_c_rate": float(np.nanstd(current_c)),
        "temperature_mean_c": float(np.nanmean(temperature)),
        "temperature_std_c": float(np.nanstd(temperature)),
        "temperature_rise_c": float(np.nanmax(temperature) - np.nanmin(temperature)),
        "temperature_slope_c_per_s": _linear_slope(time_s, temperature),
        "temperature_observed": float(observed_temp),
    }
    assert list(features) == FEATURE_COLS_V3
    return features


def make_partial_cycle_sequence(
    cycle_df: pd.DataFrame,
    *,
    ambient_temperature: float,
    rated_capacity_ah: float,
    config: PartialCycleConfig = PartialCycleConfig(),
) -> pd.DataFrame:
    """Interpolate the partial segment to a fixed 64 x 5 sequence."""
    seg = select_partial_discharge(
        cycle_df,
        upper_voltage_v=config.upper_voltage_v,
        lower_voltage_v=config.lower_voltage_v,
    )
    time_s = pd.to_numeric(seg["Time"], errors="coerce").to_numpy(dtype=float)
    time_s = time_s - time_s[0]
    if time_s[-1] <= 0:
        raise ValueError("Partial segment has non-positive duration")
    x = time_s / time_s[-1]
    grid = np.linspace(0.0, 1.0, config.sequence_bins)
    voltage = pd.to_numeric(seg["Voltage_measured"], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    current = pd.to_numeric(seg["Current_measured"], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    observed_temp = "Temperature_measured" in seg and seg["Temperature_measured"].notna().any()
    if observed_temp:
        temperature = pd.to_numeric(seg["Temperature_measured"], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    else:
        temperature = np.full(len(seg), float(ambient_temperature))
    out = pd.DataFrame({
        "normalized_time": grid,
        "voltage_v": np.interp(grid, x, voltage),
        "current_c_rate": np.abs(np.interp(grid, x, current)) / rated_capacity_ah,
        "temperature_c": np.interp(grid, x, temperature),
        "temperature_observed": float(observed_temp),
    })
    return out[SEQUENCE_FEATURE_COLS_V3]
