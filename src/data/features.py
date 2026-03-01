"""
src.data.features
=================
Feature engineering for battery lifecycle prediction.

Derived features
----------------
- **SOC** (State of Charge) via Coulomb counting per cycle
- **SOH** (State of Health) as percentage of nominal capacity
- **RUL** (Remaining Useful Life) in cycles until EOL
- **Per-cycle scalar features** for classical ML models:
    peak_voltage, min_voltage, voltage_range, avg_current, avg_temp,
    temp_rise, discharge_time, charge_time, coulombic_efficiency,
    Re_at_cycle, Rct_at_cycle, delta_capacity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import (
    NOMINAL_CAPACITY_AH,
    get_eol_threshold,
    load_cycle_csv,
    load_discharge_capacities,
    load_impedance_scalars,
    load_metadata,
)


# ── SOC via Coulomb counting ─────────────────────────────────────────────────
def compute_soc(
    cycle_df: pd.DataFrame,
    nominal_capacity_ah: float = NOMINAL_CAPACITY_AH,
) -> pd.Series:
    """Compute State of Charge (%) for a single discharge cycle via Coulomb counting.

    SOC starts at 100% and decreases as charge is consumed:
        ΔQ_i = I_i · Δt_i / 3600   (Ah, with Δt in seconds)
        SOC_i = 100 × (1 − cumulative_Q / nominal_capacity)

    Parameters
    ----------
    cycle_df : pd.DataFrame
        Must contain ``Current_measured`` (A) and ``Time`` (s).
    nominal_capacity_ah : float
        Nameplate capacity in Ah (default 2.0).

    Returns
    -------
    pd.Series
        SOC in percent [0, 100], same length as *cycle_df*.
    """
    current = cycle_df["Current_measured"].values
    time_s = cycle_df["Time"].values

    # Time deltas (first delta = 0)
    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0

    # Charge consumed (Ah); use absolute current for discharge (current < 0)
    dq = np.abs(current) * dt / 3600.0
    cumulative_q = np.cumsum(dq)

    soc = 100.0 * (1.0 - cumulative_q / nominal_capacity_ah)
    return pd.Series(soc, index=cycle_df.index, name="SoC")


# ── SOH ──────────────────────────────────────────────────────────────────────
def compute_soh(
    measured_capacity: float | np.ndarray | pd.Series,
    nominal_capacity_ah: float = NOMINAL_CAPACITY_AH,
) -> float | np.ndarray | pd.Series:
    """Compute State of Health (%) as measured capacity / nominal × 100."""
    return (measured_capacity / nominal_capacity_ah) * 100.0


# ── RUL ──────────────────────────────────────────────────────────────────────
def compute_rul_series(
    capacity_series: pd.Series,
    eol_threshold: float,
) -> pd.Series:
    """Compute Remaining Useful Life (cycles) for a capacity-fade series.

    For each cycle *i*, RUL_i = (first cycle where capacity < eol_threshold) − i.
    If the battery never reaches EOL, use the last available cycle as a censored estimate.
    """
    cap = capacity_series.values
    # Find EOL cycle index
    eol_indices = np.where(cap < eol_threshold)[0]
    if len(eol_indices) > 0:
        eol_cycle = eol_indices[0]
    else:
        eol_cycle = len(cap)  # censored — battery didn't reach EOL
    rul = eol_cycle - np.arange(len(cap))
    rul = np.clip(rul, 0, None)
    return pd.Series(rul, index=capacity_series.index, name="RUL")


# ── Degradation state classification ────────────────────────────────────────
def classify_degradation_state(soh: float | np.ndarray) -> str | np.ndarray:
    """Classify battery degradation into 4 states based on SOH %.

    States:
        0 – Healthy     (SOH ≥ 90%)
        1 – Aging        (80% ≤ SOH < 90%)
        2 – Near-EOL     (70% ≤ SOH < 80%)
        3 – EOL          (SOH < 70%)
    """
    soh_arr = np.asarray(soh)
    labels = np.full(soh_arr.shape, 3, dtype=int)  # default EOL
    labels[soh_arr >= 90] = 0
    labels[(soh_arr >= 80) & (soh_arr < 90)] = 1
    labels[(soh_arr >= 70) & (soh_arr < 80)] = 2
    if soh_arr.ndim == 0:
        return int(labels)
    return labels


DEGRADATION_LABELS = {0: "Healthy", 1: "Aging", 2: "Near-EOL", 3: "EOL"}


# ── Per-cycle scalar feature extraction ─────────────────────────────────────
def extract_cycle_features(cycle_df: pd.DataFrame) -> dict:
    """Extract scalar features from a single discharge or charge cycle.

    Parameters
    ----------
    cycle_df : pd.DataFrame
        Raw time-series for one cycle.

    Returns
    -------
    dict
        Feature dictionary with keys:
        peak_voltage, min_voltage, voltage_range, avg_current,
        avg_temp, temp_rise, cycle_duration
    """
    v = cycle_df.get("Voltage_measured")
    i = cycle_df.get("Current_measured")
    t = cycle_df.get("Temperature_measured")
    time = cycle_df.get("Time")

    features: dict = {}

    if v is not None and len(v) > 0:
        features["peak_voltage"] = float(v.max())
        features["min_voltage"] = float(v.min())
        features["voltage_range"] = float(v.max() - v.min())
    if i is not None and len(i) > 0:
        features["avg_current"] = float(np.abs(i).mean())
    if t is not None and len(t) > 0:
        features["avg_temp"] = float(t.mean())
        features["temp_rise"] = float(t.max() - t.min())
    if time is not None and len(time) > 0:
        features["cycle_duration"] = float(time.iloc[-1] - time.iloc[0])

    return features


def build_battery_feature_dataset(
    *,
    exclude_corrupt: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build full per-cycle feature dataset across all batteries.

    Combines:
    - Capacity fade information from metadata
    - Impedance scalars (Re, Rct) from impedance tests (nearest-cycle interpolated)
    - Per-cycle scalar features extracted from raw discharge CSVs
    - Derived targets: SOH (%), RUL (cycles), degradation_state (0–3)

    Returns
    -------
    pd.DataFrame
        One row per discharge cycle, with all features and targets.
    """
    from tqdm import tqdm

    # 1. Load capacity fade data
    cap_df = load_discharge_capacities(exclude_corrupt=exclude_corrupt)
    cap_df["SoH"] = compute_soh(cap_df["Capacity"])

    # 2. Compute RUL per battery
    rul_parts: list[pd.Series] = []
    for bid, group in cap_df.groupby("battery_id"):
        eol = get_eol_threshold(bid)
        rul = compute_rul_series(group["Capacity"], eol)
        rul_parts.append(rul)
    cap_df["RUL"] = pd.concat(rul_parts)

    # 3. Degradation state
    cap_df["degradation_state"] = classify_degradation_state(cap_df["SoH"].values)

    # 4. Impedance scalars — merge nearest impedance measurement per cycle
    imp_df = load_impedance_scalars(exclude_corrupt=exclude_corrupt)
    if not imp_df.empty:
        # For each battery, forward-fill impedance values across discharge cycles
        imp_pivot = imp_df.groupby("battery_id").apply(
            lambda g: g.set_index("cycle_number")[["Re", "Rct"]], include_groups=False
        )
        re_map: dict[str, pd.Series] = {}
        rct_map: dict[str, pd.Series] = {}
        for bid in imp_df["battery_id"].unique():
            if bid in imp_pivot.index.get_level_values(0):
                sub = imp_pivot.loc[bid]
                re_map[bid] = sub["Re"]
                rct_map[bid] = sub["Rct"]

        re_vals, rct_vals = [], []
        for _, row in cap_df.iterrows():
            bid = row["battery_id"]
            cn = row["cycle_number"]
            if bid in re_map and len(re_map[bid]) > 0:
                # Nearest impedance cycle
                idx = re_map[bid].index
                nearest = idx[np.argmin(np.abs(idx - cn))]
                re_vals.append(float(re_map[bid].loc[nearest]))
                rct_vals.append(float(rct_map[bid].loc[nearest]))
            else:
                re_vals.append(np.nan)
                rct_vals.append(np.nan)
        cap_df["Re"] = re_vals
        cap_df["Rct"] = rct_vals

    # 5. Extract per-cycle features from raw discharge CSVs
    meta = load_metadata(exclude_corrupt=exclude_corrupt, parse_dates=False)
    dis_meta = meta[meta["type"] == "discharge"].copy()
    dis_meta = dis_meta.sort_values(["battery_id", "test_id"]).reset_index(drop=True)
    dis_meta["cycle_number"] = dis_meta.groupby("battery_id").cumcount()

    # Build a uid lookup
    uid_lookup = dis_meta.set_index(["battery_id", "cycle_number"])["uid"].to_dict()

    extra_features: list[dict] = []
    iterator = tqdm(cap_df.iterrows(), total=len(cap_df), desc="Extracting features") if verbose else cap_df.iterrows()
    for _, row in iterator:
        uid = uid_lookup.get((row["battery_id"], row["cycle_number"]))
        if uid is not None:
            try:
                cdf = load_cycle_csv(uid)
                feats = extract_cycle_features(cdf)
            except (FileNotFoundError, Exception):
                feats = {}
        else:
            feats = {}
        extra_features.append(feats)

    feat_df = pd.DataFrame(extra_features, index=cap_df.index)
    result = pd.concat([cap_df, feat_df], axis=1)

    # 6. Compute delta_capacity (capacity change from previous cycle)
    result["delta_capacity"] = result.groupby("battery_id")["Capacity"].diff().fillna(0)

    # 7. Coulombic efficiency placeholder — needs charge data too, fill NaN for now
    if "coulombic_efficiency" not in result.columns:
        result["coulombic_efficiency"] = np.nan

    return result.reset_index(drop=True)
