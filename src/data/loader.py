"""
src.data.loader
===============
Data loading utilities for the NASA PCoE Li-ion Battery Dataset.

This module handles:
- Loading and parsing ``metadata.csv`` (including MATLAB-format date vectors)
- Loading individual cycle CSV files (charge / discharge / impedance)
- Aggregating all discharge or charge cycles into a single DataFrame
- Loading impedance scalar features (Re, Rct) from metadata

Excluded batteries: B0049–B0052 (confirmed software crash / corrupt data).
"""

from __future__ import annotations

import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "cleaned_dataset"
METADATA_PATH = DATASET_DIR / "metadata.csv"
DATA_DIR = DATASET_DIR / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ── Constants ────────────────────────────────────────────────────────────────
EXCLUDED_BATTERIES = {"B0049", "B0050", "B0051", "B0052"}
NOMINAL_CAPACITY_AH = 2.0
EOL_30PCT = 1.4  # 30 % fade → 1.4 Ah
EOL_20PCT = 1.6  # 20 % fade → 1.6 Ah

# Battery groups with their EOL thresholds
BATTERY_EOL_MAP: dict[str, float] = {}
for _bid in ("B0005", "B0006", "B0007", "B0018",
             "B0025", "B0026", "B0027", "B0028",
             "B0029", "B0030", "B0031", "B0032",
             "B0041", "B0042", "B0043", "B0044",
             "B0045", "B0046", "B0047", "B0048",
             "B0053", "B0054", "B0055", "B0056"):
    BATTERY_EOL_MAP[_bid] = EOL_30PCT
for _bid in ("B0033", "B0034", "B0036",
             "B0038", "B0039", "B0040"):
    BATTERY_EOL_MAP[_bid] = EOL_20PCT


# ── MATLAB date-vector parser ───────────────────────────────────────────────
def _parse_matlab_datevec(s: str) -> datetime | None:
    """Parse a MATLAB-style date vector string into a Python datetime.

    Handles formats like:
        ``[2010. 7. 21. 15. 0. 35.093]``
        ``[2.008e+03, 4.000e+00, 2.000e+00, ...]``
    """
    if not isinstance(s, str) or s.strip() in ("", "[]"):
        return None
    try:
        # Strip brackets and split on comma / whitespace
        inner = s.strip().strip("[]")
        # Replace multiple spaces / commas with single comma
        inner = re.sub(r"[,\s]+", ",", inner.strip())
        parts = [float(x) for x in inner.split(",") if x]
        if len(parts) < 6:
            return None
        yr, mo, dy, hr, mi, sc = parts[:6]
        return datetime(int(yr), int(mo), int(dy), int(hr), int(mi), int(sc))
    except (ValueError, OverflowError):
        return None


# ── Metadata ─────────────────────────────────────────────────────────────────
def load_metadata(
    *,
    exclude_corrupt: bool = True,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load ``metadata.csv`` with optional date parsing and corrupt-battery exclusion.

    Parameters
    ----------
    exclude_corrupt : bool
        If True, drop rows for B0049–B0052.
    parse_dates : bool
        If True, add a ``datetime`` column parsed from the raw ``start_time`` field.

    Returns
    -------
    pd.DataFrame
        One row per test/cycle.
    """
    df = pd.read_csv(METADATA_PATH)

    # Coerce Capacity to numeric (handles '[]' and empty strings)
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df["Re"] = pd.to_numeric(df["Re"], errors="coerce")
    df["Rct"] = pd.to_numeric(df["Rct"], errors="coerce")

    if exclude_corrupt:
        df = df[~df["battery_id"].isin(EXCLUDED_BATTERIES)].reset_index(drop=True)

    if parse_dates:
        df["datetime"] = df["start_time"].apply(_parse_matlab_datevec)

    return df


# ── Individual cycle data ────────────────────────────────────────────────────
def load_cycle_csv(uid: int | str) -> pd.DataFrame:
    """Load a single cycle CSV by its UID (filename number).

    Parameters
    ----------
    uid : int or str
        The global unique ID, e.g. 1 → ``00001.csv``.

    Returns
    -------
    pd.DataFrame
        Raw time-series data for that cycle.
    """
    fname = f"{int(uid):05d}.csv"
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Cycle CSV not found: {path}")
    return pd.read_csv(path)


# ── Aggregated cycle loading ─────────────────────────────────────────────────
def load_all_cycles(
    cycle_type: Literal["discharge", "charge", "impedance"],
    *,
    exclude_corrupt: bool = True,
    max_batteries: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load and concatenate all cycles of a given type across all batteries.

    Adds ``battery_id``, ``test_id``, ``uid``, ``cycle_number`` (0-based per
    battery for this cycle type), and ``Capacity`` (for discharge cycles).

    Parameters
    ----------
    cycle_type : {"discharge", "charge", "impedance"}
    exclude_corrupt : bool
    max_batteries : int or None
        Limit number of batteries processed (useful for debugging).
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Concatenated time-series data with metadata columns appended.
    """
    from tqdm import tqdm

    meta = load_metadata(exclude_corrupt=exclude_corrupt, parse_dates=False)
    subset = meta[meta["type"] == cycle_type].copy()

    if max_batteries is not None:
        keep_bats = subset["battery_id"].unique()[:max_batteries]
        subset = subset[subset["battery_id"].isin(keep_bats)]

    # Assign cycle_number per battery within this type
    subset = subset.sort_values(["battery_id", "test_id"]).reset_index(drop=True)
    subset["cycle_number"] = subset.groupby("battery_id").cumcount()

    frames: list[pd.DataFrame] = []
    iterator = tqdm(subset.iterrows(), total=len(subset), desc=f"Loading {cycle_type}") if verbose else subset.iterrows()

    for _, row in iterator:
        try:
            df = load_cycle_csv(row["uid"])
        except FileNotFoundError:
            continue

        df["battery_id"] = row["battery_id"]
        df["test_id"] = row["test_id"]
        df["uid"] = row["uid"]
        df["cycle_number"] = row["cycle_number"]

        if cycle_type == "discharge":
            df["Capacity"] = row["Capacity"]
        if cycle_type == "impedance":
            df["Re"] = row["Re"]
            df["Rct"] = row["Rct"]

        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_discharge_capacities(
    *,
    exclude_corrupt: bool = True,
    drop_zero: bool = True,
) -> pd.DataFrame:
    """Return a compact DataFrame of discharge capacity per cycle per battery.

    Columns: ``battery_id``, ``cycle_number``, ``Capacity``, ``ambient_temperature``.
    This is much faster than `load_all_cycles("discharge")` because it only
    reads metadata — no individual CSV loading.
    """
    meta = load_metadata(exclude_corrupt=exclude_corrupt, parse_dates=True)
    dis = meta[meta["type"] == "discharge"].copy()
    dis = dis.sort_values(["battery_id", "test_id"]).reset_index(drop=True)
    dis["cycle_number"] = dis.groupby("battery_id").cumcount()

    cols = ["battery_id", "cycle_number", "Capacity", "ambient_temperature"]
    if "datetime" in dis.columns:
        cols.append("datetime")
    result = dis[cols].copy()

    if drop_zero:
        result = result[result["Capacity"] > 0].dropna(subset=["Capacity"])

    return result.reset_index(drop=True)


def load_impedance_scalars(*, exclude_corrupt: bool = True) -> pd.DataFrame:
    """Return Re and Rct per cycle per battery from impedance tests (metadata only)."""
    meta = load_metadata(exclude_corrupt=exclude_corrupt, parse_dates=True)
    imp = meta[meta["type"] == "impedance"].copy()
    imp = imp.sort_values(["battery_id", "test_id"]).reset_index(drop=True)
    imp["cycle_number"] = imp.groupby("battery_id").cumcount()
    cols = ["battery_id", "cycle_number", "Re", "Rct", "ambient_temperature"]
    if "datetime" in imp.columns:
        cols.append("datetime")
    return imp[cols].dropna(subset=["Re", "Rct"]).reset_index(drop=True)


def get_battery_ids(*, exclude_corrupt: bool = True) -> list[str]:
    """Return sorted list of available battery IDs."""
    meta = load_metadata(exclude_corrupt=exclude_corrupt, parse_dates=False)
    return sorted(meta["battery_id"].unique().tolist())


def get_eol_threshold(battery_id: str) -> float:
    """Get EOL capacity threshold for a given battery."""
    return BATTERY_EOL_MAP.get(battery_id, EOL_30PCT)
