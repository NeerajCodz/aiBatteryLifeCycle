"""Adapters that normalize NASA, CALCE, and Oxford discharge cycles.

Every adapter yields the same :class:`DischargeCycle` record. Dataset-specific
parsing is isolated here so feature extraction and validation remain identical
across domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


@dataclass
class DischargeCycle:
    dataset: str
    battery_id: str
    cycle_number: int
    capacity_ah: float
    ambient_temperature_c: float
    measurements: pd.DataFrame
    source: str


def _as_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _matlab_datenum_to_elapsed_seconds(values: object) -> np.ndarray:
    """Convert MATLAB serial-day timestamps to elapsed seconds."""
    days = _as_array(values)
    return (days - days[0]) * 86_400.0


def _sanitize_temperature_channel(
    values: object,
    *,
    ambient_temperature_c: float,
    max_median_deviation_c: float = 20.0,
) -> np.ndarray:
    """Mark a grossly inconsistent whole-cycle temperature channel missing."""
    temperature = _as_array(values)
    finite = temperature[np.isfinite(temperature)]
    if finite.size == 0 or abs(float(np.median(finite)) - ambient_temperature_c) > max_median_deviation_c:
        return np.full(temperature.shape, np.nan)
    return temperature


def iter_nasa_mat_cycles(paths: Iterable[str | Path]) -> Iterator[DischargeCycle]:
    """Yield discharge cycles from original NASA PCoE MATLAB files."""
    from scipy.io import loadmat

    next_cycle_number: dict[str, int] = {}
    for raw_path in paths:
        path = Path(raw_path)
        battery_id = path.stem
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        root = payload.get(battery_id)
        if root is None:
            candidates = [value for key, value in payload.items() if not key.startswith("__")]
            if len(candidates) != 1:
                raise ValueError(f"Could not identify battery structure in {path}")
            root = candidates[0]
        cycles = np.atleast_1d(root.cycle)
        discharge_index = next_cycle_number.get(battery_id, 0)
        for cycle in cycles:
            cycle_type = str(cycle.type).strip().lower()
            if cycle_type != "discharge":
                continue
            data = cycle.data
            capacity_values = np.asarray(data.Capacity, dtype=float).reshape(-1)
            if capacity_values.size != 1 or not np.isfinite(capacity_values[0]):
                continue
            capacity = float(capacity_values[0])
            time = _as_array(data.Time)
            voltage = _as_array(data.Voltage_measured)
            current = _as_array(data.Current_measured)
            temperature = _as_array(data.Temperature_measured)
            lengths = [len(time), len(voltage), len(current)]
            n = min(lengths)
            if n < 3:
                continue
            if len(temperature) < n:
                temperature = np.full(n, np.nan)
            frame = pd.DataFrame({
                "Time": time[:n],
                "Voltage_measured": voltage[:n],
                "Current_measured": current[:n],
                "Temperature_measured": temperature[:n],
            })
            yield DischargeCycle(
                dataset="NASA",
                battery_id=battery_id,
                cycle_number=discharge_index,
                capacity_ah=capacity,
                ambient_temperature_c=float(np.asarray(cycle.ambient_temperature).squeeze()),
                measurements=frame,
                source=str(path),
            )
            discharge_index += 1
        next_cycle_number[battery_id] = discharge_index


def _calce_cycles_from_frame(
    frame: pd.DataFrame,
    *,
    battery_id: str,
    source: str,
    ambient_temperature_c: float = 25.0,
) -> list[tuple[pd.Timestamp, DischargeCycle]]:
    """Extract candidate CALCE discharges from one Arbin worksheet."""
    required = {"Test_Time(s)", "Date_Time", "Cycle_Index", "Current(A)", "Voltage(V)", "Discharge_Capacity(Ah)"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing CALCE columns: {sorted(missing)}")
    records: list[tuple[pd.Timestamp, DischargeCycle]] = []
    for raw_cycle, group in frame.groupby("Cycle_Index", sort=True):
        discharge = group[pd.to_numeric(group["Current(A)"], errors="coerce") < -0.05].copy()
        if len(discharge) < 3:
            continue
        all_capacity = pd.to_numeric(group["Discharge_Capacity(Ah)"], errors="coerce")
        # Some Arbin exports continue the counter across steps/files.  The
        # within-cycle range is invariant to that offset and is the delivered Ah.
        capacity = all_capacity.max() - all_capacity.min()
        if not np.isfinite(capacity) or not 0.3 <= capacity <= 1.5:
            continue
        test_time = pd.to_numeric(discharge["Test_Time(s)"], errors="coerce")
        measurements = pd.DataFrame({
            "Time": test_time - test_time.iloc[0],
            "Voltage_measured": pd.to_numeric(discharge["Voltage(V)"], errors="coerce"),
            "Current_measured": pd.to_numeric(discharge["Current(A)"], errors="coerce"),
            "Temperature_measured": np.nan,
        }).dropna(subset=["Time", "Voltage_measured", "Current_measured"])
        started = pd.to_datetime(discharge["Date_Time"], errors="coerce").min()
        if pd.isna(started):
            started = pd.Timestamp.min
        records.append((started, DischargeCycle(
            dataset="CALCE",
            battery_id=battery_id,
            cycle_number=int(raw_cycle),
            capacity_ah=float(capacity),
            ambient_temperature_c=float(ambient_temperature_c),
            measurements=measurements.reset_index(drop=True),
            source=source,
        )))
    return records


def iter_calce_xlsx_cycles(
    root: str | Path,
    *,
    battery_ids: tuple[str, ...] = ("CS2_35", "CS2_36", "CS2_37", "CS2_38"),
    ambient_temperature_c: float = 25.0,
) -> Iterator[DischargeCycle]:
    """Yield chronological, de-duplicated discharges from CALCE ZIP extracts."""
    root = Path(root)
    for battery_id in battery_ids:
        candidates: list[tuple[pd.Timestamp, DischargeCycle]] = []
        for path in sorted(root.glob(f"{battery_id}/**/*.xlsx")):
            book = pd.ExcelFile(path, engine="openpyxl")
            data_sheets = [name for name in book.sheet_names if name.lower() != "info"]
            for sheet in data_sheets:
                frame = pd.read_excel(book, sheet_name=sheet)
                candidates.extend(_calce_cycles_from_frame(
                    frame,
                    battery_id=battery_id,
                    source=f"{path}:{sheet}",
                    ambient_temperature_c=ambient_temperature_c,
                ))
        candidates.sort(key=lambda pair: pair[0])
        seen: set[tuple[int, int]] = set()
        ordered: list[DischargeCycle] = []
        for started, record in candidates:
            key = (int(started.value // 10**9), int(round(record.capacity_ah * 100_000)))
            if key not in seen:
                seen.add(key)
                ordered.append(record)
        for cycle_number, record in enumerate(ordered):
            record.cycle_number = cycle_number
            yield record


def iter_oxford_mat_cycles(path: str | Path) -> Iterator[DischargeCycle]:
    """Yield Oxford 1C characterization discharges from Cells 1-8."""
    from scipy.io import loadmat

    path = Path(path)
    for cell_number in range(1, 9):
        battery_id = f"Cell{cell_number}"
        payload = loadmat(
            path,
            variable_names=[battery_id],
            squeeze_me=True,
            struct_as_record=False,
        )
        cell = payload[battery_id]
        for cycle_name in sorted(cell._fieldnames, key=lambda name: int(name[3:])):
            diagnostic = getattr(cell, cycle_name)
            if not hasattr(diagnostic, "C1dc"):
                continue
            discharge = diagnostic.C1dc
            time_s = _matlab_datenum_to_elapsed_seconds(discharge.t)
            voltage = _as_array(discharge.v)
            charge_mah = _as_array(discharge.q)
            temperature = _sanitize_temperature_channel(
                discharge.T,
                ambient_temperature_c=40.0,
            )
            n = min(len(time_s), len(voltage), len(charge_mah), len(temperature))
            if n < 3:
                continue
            time_s, voltage = time_s[:n], voltage[:n]
            charge_mah, temperature = charge_mah[:n], temperature[:n]
            capacity_ah = float(np.nanmax(charge_mah) - np.nanmin(charge_mah)) / 1000.0
            with np.errstate(divide="ignore", invalid="ignore"):
                current_a = np.gradient(charge_mah, time_s) * 3.6
            current_a = (
                pd.Series(current_a)
                .replace([np.inf, -np.inf], np.nan)
                .interpolate(limit_direction="both")
                .to_numpy()
            )
            frame = pd.DataFrame({
                "Time": time_s,
                "Voltage_measured": voltage,
                "Current_measured": current_a,
                "Temperature_measured": temperature,
            })
            yield DischargeCycle(
                dataset="Oxford",
                battery_id=battery_id,
                cycle_number=int(cycle_name[3:]),
                capacity_ah=capacity_ah,
                ambient_temperature_c=40.0,
                measurements=frame,
                source=f"{path}:{battery_id}/{cycle_name}/C1dc",
            )


def cycle_inventory(cycles: Iterable[DischargeCycle]) -> pd.DataFrame:
    """Create the one-row-per-cycle audit table used by Notebook 01."""
    return pd.DataFrame([
        {
            "dataset": cycle.dataset,
            "battery_id": cycle.battery_id,
            "cycle_number": cycle.cycle_number,
            "capacity_ah": cycle.capacity_ah,
            "ambient_temperature_c": cycle.ambient_temperature_c,
            "n_measurements": len(cycle.measurements),
            "source": cycle.source,
        }
        for cycle in cycles
    ])
