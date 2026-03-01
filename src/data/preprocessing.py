"""
src.data.preprocessing
======================
Data preprocessing, windowing, splitting, and scaler management.

Provides:
- Battery-grouped train/test split (no data leakage between batteries)
- Sliding-window sequence builder for sequential models (LSTM, Transformer)
- Scaler fitting / saving / loading (StandardScaler ↔ MinMaxScaler)
- Down-sampling of per-cycle time-series to fixed-length bins
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data.loader import ARTIFACTS_DIR

SCALER_DIR = ARTIFACTS_DIR / "scalers"
SCALER_DIR.mkdir(parents=True, exist_ok=True)


# ── Train/test split by battery groups ───────────────────────────────────────
def group_battery_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42,
    battery_col: str = "battery_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/test by grouping at the battery level.

    This prevents data leakage: all cycles from a battery appear in either
    train or test, never both.

    Parameters
    ----------
    df : pd.DataFrame
    train_ratio : float
        Fraction of batteries used for training.
    random_state : int
    battery_col : str

    Returns
    -------
    (train_df, test_df) : tuple of pd.DataFrame
    """
    rng = np.random.RandomState(random_state)
    # Sort first so shuffle is deterministic regardless of insertion order
    batteries = np.array(sorted(df[battery_col].unique()))
    rng.shuffle(batteries)
    n_train = max(1, int(len(batteries) * train_ratio))
    train_bats = set(batteries[:n_train])
    test_bats = set(batteries[n_train:])
    train_df = df[df[battery_col].isin(train_bats)].reset_index(drop=True)
    test_df = df[df[battery_col].isin(test_bats)].reset_index(drop=True)
    return train_df, test_df


# ── Leave-one-battery-out split ──────────────────────────────────────────────
def leave_one_battery_out(
    df: pd.DataFrame,
    test_battery: str,
    battery_col: str = "battery_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave one battery out for testing (zero-shot generalization).

    Parameters
    ----------
    df : pd.DataFrame
    test_battery : str
        Battery ID to hold out (e.g. "B0005").

    Returns
    -------
    (train_df, test_df) : tuple of pd.DataFrame
    """
    test_df = df[df[battery_col] == test_battery].reset_index(drop=True)
    train_df = df[df[battery_col] != test_battery].reset_index(drop=True)
    return train_df, test_df


# ── Sliding window sequences ────────────────────────────────────────────────
def make_sliding_windows(
    values: np.ndarray,
    window_size: int = 32,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping sliding windows from a 1D or 2D array.

    For a 1D input of shape ``(T,)`` → windows of shape ``(N, window_size)``
    and targets of shape ``(N,)`` (the element right after each window).

    For a 2D input of shape ``(T, F)`` → windows ``(N, window_size, F)``
    and targets ``(N, F)`` or ``(N,)`` depending on downstream usage.

    Parameters
    ----------
    values : np.ndarray
        Shape ``(T,)`` or ``(T, F)``.
    window_size : int
    stride : int

    Returns
    -------
    (X, y) : tuple of np.ndarray
    """
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    T, F = values.shape
    X, y = [], []
    for i in range(0, T - window_size, stride):
        X.append(values[i : i + window_size])
        y.append(values[i + window_size])
    X = np.array(X)
    y = np.array(y)
    if F == 1:
        y = y.ravel()
    return X, y


def make_multistep_windows(
    values: np.ndarray,
    input_window: int = 32,
    output_window: int = 8,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows with multi-step targets.

    Parameters
    ----------
    values : np.ndarray
        Shape ``(T,)`` or ``(T, F)``.
    input_window : int
    output_window : int
    stride : int

    Returns
    -------
    (X, y) : tuple of np.ndarray
        X shape: ``(N, input_window, F)``, y shape: ``(N, output_window, F)`` or ``(N, output_window)``.
    """
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    T, F = values.shape
    X, y = [], []
    for i in range(0, T - input_window - output_window + 1, stride):
        X.append(values[i : i + input_window])
        y.append(values[i + input_window : i + input_window + output_window])
    X = np.array(X)
    y = np.array(y)
    if F == 1:
        y = y.squeeze(-1)
    return X, y


# ── Fixed-length bin downsampling ────────────────────────────────────────────
def downsample_to_bins(
    cycle_df: pd.DataFrame,
    n_bins: int = 20,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Downsample a single-cycle DataFrame to exactly *n_bins* rows.

    Each bin is the mean of a roughly equal-sized chunk.
    """
    if columns is not None:
        cycle_df = cycle_df[columns]
    chunks = np.array_split(cycle_df.values, n_bins)
    binned = np.array([chunk.mean(axis=0) for chunk in chunks])
    return pd.DataFrame(binned, columns=cycle_df.columns if columns is None else columns)


# ── Scaler utilities ─────────────────────────────────────────────────────────
def fit_and_save_scaler(
    data: np.ndarray | pd.DataFrame,
    scaler_type: Literal["standard", "minmax"] = "standard",
    name: str = "default",
) -> StandardScaler | MinMaxScaler:
    """Fit a scaler on training data and persist to disk.

    Parameters
    ----------
    data : array-like
        Training data.
    scaler_type : {"standard", "minmax"}
    name : str
        Filename stem for saved scaler.

    Returns
    -------
    Fitted scaler object.
    """
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    if isinstance(data, pd.DataFrame):
        data = data.values
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    scaler.fit(data)
    path = SCALER_DIR / f"{name}_{scaler_type}.joblib"
    joblib.dump(scaler, path)
    return scaler


def load_scaler(name: str, scaler_type: Literal["standard", "minmax"] = "standard"):
    """Load a previously saved scaler from disk."""
    path = SCALER_DIR / f"{name}_{scaler_type}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    return joblib.load(path)


# ── Feature/target column definitions ────────────────────────────────────────
FEATURE_COLS_SCALAR = [
    "cycle_number",
    "ambient_temperature",
    "peak_voltage",
    "min_voltage",
    "voltage_range",
    "avg_current",
    "avg_temp",
    "temp_rise",
    "cycle_duration",
    "Re",
    "Rct",
    "delta_capacity",
]

TARGET_SOH = "SoH"
TARGET_RUL = "RUL"
TARGET_DEGRADATION = "degradation_state"

SEQUENCE_FEATURE_COLS = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "SoC",
]
