"""Quantify split and target-proxy effects for the v1/v2/v3 discussion."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import regression_metrics
from src.evaluation.protocol import grouped_train_val_test_folds
from src.utils.config import FEATURE_COLS_V3

LEAKAGE_PROXY_COLS = [
    "current_capacity_retention",
    "current_cumulative_capacity",
    "current_delta_capacity",
    "current_soh_rolling_mean",
]


def _add_legacy_target_proxies(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["battery_id", "cycle_number"]).copy()
    out["current_capacity_retention"] = out["capacity_ah"] / out["reference_capacity_ah"]
    out["current_cumulative_capacity"] = out.groupby("battery_id")["capacity_ah"].cumsum()
    out["current_delta_capacity"] = out.groupby("battery_id")["capacity_ah"].diff().fillna(0)
    out["current_soh_rolling_mean"] = out.groupby("battery_id")["SoH"].transform(
        lambda values: values.rolling(5, min_periods=1).mean()
    )
    return out.reset_index(drop=True)


def _chronological_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train, test = [], []
    for _, group in frame.groupby("battery_id"):
        ordered = group.sort_values("cycle_number").index.to_numpy()
        split = max(1, int(0.8 * len(ordered)))
        train.extend(ordered[:split])
        test.extend(ordered[split:])
    return np.asarray(train), np.asarray(test)


def _score(frame: pd.DataFrame, features: list[str], train_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> dict[str, float]:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )),
    ])
    model.fit(frame.loc[train_idx, features], frame.loc[train_idx, "SoH"])
    pred = model.predict(frame.loc[test_idx, features])
    return regression_metrics(
        frame.loc[test_idx, "SoH"].to_numpy(), pred, n_predictors=len(features)
    )


def run_protocol_comparison(
    project_root: str | Path,
    *,
    seeds: tuple[int, ...] = (17, 42, 2026),
) -> pd.DataFrame:
    project_root = Path(project_root)
    frame = pd.read_csv(project_root / "artifacts" / "v3" / "features" / "nasa" / "features.csv")
    frame = _add_legacy_target_proxies(frame)
    groups = frame["battery_id"].astype(str).to_numpy()
    rows = []
    for seed in seeds:
        all_idx = np.arange(len(frame))
        v1_train, v1_test = train_test_split(all_idx, test_size=0.2, random_state=seed)
        v2_train, v2_test = _chronological_indices(frame)
        protocols = [("v1_random_cycle", 1, v1_train, v1_test), ("v2_chronological", 1, v2_train, v2_test)]
        for fold, (train, _, test) in enumerate(
            grouped_train_val_test_folds(groups, n_splits=5, random_state=seed), start=1
        ):
            protocols.append(("v3_grouped_battery", fold, train, test))
        for protocol, fold, train_idx, test_idx in protocols:
            for feature_contract, features in [
                ("safe_partial_cycle", FEATURE_COLS_V3),
                ("legacy_target_proxies", FEATURE_COLS_V3 + LEAKAGE_PROXY_COLS),
            ]:
                metrics = _score(frame, features, train_idx, test_idx, seed)
                rows.append({
                    "seed": seed,
                    "protocol": protocol,
                    "fold": fold,
                    "feature_contract": feature_contract,
                    "train_batteries": frame.loc[train_idx, "battery_id"].nunique(),
                    "test_batteries": frame.loc[test_idx, "battery_id"].nunique(),
                    **metrics,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.project_root / "artifacts" / "v3" / "results" / "protocol_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = run_protocol_comparison(args.project_root)
    frame.to_csv(output, index=False)
    print(frame.groupby(["protocol", "feature_contract"])[["mae", "rmse", "r2", "within_5pp"]].mean())
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
