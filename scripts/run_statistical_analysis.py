"""Generate uncertainty, paired tests, residual, importance, and robustness tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import per_battery_evaluation, regression_metrics
from src.evaluation.protocol import (
    battery_cluster_bootstrap,
    grouped_train_val_test_folds,
    paired_battery_wilcoxon,
)
from src.experiments.classical import classical_factories
from src.utils.config import FEATURE_COLS_V3


def _within_prediction_files(results: Path, dataset: str) -> list[Path]:
    return [
        path for path in sorted(results.glob(f"{dataset}_*_predictions.csv"))
        if not path.name.startswith("nasa_to_")
    ]


def _averaged_predictions(results: Path, dataset: str) -> pd.DataFrame:
    files = _within_prediction_files(results, dataset)
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    keys = ["model", "battery_id", "cycle_number", "y_true"]
    if "row_index" in frame.columns:
        keys.insert(1, "row_index")
    averaged = frame.groupby(keys, as_index=False)["y_pred"].mean()
    averaged["residual"] = averaged["y_true"] - averaged["y_pred"]
    averaged["dataset"] = dataset.upper() if dataset == "nasa" else dataset.title()
    return averaged


def _bootstrap_best_models(
    averaged: dict[str, pd.DataFrame], n_bootstrap: int
) -> pd.DataFrame:
    rows = []
    for dataset, frame in averaged.items():
        if frame.empty:
            continue
        scores = frame.groupby("model").apply(
            lambda g: np.mean(np.abs(g["y_true"] - g["y_pred"])),
            include_groups=False,
        )
        best = str(scores.idxmin())
        selected = frame[frame["model"] == best]
        ci = battery_cluster_bootstrap(
            selected["y_true"].to_numpy(),
            selected["y_pred"].to_numpy(),
            selected["battery_id"].to_numpy(),
            n_predictors=18,
            n_bootstrap=n_bootstrap,
            random_state=42,
        )
        ci.insert(0, "model", best)
        ci.insert(0, "dataset", dataset.upper() if dataset == "nasa" else dataset.title())
        rows.append(ci)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _battery_scores_and_tests(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model, group in frame.groupby("model"):
        per = per_battery_evaluation(
            group["y_true"].to_numpy(), group["y_pred"].to_numpy(), group["battery_id"]
        )
        per["model"] = model
        rows.append(per)
    scores = pd.concat(rows, ignore_index=True)
    reference = scores.groupby("model")["mae"].mean().idxmin()
    tests = paired_battery_wilcoxon(scores, reference_model=str(reference))
    return scores, tests


def _importance_ablation_and_stress(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = frame[FEATURE_COLS_V3].to_numpy(dtype=float)
    y = frame["SoH"].to_numpy(dtype=float)
    groups = frame["battery_id"].astype(str).to_numpy()
    folds = list(grouped_train_val_test_folds(groups, n_splits=5, random_state=42))

    importance_rows = []
    ablation_rows = []
    stress_rows = []
    ablations = {
        "all_safe_features": FEATURE_COLS_V3,
        "without_temperature": [c for c in FEATURE_COLS_V3 if "temperature" not in c],
        "without_voltage": [c for c in FEATURE_COLS_V3 if "voltage" not in c],
        "without_current": [c for c in FEATURE_COLS_V3 if "current" not in c],
        "time_and_usage_only": ["cycle_index", "prior_equivalent_full_cycles", "segment_duration_s"],
    }
    for fold, (train_idx, _, test_idx) in enumerate(folds, start=1):
        base = classical_factories(42)["extra_trees"]()
        base.fit(X[train_idx], y[train_idx])
        permutation = permutation_importance(
            base, X[test_idx], y[test_idx], scoring="neg_mean_absolute_error",
            n_repeats=10, random_state=42, n_jobs=-1,
        )
        for name, mean, std in zip(FEATURE_COLS_V3, permutation.importances_mean, permutation.importances_std):
            importance_rows.append({"fold": fold, "feature": name, "mae_increase": mean, "repeat_std": std})

        for name, columns in ablations.items():
            positions = [FEATURE_COLS_V3.index(column) for column in columns]
            model = classical_factories(42)["extra_trees"]()
            model.fit(X[train_idx][:, positions], y[train_idx])
            pred = model.predict(X[test_idx][:, positions])
            ablation_rows.append({"fold": fold, "ablation": name, "n_features": len(columns), **regression_metrics(y[test_idx], pred, n_predictors=len(columns))})

        rng = np.random.default_rng(42 + fold)
        scenarios: dict[str, np.ndarray] = {"unmodified": X[test_idx].copy()}
        missing_temperature = X[test_idx].copy()
        for col in [c for c in FEATURE_COLS_V3 if "temperature" in c]:
            missing_temperature[:, FEATURE_COLS_V3.index(col)] = np.nan
        scenarios["temperature_missing"] = missing_temperature
        noisy = X[test_idx].copy()
        for column in FEATURE_COLS_V3:
            if "voltage" in column or "current" in column:
                position = FEATURE_COLS_V3.index(column)
                scale = np.nanstd(X[train_idx, position])
                noisy[:, position] += rng.normal(0.0, 0.05 * (scale or 1.0), len(noisy))
        scenarios["voltage_current_noise_5pct_sd"] = noisy
        for scenario, values in scenarios.items():
            pred = base.predict(values)
            stress_rows.append({"fold": fold, "scenario": scenario, **regression_metrics(y[test_idx], pred, n_predictors=len(FEATURE_COLS_V3))})
    return pd.DataFrame(importance_rows), pd.DataFrame(ablation_rows), pd.DataFrame(stress_rows)


def run_statistical_analysis(
    project_root: str | Path,
    *,
    bootstrap_samples: int = 10_000,
) -> dict[str, pd.DataFrame]:
    root = Path(project_root)
    results = root / "artifacts" / "v3" / "results"
    results.mkdir(parents=True, exist_ok=True)
    averaged = {name: _averaged_predictions(results, name) for name in ("nasa", "calce", "oxford")}
    available = {name: frame for name, frame in averaged.items() if not frame.empty}
    if "nasa" not in available:
        raise FileNotFoundError("NASA prediction files are required before statistical analysis")

    all_predictions = pd.concat(available.values(), ignore_index=True)
    residuals = (
        all_predictions.groupby(["dataset", "model"])["residual"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    battery_scores, paired_tests = _battery_scores_and_tests(available["nasa"])
    intervals = _bootstrap_best_models(available, bootstrap_samples)
    nasa_features = pd.read_csv(root / "artifacts" / "v3" / "features" / "nasa" / "features.csv")
    importance, ablations, stress = _importance_ablation_and_stress(nasa_features)

    outputs = {
        "averaged_predictions": all_predictions,
        "residual_diagnostics": residuals,
        "nasa_per_battery_metrics": battery_scores,
        "nasa_wilcoxon_holm": paired_tests,
        "best_model_cluster_bootstrap_ci": intervals,
        "extra_trees_permutation_importance": importance,
        "feature_ablation": ablations,
        "sensor_stress": stress,
    }
    for name, table in outputs.items():
        table.to_csv(results / f"{name}.csv", index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    outputs = run_statistical_analysis(args.project_root, bootstrap_samples=args.bootstrap_samples)
    for name, table in outputs.items():
        print(name, table.shape)


if __name__ == "__main__":
    main()
