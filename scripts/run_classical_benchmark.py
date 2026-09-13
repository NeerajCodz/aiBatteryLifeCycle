"""Run and overwrite the grouped classical/ensemble benchmark for one dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.classical import run_grouped_tabular_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("nasa", "calce", "oxford"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 2026))
    args = parser.parse_args()

    frame = pd.read_csv(
        args.project_root / "artifacts" / "v3" / "features" / args.dataset / "features.csv"
    )
    metrics, predictions = run_grouped_tabular_benchmark(
        frame,
        dataset_name=args.dataset.upper() if args.dataset == "nasa" else args.dataset.title(),
        n_splits=min(5, frame["battery_id"].nunique()),
        seeds=tuple(args.seeds),
    )
    results = args.project_root / "artifacts" / "v3" / "results"
    results.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results / f"{args.dataset}_classical_fold_metrics.csv", index=False)
    predictions.to_csv(results / f"{args.dataset}_classical_predictions.csv", index=False)
    print(metrics.groupby("model")[["mae", "rmse", "r2", "within_5pp"]].mean().sort_values("mae"))


if __name__ == "__main__":
    main()
