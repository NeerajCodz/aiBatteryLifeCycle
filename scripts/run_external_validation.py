"""Within-dataset and NASA-to-external validation for CALCE or Oxford."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.classical import run_grouped_tabular_benchmark, run_zero_shot_tabular
from src.experiments.deep import run_grouped_sequence_benchmark, run_zero_shot_sequence
from scripts.run_sequence_benchmark import (
    MODEL_FAMILIES,
    _atomic_write_csv,
    _model_is_complete,
    _read_csv_or_empty,
)
from scripts.run_zero_shot_benchmark import _write_target_results, _zero_shot_model_is_complete


def _load_sequence(root: Path, dataset: str) -> tuple[np.ndarray, pd.DataFrame]:
    folder = root / "artifacts" / "v3" / "features" / dataset
    return np.load(folder / "sequences.npz")["X"], pd.read_csv(folder / "sequence_index.csv")


def _ensure_grouped_results(
    root: Path,
    dataset: str,
    seeds: tuple[int, ...],
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> None:
    results = root / "artifacts" / "v3" / "results"
    features = root / "artifacts" / "v3" / "features" / dataset
    classical_metrics = results / f"{dataset}_classical_fold_metrics.csv"
    classical_predictions = results / f"{dataset}_classical_predictions.csv"
    if not classical_metrics.exists() or not classical_predictions.exists():
        frame = pd.read_csv(features / "features.csv")
        metrics, predictions = run_grouped_tabular_benchmark(
            frame,
            dataset_name=dataset.title(),
            n_splits=min(5, frame["battery_id"].nunique()),
            seeds=seeds,
        )
        metrics.to_csv(classical_metrics, index=False)
        predictions.to_csv(classical_predictions, index=False)

    X, index = _load_sequence(root, dataset)
    for family, model_ids in MODEL_FAMILIES.items():
        metric_path = results / f"{dataset}_{family}_fold_metrics.csv"
        prediction_path = results / f"{dataset}_{family}_predictions.csv"
        metrics = _read_csv_or_empty(metric_path)
        predictions = _read_csv_or_empty(prediction_path)
        n_splits = min(5, index["battery_id"].nunique())
        for model_id in model_ids:
            if _model_is_complete(metrics, predictions, model_id, seeds, n_splits):
                continue
            model_metrics, model_predictions = run_grouped_sequence_benchmark(
                X,
                index,
                dataset_name=dataset.title(),
                n_splits=n_splits,
                seeds=seeds,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                model_ids=(model_id,),
            )
            if not metrics.empty and "model" in metrics:
                metrics = metrics[metrics["model"] != model_id]
            if not predictions.empty and "model" in predictions:
                predictions = predictions[predictions["model"] != model_id]
            metrics = pd.concat([metrics, model_metrics], ignore_index=True)
            predictions = pd.concat([predictions, model_predictions], ignore_index=True)
            _atomic_write_csv(metrics, metric_path)
            _atomic_write_csv(predictions, prediction_path)


def _ensure_zero_shot_results(
    root: Path,
    dataset: str,
    seeds: tuple[int, ...],
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> None:
    results = root / "artifacts" / "v3" / "results"
    feature_root = root / "artifacts" / "v3" / "features"
    source = pd.read_csv(feature_root / "nasa" / "features.csv")
    target = pd.read_csv(feature_root / dataset / "features.csv")
    tabular_metrics = results / f"nasa_to_{dataset}_classical_metrics.csv"
    tabular_predictions = results / f"nasa_to_{dataset}_classical_predictions.csv"
    if not tabular_metrics.exists() or not tabular_predictions.exists():
        metrics, predictions = run_zero_shot_tabular(
            source,
            target,
            target_name=dataset.title(),
            random_state=42,
        )
        metrics.to_csv(tabular_metrics, index=False)
        predictions.to_csv(tabular_predictions, index=False)

    source_X, source_index = _load_sequence(root, "nasa")
    targets = {
        target_name.title(): _load_sequence(root, target_name)
        for target_name in ("calce", "oxford")
    }
    for family, model_ids in MODEL_FAMILIES.items():
        for model_id in model_ids:
            if _zero_shot_model_is_complete(results, family, model_id, seeds):
                continue
            metrics, predictions = run_zero_shot_sequence(
                source_X,
                source_index,
                targets,
                seeds=seeds,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                model_ids=(model_id,),
            )
            _write_target_results(results, family, metrics, predictions, merge=True)


def run_external_validation(
    project_root: str | Path,
    *,
    dataset: str,
    seeds: tuple[int, ...] = (17, 42, 2026),
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 64,
) -> pd.DataFrame:
    dataset = dataset.lower()
    if dataset not in {"calce", "oxford"}:
        raise ValueError("dataset must be 'calce' or 'oxford'")
    root = Path(project_root)
    result_dir = root / "artifacts" / "v3" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    _ensure_grouped_results(root, dataset, seeds, max_epochs, patience, batch_size)
    _ensure_zero_shot_results(root, dataset, seeds, max_epochs, patience, batch_size)

    frames = []
    for path in sorted(result_dir.glob(f"{dataset}_*_fold_metrics.csv")):
        frame = pd.read_csv(path)
        frame["validation"] = "within_dataset_grouped"
        frames.append(frame)
    for path in sorted(result_dir.glob(f"nasa_to_{dataset}_*_metrics.csv")):
        frame = pd.read_csv(path)
        frame["validation"] = "nasa_zero_shot"
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    summary = (
        combined.groupby(["validation", "model"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            adjusted_r2=("adjusted_r2", "mean"),
            mape=("mape", "mean"),
            within_5pp=("within_5pp", "mean"),
        )
        .sort_values(["validation", "mae"])
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("calce", "oxford"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 2026))
    args = parser.parse_args()
    summary = run_external_validation(
        args.project_root,
        dataset=args.dataset,
        seeds=tuple(args.seeds),
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    output = args.project_root / "artifacts" / "v3" / "results" / f"{args.dataset}_validation_summary.csv"
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
