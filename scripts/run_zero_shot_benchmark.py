"""Train once on NASA and score frozen models on CALCE and Oxford."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sequence_benchmark import MODEL_FAMILIES, _atomic_write_csv, _read_csv_or_empty
from src.experiments.classical import run_zero_shot_tabular
from src.experiments.deep import run_zero_shot_sequence


def _load_sequence(root: Path, dataset: str) -> tuple[np.ndarray, pd.DataFrame]:
    folder = root / "artifacts" / "v3" / "features" / dataset
    return np.load(folder / "sequences.npz")["X"], pd.read_csv(folder / "sequence_index.csv")


def _write_target_results(
    result_dir: Path,
    family: str,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    merge: bool = False,
) -> None:
    for target in ("calce", "oxford"):
        label = target.title()
        target_metrics = metrics[metrics["target_dataset"] == label].copy()
        target_predictions = predictions[predictions["target_dataset"] == label].copy()
        metrics_path = result_dir / f"nasa_to_{target}_{family}_metrics.csv"
        predictions_path = result_dir / f"nasa_to_{target}_{family}_predictions.csv"
        if merge:
            model_ids = set(target_metrics["model"])
            old_metrics = _read_csv_or_empty(metrics_path)
            old_predictions = _read_csv_or_empty(predictions_path)
            if not old_metrics.empty and "model" in old_metrics:
                old_metrics = old_metrics[~old_metrics["model"].isin(model_ids)]
            if not old_predictions.empty and "model" in old_predictions:
                old_predictions = old_predictions[~old_predictions["model"].isin(model_ids)]
            target_metrics = pd.concat([old_metrics, target_metrics], ignore_index=True)
            target_predictions = pd.concat([old_predictions, target_predictions], ignore_index=True)
        _atomic_write_csv(target_metrics, metrics_path)
        _atomic_write_csv(target_predictions, predictions_path)


def _zero_shot_model_is_complete(
    result_dir: Path,
    family: str,
    model_id: str,
    seeds: tuple[int, ...],
) -> bool:
    for target in ("calce", "oxford"):
        metrics = _read_csv_or_empty(result_dir / f"nasa_to_{target}_{family}_metrics.csv")
        predictions = _read_csv_or_empty(result_dir / f"nasa_to_{target}_{family}_predictions.csv")
        if metrics.empty or predictions.empty or "model" not in metrics or "model" not in predictions:
            return False
        model_metrics = metrics[metrics["model"] == model_id]
        model_predictions = predictions[predictions["model"] == model_id]
        if set(model_metrics.get("seed", [])) != set(seeds) or len(model_metrics) != len(seeds):
            return False
        if model_predictions.empty:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("classical", *MODEL_FAMILIES))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 2026))
    args = parser.parse_args()

    root = args.project_root
    result_dir = root / "artifacts" / "v3" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.family == "classical":
        source = pd.read_csv(root / "artifacts" / "v3" / "features" / "nasa" / "features.csv")
        for target in ("calce", "oxford"):
            frame = pd.read_csv(root / "artifacts" / "v3" / "features" / target / "features.csv")
            metrics, predictions = run_zero_shot_tabular(
                source, frame, target_name=target.title(), random_state=42,
            )
            metrics.to_csv(result_dir / f"nasa_to_{target}_classical_metrics.csv", index=False)
            predictions.to_csv(result_dir / f"nasa_to_{target}_classical_predictions.csv", index=False)
            print(target, metrics[["model", "mae", "rmse", "r2", "within_5pp"]].sort_values("mae").to_string(index=False))
        return

    source_X, source_index = _load_sequence(root, "nasa")
    targets = {
        dataset.title(): _load_sequence(root, dataset)
        for dataset in ("calce", "oxford")
    }
    seeds = tuple(args.seeds)
    for model_id in MODEL_FAMILIES[args.family]:
        if _zero_shot_model_is_complete(result_dir, args.family, model_id, seeds):
            print(f"[NASA->external] model={model_id} checkpoint complete; skipping", flush=True)
            continue
        metrics, predictions = run_zero_shot_sequence(
            source_X,
            source_index,
            targets,
            seeds=seeds,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            model_ids=(model_id,),
        )
        _write_target_results(result_dir, args.family, metrics, predictions, merge=True)
        print(f"[NASA->external] model={model_id} checkpoint written", flush=True)

    all_metrics = []
    for target in ("calce", "oxford"):
        all_metrics.append(_read_csv_or_empty(result_dir / f"nasa_to_{target}_{args.family}_metrics.csv"))
    metrics = pd.concat(all_metrics, ignore_index=True)
    print(metrics.groupby(["target_dataset", "model"])[["mae", "rmse", "r2", "within_5pp", "epochs"]].mean().to_string())


if __name__ == "__main__":
    main()
