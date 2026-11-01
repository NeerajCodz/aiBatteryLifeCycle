"""Run a restartable sequence-model benchmark family from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.deep import run_grouped_sequence_benchmark

MODEL_FAMILIES = {
    "recurrent": ("vanilla_lstm", "bidirectional_lstm", "gru", "attention_lstm"),
    "transformer": ("battery_gpt", "temporal_fusion_transformer", "itransformer", "physics_itransformer"),
    "graph_variational": ("dynamic_graph_itransformer", "vae_lstm"),
}


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _model_is_complete(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    model_id: str,
    seeds: tuple[int, ...],
    n_splits: int,
) -> bool:
    if metrics.empty or predictions.empty or "model" not in metrics or "model" not in predictions:
        return False
    subset = metrics[metrics["model"] == model_id]
    pred_subset = predictions[predictions["model"] == model_id]
    expected = {(seed, fold) for seed in seeds for fold in range(1, n_splits + 1)}
    observed = set(zip(subset.get("seed", []), subset.get("fold", [])))
    return observed == expected and not pred_subset.empty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("nasa", "calce", "oxford"))
    parser.add_argument("family", choices=tuple(MODEL_FAMILIES))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 42, 2026))
    parser.add_argument("--result-dir", type=Path, help="Separate output directory for new audited reruns")
    parser.add_argument("--audit-dir", type=Path, help="Fit histories and role predictions (default: result-dir/training_audit)")
    args = parser.parse_args()

    feature_dir = args.project_root / "artifacts" / "v3" / "features" / args.dataset
    result_dir = args.result_dir or args.project_root / "artifacts" / "v3" / "results"
    audit_dir = args.audit_dir or result_dir / 'training_audit'
    result_dir.mkdir(parents=True, exist_ok=True)
    X = np.load(feature_dir / "sequences.npz")["X"]
    index = pd.read_csv(feature_dir / "sequence_index.csv")
    prefix = result_dir / f"{args.dataset}_{args.family}"
    metrics_path = Path(f"{prefix}_fold_metrics.csv")
    predictions_path = Path(f"{prefix}_predictions.csv")
    metrics = _read_csv_or_empty(metrics_path)
    predictions = _read_csv_or_empty(predictions_path)
    seeds = tuple(args.seeds)
    n_splits = min(5, index["battery_id"].nunique())
    dataset_name = args.dataset.upper() if args.dataset == "nasa" else args.dataset.title()

    for model_id in MODEL_FAMILIES[args.family]:
        if _model_is_complete(metrics, predictions, model_id, seeds, n_splits):
            print(f"[{dataset_name}] model={model_id} checkpoint complete; skipping", flush=True)
            continue
        model_metrics, model_predictions = run_grouped_sequence_benchmark(
            X,
            index,
            dataset_name=dataset_name,
            n_splits=n_splits,
            seeds=seeds,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            model_ids=(model_id,),
            audit_dir=audit_dir,
        )
        if not metrics.empty and "model" in metrics:
            metrics = metrics[metrics["model"] != model_id]
        if not predictions.empty and "model" in predictions:
            predictions = predictions[predictions["model"] != model_id]
        metrics = pd.concat([metrics, model_metrics], ignore_index=True)
        predictions = pd.concat([predictions, model_predictions], ignore_index=True)
        _atomic_write_csv(metrics, metrics_path)
        _atomic_write_csv(predictions, predictions_path)
        print(f"[{dataset_name}] model={model_id} checkpoint written", flush=True)

    print(metrics.groupby("model")[["mae", "rmse", "r2", "within_5pp", "epochs"]].mean().sort_values("mae"))


if __name__ == "__main__":
    main()
