"""Train deployable v3 pipelines and assemble a consistent model-repo release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import per_battery_evaluation
from src.experiments.classical import classical_factories
from src.models.catalog import get_model_spec
from src.utils.config import FEATURE_COLS_V3, RATED_CAPACITY_AH


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_result_package(source_dir: Path, destination_dir: Path) -> None:
    """Copy compact metrics verbatim and gzip row-level prediction tables."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.iterdir()):
        if not source.is_file():
            continue
        if source.name.endswith("_predictions.csv"):
            target = destination_dir / f"{source.name}.gz"
            with source.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy2(source, destination_dir / source.name)


def _grouped_summary(results: Path) -> pd.DataFrame:
    metrics = pd.read_csv(results / "nasa_classical_fold_metrics.csv")
    return metrics.groupby("model", as_index=False).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), r2=("r2", "mean"),
        adjusted_r2=("adjusted_r2", "mean"), mape=("mape", "mean"),
        within_5pp=("within_5pp", "mean"), generalization_gap_mae=("generalization_gap_mae", "mean"),
    )


def _macro_battery_mae(results: Path) -> pd.Series:
    pred = pd.read_csv(results / "nasa_classical_predictions.csv")
    averaged = pred.groupby(["model", "row_index", "battery_id", "y_true"], as_index=False)["y_pred"].mean()
    rows = []
    for model, group in averaged.groupby("model"):
        per = per_battery_evaluation(group["y_true"].to_numpy(), group["y_pred"].to_numpy(), group["battery_id"])
        rows.append((model, per["mae"].mean()))
    return pd.Series(dict(rows), name="macro_battery_mae")


def prepare_model_release(project_root: Path, asset_repo: Path | None = None) -> dict[str, object]:
    results = project_root / "artifacts" / "v3" / "results"
    feature_path = project_root / "artifacts" / "v3" / "features" / "nasa" / "features.csv"
    release = project_root / "artifacts" / "v3"
    model_dir = release / "models" / "classical"
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(feature_path)
    X = frame[FEATURE_COLS_V3].to_numpy(dtype=float)
    y = frame["SoH"].to_numpy(dtype=float)
    summary = _grouped_summary(results).set_index("model")
    macro = _macro_battery_mae(results)

    catalog = {}
    latency_batch = X[: min(1000, len(X))]
    for model_id, factory in classical_factories(42).items():
        model = factory()
        model.fit(X, y)
        path = model_dir / f"{model_id}.joblib"
        joblib.dump(model, path, compress=3)
        start = time.perf_counter()
        for _ in range(5):
            model.predict(latency_batch)
        latency_ms = (time.perf_counter() - start) * 1000 / (5 * len(latency_batch))
        metrics = summary.loc[model_id].to_dict()
        spec = get_model_spec(model_id)
        catalog[model_id] = {
            "display_name": spec.display_name,
            "family": "classical",
            "algorithm": type(model.named_steps["model"]).__name__,
            "version": "3.0.0",
            "requires_scaling": "scaler" in model.named_steps,
            "file": f"models/classical/{model_id}.joblib",
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "macro_per_battery_mae": float(macro[model_id]),
            "cpu_latency_ms_per_sample": float(latency_ms),
            **{key: float(value) for key, value in metrics.items()},
        }

    # Apply the preregistered lexicographic selection rule.  Size and latency
    # matter only if the accuracy criteria are equal, so scientific performance
    # remains the primary basis for release selection.
    champion = min(
        catalog,
        key=lambda name: (
            catalog[name]["macro_per_battery_mae"],
            catalog[name]["rmse"],
            catalog[name]["bytes"],
            catalog[name]["cpu_latency_ms_per_sample"],
        ),
    )

    nasa_cells = int(frame["battery_id"].nunique())
    nasa_cycles = int(len(frame))
    manifest = {
        "version": "v3",
        "display": "v3.0 reviewer revision",
        "description": "Leakage-safe current-cycle SOH estimation from a 4.0-3.6 V partial discharge window.",
        "status": "research prototype; not safety certified",
        "split_strategy": "battery-grouped train/validation/test; 5 folds; seeds 17,42,2026",
        "target": "current-cycle SOH",
        "features": len(FEATURE_COLS_V3),
        "feature_set": FEATURE_COLS_V3,
        "rated_capacity_ah": RATED_CAPACITY_AH,
        "sequence_length": 64,
        "dataset": f"NASA PCoE; {nasa_cells} retained batteries; {nasa_cycles} eligible discharge cycles",
        "champion": champion,
        "default_model": champion,
        "selection_metric": "macro per-battery MAE",
        "evaluated_model_count": 20,
        "deployable_model_count": len(catalog),
        "models": catalog,
    }
    (release / "models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checksums = (project_root / "datasets" / "checksums.sha256").read_text(encoding="utf-8")
    datamap = {
        "target_definition": "SOH_t = 100 * Q_t / median(Q_1,Q_2,Q_3) after robust characterization selection",
        "input_window": "current discharge from 4.0 V through 3.6 V",
        "forbidden_inputs": ["current-cycle full capacity", "current-cycle SOH", "rolling SOH", "capacity retention computed from Q_t"],
        "feature_set": FEATURE_COLS_V3,
        "rated_capacity_ah": RATED_CAPACITY_AH,
        "source_checksums_sha256": checksums.splitlines(),
        "quality_exclusions": str((project_root / "artifacts" / "v3" / "features" / "nasa" / "exclusions.csv").relative_to(project_root)),
    }
    (release / "datamap.json").write_text(json.dumps(datamap, indent=2), encoding="utf-8")
    dataset_rows = []
    for dataset in ("nasa", "calce", "oxford"):
        dataset_frame = pd.read_csv(release / "features" / dataset / "features.csv")
        exclusions = pd.read_csv(release / "features" / dataset / "exclusions.csv")
        dataset_rows.append({
            "dataset": dataset.upper() if dataset == "nasa" else dataset.title(),
            "rated_capacity_ah": RATED_CAPACITY_AH[dataset.upper() if dataset != "oxford" else "Oxford"],
            "retained_batteries": int(dataset_frame["battery_id"].nunique()),
            "retained_cycles": int(len(dataset_frame)),
            "excluded_records": int(len(exclusions)),
            "soh_min": float(dataset_frame["SoH"].min()),
            "soh_median": float(dataset_frame["SoH"].median()),
            "soh_max": float(dataset_frame["SoH"].max()),
        })
    dataset_manifest = {
        "version": "3.0.0",
        "generated_by": "scripts/prepare_model_release.py",
        "target": "current-cycle SOH; full-cycle capacity is label-only",
        "input_window": "4.0-3.6 V partial discharge",
        "datasets": dataset_rows,
        "source_checksums_sha256": checksums.splitlines(),
        "feature_set": FEATURE_COLS_V3,
    }
    (release / "dataset.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")

    if asset_repo is not None:
        destination = asset_repo / "v3"
        destination.mkdir(parents=True, exist_ok=True)
        for relative in ("models", "figures", "tables"):
            source_dir = release / relative
            if source_dir.exists():
                target_dir = destination / relative
                target_dir.mkdir(parents=True, exist_ok=True)
                for source in source_dir.rglob("*"):
                    if source.is_file():
                        target = target_dir / source.relative_to(source_dir)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
        for filename in ("models.json", "datamap.json", "dataset.json", "paper_artifact_manifest.json", "paper_artifact_manifest.csv"):
            source = release / filename
            if source.exists():
                shutil.copy2(source, destination / filename)
        shutil.copy2(release / "dataset.json", asset_repo / "dataset.json")
        _copy_result_package(results, destination / "results")
        release_summary = _grouped_summary(results)
        release_summary.to_csv(destination / "results" / "nasa_grouped_summary.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--asset-repo", type=Path, default=None)
    args = parser.parse_args()
    manifest = prepare_model_release(args.project_root, args.asset_repo)
    print(json.dumps({"default_model": manifest["default_model"], "models": len(manifest["models"])}, indent=2))


if __name__ == "__main__":
    main()
