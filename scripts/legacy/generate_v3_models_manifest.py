from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FEATURE_SET_V3 = [
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
    "capacity_retention",
    "cumulative_energy",
    "dRe_dn",
    "dRct_dn",
    "soh_rolling_mean",
    "voltage_slope",
]


MODEL_FILE_MAP = {
    "xgboost": "models/classical/xgboost.joblib",
    "gradient_boosting": "models/classical/gradient_boosting.joblib",
    "lightgbm": "models/classical/lightgbm.joblib",
    "random_forest": "models/classical/random_forest.joblib",
    "extra_trees": "models/classical/extra_trees.joblib",
    "svr": "models/classical/svr.joblib",
    "ridge": "models/classical/ridge.joblib",
    "knn_k5": "models/classical/knn_k5.joblib",
    "vanilla_lstm": "models/deep/vanilla_lstm.pt",
    "bidirectional_lstm": "models/deep/bidirectional_lstm.pt",
    "gru": "models/deep/gru.pt",
    "attention_lstm": "models/deep/attention_lstm.pt",
    "batterygpt": "models/deep/batterygpt.pt",
    "tft": "models/deep/tft.pt",
    "itransformer": "models/deep/itransformer.keras",
    "physics_itransformer": "models/deep/physics_itransformer.keras",
    "dynamic_graph_itransformer": "models/deep/dynamic_graph_itransformer.keras",
    "vae_lstm": "models/deep/vae_lstm.pt",
    "stacking_ensemble": "models/ensemble/ensemble_stacking.joblib",
}


MODEL_META_DEFAULTS = {
    "xgboost": ("XGBoost", "classical", "XGBRegressor", False),
    "gradient_boosting": ("GradientBoosting", "classical", "GradientBoostingRegressor", False),
    "lightgbm": ("LightGBM", "classical", "LGBMRegressor", False),
    "random_forest": ("Random Forest", "classical", "RandomForestRegressor", False),
    "extra_trees": ("ExtraTrees", "classical", "ExtraTreesRegressor", False),
    "svr": ("SVR (RBF)", "classical", "SVR", True),
    "ridge": ("Ridge Regression", "classical", "Ridge", True),
    "knn_k5": ("KNN (k=5)", "classical", "KNeighborsRegressor", True),
    "vanilla_lstm": ("Vanilla LSTM", "deep_pytorch", "VanillaLSTM", True),
    "bidirectional_lstm": ("Bidirectional LSTM", "deep_pytorch", "BidirectionalLSTM", True),
    "gru": ("GRU", "deep_pytorch", "GRUModel", True),
    "attention_lstm": ("Attention LSTM", "deep_pytorch", "AttentionLSTM", True),
    "batterygpt": ("BatteryGPT", "deep_pytorch", "BatteryGPT", True),
    "tft": ("Temporal Fusion Transformer", "deep_pytorch", "TemporalFusionTransformer", True),
    "itransformer": ("iTransformer", "deep_keras", "iTransformer", True),
    "physics_itransformer": ("Physics iTransformer", "deep_keras", "PhysicsITransformer", True),
    "dynamic_graph_itransformer": ("DG-iTransformer", "deep_keras", "DynamicGraphITransformer", True),
    "vae_lstm": ("VAE-LSTM", "deep_pytorch", "VAE_LSTM", True),
    "stacking_ensemble": ("Stacking Ensemble", "ensemble", "RidgeStacking", False),
    "best_ensemble": ("Weighted Avg Ensemble", "ensemble", "WeightedAverage", False),
}


CSV_NAME_TO_ID = {
    "GradientBoosting": "gradient_boosting",
    "XGBoost": "xgboost",
    "RandomForest": "random_forest",
    "LightGBM": "lightgbm",
    "Ridge": "ridge",
    "SVR": "svr",
    "ExtraTrees": "extra_trees",
    "KNN-5": "knn_k5",
    "Bidirectional LSTM": "bidirectional_lstm",
    "GRU": "gru",
    "Vanilla LSTM": "vanilla_lstm",
    "Attention LSTM": "attention_lstm",
    "TFT": "tft",
    "BatteryGPT": "batterygpt",
    "Physics iTransformer": "physics_itransformer",
    "iTransformer": "itransformer",
    "Weighted Avg Ensemble": "best_ensemble",
    "Stacking Ensemble": "stacking_ensemble",
    "xgboost": "xgboost",
    "random_forest": "random_forest",
    "extra_trees": "extra_trees",
    "tft": "tft",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(v3_root: Path) -> dict[str, dict[str, float]]:
    results_dir = v3_root / "results"
    metrics: dict[str, dict[str, float]] = {}

    csv_files = [
        "classical_soh_results.csv",
        "lstm_soh_results.csv",
        "transformer_soh_results.csv",
        "ensemble_results.csv",
    ]
    for fname in csv_files:
        p = results_dir / fname
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_name = row.get("model", "")
                model_id = CSV_NAME_TO_ID.get(raw_name, raw_name.strip().lower().replace(" ", "_"))
                d: dict[str, float] = {}
                for k, v in row.items():
                    if k == "model" or v in (None, ""):
                        continue
                    try:
                        d[k] = float(v)
                    except ValueError:
                        continue
                if d:
                    metrics[model_id] = d

    json_files = {
        "dg_itransformer_results.json": "dynamic_graph_itransformer",
        "vae_lstm_results.json": "vae_lstm",
    }
    for fname, model_id in json_files.items():
        p = results_dir / fname
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        d = {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}
        if d:
            metrics[model_id] = d

    return metrics


def scalar_from_metrics(d: dict[str, float], *keys: str) -> float | None:
    for k in keys:
        if k in d:
            return float(d[k])
    return None


def build_models_block(
    v3_root: Path,
    existing_models: dict[str, Any],
    metrics: dict[str, dict[str, float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    models: dict[str, Any] = {}

    # File-backed models.
    for model_id, rel_path in MODEL_FILE_MAP.items():
        abs_path = v3_root / rel_path
        if not abs_path.exists():
            continue

        old = existing_models.get(model_id, {})
        m = metrics.get(model_id, {})
        display, family, algo, requires_scaling = MODEL_META_DEFAULTS[model_id]
        entry: dict[str, Any] = {
            "display_name": old.get("display_name", display),
            "family": old.get("family", family),
            "algorithm": old.get("algorithm", algo),
            "version": "3.0",
            "requires_scaling": old.get("requires_scaling", requires_scaling),
            "file": rel_path,
            "sha256": sha256_file(abs_path),
            "bytes": abs_path.stat().st_size,
        }

        r2 = scalar_from_metrics(m, "R2", "r2")
        mae = scalar_from_metrics(m, "MAE", "mae")
        within_5 = scalar_from_metrics(m, "within_5pct")
        if r2 is not None:
            entry["r2"] = r2
        if mae is not None:
            entry["mae"] = mae
        if within_5 is not None:
            entry["within_5pct"] = within_5
        if "f1_macro" in m:
            entry["f1_macro"] = float(m["f1_macro"])
        if "f1_weighted" in m:
            entry["f1_weighted"] = float(m["f1_weighted"])

        if model_id in ("best_ensemble", "stacking_ensemble") and "tol_2pct" in m:
            entry["tol_2pct"] = float(m["tol_2pct"])

        if model_id in ("stacking_ensemble",):
            entry["components"] = [
                "xgboost",
                "random_forest",
                "extra_trees",
                "attention_lstm",
                "tft",
            ]

        models[model_id] = entry

    # Virtual weighted-average ensemble metadata.
    weighted = metrics.get("best_ensemble", {})
    old_ens = existing_models.get("best_ensemble", {})
    dname, fam, algo, req = MODEL_META_DEFAULTS["best_ensemble"]
    ens_entry: dict[str, Any] = {
        "display_name": old_ens.get("display_name", dname),
        "family": old_ens.get("family", fam),
        "algorithm": old_ens.get("algorithm", algo),
        "version": "3.0",
        "requires_scaling": old_ens.get("requires_scaling", req),
        "components": old_ens.get(
            "components",
            ["xgboost", "random_forest", "extra_trees", "attention_lstm", "tft"],
        ),
        "weights_method": "optimized_l_bfgs_b",
        "weights_file": "models/ensemble/ensemble_weights.json",
        "file": None,
    }
    if weighted:
        r2 = scalar_from_metrics(weighted, "R2", "r2")
        mae = scalar_from_metrics(weighted, "MAE", "mae")
        within_5 = scalar_from_metrics(weighted, "within_5pct")
        if r2 is not None:
            ens_entry["r2"] = r2
        if mae is not None:
            ens_entry["mae"] = mae
        if within_5 is not None:
            ens_entry["within_5pct"] = within_5
        if "tol_2pct" in weighted:
            ens_entry["tol_2pct"] = float(weighted["tol_2pct"])
        if "f1_macro" in weighted:
            ens_entry["f1_macro"] = float(weighted["f1_macro"])
        if "f1_weighted" in weighted:
            ens_entry["f1_weighted"] = float(weighted["f1_weighted"])

    weights_path = v3_root / "models/ensemble/ensemble_weights.json"
    if weights_path.exists():
        ens_entry["weights_sha256"] = sha256_file(weights_path)

    models["best_ensemble"] = ens_entry

    # Auxiliary artifacts that should not be used as predictors.
    auxiliary_artifacts = {
        "re_rct_progression": {
            "display_name": "Re/Rct Progression Regressors",
            "family": "auxiliary",
            "algorithm": "LinearRegressionBundle",
            "version": "3.0",
            "file": "models/classical/re_rct_progression.joblib",
        }
    }
    aux_path = v3_root / auxiliary_artifacts["re_rct_progression"]["file"]
    if aux_path.exists():
        auxiliary_artifacts["re_rct_progression"]["sha256"] = sha256_file(aux_path)
        auxiliary_artifacts["re_rct_progression"]["bytes"] = aux_path.stat().st_size

    return models, auxiliary_artifacts


def collect_checksums(v3_root: Path) -> dict[str, Any]:
    checksums: dict[str, Any] = {
        "models": {},
        "scalers": {},
        "results": {},
        "features": {},
    }

    for rel in sorted((v3_root / "models").rglob("*")):
        if rel.is_file():
            key = rel.relative_to(v3_root).as_posix()
            checksums["models"][key] = sha256_file(rel)

    for rel in sorted((v3_root / "scalers").glob("*")):
        if rel.is_file():
            key = rel.relative_to(v3_root).as_posix()
            checksums["scalers"][key] = sha256_file(rel)

    for rel in sorted((v3_root / "results").glob("*")):
        if rel.is_file():
            key = rel.relative_to(v3_root).as_posix()
            checksums["results"][key] = sha256_file(rel)

    for rel in sorted((v3_root / "features").glob("*")):
        if rel.is_file():
            key = rel.relative_to(v3_root).as_posix()
            checksums["features"][key] = sha256_file(rel)

    checksums["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return checksums


def pick_champion(models: dict[str, Any]) -> str:
    best = "xgboost"
    best_r2 = -1e9
    for model_id, meta in models.items():
        if model_id in ("best_ensemble", "auxiliary"):
            continue
        r2 = meta.get("r2")
        if isinstance(r2, (int, float)) and r2 > best_r2:
            best_r2 = float(r2)
            best = model_id
    return best


def build_manifest(v3_root: Path) -> dict[str, Any]:
    existing = load_existing(v3_root / "models.json")
    existing_models = existing.get("models", {})
    metrics = load_metrics(v3_root)
    models, auxiliary_artifacts = build_models_block(v3_root, existing_models, metrics)
    champion = pick_champion(models)

    manifest: dict[str, Any] = {
        "version": "v3",
        "display": "v3.0",
        "description": existing.get(
            "description",
            "Production models with cross-battery split, 18 engineered features, and full artifact integrity checks.",
        ),
        "split_strategy": existing.get("split_strategy", "cross-battery grouped split (no data leakage)"),
        "features": 18,
        "feature_set": existing.get("feature_set", FEATURE_SET_V3),
        "sequence_length": int(existing.get("sequence_length", 32)),
        "dataset": existing.get("dataset", "NASA PCoE Li-ion 18650"),
        "default_model": "best_ensemble",
        "models": models,
        "auxiliary_artifacts": auxiliary_artifacts,
        "scalers": {
            "features_standard": "scalers/features_standard.joblib",
            "features_minmax": "scalers/features_minmax.joblib",
        },
        "scaler_checksums": {
            "features_standard": sha256_file(v3_root / "scalers/features_standard.joblib")
            if (v3_root / "scalers/features_standard.joblib").exists()
            else None,
            "features_minmax": sha256_file(v3_root / "scalers/features_minmax.joblib")
            if (v3_root / "scalers/features_minmax.joblib").exists()
            else None,
        },
        "champion": champion,
        "framework": existing.get(
            "framework",
            ["scikit-learn", "xgboost", "lightgbm", "pytorch", "tensorflow"],
        ),
        "training_date": datetime.now(timezone.utc).date().isoformat(),
        "checksums": collect_checksums(v3_root),
        "verification": {
            "hash_algorithm": "sha256",
            "required": True,
            "notes": "Verify checksums before serving or deploying artifacts.",
        },
    }

    # Keep optional documentation fields if present.
    for opt_key in ("engineered_features", "improvements_over_v2"):
        if opt_key in existing:
            manifest[opt_key] = existing[opt_key]

    return manifest


def write_manifest(v3_root: Path) -> Path:
    out_path = v3_root / "models.json"
    manifest = build_manifest(v3_root)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate v3 models.json with SHA256 checksums")
    parser.add_argument(
        "--v3-root",
        default="artifacts/v3",
        help="Path to v3 artifact root (default: artifacts/v3)",
    )
    args = parser.parse_args()

    v3_root = Path(args.v3_root).resolve()
    if not v3_root.exists():
        raise SystemExit(f"v3 root does not exist: {v3_root}")

    out = write_manifest(v3_root)
    print(f"Wrote manifest: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
