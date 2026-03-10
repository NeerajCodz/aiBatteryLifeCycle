"""
api.routers.visualize
=====================
Endpoints that serve pre-computed or on-demand visualisation data
consumed by the React frontend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.model_registry import registry, classify_degradation, soh_to_color
from api.schemas import BatteryVizData, DashboardData
from scripts.download_models import ensure_metadata_first, download_metrics_bundle

router = APIRouter(prefix="/api", tags=["visualization"])

_PROJECT = Path(__file__).resolve().parents[2]
_ARTIFACTS = _PROJECT / "artifacts"
_FIGURES = _ARTIFACTS / "figures"
_DATASET = _PROJECT / "cleaned_dataset"

_SUPPORTED_VERSIONS = {"v1", "v2", "v3"}


# ── Dashboard aggregate ──────────────────────────────────────────────────────
@router.get("/dashboard", response_model=DashboardData)
async def dashboard():
    """Return full dashboard payload for the frontend."""
    # Battery summary
    metadata_path = _DATASET / "metadata.csv"
    batteries: list[BatteryVizData] = []
    capacity_fade: dict[str, list[float]] = {}

    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        for bid in meta["battery_id"].unique():
            sub = meta[meta["battery_id"] == bid].sort_values("start_time")
            caps_s = pd.to_numeric(sub["Capacity"], errors="coerce").dropna()
            if caps_s.empty:
                continue
            caps = caps_s.tolist()
            last_cap = float(caps[-1])
            soh = (last_cap / 2.0) * 100
            avg_temp = float(sub["ambient_temperature"].mean())
            cycle = len(sub)
            batteries.append(BatteryVizData(
                battery_id=bid,
                soh_pct=round(soh, 1),
                temperature=round(avg_temp, 1),
                cycle_number=cycle,
                degradation_state=classify_degradation(soh),
                color_hex=soh_to_color(soh),
            ))
            capacity_fade[bid] = caps

    model_metrics = registry.get_metrics()
    # Find best model
    best_model = "none"
    best_r2 = -999
    for name, m in model_metrics.items():
        r2 = m.get("R2", -999)
        if r2 > best_r2:
            best_r2 = r2
            best_model = name

    return DashboardData(
        batteries=batteries,
        capacity_fade=capacity_fade,
        model_metrics=model_metrics,
        best_model=best_model,
    )


# ── Capacity fade for a specific battery ─────────────────────────────────────
@router.get("/battery/{battery_id}/capacity")
async def battery_capacity(battery_id: str):
    """Return cycle-by-cycle capacity for one battery."""
    meta_path = _DATASET / "metadata.csv"
    if not meta_path.exists():
        raise HTTPException(404, "Metadata not found")
    meta = pd.read_csv(meta_path)
    sub = meta[meta["battery_id"] == battery_id].sort_values("start_time")
    if sub.empty:
        raise HTTPException(404, f"Battery {battery_id} not found")
    caps = pd.to_numeric(sub["Capacity"], errors="coerce").dropna().tolist()
    cycles = list(range(1, len(caps) + 1))
    soh_list = [(float(c) / 2.0) * 100 for c in caps]
    return {"battery_id": battery_id, "cycles": cycles, "capacity_ah": caps, "soh_pct": soh_list}


# ── Serve saved figures ──────────────────────────────────────────────────────
@router.get("/figures/{filename}")
async def get_figure(filename: str):
    """Serve a saved matplotlib/plotly figure from artifacts/figures."""
    path = _FIGURES / filename
    if not path.exists():
        raise HTTPException(404, f"Figure {filename} not found")
    content_type = "image/png"
    if path.suffix == ".html":
        content_type = "text/html"
    elif path.suffix == ".svg":
        content_type = "image/svg+xml"
    return FileResponse(path, media_type=content_type)


# ── Figures listing ──────────────────────────────────────────────────────────
@router.get("/figures")
async def list_figures():
    """List all available figures."""
    if not _FIGURES.exists():
        return []
    return sorted([f.name for f in _FIGURES.iterdir() if f.is_file()])


# ── Battery list ─────────────────────────────────────────────────────────────
@router.get("/batteries")
async def list_batteries():
    """Return all battery IDs and basic stats."""
    meta_path = _DATASET / "metadata.csv"
    if not meta_path.exists():
        return []
    meta = pd.read_csv(meta_path)
    out = []
    for bid in sorted(meta["battery_id"].unique()):
        sub = meta[meta["battery_id"] == bid]
        caps = pd.to_numeric(sub["Capacity"], errors="coerce").dropna()
        out.append({
            "battery_id": bid,
            "n_cycles": len(sub),
            "last_capacity": round(float(caps.iloc[-1]), 4) if len(caps) else None,
            "soh_pct": round((float(caps.iloc[-1]) / 2.0) * 100, 1) if len(caps) else None,
            "ambient_temperature": round(float(sub["ambient_temperature"].mean()), 1),
        })
    return out


# ── Comprehensive metrics endpoint ───────────────────────────────────────────
def _safe_read_csv(path: Path) -> list[dict]:
    """Read a CSV file into a list of dicts, replacing NaN with None."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


def _safe_read_json(path: Path) -> dict:
    """Read a JSON file, returning empty dict on failure."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _safe_read_csv_first(paths: list[Path]) -> list[dict]:
    for path in paths:
        if path.exists():
            return _safe_read_csv(path)
    return []


def _safe_read_json_first(paths: list[Path]) -> dict:
    for path in paths:
        if path.exists():
            return _safe_read_json(path)
    return {}


def _version_root(version: str) -> Path:
    return _ARTIFACTS / version


def _ensure_version(version: str) -> None:
    if version not in _SUPPORTED_VERSIONS:
        raise HTTPException(400, f"Unknown version '{version}'")


def _version_figures(version: str) -> list[str]:
    fig_dir = _version_root(version) / "figures"
    if not fig_dir.exists():
        return []
    return sorted([f.name for f in fig_dir.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".svg", ".jpg", ".jpeg", ".webp")])


def _battery_stats_for_version(version: str) -> dict:
    root = _version_root(version)
    features_candidates = [
        root / "features" / "battery_features.csv",
        root / "results" / "battery_features.csv",
    ]
    features_path = next((p for p in features_candidates if p.exists()), None)
    if not features_path:
        return {}
    df = pd.read_csv(features_path)
    stats = {
        "total_samples": len(df),
        "batteries": int(df["battery_id"].nunique()) if "battery_id" in df.columns else 0,
        "feature_columns": [
            c for c in df.columns.tolist()
            if c not in ["battery_id", "datetime", "SoH", "RUL", "degradation_state"]
        ],
    }
    if "SoH" in df.columns:
        stats.update({
            "avg_soh": round(float(df["SoH"].mean()), 2),
            "min_soh": round(float(df["SoH"].min()), 2),
            "max_soh": round(float(df["SoH"].max()), 2),
        })
    if "RUL" in df.columns:
        stats["avg_rul"] = round(float(df["RUL"].mean()), 1)
    if "degradation_state" in df.columns:
        stats["degradation_distribution"] = json.loads(df["degradation_state"].value_counts().to_json())
    if "ambient_temperature" in df.columns:
        stats["temp_groups"] = sorted(df["ambient_temperature"].dropna().unique().tolist())
    return stats


def _build_metrics_payload(version: str) -> dict:
    _ensure_version(version)
    root = _version_root(version)

    # Ensure artifacts required by metrics exist locally for this version.
    try:
        ensure_metadata_first([version])
        results_dir = root / "results"
        figures_dir = root / "figures"
        has_results = results_dir.exists() and any(results_dir.glob("*"))
        has_figures = figures_dir.exists() and any(figures_dir.glob("*"))
        if not has_results and not has_figures:
            download_metrics_bundle(version)
    except Exception:
        # Keep endpoint resilient; payload will still be built from whatever exists.
        pass

    results = root / "results"
    reports = root / "reports"
    models_meta = _safe_read_json(root / "models.json")
    datamap = _safe_read_json(root / "datamap.json")

    unified = _safe_read_csv_first([results / "unified_results.csv"])
    classical_results = _safe_read_csv_first([
        results / "classical_results.csv",
        results / "classical_soh_results.csv",
    ])
    classical_soh = _safe_read_csv_first([results / "classical_soh_results.csv"])
    lstm_results = _safe_read_csv_first([results / "lstm_soh_results.csv"])
    ensemble_results = _safe_read_csv_first([results / "ensemble_results.csv"])
    transformer_results = _safe_read_csv_first([results / "transformer_soh_results.csv"])
    validation = _safe_read_csv_first([
        results / "model_validation.csv",
        reports / "model_validation.csv",
    ])
    rankings = _safe_read_csv_first([results / "final_rankings.csv"])
    classical_rul = _safe_read_csv_first([results / "classical_rul_results.csv"])

    training_summary = _safe_read_json_first([
        results / "training_summary.json",
        reports / "training_summary.json",
    ])
    validation_summary = _safe_read_json_first([
        results / "validation_summary.json",
        reports / "validation_summary.json",
    ])
    intra_battery = _safe_read_json_first([
        results / "intra_battery.json",
        reports / "intra_battery.json",
    ])
    vae_lstm = _safe_read_json_first([results / "vae_lstm_results.json"])
    dg_itransformer = _safe_read_json_first([results / "dg_itransformer_results.json"])

    # Fallback: build unified/classical-like rows directly from models.json when
    # result CSVs are not yet downloaded for a version.
    if not unified and isinstance(models_meta, dict):
        model_rows = []
        for name, info in (models_meta.get("models") or {}).items():
            if not isinstance(info, dict):
                continue
            model_rows.append({
                "model": name,
                "family": info.get("family"),
                "R2": info.get("r2"),
                "MAE": info.get("mae"),
                "RMSE": info.get("rmse"),
                "MAPE": info.get("mape"),
                "within_5pct": info.get("within_5pct"),
                "f1_macro": info.get("f1_macro"),
                "f1_weighted": info.get("f1_weighted"),
            })
        unified = model_rows
        if not classical_results:
            classical_results = [r for r in model_rows if (r.get("family") or "").startswith("classical")]

    # Fallback summaries derived from unified rows
    if not training_summary and unified:
        valid_r2 = [r.get("R2") for r in unified if isinstance(r.get("R2"), (int, float))]
        valid_w5 = [r.get("within_5pct") for r in unified if isinstance(r.get("within_5pct"), (int, float))]
        best = max(unified, key=lambda r: r.get("R2") if isinstance(r.get("R2"), (int, float)) else -999)
        training_summary = {
            "best_model": best.get("model"),
            "best_r2": best.get("R2"),
            "best_within_5pct": best.get("within_5pct"),
            "total_models": len(unified),
            "mean_within_5pct": (sum(valid_w5) / len(valid_w5)) if valid_w5 else None,
            "passed_models": sum(1 for v in valid_w5 if v >= 95.0),
            "pass_rate_pct": (sum(1 for v in valid_w5 if v >= 95.0) / len(valid_w5) * 100.0) if valid_w5 else 0.0,
            "mean_r2": (sum(valid_r2) / len(valid_r2)) if valid_r2 else None,
        }

    return {
        "version": version,
        "models_meta": models_meta,
        "datamap": datamap,
        "unified_results": unified,
        "classical_results": classical_results,
        "classical_soh": classical_soh,
        "lstm_results": lstm_results,
        "ensemble_results": ensemble_results,
        "transformer_results": transformer_results,
        "validation": validation,
        "rankings": rankings,
        "classical_rul": classical_rul,
        "training_summary": training_summary,
        "validation_summary": validation_summary,
        "intra_battery": intra_battery,
        "vae_lstm": vae_lstm,
        "dg_itransformer": dg_itransformer,
        "figures": _version_figures(version),
        "battery_stats": _battery_stats_for_version(version),
    }


@router.get("/metrics")
async def get_metrics():
    """Default metrics endpoint: latest version (v3)."""
    return _build_metrics_payload("v3")


@router.get("/{version}/metrics")
async def get_metrics_for_version(version: str):
    """Return version-aware metrics payload from artifacts/{version}."""
    return _build_metrics_payload(version)


@router.get("/{version}/figures")
async def list_version_figures(version: str):
    _ensure_version(version)
    return _version_figures(version)


@router.get("/{version}/figures/{filename}")
async def get_version_figure(version: str, filename: str):
    """Serve saved figures from artifacts/{version}/figures."""
    _ensure_version(version)
    path = _version_root(version) / "figures" / filename
    if not path.exists():
        raise HTTPException(404, f"Figure {filename} not found for {version}")
    content_type = "image/png"
    if path.suffix == ".html":
        content_type = "text/html"
    elif path.suffix == ".svg":
        content_type = "image/svg+xml"
    elif path.suffix.lower() in (".jpg", ".jpeg"):
        content_type = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        content_type = "image/webp"
    return FileResponse(path, media_type=content_type)


@router.get("/v2/figures/{filename}")
async def get_v2_figure(filename: str):
    """Backward-compatible alias for v2 figure endpoint."""
    return await get_version_figure("v2", filename)
