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

router = APIRouter(prefix="/api", tags=["visualization"])

_PROJECT = Path(__file__).resolve().parents[2]
_ARTIFACTS = _PROJECT / "artifacts"
_FIGURES = _ARTIFACTS / "figures"
_DATASET = _PROJECT / "cleaned_dataset"
_V2_RESULTS = _ARTIFACTS / "v2" / "results"
_V2_REPORTS = _ARTIFACTS / "v2" / "reports"
_V2_FIGURES = _ARTIFACTS / "v2" / "figures"


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


@router.get("/metrics")
async def get_metrics():
    """Return comprehensive model metrics data from v2 artifacts for the Metrics dashboard."""
    # Unified results (all models)
    unified = _safe_read_csv(_V2_RESULTS / "unified_results.csv")
    # Classical results (v2 retrained)
    classical_v2 = _safe_read_csv(_V2_RESULTS / "v2_classical_results.csv")
    # Classical SOH results (v1)
    classical_soh = _safe_read_csv(_V2_RESULTS / "classical_soh_results.csv")
    # LSTM results
    lstm_results = _safe_read_csv(_V2_RESULTS / "lstm_soh_results.csv")
    # Ensemble results
    ensemble_results = _safe_read_csv(_V2_RESULTS / "ensemble_results.csv")
    # Transformer results
    transformer_results = _safe_read_csv(_V2_RESULTS / "transformer_soh_results.csv")
    # Validation
    validation = _safe_read_csv(_V2_RESULTS / "v2_model_validation.csv")
    # Final rankings
    rankings = _safe_read_csv(_V2_RESULTS / "final_rankings.csv")
    # Classical RUL results
    classical_rul = _safe_read_csv(_V2_RESULTS / "classical_rul_results.csv")

    # JSON summaries
    training_summary = _safe_read_json(_V2_RESULTS / "v2_training_summary.json")
    validation_summary = _safe_read_json(_V2_RESULTS / "v2_validation_summary.json")
    intra_battery = _safe_read_json(_V2_RESULTS / "v2_intra_battery.json")
    vae_lstm = _safe_read_json(_V2_RESULTS / "vae_lstm_results.json")
    dg_itransformer = _safe_read_json(_V2_RESULTS / "dg_itransformer_results.json")

    # Available v2 figures
    v2_figures = []
    if _V2_FIGURES.exists():
        v2_figures = sorted([f.name for f in _V2_FIGURES.iterdir() if f.is_file() and f.suffix in ('.png', '.svg')])

    # Battery features summary
    features_path = _V2_RESULTS / "battery_features.csv"
    battery_stats = {}
    if features_path.exists():
        df = pd.read_csv(features_path)
        battery_stats = {
            "total_samples": len(df),
            "batteries": int(df["battery_id"].nunique()),
            "avg_soh": round(float(df["SoH"].mean()), 2),
            "min_soh": round(float(df["SoH"].min()), 2),
            "max_soh": round(float(df["SoH"].max()), 2),
            "avg_rul": round(float(df["RUL"].mean()), 1),
            "feature_columns": [c for c in df.columns.tolist() if c not in ["battery_id", "datetime"]],
            "degradation_distribution": json.loads(df["degradation_state"].value_counts().to_json()),
            "temp_groups": sorted(df["ambient_temperature"].unique().tolist()),
        }

    return {
        "unified_results": unified,
        "classical_v2": classical_v2,
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
        "v2_figures": v2_figures,
        "battery_stats": battery_stats,
    }


@router.get("/v2/figures/{filename}")
async def get_v2_figure(filename: str):
    """Serve a saved figure from artifacts/v2/figures."""
    path = _V2_FIGURES / filename
    if not path.exists():
        raise HTTPException(404, f"Figure {filename} not found")
    content_type = "image/png"
    if path.suffix == ".html":
        content_type = "text/html"
    elif path.suffix == ".svg":
        content_type = "image/svg+xml"
    return FileResponse(path, media_type=content_type)
