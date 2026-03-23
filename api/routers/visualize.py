"""
api.routers.visualize
=====================
Endpoints that serve pre-computed or on-demand visualisation data
consumed by the React frontend.
"""

from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from api.model_registry import registry, classify_degradation, soh_to_color
from api.schemas import BatteryVizData, DashboardData

router = APIRouter(prefix="/api", tags=["visualization"])

_PROJECT = Path(__file__).resolve().parents[2]
_ARTIFACTS = _PROJECT / "artifacts"
_FIGURES = _ARTIFACTS / "figures"
_DATASET = _PROJECT / "cleaned_dataset"
_HF_RAW_BASE = os.getenv(
    "HF_ARTIFACTS_RAW_BASE",
    "https://huggingface.co/NeerajCodz/aiBatteryLifeCycle/resolve/main",
)
_HF_TREE_API_BASE = os.getenv(
    "HF_ARTIFACTS_TREE_API_BASE",
    "https://huggingface.co/api/models/NeerajCodz/aiBatteryLifeCycle/tree/main",
)

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
def _hf_url(rel_path: str) -> str:
    return f"{_HF_RAW_BASE.rstrip('/')}/{rel_path.lstrip('/')}"


def _hf_version_url(version: str, rel_path: str) -> str:
    return _hf_url(f"{version}/{rel_path.lstrip('/')}")


def _hf_tree_api_url(rel_path: str = "") -> str:
    base = _HF_TREE_API_BASE.rstrip("/")
    rel = rel_path.lstrip("/")
    return f"{base}/{rel}" if rel else base


def _read_remote_text(url: str) -> str | None:
    req = Request(url, headers={"User-Agent": "aiBatteryLifecycle/metrics"})
    try:
        with urlopen(req, timeout=15) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None


def _read_remote_json(url: str) -> dict:
    text = _read_remote_text(url)
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_figure_names_from_datamap(datamap: dict) -> list[str]:
    files = datamap.get("files", []) if isinstance(datamap, dict) else []
    names: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("path", ""))
        if not rel.startswith("figures/"):
            continue
        name = Path(rel).name
        if Path(name).suffix.lower() in (".png", ".svg", ".jpg", ".jpeg", ".webp"):
            names.append(name)
    return sorted(set(names))


def _list_tree_api_figures(version: str) -> list[str]:
    text = _read_remote_text(_hf_tree_api_url(f"{version}/figures"))
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    names: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("path") or item.get("name") or "")
        name = Path(rel).name
        if Path(name).suffix.lower() in (".png", ".svg", ".jpg", ".jpeg", ".webp"):
            names.append(name)
    return sorted(set(names))


def _safe_read_json_any(path: Path, remote_url: str | None = None) -> Any:
    """Read JSON (dict/list) from disk, then optional remote fallback."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if remote_url:
        text = _read_remote_text(remote_url)
        if text:
            try:
                return json.loads(text)
            except Exception:
                pass
    return None


def _looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _location_filename(location: str) -> str:
    loc = location.strip()
    if not loc:
        return ""
    if _looks_like_url(loc):
        return Path(urlparse(loc).path).name
    if loc.startswith("figures/"):
        return Path(loc).name
    return Path(loc).name


def _prettify_figure_name(raw: str) -> str:
    if not raw:
        return ""
    base = Path(raw).stem if Path(raw).suffix else raw
    return base.replace("_", " ").replace("-", " ").strip()


def _tags_from_label(label: str) -> list[str]:
    base = Path(label).stem if Path(label).suffix else label
    raw_parts = (
        base.replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .lower()
        .split("_")
    )
    stop = {"and", "the", "all", "by", "vs", "with", "for"}
    out: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        token = part.strip()
        if len(token) < 2 or token in stop or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _location_to_serve_url(version: str, location: str) -> str:
    loc = location.strip()
    if not loc:
        return ""
    if _looks_like_url(loc) or loc.startswith("/"):
        return loc
    filename = _location_filename(loc)
    if not filename:
        return ""
    return f"/api/{version}/figures/{quote(filename)}"


def _normalize_figure_manifest_item(version: str, raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        loc = raw.strip()
        if not loc:
            return None
        pretty = _prettify_figure_name(_location_filename(loc) or loc)
        return {
            "name": pretty or loc,
            "tags": _tags_from_label(pretty or loc),
            "location": loc,
            "url": _location_to_serve_url(version, loc),
        }

    if not isinstance(raw, dict):
        return None

    location = str(
        raw.get("location")
        or raw.get("url")
        or raw.get("path")
        or raw.get("file")
        or raw.get("src")
        or ""
    ).strip()
    name = str(raw.get("name") or "").strip()

    if not location and name and Path(name).suffix:
        location = name
    if not location:
        return None

    if not name:
        name = _prettify_figure_name(_location_filename(location) or location)
    if not name:
        name = location

    raw_tags = raw.get("tags", [])
    if isinstance(raw_tags, str):
        tags = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
    elif isinstance(raw_tags, list):
        tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
    else:
        tags = []
    if not tags:
        tags = _tags_from_label(name)

    # De-duplicate while preserving order
    deduped_tags: list[str] = []
    seen_tags: set[str] = set()
    for t in tags:
        if t in seen_tags:
            continue
        seen_tags.add(t)
        deduped_tags.append(t)

    return {
        "name": name,
        "tags": deduped_tags,
        "location": location,
        "url": _location_to_serve_url(version, location),
    }


def _version_figures_manifest(version: str) -> list[dict[str, Any]]:
    payload = _safe_read_json_any(
        _version_root(version) / "figures.json",
        _hf_version_url(version, "figures.json"),
    )

    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        figures = payload.get("figures")
        items = payload.get("items")
        if isinstance(figures, list):
            raw_items = figures
        elif isinstance(items, list):
            raw_items = items
        else:
            raw_items = []
    else:
        raw_items = []

    manifest: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_items:
        normalized = _normalize_figure_manifest_item(version, item)
        if not normalized:
            continue
        key = (normalized["name"].lower(), normalized["location"])
        if key in seen:
            continue
        seen.add(key)
        manifest.append(normalized)

    if manifest:
        return manifest

    # Fallback manifest generated from discovered figure filenames.
    fallback: list[dict[str, Any]] = []
    for filename in _version_figures(version):
        normalized = _normalize_figure_manifest_item(version, filename)
        if normalized:
            fallback.append(normalized)
    return fallback


def _safe_read_csv(path: Path, remote_url: str | None = None) -> list[dict]:
    """Read CSV from disk, then fallback to HF raw URL."""
    try:
        if path.exists():
            df = pd.read_csv(path)
            return json.loads(df.to_json(orient="records"))
    except Exception:
        pass
    if not remote_url:
        return []
    text = _read_remote_text(remote_url)
    if text is None:
        return []
    try:
        df = pd.read_csv(StringIO(text))
        return json.loads(df.to_json(orient="records"))
    except Exception:
        return []


def _safe_read_json(path: Path, remote_url: str | None = None) -> dict:
    """Read JSON from disk, then fallback to HF raw URL."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    if not remote_url:
        return {}
    text = _read_remote_text(remote_url)
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_dataset_info() -> dict:
    """Load global dataset metadata used by Dataset/Validation tabs."""
    return _safe_read_json(
        _ARTIFACTS / "dataset.json",
        _hf_url("dataset.json"),
    )


def _safe_read_csv_first(version: str, rel_paths: list[str]) -> list[dict]:
    for rel in rel_paths:
        path = _version_root(version) / rel
        rows = _safe_read_csv(path, _hf_version_url(version, rel))
        if rows:
            return rows
    return []


def _safe_read_json_first(version: str, rel_paths: list[str]) -> dict:
    for rel in rel_paths:
        path = _version_root(version) / rel
        data = _safe_read_json(path, _hf_version_url(version, rel))
        if data:
            return data
    return {}


def _version_root(version: str) -> Path:
    return _ARTIFACTS / version


def _ensure_version(version: str) -> None:
    if version not in _SUPPORTED_VERSIONS:
        raise HTTPException(400, f"Unknown version '{version}'")


def _version_figures(version: str) -> list[str]:
    fig_dir = _version_root(version) / "figures"
    if not fig_dir.exists():
        local = []
    else:
        local = sorted([
            f.name
            for f in fig_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".png", ".svg", ".jpg", ".jpeg", ".webp")
        ])
    if local:
        return local

    # Local datamap may only list downloaded subsets on Spaces.
    # If it does not contain figure entries, explicitly query HF raw datamap.
    local_datamap = _safe_read_json(_version_root(version) / "datamap.json", None)
    local_map_figs = _extract_figure_names_from_datamap(local_datamap)
    if local_map_figs:
        return local_map_figs

    remote_datamap = _read_remote_json(_hf_version_url(version, "datamap.json"))
    remote_map_figs = _extract_figure_names_from_datamap(remote_datamap)
    if remote_map_figs:
        return remote_map_figs

    # Last-resort fallback: Hugging Face tree API listing for /<version>/figures
    return _list_tree_api_figures(version)


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

    results = root / "results"
    reports = root / "reports"
    models_meta = _safe_read_json(root / "models.json", _hf_version_url(version, "models.json"))
    datamap = _safe_read_json(root / "datamap.json", _hf_version_url(version, "datamap.json"))
    dataset_info = _load_dataset_info()

    unified = _safe_read_csv_first(version, ["results/unified_results.csv"])
    classical_results = _safe_read_csv_first(version, [
        "results/classical_results.csv",
        "results/classical_soh_results.csv",
    ])
    classical_soh = _safe_read_csv_first(version, ["results/classical_soh_results.csv"])
    lstm_results = _safe_read_csv_first(version, ["results/lstm_soh_results.csv"])
    ensemble_results = _safe_read_csv_first(version, ["results/ensemble_results.csv"])
    transformer_results = _safe_read_csv_first(version, ["results/transformer_soh_results.csv"])
    validation = _safe_read_csv_first(version, [
        "results/model_validation.csv",
        "reports/model_validation.csv",
    ])
    rankings = _safe_read_csv_first(version, ["results/final_rankings.csv"])
    classical_rul = _safe_read_csv_first(version, ["results/classical_rul_results.csv"])

    training_summary = _safe_read_json_first(version, [
        "results/training_summary.json",
        "reports/training_summary.json",
    ])
    validation_summary = _safe_read_json_first(version, [
        "results/validation_summary.json",
        "reports/validation_summary.json",
    ])
    intra_battery = _safe_read_json_first(version, [
        "results/intra_battery.json",
        "reports/intra_battery.json",
    ])
    vae_lstm = _safe_read_json_first(version, ["results/vae_lstm_results.json"])
    dg_itransformer = _safe_read_json_first(version, ["results/dg_itransformer_results.json"])

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

    figures_manifest = _version_figures_manifest(version)
    figures: list[str] = []
    seen_names: set[str] = set()
    for item in figures_manifest:
        loc = str(item.get("location", ""))
        name = _location_filename(loc) or str(item.get("name", "")).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        figures.append(name)

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
        "figures": figures,
        "figures_manifest": figures_manifest,
        "battery_stats": _battery_stats_for_version(version),
        "dataset_info": dataset_info,
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
    manifest = _version_figures_manifest(version)
    names: list[str] = []
    seen: set[str] = set()
    for item in manifest:
        loc = str(item.get("location", ""))
        name = _location_filename(loc) or str(item.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


@router.get("/{version}/figures.json")
async def get_version_figures_manifest(version: str):
    """Return versioned figure manifest (name/tags/location/url)."""
    _ensure_version(version)
    return _version_figures_manifest(version)


@router.get("/{version}/figures/{filename}")
async def get_version_figure(version: str, filename: str):
    """Serve saved figures from artifacts/{version}/figures."""
    _ensure_version(version)
    path = _version_root(version) / "figures" / filename
    if not path.exists():
        requested = Path(filename).name
        if requested != filename:
            raise HTTPException(400, "Invalid figure filename")
        if requested not in set(_version_figures(version)):
            raise HTTPException(404, f"Figure {filename} not found for {version}")
        return RedirectResponse(
            url=_hf_version_url(version, f"figures/{requested}"),
            status_code=307,
        )
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
