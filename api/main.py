"""
api.main
========
FastAPI application entry-point for the AI Battery Lifecycle Predictor.

Architecture
------------
- **v1 (Classical)**    : Ridge, Lasso, ElasticNet, KNN ×3, SVR,
                          Random Forest, XGBoost, LightGBM
- **v2 (Deep)**         : Vanilla LSTM, BiLSTM, GRU, Attention LSTM,
                          BatteryGPT, TFT, iTransformer ×3, VAE-LSTM
- **v2.6 (Ensemble)**   : BestEnsemble — weighted average of RF + XGB + LGB
                          (weights proportional to R²)

Mounted routes
--------------
- ``/api/*``      REST endpoints  (predict, batch, recommend, models, visualize)
- ``/gradio``     Gradio interactive demo  (optional, requires *gradio* package)
- ``/``           React SPA  (served from ``frontend/dist/``)

Key endpoints
-------------
- ``POST /api/predict``          — single-cycle SOH + RUL prediction
- ``POST /api/predict/ensemble`` — always uses BestEnsemble (v2.6)
- ``POST /api/predict/batch``    — batch prediction from JSON array
- ``GET  /api/models``           — list all models with version / R² metadata
- ``GET  /api/models/versions``  — group models by generation (v1/v2)
- ``GET  /health``               — liveness probe

Run locally
-----------
::

    uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload

Docker
------
::

    docker compose up --build
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.model_registry import registry, registry_v1, registry_v2, registry_v3
from api.schemas import HealthResponse
from src.utils.logger import get_logger

log = get_logger(__name__)

__version__ = "3.0.0"

# ── Static frontend path ────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_FRONTEND_DIST = _HERE.parent / "frontend" / "dist"


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    log.info("Loading model registries …")
    registry_v1.load_all()
    log.info("v1 registry ready — %d models loaded", registry_v1.model_count)
    registry_v2.load_all()
    log.info("v2 registry ready — %d models loaded", registry_v2.model_count)
    registry_v3.load_all()
    log.info("v3 registry ready — %d models loaded", registry_v3.model_count)
    yield
    log.info("Shutting down battery-lifecycle API")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Battery Lifecycle Predictor",
    description=(
        "Predict SOH, RUL, and degradation state of Li-ion batteries "
        "using models trained on the NASA PCoE dataset."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    return HealthResponse(
        status="ok",
        version=__version__,
        models_loaded=registry_v1.model_count + registry_v2.model_count + registry_v3.model_count,
        device=registry.device,
    )


# ── Version management ───────────────────────────────────────────────────────
_REGISTRIES = {"v1": registry_v1, "v2": registry_v2, "v3": registry_v3}
_version_status: dict[str, str] = {}   # "downloading" | "ready" | "error"


def _artifacts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "artifacts"


def _version_loaded(version: str) -> bool:
    base = _artifacts_dir() / version / "models" / "classical"
    return any(base.glob("*.joblib")) if base.exists() else False


@app.get("/api/versions", tags=["meta"])
async def list_versions():
    """Return all known versions with loaded / downloading status."""
    return [
        {
            "id": v,
            "display": f"Version {v[1]}",
            "loaded": _version_loaded(v),
            "model_count": _REGISTRIES[v].model_count,
            "status": _version_status.get(v, "ready" if _version_loaded(v) else "not_downloaded"),
        }
        for v in ["v3", "v2", "v1"]
    ]


async def _bg_load_version(version: str) -> None:
    import subprocess, sys as _sys
    try:
        proc = await asyncio.create_subprocess_exec(
            _sys.executable, "scripts/download_models.py", "--version", version,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        await proc.wait()
        if proc.returncode == 0:
            _REGISTRIES[version].load_all()
            _version_status[version] = "ready"
            log.info("Version %s loaded on demand — %d models", version,
                     _REGISTRIES[version].model_count)
        else:
            _version_status[version] = "error"
            log.error("download_models.py failed for version %s", version)
    except Exception as exc:
        _version_status[version] = "error"
        log.error("Failed to load version %s: %s", version, exc)


@app.post("/api/versions/{version}/load", tags=["meta"])
async def load_version(version: str, background_tasks: BackgroundTasks):
    """Download + activate a model version from HF Hub (runs in background)."""
    if version not in _REGISTRIES:
        raise HTTPException(status_code=400, detail=f"Unknown version '{version}'")
    if _version_status.get(version) == "downloading":
        return {"status": "downloading", "version": version}
    _version_status[version] = "downloading"
    background_tasks.add_task(_bg_load_version, version)
    return {"status": "downloading", "version": version}


# ── Include routers ──────────────────────────────────────────────────────────
from api.routers.predict import router as predict_router, v1_router
from api.routers.predict_v2 import router as predict_v2_router
from api.routers.predict_v3 import router as predict_v3_router
from api.routers.visualize import router as viz_router
from api.routers.simulate import router as simulate_router

app.include_router(predict_router)    # /api/* (default, uses v2 registry)
app.include_router(v1_router)         # /api/v1/* (legacy v1 models)
app.include_router(predict_v2_router) # /api/v2/* (v2 models)
app.include_router(predict_v3_router) # /api/v3/* (v3 models, best accuracy)
app.include_router(simulate_router)   # /api/v3/simulate (ML-driven simulation)
app.include_router(viz_router)


# ── Mount Gradio ─────────────────────────────────────────────────────────────
try:
    import gradio as gr
    from api.gradio_app import create_gradio_app

    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
    log.info("Gradio UI mounted at /gradio")
except ImportError:
    log.warning("Gradio not installed — /gradio endpoint unavailable")


# ── Serve React SPA ──────────────────────────────────────────────────────────
if _FRONTEND_DIST.exists() and (_FRONTEND_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="static-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_catch_all(full_path: str):
        """Serve React SPA for any path not matched by API routes."""
        file_path = _FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_FRONTEND_DIST / "index.html")

    log.info("React SPA served from %s", _FRONTEND_DIST)
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "message": "AI Battery Lifecycle Predictor API",
            "docs": "/docs",
            "gradio": "/gradio",
            "health": "/health",
        }
