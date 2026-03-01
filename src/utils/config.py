"""
src.utils.config
================
Central project configuration — paths, constants, hyperparameters.

Artifact Versioning
-------------------
Trained models, scalers, figures, and result files live under
``artifacts/<version>/`` (e.g. ``artifacts/v1/``, ``artifacts/v2/``).

Use :func:`get_version_paths` and :func:`ensure_version_dirs` to work
with versioned artifact directories consistently.  The module-level
``MODELS_DIR``, ``SCALERS_DIR``, ``FIGURES_DIR`` variables point to the
repository-root artifacts folder and are kept for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "cleaned_dataset"
DATA_DIR = DATASET_DIR / "data"
METADATA_PATH = DATASET_DIR / "metadata.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LOGS_DIR = ARTIFACTS_DIR / "logs"

# Currently active artifact version  (changed when v2 is validated)
ACTIVE_VERSION: str = "v2"

# Ensure all legacy artifact directories exist (backward compat)
for _d in (MODELS_DIR, SCALERS_DIR, FIGURES_DIR, LOGS_DIR,
           MODELS_DIR / "classical", MODELS_DIR / "deep", MODELS_DIR / "ensemble"):
    _d.mkdir(parents=True, exist_ok=True)


# ── Artifact versioning helpers ──────────────────────────────────────────────
def get_version_paths(version: str = ACTIVE_VERSION) -> Dict[str, Path]:
    """Return a dict of typed paths for a given artifact version.

    Keys
    ----
    root, models_classical, models_deep, models_ensemble,
    scalers, figures, results, logs

    Example
    -------
    >>> v2 = get_version_paths("v2")
    >>> joblib.dump(model, v2["models_classical"] / "rf.joblib")
    """
    root = ARTIFACTS_DIR / version
    return {
        "root":             root,
        "models_classical": root / "models" / "classical",
        "models_deep":      root / "models" / "deep",
        "models_ensemble":  root / "models" / "ensemble",
        "scalers":          root / "scalers",
        "figures":          root / "figures",
        "results":          root / "results",
        "logs":             root / "logs",
    }


def ensure_version_dirs(version: str = ACTIVE_VERSION) -> Dict[str, Path]:
    """Create all subdirectories for a given version and return paths dict."""
    paths = get_version_paths(version)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

# ── Battery dataset constants ────────────────────────────────────────────────
NOMINAL_CAPACITY_AH = 2.0
EOL_30PCT_AH = 1.4
EOL_20PCT_AH = 1.6
EXCLUDED_BATTERIES = {"B0049", "B0050", "B0051", "B0052"}

# ── Training defaults ────────────────────────────────────────────────────────
RANDOM_STATE = 42
TRAIN_RATIO = 0.8
WINDOW_SIZE = 32
N_BINS = 20               # For fixed-length downsample of within-cycle data
BATCH_SIZE = 32
MAX_EPOCHS = 150
EARLY_STOP_PATIENCE = 20
MC_DROPOUT_SAMPLES = 50   # For uncertainty estimation

# ── Classical ML ─────────────────────────────────────────────────────────────
N_OPTUNA_TRIALS = 100
CV_FOLDS = 5

# ── Deep learning ────────────────────────────────────────────────────────────
LEARNING_RATE = 1e-3
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2
TRANSFORMER_LAYERS = TRANSFORMER_NLAYERS  # alias for convenience
DROPOUT = 0.2
LATENT_DIM = 16            # For VAE

# ── Feature col lists (duplicated from preprocessing for easy import) ────────
FEATURE_COLS_SCALAR = [
    "cycle_number", "ambient_temperature",
    "peak_voltage", "min_voltage", "voltage_range",
    "avg_current", "avg_temp", "temp_rise",
    "cycle_duration", "Re", "Rct", "delta_capacity",
]

# ── Visualization ────────────────────────────────────────────────────────────
MATPLOTLIB_STYLE = "seaborn-v0_8-whitegrid"
FIG_DPI = 150
FIG_SIZE = (12, 7)
CMAP_DIVERGING = "RdYlGn"
CMAP_SEQUENTIAL = "viridis"
