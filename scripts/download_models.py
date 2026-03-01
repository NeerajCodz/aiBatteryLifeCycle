"""
Download model artifacts from Hugging Face Hub at container startup.

Called automatically by the Docker entrypoint before uvicorn starts.
Downloads only if artifacts are missing (idempotent — skips already-present files).
"""

import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
REPO_ID   = "NeerajCodz/aiBatteryLifeCycle"
REPO_TYPE = "model"
# Token read from the HF_TOKEN Space Secret (set in Space Settings → Secrets)
# For local use: set HF_TOKEN in your shell or .env before running
HF_TOKEN  = os.getenv("HF_TOKEN", "")

# Project root — snapshot_download uses this as local_dir so that files stored
# at "artifacts/v1/..." in the HF repo land at <PROJECT_ROOT>/artifacts/v1/...
# (NOT at <PROJECT_ROOT>/artifacts/artifacts/v1/... which would happen if we
# pointed local_dir at the artifacts/ subfolder directly).
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # /app  (or repo root locally)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Sentinel file — written after a successful download
SENTINEL = ARTIFACTS_DIR / ".hf_downloaded"

# ──────────────────────────────────────────────────────────────────────────────


def already_downloaded() -> bool:
    """Return True only when all three BestEnsemble component models are present.

    These three are required for ML simulation to work.  Any other models are
    optional bonuses, but if these three are absent the Container must download.
    """
    required = [
        ARTIFACTS_DIR / "v2" / "models" / "classical" / "random_forest.joblib",
        ARTIFACTS_DIR / "v2" / "models" / "classical" / "xgboost.joblib",
        ARTIFACTS_DIR / "v2" / "models" / "classical" / "lightgbm.joblib",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        if SENTINEL.exists():
            SENTINEL.unlink()   # stale sentinel — remove so next run re-downloads
            print(f"[download_models] Sentinel was stale ({len(missing)} key models missing) — will re-download")
        return False
    return True


def download_artifacts() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[download_models] huggingface_hub not installed — installing now…")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.23", "-q"])
        from huggingface_hub import snapshot_download

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[download_models] Downloading from {REPO_ID} → {ARTIFACTS_DIR}")

    # Repo is public — only pass token when non-empty.
    # Passing an empty string causes a 401 even on public repos.
    # IMPORTANT: local_dir must be PROJECT_ROOT (not artifacts/) because the HF
    # repo stores files under "artifacts/v1/..." — pointing local_dir at the
    # project root makes them land at <root>/artifacts/v1/... as expected.
    kwargs: dict = dict(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(PROJECT_ROOT),
        ignore_patterns=["*.png", "*.jpg", "*.pdf", "*.log", "figures/**"],
    )
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    snapshot_download(**kwargs)

    # Write sentinel
    SENTINEL.write_text("downloaded\n")
    print("[download_models] ✅ Artifacts ready")


def main():
    if already_downloaded():
        print("[download_models] Artifacts already present — skipping download")
        return

    download_artifacts()


if __name__ == "__main__":
    main()
