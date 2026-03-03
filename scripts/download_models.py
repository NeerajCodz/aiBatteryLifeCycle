"""
Download model artifacts from Hugging Face Hub at container startup.

Called automatically by the Docker entrypoint before uvicorn starts.
Can also download a specific version on-demand (e.g. from the API).

HF model repo layout  (v1/ and v2/ at repo root):
    v1/models/classical/*.joblib
    v1/models/deep/*.pt  *.keras
    v1/scalers/*.joblib
    v2/models/classical/*.joblib
    v2/models/deep/*.pt  *.keras
    v2/scalers/*.joblib
    v2/results/*.json

Local layout after download  (local_dir = ARTIFACTS_DIR):
    artifacts/v1/...
    artifacts/v2/...
"""

import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
REPO_ID   = "NeerajCodz/aiBatteryLifeCycle"
REPO_TYPE = "model"
# Token read from the HF_TOKEN Space Secret (set in Space Settings -> Secrets)
# For local use: set HF_TOKEN in your shell or .env before running
HF_TOKEN  = os.getenv("HF_TOKEN", "")

# HF repo stores v1/ and v2/ at root → local_dir=ARTIFACTS_DIR maps them to
#   artifacts/v1/...  and  artifacts/v2/...
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Sentinel file — written after a successful full download
SENTINEL = ARTIFACTS_DIR / ".hf_downloaded"

# ──────────────────────────────────────────────────────────────────────────────


def _hf_kwargs(allow_patterns: list | None = None,
               ignore_patterns: list | None = None) -> dict:
    """Build kwargs for snapshot_download; inject token only when non-empty."""
    kwargs: dict = dict(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(ARTIFACTS_DIR),
    )
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    return kwargs


def _key_models(version: str = "v2") -> list:
    base = ARTIFACTS_DIR / version / "models" / "classical"
    return [base / f"{m}.joblib" for m in ("random_forest", "xgboost", "lightgbm")]


def version_loaded(version: str) -> bool:
    """Return True when the given version's key models exist on disk."""
    return all(p.exists() for p in _key_models(version))


def already_downloaded(version: str = "v2") -> bool:
    """Return True only when all three BestEnsemble component models are present."""
    missing = [p for p in _key_models(version) if not p.exists()]
    if missing:
        if SENTINEL.exists():
            SENTINEL.unlink()
            print(f"[download_models] Sentinel stale ({len(missing)} key models missing) — will re-download")
        return False
    return True


def _ensure_hub():
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "huggingface_hub>=0.23", "-q"])


def download_version(version: str) -> None:
    """Download a single version (e.g. 'v1' or 'v2') from HF Hub into artifacts/."""
    _ensure_hub()
    from huggingface_hub import snapshot_download
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download_models] Downloading {version}/ from {REPO_ID} -> {ARTIFACTS_DIR}")
    snapshot_download(**_hf_kwargs(
        allow_patterns=[f"{version}/**"],
        ignore_patterns=["*.log"],
    ))
    print(f"[download_models] {version}/ ready")


def download_all() -> None:
    """Download all versions (v1 + v2) from HF Hub."""
    _ensure_hub()
    from huggingface_hub import snapshot_download
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download_models] Downloading all versions from {REPO_ID} -> {ARTIFACTS_DIR}")
    snapshot_download(**_hf_kwargs(ignore_patterns=["*.log"]))
    SENTINEL.write_text("downloaded\n")
    print("[download_models] Artifacts ready")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None,
                        help="Download only this version, e.g. v1 or v2")
    args = parser.parse_args()

    if args.version:
        if version_loaded(args.version):
            print(f"[download_models] {args.version} already present — skipping")
        else:
            download_version(args.version)
        return

    # Default: ensure v2 (latest) is present
    if already_downloaded("v2"):
        print("[download_models] Artifacts already present — skipping download")
        return

    download_all()


if __name__ == "__main__":
    main()
