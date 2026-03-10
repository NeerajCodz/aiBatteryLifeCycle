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
import json
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
DEFAULT_STARTUP_TOP_MODELS = 3

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


def _key_models(version: str = "v3") -> list:
    base = ARTIFACTS_DIR / version / "models" / "classical"
    return [base / f"{m}.joblib" for m in ("random_forest", "xgboost", "lightgbm")]


def version_loaded(version: str) -> bool:
    """Return True when the given version's key models exist on disk."""
    return all(p.exists() for p in _key_models(version))


def already_downloaded(version: str = "v3") -> bool:
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


def _read_models_meta(version: str) -> dict:
    meta_path = ARTIFACTS_DIR / version / "models.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _model_score(info: dict) -> float:
    """Return a comparable score for a model metadata record.

    Priority:
      1) R2 (higher is better)
      2) f1_weighted / f1_macro (higher is better)
      3) within_5pct (higher is better)
      4) mae (lower is better; converted to negative contribution)
    """
    if not isinstance(info, dict):
        return float("-inf")

    r2 = info.get("r2")
    if isinstance(r2, (int, float)):
        return float(r2)

    # Fallback when r2 is unavailable.
    f1w = info.get("f1_weighted")
    if isinstance(f1w, (int, float)):
        return float(f1w)

    f1m = info.get("f1_macro")
    if isinstance(f1m, (int, float)):
        return float(f1m)

    w5 = info.get("within_5pct")
    if isinstance(w5, (int, float)):
        return float(w5) / 100.0

    mae = info.get("mae")
    if isinstance(mae, (int, float)):
        return -float(mae)

    return float("-inf")


def select_top_models(version: str, top_k: int = DEFAULT_STARTUP_TOP_MODELS) -> list[str]:
    """Pick top-k concrete models from models.json by best available metric score.

    Excludes virtual entries without a model file.
    """
    meta = _read_models_meta(version)
    models = meta.get("models", {}) if isinstance(meta, dict) else {}
    scored: list[tuple[float, str]] = []
    for name, info in models.items():
        if not isinstance(info, dict):
            continue
        if not info.get("file"):
            continue
        score = _model_score(info)
        if score != float("-inf"):
            scored.append((score, name))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in scored[:top_k]]


def _collect_model_patterns(version: str, model_names: list[str]) -> list[str]:
    """Build allow_patterns for model files + required shared artifacts."""
    meta = _read_models_meta(version)
    models = meta.get("models", {}) if isinstance(meta, dict) else {}
    patterns = {
        f"{version}/models.json",
        f"{version}/features/train_split.csv",
        f"{version}/scalers/features_standard.joblib",
        f"{version}/scalers/sequence_scaler.joblib",
        f"{version}/scalers/features_minmax.joblib",
    }
    for model_name in model_names:
        info = models.get(model_name, {})
        rel_file = info.get("file") if isinstance(info, dict) else None
        if rel_file:
            patterns.add(f"{version}/{rel_file}")
    return sorted(patterns)


def download_models(version: str, model_names: list[str]) -> None:
    """Download selected model artifacts for one version."""
    _ensure_hub()
    from huggingface_hub import snapshot_download
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    allow = _collect_model_patterns(version, model_names)
    print(f"[download_models] Downloading selected models for {version}: {model_names}")
    snapshot_download(**_hf_kwargs(
        allow_patterns=allow,
        ignore_patterns=["*.log"],
    ))
    print(f"[download_models] Selected model artifacts ready for {version}")


def download_model(version: str, model_name: str) -> None:
    """Download one model artifact (plus required metadata/scalers)."""
    download_models(version, [model_name])


def download_models_meta_only(versions: list[str]) -> None:
    """Download only models.json for listed versions."""
    _ensure_hub()
    from huggingface_hub import snapshot_download
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    allow = [f"{v}/models.json" for v in versions]
    print(f"[download_models] Downloading metadata files: {allow}")
    snapshot_download(**_hf_kwargs(
        allow_patterns=allow,
        ignore_patterns=["*.log"],
    ))
    print("[download_models] Metadata ready")


def download_all() -> None:
    """Download all versions (v1 + v2 + v3) from HF Hub."""
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
    parser.add_argument("--model", default=None,
                        help="Download only this model key inside --version")
    args = parser.parse_args()

    if args.version and args.model:
        download_models_meta_only([args.version])
        download_model(args.version, args.model)
        return

    if args.version:
        if version_loaded(args.version):
            print(f"[download_models] {args.version} already present — skipping")
        else:
            download_version(args.version)
        return

    # Default: ensure models.json exists for all versions, but download only
    # latest (v3) heavy model artifacts at startup.
    download_models_meta_only(["v1", "v2", "v3"])

    # Then fetch only top-N (default=3) v3 model artifacts by best score.
    top_models = select_top_models("v3")
    if not top_models:
        print("[download_models] Could not resolve top models from models.json, falling back to full v3 download")
        if already_downloaded("v3"):
            print("[download_models] v3 artifacts already present — skipping download")
            return
        download_version("v3")
        return

    print(f"[download_models] Default startup model set (v3 top-{DEFAULT_STARTUP_TOP_MODELS}): {top_models}")
    download_models("v3", top_models)


if __name__ == "__main__":
    main()
