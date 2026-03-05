"""
Upload all trained model artifacts to Hugging Face Hub model repository.

Usage:
    python scripts/upload_models_to_hub.py

This script:
  - Creates the HF model repo NeerajCodz/aiBatteryLifeCycle if it doesn't exist
  - Uploads artifacts/v1/ and artifacts/v2/ preserving folder structure
  - Writes a proper README / model card
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN")          # set HF_TOKEN in your shell before running
REPO_ID    = "NeerajCodz/aiBatteryLifeCycle"
REPO_TYPE  = "model"

ROOT       = Path(__file__).resolve().parent.parent   # repo root
ARTIFACTS  = ROOT / "artifacts"

MODEL_CARD = """\
---
license: mit
language:
  - en
tags:
  - battery
  - state-of-health
  - remaining-useful-life
  - time-series
  - regression
  - lstm
  - transformer
  - xgboost
  - lightgbm
  - random-forest
  - ensemble
datasets:
  - NASA-PCoE-Battery
metrics:
  - r2
  - mae
  - rmse
pipeline_tag: tabular-regression
---

# AI Battery Lifecycle — Model Repository

Trained model artifacts for the [aiBatteryLifeCycle](https://huggingface.co/spaces/NeerajCodz/aiBatteryLifeCycle) project.

SOH (State-of-Health) and RUL (Remaining Useful Life) prediction for lithium-ion batteries
trained on the NASA PCoE Battery Dataset.

## Repository Layout

```
artifacts/
├── v1/
│   ├── models/
│   │   ├── classical/   # Ridge, Lasso, ElasticNet, KNN ×3, SVR, XGBoost, LightGBM, RF
│   │   └── deep/        # Vanilla LSTM, Bi-LSTM, GRU, Attention-LSTM, TFT,
│   │                    # BatteryGPT, iTransformer, Physics-iTransformer,
│   │                    # DG-iTransformer, VAE-LSTM
│   └── scalers/         # MinMax, Standard, Linear, Sequence scalers
└── v2/
    ├── models/
    │   ├── classical/   # Same family + Extra Trees, Gradient Boosting, best_rul_model
    │   └── deep/        # Same deep models re-trained on v2 feature set
    ├── scalers/         # Per-model feature scalers
    └── results/         # Validation JSONs
```

## Model Performance Summary (v3)

| Rank | Model | R² | MAE | Family |
|------|-------|----|-----|--------|
| 1 | XGBoost | 0.9866 | 1.58 | Classical |
| 2 | GradientBoosting | 0.9860 | 1.38 | Classical |
| 3 | LightGBM | 0.9826 | 1.98 | Classical |
| 4 | RandomForest | 0.9814 | 1.83 | Classical |
| 5 | ExtraTrees | 0.9701 | 3.20 | Classical |
| 6 | TFT | 0.8751 | 3.88 | Transformer |
| 7 | Weighted Avg Ensemble | 0.8991 | 3.51 | Ensemble |

## Usage

These artifacts are automatically downloaded by the Space on startup via
`scripts/download_models.py`. You can also use them directly:

```python
from huggingface_hub import snapshot_download

local = snapshot_download(
    repo_id="NeerajCodz/aiBatteryLifeCycle",
    repo_type="model",
    local_dir="artifacts",
    token="<your-token>",   # only needed if private
)
```

## Framework

- **Classical models:** scikit-learn / XGBoost / LightGBM `.joblib`
- **Deep models (PyTorch):** `.pt` state-dicts (CPU weights)
- **Deep models (Keras):** `.keras` SavedModel format
- **Scalers:** scikit-learn `.joblib`

## Citation

```bibtex
@misc{aiBatteryLifeCycle2025,
  author  = {Neeraj},
  title   = {AI Battery Lifecycle — SOH/RUL Prediction},
  year    = {2025},
  url     = {https://huggingface.co/spaces/NeerajCodz/aiBatteryLifeCycle}
}
```
"""

# ──────────────────────────────────────────────────────────────────────────────

def main():
    api = HfApi(token=HF_TOKEN)

    # 1. Create repo (no-op if already exists)
    print(f"Creating / verifying repo: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=HF_TOKEN,
        exist_ok=True,
        private=False,
    )

    # 2. Upload model card
    print("Uploading README / model card...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="chore: update model card",
    )

    # 3. Upload each version directly at repo root: v1/ and v2/ (NOT under artifacts/)
    #    Split into one commit per subdirectory so no single commit is too large
    #    (the 100 MB random_forest.joblib would time out a combined commit).
    for version in ["v1", "v2", "v3"]:
        version_path = ARTIFACTS / version
        if not version_path.exists():
            print(f"  [skip] {version_path} does not exist")
            continue

        # Gather all subdirectories that contain files (plus version root files)
        subdirs = sorted({
            p.parent
            for p in version_path.rglob("*")
            if p.is_file()
            and ".log" not in p.suffixes
            and "__pycache__" not in p.parts
        })

        for subdir in subdirs:
            rel = subdir.relative_to(version_path)
            # Use as_posix() to ensure forward slashes on Windows
            rel_posix = rel.as_posix()
            repo_path = version if rel_posix == "." else f"{version}/{rel_posix}"

            files_in_sub = [
                f for f in subdir.iterdir()
                if f.is_file()
                and ".log" not in f.suffixes
                and f.name != ".hf_downloaded"
            ]
            if not files_in_sub:
                continue

            print(f"  Uploading {len(files_in_sub)} file(s) → {repo_path}/")
            upload_folder(
                folder_path=str(subdir),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=HF_TOKEN,
                ignore_patterns=["*.log"],
                commit_message=f"feat: {repo_path}",
                run_as_future=False,
            )
            print(f"    [OK] {repo_path}/")

    # 4. Remove old artifacts/ tree from previous uploads
    print("\nCleaning up legacy artifacts/ folder in HF repo (if any)...")
    try:
        api.delete_folder(
            path_in_repo="artifacts",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="chore: remove legacy artifacts/ folder (moved to repo root)",
        )
        print("  [OK] artifacts/ removed")
    except Exception as e:
        print(f"  [skip] No legacy artifacts/ to remove ({e})")

    print("\n[OK] All artifacts uploaded to", f"https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
