# Project Structure (v2.0)

## Root Level Organization

```
aiBatteryLifecycle/
├── 📂 api/                    FastAPI backend with model registry
├── 📂 artifacts/              Versioned model artifacts & results
│   ├── v1/                    Legacy models (cross-battery train/test)
│   └── v2/                    Production models (intra-battery split) ✓ ACTIVE
│       ├── models/            Trained models: {classical, deep, ensemble}
│       ├── scalers/           Feature scalers (StandardScaler for linear models)
│       ├── results/           CSV results, feature matrices, metrics
│       ├── figures/           Visualizations: PNG charts, HTML reports
│       ├── logs/              Training/inference logs
│       └── features/          Feature engineering artifacts
├── 📂 cleaned_dataset/        Raw battery test data
│   ├── data/                  CSV files per battery (00001.csv - 00137.csv)
│   ├── extra_infos/           Supplementary metadata
│   └── metadata.csv           Battery inventory
├── 📂 docs/                   Documentation markdown files
├── 📂 frontend/               React SPA (TypeScript, Vite)
├── 📂 notebooks/              Jupyter analysis & training (01-09)
├── 📂 reference/              External papers & reference notebooks
├── 📂 scripts/                Organized utility scripts
│   ├── data/                  Data processing (write_nb03_v2, patch_dl_notebooks_v2)
│   ├── models/                Model training (retrain_classical)
│   ├── __init__.py            Package marker
│   └── README (in data/models/)
├── 📂 src/                    Core Python library
│   ├── data/                  Data loading & preprocessing
│   ├── evaluation/            Metrics & validation
│   ├── models/                Model architectures
│   ├── utils/                 Config, logging, helpers
│   └── __init__.py
├── 📂 tests/                  ✓ NEW: Test & validation scripts
│   ├── test_v2_models.py      Comprehensive v2 validation
│   ├── test_predictions.py    Quick endpoint test
│   ├── __init__.py
│   └── README.md
├── 📄 CHANGELOG.md            Version history & updates  
├── 📄 VERSION.md              ✓ NEW: Versioning & versioning guide
├── 📄 README.md               Project overview
├── 📄 requirements.txt        Python dependencies
├── 📄 package.json            Node.js dependencies (frontend)
├── 📄 Dockerfile              Docker configuration
├── 📄 docker-compose.yml      Multi-container orchestration
├── 📄 tsconfig.json           TypeScript config (frontend)
└── 📄 vite.config.ts          Vite bundler config (frontend)
```

## Key Changes in V2 Reorganization

### ✓ Completed
1. **Versioned Artifacts**
   - Moved `artifacts/models/` → `artifacts/v2/models/`
   - Moved `artifacts/scalers/` → `artifacts/v2/scalers/`
   - Moved `artifacts/figures/` → `artifacts/v2/figures/`
   - All result CSVs → `artifacts/v2/results/`
   - Clean `artifacts/` root (only v1 and v2 subdirs)

2. **Organized Scripts**
   - Created `scripts/data/` for data processing utilities
   - Created `scripts/models/` for model training scripts
   - All scripts now using `get_version_paths('v2')`
   - Path: `scripts/retrain_classical.py` → `scripts/models/retrain_classical.py`

3. **Centralized Tests**
   - Created `tests/` folder at project root
   - Moved `test_v2_models.py` → `tests/`
   - Moved `test_predictions.py` → `tests/`
   - Added `tests/README.md` with usage guide
   - All tests now using `artifacts/v2/` paths

4. **Updated Imports & Paths**
   - `test_v2_models.py`: Uses `v2['results']` for data loading
   - `retrain_classical.py`: Uses `get_version_paths('v2')` for artifact saving
   - API: Already defaults to `registry_v2`
   - Notebooks NB03-09: Already use `get_version_paths()`

### Code Changes Summary

| File | Change | Result |
|------|--------|--------|
| `tests/test_v2_models.py` | Updated artifact paths to use v2 | Output → `artifacts/v2/{results,figures}` |
| `scripts/models/retrain_classical.py` | Uses `get_version_paths('v2')` | Models saved to `artifacts/v2/models/classical/` |
| `api/model_registry.py` | Already has versioning support | No changes needed |
| `src/utils/config.py` | Already supports versioning | No changes needed |
| Notebooks NB03-09 | Already use `get_version_paths()` | No changes needed |

## Running Tests After Reorganization

```bash
# From project root
python tests/test_v2_models.py      # Full v2 validation
python tests/test_predictions.py    # Quick endpoint test
python scripts/models/retrain_classical.py  # Retrain models
```

## Artifact Access in Code

### Before (V1 - hardcoded paths)
```python
model_path = "artifacts/models/classical/rf.joblib"
results_csv = "artifacts/results.csv"
```

### After (V2 - versioned paths via config)
```python
from src.utils.config import get_version_paths
v2 = get_version_paths('v2')

model_path = v2['models_classical'] / 'rf.joblib'
results_csv = v2['results'] / 'results.csv'
```

## Production Readiness

| Aspect | Status | Notes |
|--------|--------|-------|
| Versioning | ✓ Complete | All artifacts under `v2/` |
| Structure | ✓ Organized | Scripts, tests, notebooks organized |
| Configuration | ✓ Active | `ACTIVE_VERSION = 'v2'` in config |
| API | ✓ Ready | Defaults to `registry_v2` |
| Tests | ✓ Available | `tests/test_v2_models.py` for validation |
| Documentation | ✓ Added | VERSION.md and README files created |

## Forward Compatibility

### For Future Versions (v3, v4, etc.)

Simply copy the v2 folder structure and update:
```python
# In src/utils/config.py
ACTIVE_VERSION: str = "v3"

# In scripts
v3 = get_version_paths('v3')
ensure_version_dirs('v3')
```

The system will automatically create versioned paths and maintain backward compatibility.

