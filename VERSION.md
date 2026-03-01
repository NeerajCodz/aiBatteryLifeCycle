# Project Versioning & Structure

## Current Active Version: v2.0

All active models, artifacts, and features use the **v2.0** versioning scheme.

### Directory Structure

```
artifacts/
├── v1/                     ← Legacy models (cross-battery split)
│   ├── models/
│   │   ├── classical/
│   │   ├── deep/
│   │   └── ensemble/
│   ├── scalers/
│   ├── figures/
│   ├── results/
│   ├── logs/
│   └── features/
│       
├── v2/                     ← Current production (intra-battery split) ✓
│   ├── models/
│   │   ├── classical/      ← 14 classical ML models
│   │   ├── deep/           ← 8 deep learning models  
│   │   └── ensemble/       ← Weighted ensemble
│   ├── scalers/            ← Feature scalers for linear models
│   ├── figures/            ← All validation visualizations (PNG, HTML)
│   ├── results/            ← CSV/JSON results and feature matrices
│   ├── logs/               ← Training logs
│   └── features/           ← Feature engineering artifacts
```

### V2 Key Changes from V1

| Aspect | V1 | V2 |
|--------|----|----|
| **Data Split** | Cross-battery (groups of batteries) | Intra-battery chronological (first 80% cycles per battery) |
| **Train/Test Contamination** | ⚠️ YES (same batteries in both) | ✓ NO (different time periods per battery) |
| **Generalization** | Poor (batteries see same time periods) | Better (true temporal split) |
| **Test Realism** | Interpolation (within-cycle prediction) | Extrapolation (future cycles) |
| **Classical Models** | 6 standard models | 14 models (added ExtraTrees, GradientBoosting, KNN ×3) |
| **Deep Models** | 8 models | Retraining in progress |
| **Ensemble** | RF + XGB + LGB (v1 trained) | RF + XGB + LGB (v2 trained when available) |

### Model Statistics

#### Classical Models (V2)
- **Total:** 14 models
- **Target Metric:** Within-±5% SOH accuracy ≥ 95%
- **Current Pass Rate:** See `artifacts/v2/results/v2_validation_report.html`

#### Configuration

**Active version is set in** `src/utils/config.py`:
```python
ACTIVE_VERSION: str = "v2"
```

**API defaults to v2:**
```python
registry = registry_v2  # Default registry (v2.0.0 models)
```

### Migration Checklist ✓

- ✓ Created versioned artifact directories under `artifacts/v2/`
- ✓ Moved all v2 models to `artifacts/v2/models/classical/` etc.
- ✓ Moved all results to `artifacts/v2/results/`
- ✓ Moved all figures to `artifacts/v2/figures/`
- ✓ Moved all scalers to `artifacts/v2/scalers/`
- ✓ Updated notebooks (NB03-09) to use `get_version_paths('v2')`
- ✓ Updated API to default to v2 registry
- ✓ Organized scripts into `scripts/data/`, `scripts/models/`
- ✓ Moved tests to `tests/` folder
- ✓ Cleaned up legacy artifact directories

### File Locations

| Content | Path |
|---------|------|
| Models (classical) | `artifacts/v2/models/classical/*.joblib` |
| Models (deep) | `artifacts/v2/models/deep/*.pth` |
| Models (ensemble) | `artifacts/v2/models/ensemble/*.joblib` |
| Scalers | `artifacts/v2/scalers/*.joblib` |
| Results CSV | `artifacts/v2/results/*.csv` |
| Feature matrix | `artifacts/v2/results/battery_features.csv` |
| Visualizations | `artifacts/v2/figures/*.{png,html}` |
| Logs | `artifacts/v2/logs/*.log` |

### Running Scripts

```bash
# Run v2 model validation test
python tests/test_v2_models.py

# Run quick prediction test  
python tests/test_predictions.py

# Retrain classical models (WARNING: takes ~30 min)
python scripts/models/retrain_classical.py

# Generate/patch notebooks (one-time utilities)
python scripts/data/write_nb03_v2.py
python scripts/data/patch_dl_notebooks_v2.py
```

### Next Steps

1. ✓ Verify v2 model accuracy meets thresholds
2. ✓ Update research paper with v2 results
3. ✓ Complete research notes for all notebooks
4. ✓ Test cycle recommendation engine
5. Deploy v2 to production

### Version History

| Version | Date | Status | Notes |
|---------|------|--------|-------|
| v1.0 | 2025-Q1 | ✓ Complete | Classical + Deep models, cross-battery split |
| v2.0 | 2026-02-25 | ✓ Active | Intra-battery split, improved generalization |
| v3.0 | TBD | -- | Physics-informed models, uncertainty quantification |

