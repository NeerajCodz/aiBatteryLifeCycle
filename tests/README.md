# Tests Directory

Automated test and validation scripts for the AI Battery Lifecycle Predictor.

## Test Scripts

### 1. `test_v2_models.py` — Comprehensive V2 Validation
Validates all 14 v2 classical models against test set with detailed reporting.

**Usage:**
```bash
python tests/test_v2_models.py
```

**Output:**
- `artifacts/v2/results/v2_model_validation.csv` — Full metrics table
- `artifacts/v2/results/v2_validation_summary.json` — Summary statistics
- `artifacts/v2/figures/validation_accuracy_bars.png` — Accuracy ranking chart
- `artifacts/v2/figures/r2_vs_accuracy.png` — R² vs accuracy scatter plot
- `artifacts/v2/figures/best_model_analysis.png` — Best model performance
- `artifacts/v2/figures/per_battery_accuracy.png` — Per-battery accuracy heatmap
- `artifacts/v2/results/v2_validation_report.html` — HTML report
- `artifacts/v2/results/v2_validation_report.md` — Markdown report

**Target Metrics:**
- Within-±5% SOH Accuracy ≥ 95% (primary success metric)
- Within-±2% SOH Accuracy (secondary)
- R² ≥ 0.95 (correlation)
- MAE ≤ 3% (mean absolute error)

### 2. `test_predictions.py` — Quick Endpoint Validation
Quick validation of prediction endpoint with sample features on model registry.

**Usage:**
```bash
python tests/test_predictions.py
```

**Output:** Console output showing predictions for 4 test scenarios:
- Early life cycle (SOH ≈ 99%)
- Healthy cycle (SOH ≈ 97%)
- Degraded cycle (SOH ≈ 80%)
- End-of-life cycle (SOH ≈ 40%)

## Running Tests

### Run all tests
```bash
python tests/test_v2_models.py
python tests/test_predictions.py
```

### Run individual test
```bash
python tests/test_v2_models.py    # V2 model validation
python tests/test_predictions.py  # Endpoint test
```

## Test Data

Tests use:
- **Features:** `artifacts/v2/results/battery_features.csv` (2,678 samples)
- **Target:** SOH (State of Health) percentage
- **Split:** Intra-battery chronological (first 80% cycles → train, last 20% → test)
- **Batteries:** All 30 usable batteries in both train and test

## Model Versions Tested

| Model | Category | V2 Status |
|-------|----------|-----------|
| ExtraTrees | Tree | ✓ v2.0.0 |
| GradientBoosting | Tree | ✓ v2.0.0 |  
| RandomForest | Tree | v1.0.0 |
| XGBoost | Tree | v1.0.0 |
| LightGBM | Tree | v1.0.0 |
| SVR | Linear | v1.0.0 |
| Ridge | Linear | v1.0.0 |
| Lasso | Linear | v1.0.0 |
| ElasticNet | Linear | v1.0.0 |
| KNN (k=5, 10, 20) | Linear | v1.0.0 |

## Success Criteria

✓ Model passes if: **Within-±5% Accuracy ≥ 95%**

Current pass rate: See latest report in `artifacts/v2/results/v2_validation_report.html`
