# API Documentation

## Base URL

- **Local:** `http://localhost:7860`
- **Docker:** `http://localhost:7860`
- **Hugging Face Spaces:** `https://neerajcodz-aibatterylifecycle.hf.space`

## Interactive Docs

- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **Gradio UI:** `/gradio`

## API Versioning (v2.1.0)

The API supports two model generations served in parallel:

| Prefix | Models | Split Strategy | Notes |
|--------|--------|---------------|-------|
| `/api/v1/*` | v1 models (cross-battery split) | Group-battery 80/20 | Legacy |
| `/api/v2/*` | v2 models (chrono split, bug-fixed) | Intra-battery 80/20 | **Recommended** |
| `/api/*` | Default (v2) | Same as v2 | Backward-compatible |

### v2 Bug Fixes
- **avg_temp auto-correction removed** — v1 silently added 8°C to avg_temp
- **Recommendation baseline fixed** — v1 re-predicted SOH, yielding ~0 improvement

---

## Endpoints

### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "ok",
  "version": "2.0.0",
  "models_loaded": 12,
  "device": "cpu"
}
```

---

### Single Prediction

```http
POST /api/predict
Content-Type: application/json
```

Request:
```json
{
  "battery_id": "B0005",
  "cycle_number": 100,
  "ambient_temperature": 24.0,
  "peak_voltage": 4.2,
  "min_voltage": 2.7,
  "avg_current": 2.0,
  "avg_temp": 25.0,
  "temp_rise": 3.0,
  "cycle_duration": 3600,
  "Re": 0.04,
  "Rct": 0.02,
  "delta_capacity": -0.005
}
```

Optionally include `"model_name"` to select a specific model (leave null to use the registry default):

```json
{
  ...
  "model_name": "random_forest"
}
```

Response:
```json
{
  "battery_id": "B0005",
  "cycle_number": 100,
  "soh_pct": 92.5,
  "rul_cycles": 450,
  "degradation_state": "Healthy",
  "confidence_lower": 90.5,
  "confidence_upper": 94.5,
  "model_used": "random_forest",
  "model_version": "v1.0.0"
}
```

---

### Ensemble Prediction

```http
POST /api/predict/ensemble
Content-Type: application/json
```

Always uses the **BestEnsemble (v3.0.0)** — weighted average of Random Forest, XGBoost, and
LightGBM (weights proportional to R²). Body is identical to single prediction.

Response includes `"model_version": "v3.0.0"`.

---

### Batch Prediction

```http
POST /api/predict/batch
Content-Type: application/json
```

Request:
```json
{
  "battery_id": "B0005",
  "cycles": [
    {"cycle_number": 1, "ambient_temperature": 24, ...},
    {"cycle_number": 2, "ambient_temperature": 24, ...}
  ]
}
```

---

### Recommendations

```http
POST /api/recommend
Content-Type: application/json
```

Request:
```json
{
  "battery_id": "B0005",
  "current_cycle": 100,
  "current_soh": 85.0,
  "ambient_temperature": 24.0,
  "top_k": 5
}
```

Response:
```json
{
  "battery_id": "B0005",
  "current_soh": 85.0,
  "recommendations": [
    {
      "rank": 1,
      "ambient_temperature": 24.0,
      "discharge_current": 0.5,
      "cutoff_voltage": 2.7,
      "predicted_rul": 500,
      "rul_improvement": 50,
      "rul_improvement_pct": 11.1,
      "explanation": "Operate at 24°C, 0.5A, cutoff 2.7V for ~500 cycles RUL"
    }
  ]
}
```

---

### Dashboard Data

```http
GET /api/dashboard
```

Returns full dashboard payload with battery fleet stats, capacity fade curves, and model metrics.

---

### Battery List

```http
GET /api/batteries
```

---

### Battery Capacity

```http
GET /api/battery/{battery_id}/capacity
```

---

### Model List

```http
GET /api/models
```

Returns every registered model with version, family, R², and load status.

---

### Model Versions

```http
GET /api/models/versions
```

Groups models by generation:

```json
{
  "v1_classical":  ["ridge", "lasso", "random_forest", "xgboost", "lightgbm", ...],
  "v2_deep":       ["vanilla_lstm", "bilstm", "gru", "attention_lstm", "tft", ...],
  "v2_ensemble":   ["best_ensemble"],
  "other":         [],
  "default_model": "best_ensemble"
}
```

---

### Figures

```http
GET /api/figures          # List all
GET /api/figures/{name}   # Serve a figure
```
