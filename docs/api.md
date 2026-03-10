# API Documentation

## Base URL

- **Local:** `http://localhost:7860`
- **Docker:** `http://localhost:7860`
- **Hugging Face Spaces:** `https://neerajcodz-aibatterylifecycle.hf.space`

## Interactive Docs

- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **Gradio UI:** `/gradio`

## API Versioning (v3.0.0)

The API supports three model generations served in parallel, with dynamic on-demand loading:

| Prefix | Models | Features | Default Model | Status |
|--------|--------|:--------:|---------------|--------|
| `/api/v1/*` | v1 classical (12 models) | 12 | Random Forest | On-demand |
| `/api/v2/*` | v2 classical + deep (20+ models) | 12 | ExtraTrees | On-demand |
| `/api/v3/*` | v3 optimized + ensemble (18 models) | 18 | XGBoost | **Loaded at startup** |
| `/api/*` | Default (v3) | 18 | XGBoost | Same as v3 |

### Startup Behavior
- Only **v3** models are loaded at startup to minimize memory and boot time.
- v1 and v2 models can be loaded on-demand via `POST /api/versions/{v}/load`.
- Each version's metadata is read from `artifacts/{version}/models.json`.

### Version Management Endpoints

```http
GET /api/versions
```
Returns all versions with `on_disk`, `loaded`, `model_count`, `catalog_count`, `description`, `features`, `champion`, and `status`.

```http
POST /api/versions/{version}/load
```
Downloads (if needed) and loads a version's models into memory. Returns immediately if already on disk.

```http
GET /api/versions/{version}/models-meta
```
Returns the full `models.json` metadata for a version.

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
