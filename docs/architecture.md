# Architecture Overview

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Docker Container (port 7860)                   │
├──────────────┬───────────────┬───────────────────────────────────┤
│  React SPA   │  Gradio UI    │  FastAPI Backend                  │
│  (static)    │  /gradio      │  /api/*     /docs     /health     │
│  /           │               │                                    │
├──────────────┴───────────────┴───────────────────────────────────┤
│                      Model Registry                               │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │Classical │  │ LSTM×4   │  │Transform.│  │ Ensemble │          │
│  │ models   │  │ GRU      │  │ GPT, TFT │  │ Stack/WA │          │
│  └─────────┘  └──────────┘  └──────────┘  └──────────┘          │
├──────────────────────────────────────────────────────────────────┤
│                  Data Pipeline (src/)                              │
│  loader.py → features.py → preprocessing.py → model training     │
├──────────────────────────────────────────────────────────────────┤
│                 NASA PCoE Dataset (cleaned_dataset/)              │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Ingestion:** `loader.py` reads metadata.csv + per-cycle CSVs
2. **Feature Engineering:** `features.py` computes SOC, SOH, RUL, scalar features per cycle
3. **Preprocessing:** `preprocessing.py` creates sliding windows, scales features, splits by battery
4. **Training:** Notebooks train each model family, save checkpoints to `artifacts/models/`
5. **Serving:** `model_registry.py` loads all models at startup
6. **Prediction:** API receives features → registry dispatches to best model → returns SOH/RUL
7. **Simulation:** `POST /api/v2/simulate` receives multi-battery config → vectorized Arrhenius degradation + ML via `predict_array()` → returns per-step SOH, RUL, and degradation-state history for each battery
8. **Visualization:** Frontend fetches results and renders analytics (fleet overview, compare, temperature analysis, recommendations)

## Model Registry

The `ModelRegistry` singleton:
- Scans `artifacts/models/classical/` for `.joblib` files (sklearn/xgb/lgbm)
- Scans `artifacts/models/deep/` for `.pt` (PyTorch) and `.keras` (TF) files
- Loads classical models eagerly; deep models registered lazily
- Selects default model by priority: XGBoost > LightGBM > RandomForest > Ridge > deep models
- Provides unified `predict()` interface regardless of framework
- `predict_array(X: np.ndarray, model_name: str)` batch method enables vectorized simulation: accepts an (N, n_features) array and returns predictions for all N cycles in one call, avoiding Python loops
- `_x_for_model()` normalizes input feature extraction for both single-cycle and batch paths
- `_load_scaler()` lazily loads per-model scalers from `artifacts/scalers/`

## Frontend Architecture

- **Vite 7** build tool with React 19 + TypeScript 5.9
- **lucide-react 0.575** for all icons — no emojis used anywhere in the UI
- **Recharts 3** for all 2D charts (BarChart, AreaChart, LineChart, ScatterChart, RadarChart, PieChart)
- **TailwindCSS 4** for styling
- Tabs: Simulation | Predict | Metrics | Analytics | Recommendations | Research Paper
- API proxy in dev mode (`/api` → `localhost:7860`) → same-origin in production (served by FastAPI)
- **Analytics (GraphPanel):** 4-section dashboard — Fleet Overview (health kpi, fleet SOH bar, bubble scatter), Single Battery (SOH + RUL projection, capacity fade, degradation rate), Compare (multi-battery overlay), Temperature Analysis
- **Metrics (MetricsPanel):** 6-section interactive dashboard — Overview KPIs, Models (sort/filter/chart-type controls), Validation, Deep Learning, Dataset stats, Figures searchable gallery
- **Recommendations (RecommendationPanel):** Slider inputs for SOH/temp, 3 chart tabs (RUL bar, params bar, top-3 radar), expandable table rows with per-recommendation explanation
