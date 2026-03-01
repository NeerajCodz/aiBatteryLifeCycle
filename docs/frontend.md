# Frontend Documentation

## Technology Stack

| Technology | Version | Purpose |
|-----------|---------|----------|
| Vite | 7.x | Build tool & dev server |
| React | 19.x | UI framework |
| TypeScript | 5.9.x | Type safety |
| Recharts | 3.7.x | Interactive 2D charts (BarChart, LineChart, AreaChart, RadarChart, ScatterChart, PieChart) |
| lucide-react | 0.575.x | Icon system — **no emojis in UI** |
| TailwindCSS | 4.x | Utility-first CSS |
| Axios | 1.x | HTTP client |

## Project Structure

```
frontend/
├── index.html
├── vite.config.ts          # Vite + /api proxy
├── tsconfig.json
├── package.json
└── src/
    ├── main.tsx
    ├── App.tsx             # Root, tab navigation, v1/v2 selector
    ├── api.ts              # All API calls + TypeScript types
    ├── index.css
    └── components/
        ├── Dashboard.tsx           # Fleet overview heatmap + capacity charts
        ├── PredictionForm.tsx      # Single-cycle SOH prediction + gauge
        ├── SimulationPanel.tsx     # Multi-battery lifecycle simulation
        ├── MetricsPanel.tsx        # Full model metrics dashboard
        ├── GraphPanel.tsx          # Analytics — fleet, single battery, compare, temperature
        └── RecommendationPanel.tsx # Operating condition optimizer with charts
```

## Tab Order

| Tab | Component | Description |
|-----|-----------|-------------|
| Simulation | `SimulationPanel` | ML-backed multi-battery lifecycle forecasting |
| Predict | `PredictionForm` | Single-cycle SOH + RUL prediction |
| Metrics | `MetricsPanel` | Full model evaluation dashboard |
| Analytics | `GraphPanel` | Fleet & per-battery interactive analytics |
| Recommendations | `RecommendationPanel` | Operating condition optimizer |
| Research Paper | — | Embedded research PDF |

## Components

### MetricsPanel
Full interactive model evaluation dashboard with 6 switchable sections:

- **Overview** — KPI stat cards with lucide icons, R² ranking bar chart, model family pie chart, normalized radar chart (top 5), R² vs MAE scatter trade-off plot, Top-3 rankings podium
- **Models** — Interactive sort (R²/MAE/RMSE/MAPE, asc/desc), family filter dropdown, chart-type toggle (bar/radar/scatter), multi-select compare mode, colour-coded metric badges, full metrics table with per-row highlighting
- **Validation** — Within-5% / within-10% grouped bar chart, full validation table with pass/fail badges
- **Deep Learning** — LSTM/ensemble/VAE-LSTM/DG-iTransformer results with charts and metric tables
- **Dataset** — Battery stats cards, engineered features list, temperature groups, degradation distribution bar chart, SOH range gauge
- **Figures** — Searchable grid of all artifact figures with modal lightbox on click

**Key features:**
- All icons via `lucide-react` (no emojis)
- `filteredModels` useMemo respects active sort/filter state
- `MetricBadge` component colour-codes values green/yellow/red based on model quality thresholds
- `SectionBadge` nav bar with icon + label

### GraphPanel (Analytics)
Four-section analytics dashboard:

- **Fleet Overview** — SOH bar chart sorted by health (colour-coded green/yellow/red), SOH vs cycles bubble scatter (bubble = temperature), fleet status KPI cards (healthy/degraded/near-EOL), filter controls (min SOH slider, temp range), clickable battery roster table
- **Single Battery** — SOH trajectory + linear RUL projection overlay, capacity fade area chart, smoothed degradation rate area chart, show/hide EOL reference line toggle
- **Compare** — Multi-select up to 5 batteries; SOH overlay line chart with distinct colours per battery, capacity fade overlay, summary comparison table (final SOH, cycles, min capacity)
- **Temperature Analysis** — Temperature vs SOH scatter, temperature distribution histogram

**Key features:**
- Multi-battery data loaded in parallel using `Promise.all`
- RUL projection via least-squares on last 20 cycles → extrapolated to 70% SOH
- `SohBadge` component with dynamic colour + icon

### RecommendationPanel
Interactive optimizer replacing the previous plain form + table:

- **Input form** — Text input for battery ID, range sliders for SOH and ambient temperature, numeric inputs for cycle and top-k
- **Summary cards** — Battery ID, best predicted RUL, best improvement, config count
- **Visual Analysis tabs:**
  - *RUL Comparison* — bar chart comparing predicted RUL across all recommendations
  - *Parameters* — grouped bar chart showing temp/current/cutoff per rank
  - *Radar* — normalized multi-metric radar chart for top-3 configs
- **Recommendations table** — rank icons (Trophy/Award/Medal from lucide), colour-coded improvement badges, expandable rows showing per-recommendation explanation and parameter details

### SimulationPanel
- Configure up to N battery simulations with individual parameters (temp, voltage, current, EOL threshold)
- Select ML model for lifecycle prediction (or pure physics fallback)
- Animated SOH trajectory charts, final stats table, degradation state timeline

### Dashboard
- Fleet battery grid with SOH colour coding
- Capacity fade line chart per selected battery
- Model metrics bar chart

### PredictionForm
- 12-input form with all engineered cycle features
- SOH gauge visualization (SVG ring) with degradation state colour
- Confidence interval display
- Model selector (v2 models / best_ensemble)

## API Integration (`api.ts`)

All API calls return typed TypeScript response objects:

| Function | Endpoint | Description |
|----------|----------|--------------|
| `fetchDashboard()` | `GET /api/dashboard` | Fleet overview + capacity fade data |
| `fetchBatteries()` | `GET /api/batteries` | All battery metadata |
| `fetchBatteryCapacity(id)` | `GET /api/battery/{id}/capacity` | Per-battery cycles, capacity, SOH arrays |
| `predictSoh(req)` | `POST /api/v2/predict` | Single-cycle SOH + RUL prediction |
| `fetchRecommendations(req)` | `POST /api/v2/recommend` | Operating condition optimization |
| `simulateBatteries(req)` | `POST /api/v2/simulate` | Multi-battery lifecycle simulation |
| `fetchMetrics()` | `GET /api/metrics` | Full model evaluation metrics |
| `fetchModels()` | `GET /api/v2/models` | All loaded models with metadata |

## Development

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

API requests proxy to `http://localhost:7860` in dev mode (see `vite.config.ts`).

## Build

```bash
npm run build      # outputs to dist/
```

The built `dist/` folder is served as static files by FastAPI at the root path.
| Vite | 6.x | Build tool & dev server |
| React | 18.x | UI framework |
| TypeScript | 5.x | Type safety |
| Three.js | latest | 3D rendering |
| @react-three/fiber | latest | React renderer for Three.js |
| @react-three/drei | latest | Three.js helpers |
| Recharts | latest | 2D charts |
| TailwindCSS | 4.x | Utility-first CSS |
| Axios | latest | HTTP client |

## Project Structure

```
frontend/
├── index.html              # Entry HTML
├── vite.config.ts          # Vite + proxy config
├── tsconfig.json           # TypeScript config
├── package.json
├── public/
│   └── vite.svg            # Favicon
└── src/
    ├── main.tsx            # React entry point
    ├── App.tsx             # Root component + tab navigation
    ├── api.ts              # API client + types
    ├── index.css           # TailwindCSS import
    └── components/
        ├── Dashboard.tsx       # Fleet overview + charts
        ├── PredictionForm.tsx  # SOH prediction form + gauge
        ├── BatteryViz3D.tsx    # 3D battery pack heatmap
        ├── GraphPanel.tsx      # Analytics / per-battery graphs
        └── RecommendationPanel.tsx  # Operating condition optimizer
```

## Components

### Dashboard
- Stat cards (battery count, models, best R²)
- Battery fleet grid with SOH color coding
- SOH capacity fade line chart (Recharts)
- Model R² comparison bar chart

### PredictionForm
- 12-input form with all cycle features
- SOH gauge visualization (SVG ring)
- Result display with degradation state coloring
- Confidence interval display

### BatteryViz3D
- 3D battery pack with cylindrical cells
- SOH-based fill level and color mapping
- Click-to-inspect with side panel details
- Auto-rotation with orbit controls
- Health legend and battery list sidebar

### GraphPanel
- Battery selector dropdown
- Per-battery SOH trajectory (line chart)
- Per-battery capacity fade curve
- Fleet scatter plot (SOH vs cycles, bubble size = temperature)

### RecommendationPanel
- Input: battery ID, current cycle, SOH, temperature, top_k
- Table of ranked recommendations
- Shows temperature, current, cutoff voltage, predicted RUL, improvement %

## Development

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

API requests are proxied to `http://localhost:7860` during development.

## Production Build

```bash
npm run build      # Outputs to dist/
```

The built files are served by FastAPI as static assets.
