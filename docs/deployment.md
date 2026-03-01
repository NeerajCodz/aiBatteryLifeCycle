# Deployment Guide

## Local Development

### Backend
```bash
cd aiBatteryLifecycle
.\venv\Scripts\activate          # Windows
source venv/bin/activate          # Linux/Mac

uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload
```

### Frontend (dev mode)
```bash
cd frontend
npm install
npm run dev
```

Frontend proxies `/api/*` to `localhost:7860` in dev mode.

### Frontend (production build)
```bash
cd frontend
npm run build
```

Built files go to `frontend/dist/` and are served by FastAPI.

---

## Docker

### Build
```bash
docker build -t battery-predictor .
```

### Run
```bash
docker run -p 7860:7860 battery-predictor
```

### Build stages
1. **frontend-build:** `node:20-slim` — installs npm deps and builds React SPA
2. **runtime:** `python:3.11-slim` — installs Python deps, copies source and built frontend

### Docker Compose (recommended)

```bash
# Production — single container (frontend + API)
docker compose up --build

# Development — backend only with hot-reload  
docker compose --profile dev up api-dev
# then separately:
cd frontend && npm run dev
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG` / `INFO` / `WARNING` / `ERROR`) |
| `WORKERS` | `1` | Uvicorn worker count |

---

## Hugging Face Spaces

### Setup
1. Create a new Space on Hugging Face (SDK: Docker)
2. Push the repository to the Space
3. The Dockerfile exposes port 7860 (HF Spaces default)

### Dockerfile Requirements
- Must expose port **7860**
- Must respond to health checks at `/health`
- Keep image size manageable (use CPU-only PyTorch/TF)

### Files to include
```
Dockerfile
requirements.txt
api/
src/
frontend/               # Vite builds during Docker image creation
cleaned_dataset/
artifacts/v1/            # v1 model checkpoints (legacy)
artifacts/v2/            # v2 model checkpoints (recommended)
artifacts/models/        # Root-level models (backward compat)
```

### HuggingFace Space URL
```
https://huggingface.co/spaces/NeerajCodz/aiBatteryLifeCycle
```

### Space configuration (README.md header)
```yaml
---
title: AI Battery Lifecycle Predictor
emoji: 🔋
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---
```

---

## Production Considerations

### Performance
- Use `--workers N` for multi-core deployment
- Enable GPU passthrough for deep model inference: `docker run --gpus all`
- Consider preloading all models (not lazy loading)

### Security
- Set `CORS_ORIGINS` to specific domains in production
- Add authentication middleware if needed
- Use HTTPS reverse proxy (nginx, Caddy)

### Monitoring
- Health endpoint: `/health`
- Logs: JSON-per-line rotating log at `artifacts/logs/battery_lifecycle.log` (10 MB × 5 backups)
  — set `LOG_LEVEL=DEBUG` for verbose output, mount volume to persist across container restarts
- Metrics: Add Prometheus endpoint if needed
