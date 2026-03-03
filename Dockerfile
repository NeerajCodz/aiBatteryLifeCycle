# ─────────────────────────────────────────────────────────────
# Stage 1: Build React frontend
# ─────────────────────────────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ─────────────────────────────────────────────────────────────
# Stage 2: Python runtime
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_LEVEL=INFO

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Install torch CPU-only and tensorflow-cpu FIRST so requirements.txt
# finds them already satisfied (avoids downloading 3+ GB of CUDA deps)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install tensorflow-cpu && \
    pip install -r requirements.txt

# Ensure writable artifact directories exist
RUN mkdir -p artifacts/v1/models/classical artifacts/v1/models/deep \
             artifacts/v1/scalers \
             artifacts/v2/models/classical artifacts/v2/models/deep \
             artifacts/v2/scalers artifacts/v2/results artifacts/v2/reports \
             artifacts/logs

# Copy project source (artifacts/ is NOT in git — downloaded at runtime)
COPY src/ src/
COPY api/ api/
COPY scripts/ scripts/
COPY cleaned_dataset/ cleaned_dataset/

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist frontend/dist

# Expose port (Hugging Face Spaces expects 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Entrypoint: download models from HF Hub if absent, then start the server
CMD ["sh", "-c", "python scripts/download_models.py && uvicorn api.main:app --host 0.0.0.0 --port 7860 --workers 1"]
