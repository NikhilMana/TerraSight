# ==============================================================================
# TerraSight // Fovea-LiDAR Production Multi-Stage Dockerfile
# Problem Statement: SIH26053 - DRDO (Smart India Hackathon 2026)
# ==============================================================================

# ----------------- Stage 1: Build React + Vite Frontend -----------------
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ----------------- Stage 2: Python Production Runtime -------------------
FROM python:3.11-slim AS runtime

# System libraries for numerical computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and model weights
COPY core/ ./core/
COPY segmentation/ ./segmentation/
COPY data/ ./data/
COPY config/ ./config/
COPY backend/ ./backend/
COPY visualization/ ./visualization/
COPY benchmarks/ ./benchmarks/
COPY experiments/ ./experiments/
COPY run_production.py ./

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose server port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

ENTRYPOINT ["python", "run_production.py", "--host", "0.0.0.0", "--port", "8000"]
