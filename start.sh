#!/bin/bash
set -e

echo "=============================================================================="
echo "  TERRASIGHT // FOVEA-LIDAR"
echo "  DRDO SIH26053 - Smart India Hackathon 2026"
echo "=============================================================================="
echo ""

# 1. Build frontend if missing
if [ ! -f "frontend/dist/index.html" ]; then
    echo "[INFO] Frontend build not found. Building production bundle..."
    cd frontend
    npm install
    npm run build
    cd ..
fi

echo "[INFO] Starting TerraSight unified production server on http://localhost:8000 ..."
echo "[INFO] API Documentation: http://localhost:8000/docs"
echo ""

python run_production.py --host 0.0.0.0 --port 8000 --reload
