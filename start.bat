@echo off
echo ==============================================================================
echo   TERRASIGHT // FOVEA-LIDAR
echo   DRDO SIH26053 - Smart India Hackathon 2026
echo ==============================================================================
echo.

REM 1. Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

REM 2. Check if frontend build exists, build if missing
if not exist "frontend\dist\index.html" (
    echo [INFO] Frontend build not found. Building production bundle...
    cd frontend
    call npm install
    call npm run build
    cd ..
)

echo [INFO] Starting TerraSight unified production server on http://localhost:8000 ...
echo [INFO] API Documentation: http://localhost:8000/docs
echo.

python run_production.py --host 0.0.0.0 --port 8000 --reload
pause
