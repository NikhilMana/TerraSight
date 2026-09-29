"""
TerraSight // Fovea-LiDAR Production Server Entrypoint.
Runs the unified FastAPI application serving both the REST perception API
and the built React + Three.js tactical frontend.
"""

import sys
import os
import argparse
from pathlib import Path
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    default_port = int(os.environ.get("PORT", 8000))
    default_host = os.environ.get("HOST", "0.0.0.0")

    parser = argparse.ArgumentParser(description="TerraSight // Fovea-LiDAR Production Server")
    parser.add_argument("--host", default=default_host, help=f"Host interface to bind (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to listen on (default: {default_port})")
    parser.add_argument("--workers", type=int, default=1, help="Worker count (default: 1)")
    parser.add_argument("--reload", action="store_true", default=False, help="Enable hot reload for development")

    args = parser.parse_args()

    frontendDist = PROJECT_ROOT / "frontend" / "dist"
    print("=" * 80)
    print("  TERRASIGHT // FOVEA-LIDAR PRODUCTION PERCEPTION SERVER")
    print("  DRDO SIH26053 - Smart India Hackathon 2026")
    print("=" * 80)
    print(f"  -> Host:            http://{args.host}:{args.port}")
    print(f"  -> API Docs:        http://{args.host}:{args.port}/docs")
    if frontendDist.is_dir():
        print(f"  -> Frontend:        Mounted from {frontendDist}")
    else:
        print(f"  -> [WARNING] Frontend dist not found at {frontendDist}. Run 'npm run build' inside frontend/")
    print("=" * 80)

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
