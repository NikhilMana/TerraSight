"""
Verification & Benchmarking Script for Phase 2 & Phase 3.
Evaluates the baseline Uniform 2.5D Elevation Grid across multiple resolutions (5cm, 10cm, 25cm).
Measures:
  - Exact memory footprint (MB)
  - Processing latency (ms) and FPS
  - Cell counts and spatial sparsity
  - Generates publication-ready 4-panel elevation and semantic grid visualization
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.point_cloud import PointCloud
from data.kitti_loader import loadSemanticKittiScan
from core.grid_uniform import UniformElevationGrid
from segmentation.kitti_classes import TargetClass


def runPhase2And3Verification() -> bool:
    print("=" * 65)
    print("  PHASE 2 & 3: 2.5D Elevation Mapping & Uniform Baseline Benchmarking")
    print("=" * 65)

    # 1. Load sample scan
    sampleBin = PROJECT_ROOT / "data" / "sample_scans" / "sample_000000.bin"
    sampleLabel = PROJECT_ROOT / "data" / "sample_scans" / "sample_000000.label"
    if not sampleBin.exists():
        print(f"Error: Sample scan not found at {sampleBin}. Run verify_phase1.py first.")
        return False

    print(f"\n[Step 1] Loading sample LiDAR scan from {sampleBin.name}...")
    scan = loadSemanticKittiScan(sampleBin, sampleLabel)
    print(f"  -> Successfully loaded {scan.pointCount:,} points.")

    # 2. Benchmark Uniform Grids across multiple resolutions
    resolutions = [0.05, 0.10, 0.25]  # 5cm, 10cm, 25cm
    benchmarkResults = []

    print("\n[Step 2] Benchmarking Baseline Uniform Grids (ROI: [-50m, 50m] x [-50m, 50m]):")
    print(f"{'Resolution':<12} | {'Shape':<12} | {'Total Cells':<12} | {'Occupied':<10} | {'Sparsity %':<10} | {'RAM (MB)':<10} | {'Latency (ms)':<12} | {'FPS':<8}")
    print("-" * 95)

    gridsDict = {}

    for res in resolutions:
        grid = UniformElevationGrid(
            resolution=res,
            minX=-50.0,
            maxX=50.0,
            minY=-50.0,
            maxY=50.0,
        )

        # Warm-up run then 5 timed runs for statistical confidence
        grid.update(scan)
        timings = []
        for _ in range(5):
            grid.clear()
            t0 = time.perf_counter()
            grid.update(scan)
            timings.append((time.perf_counter() - t0) * 1000.0)

        meanLatencyMs = float(np.mean(timings))
        grid.lastUpdateLatencyMs = meanLatencyMs
        metrics = grid.getMetrics()
        benchmarkResults.append(metrics)
        gridsDict[res] = grid

        print(
            f"{int(res*100):2d} cm        | "
            f"{metrics['gridShape'][0]}x{metrics['gridShape'][1]:<7} | "
            f"{metrics['totalAllocatedCells']:12,d} | "
            f"{metrics['occupiedCells']:10,d} | "
            f"{metrics['sparsityPercent']:9.2f}% | "
            f"{metrics['memoryUsageMb']:9.2f} | "
            f"{meanLatencyMs:11.2f} ms | "
            f"{1000.0/meanLatencyMs:7.1f}"
        )

    # Save metrics JSON
    resultsDir = PROJECT_ROOT / "experiments" / "results"
    resultsDir.mkdir(parents=True, exist_ok=True)
    metricsJsonPath = resultsDir / "phase2_3_uniform_metrics.json"
    with open(metricsJsonPath, "w") as f:
        json.dump(benchmarkResults, f, indent=2)
    print(f"\n  -> Baseline benchmark metrics saved to: {metricsJsonPath}")

    # 3. Generate 4-Panel Visualization for Uniform 10cm Grid
    print("\n[Step 3] Rendering 4-Panel 2.5D Elevation & Semantic Map (10cm Grid)...")
    visGrid = gridsDict[0.10]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), facecolor="#121212")
    extent = [visGrid.minX, visGrid.maxX, visGrid.minY, visGrid.maxY]

    # Panel 1: Point Cloud BEV
    ax1 = axes[0, 0]
    ax1.set_facecolor("#181818")
    ax1.scatter(scan.points[:, 0], scan.points[:, 1], c=scan.points[:, 2], cmap="viridis", s=1, alpha=0.5)
    ax1.set_title("1. Raw LiDAR Point Cloud (Color by Z)", color="white", fontsize=12)
    ax1.set_xlim(-20, 60)
    ax1.set_ylim(-30, 30)
    ax1.set_aspect("equal")

    # Panel 2: 2.5D Mean Elevation (zMean)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#181818")
    elevData = np.ma.masked_invalid(visGrid.elevationMean)
    imElev = ax2.imshow(elevData, extent=extent, origin="lower", cmap="plasma", vmin=-2.0, vmax=2.5)
    cbarElev = plt.colorbar(imElev, ax=ax2, fraction=0.046, pad=0.04)
    cbarElev.set_label("Elevation (m)", color="white")
    cbarElev.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbarElev.ax.yaxis.get_ticklabels(), color="white")
    ax2.set_title("2. 2.5D Mean Elevation Grid ($z_{mean}$)", color="white", fontsize=12)
    ax2.set_xlim(-20, 60)
    ax2.set_ylim(-30, 30)

    # Panel 3: 2.5D Height Difference / Obstacle Clearance (Delta Z)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#181818")
    hDiffData = np.ma.masked_where(~visGrid.isOccupied, visGrid.heightDiff)
    imHDiff = ax3.imshow(hDiffData, extent=extent, origin="lower", cmap="hot", vmin=0.0, vmax=2.5)
    cbarHDiff = plt.colorbar(imHDiff, ax=ax3, fraction=0.046, pad=0.04)
    cbarHDiff.set_label(r"Obstacle Height $\Delta z$ (m)", color="white")
    cbarHDiff.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbarHDiff.ax.yaxis.get_ticklabels(), color="white")
    ax3.set_title(r"3. 2.5D Obstacle Height ($\Delta z = z_{max} - z_{min}$)", color="white", fontsize=12)
    ax3.set_xlim(-20, 60)
    ax3.set_ylim(-30, 30)

    # Panel 4: Semantic & Dynamic Occupancy Grid
    ax4 = axes[1, 1]
    ax4.set_facecolor("#181818")
    # Build RGB map
    rgbMap = np.zeros((visGrid.heightCells, visGrid.widthCells, 3), dtype=np.float32)
    # Background is dark gray
    rgbMap[:, :] = [0.10, 0.10, 0.10]
    occ = visGrid.isOccupied
    sem = visGrid.semanticClass

    # Terrain = Green (0.18, 0.65, 0.34)
    terrainMask = occ & (sem == TargetClass.TERRAIN.value)
    rgbMap[terrainMask] = [0.18, 0.65, 0.34]

    # Static = Blue (0.25, 0.55, 0.90)
    staticMask = occ & (sem == TargetClass.STATIC_OBSTACLE.value)
    rgbMap[staticMask] = [0.25, 0.55, 0.90]

    # Dynamic = Bright Red (1.0, 0.20, 0.15)
    dynamicMask = occ & (visGrid.isDynamic | (sem == TargetClass.DYNAMIC_OBSTACLE.value))
    rgbMap[dynamicMask] = [1.0, 0.20, 0.15]

    ax4.imshow(rgbMap, extent=extent, origin="lower")
    ax4.set_title("4. 2.5D Semantic Occupancy (Green: Terrain, Blue: Static, Red: Dynamic)", color="white", fontsize=11)
    ax4.set_xlim(-20, 60)
    ax4.set_ylim(-30, 30)

    for ax in axes.flatten():
        ax.set_xlabel("X (m)", color="white", fontsize=10)
        ax.set_ylabel("Y (m)", color="white", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.grid(True, color="#333333", linestyle="--", alpha=0.3)

    plt.suptitle("Fovea-LiDAR Phase 2 & 3: Baseline Uniform 2.5D Elevation & Semantic Grid", color="white", fontsize=14, y=0.98)
    plt.tight_layout()
    plotPath = resultsDir / "phase2_3_uniform_elevation.png"
    plt.savefig(plotPath, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> Visual verification plot saved to: {plotPath}")

    print("\n" + "=" * 65)
    print("  PHASE 2 & 3 STATUS: VERIFICATION & BENCHMARK SUCCESSFUL")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = runPhase2And3Verification()
    sys.exit(0 if success else 1)
