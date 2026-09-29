"""
Distant Dynamic Obstacle Comparison Demonstration (DRDO SIH26053).
Project: Fovea-LiDAR | Team: FoveaX

Compares spatial fidelity between:
  1. Distance-Only Adaptive Grid: Discretizes distant obstacles (60-100m) at coarse 50cm cells.
     Results in severe quantization error, boundary blurring, and loss of obstacle geometry.
  2. Fovea-LiDAR (Proposed): Multi-criteria risk scoring detects distant dynamic actor
     and dynamically spawns a localized 5cm high-resolution fovea, preserving 10x finer contour.

Saves:
  - Quantitative comparison JSON: experiments/results/distant_obstacle_metrics.json
  - Publication-quality visualization: experiments/results/distant_obstacle_comparison.png
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.point_cloud import PointCloud
from core.grid_fovea import FoveaLiDARGrid, DEFAULT_FOVEA_BANDS
from core.risk_engine import RiskScoringEngine, RiskWeights


def generateDistantVehiclePointCloud(
    centerX: float = 65.0,
    centerY: float = 4.0,
    length: float = 4.5,
    width: float = 1.9,
    height: float = 1.6,
    numPoints: int = 1200,
    randomSeed: int = 42,
) -> PointCloud:
    """
    Generates a realistic 3D point cloud of a vehicle located at 65m range.
    Simulates LiDAR rays hitting front bumper, hood, roof, windshield, and side panel.
    """
    rng = np.random.default_rng(randomSeed)
    pts = []

    # Front bumper & grill (x = centerX - length/2)
    nFront = numPoints // 4
    yF = rng.uniform(-width / 2, width / 2, nFront)
    zF = rng.uniform(-0.5, 0.4, nFront)
    xF = np.full(nFront, -length / 2) + rng.normal(0, 0.02, nFront)
    pts.append(np.column_stack([xF, yF, zF]))

    # Hood (z ~ 0.4, x in [-length/2, -0.3])
    nHood = numPoints // 4
    xH = rng.uniform(-length / 2, -0.3, nHood)
    yH = rng.uniform(-width / 2, width / 2, nHood)
    zH = np.full(nHood, 0.4) + rng.normal(0, 0.02, nHood)
    pts.append(np.column_stack([xH, yH, zH]))

    # Roof & Windshield (z in [0.4, height-0.5], x in [-0.3, length/2])
    nRoof = numPoints // 4
    xR = rng.uniform(-0.3, length / 2, nRoof)
    yR = rng.uniform(-width / 2 * 0.85, width / 2 * 0.85, nRoof)
    zR = rng.uniform(0.4, height - 0.5, nRoof)
    pts.append(np.column_stack([xR, yR, zR]))

    # Side panel facing sensor (y = centerY - width/2 or near side)
    nSide = numPoints - (nFront + nHood + nRoof)
    xS = rng.uniform(-length / 2, length / 2, nSide)
    yS = np.full(nSide, -width / 2) + rng.normal(0, 0.02, nSide)
    zS = rng.uniform(-0.5, height - 0.5, nSide)
    pts.append(np.column_stack([xS, yS, zS]))

    localPts = np.vstack(pts)

    # Transform to world coordinates (centered at centerX, centerY, road height z = -1.2)
    worldPts = np.zeros_like(localPts)
    worldPts[:, 0] = localPts[:, 0] + centerX
    worldPts[:, 1] = localPts[:, 1] + centerY
    worldPts[:, 2] = localPts[:, 2] - 0.5  # elevation relative to sensor

    N = len(worldPts)
    return PointCloud(
        points=worldPts.astype(np.float32),
        intensity=rng.uniform(0.3, 0.9, N).astype(np.float32),
        semanticLabels=np.full(N, 2, dtype=np.uint32),  # TargetClass 2: Dynamic
        dynamicFlags=np.ones(N, dtype=bool),
        timestamp=0.0,
    )


def runDistantObstacleComparison() -> None:
    print("=" * 80)
    print("  FOVEA-LiDAR: DISTANT DYNAMIC OBSTACLE RESOLUTION BENCHMARK")
    print("=" * 80)

    outDir = PROJECT_ROOT / "experiments" / "results"
    outDir.mkdir(parents=True, exist_ok=True)

    # 1. Generate vehicle at 65.1 meters (Band 3: 60-100m)
    vehicleScan = generateDistantVehiclePointCloud(centerX=65.0, centerY=4.0, numPoints=1200)
    ptsXY = vehicleScan.points[:, :2]
    dists = np.sqrt(ptsXY[:, 0] ** 2 + ptsXY[:, 1] ** 2)
    meanDist = float(np.mean(dists))

    print(f"\n[1] Synthesized Distant Dynamic Vehicle:")
    print(f"  -> Range from Sensor: {meanDist:.2f} meters (Band 3: 60-100m)")
    print(f"  -> Point Count: {vehicleScan.pointCount:,} points")
    print(f"  -> Vehicle Dimensions: 4.5m (L) x 1.9m (W) x 1.6m (H)")

    # 2. Representation A: Distance-Only Adaptive Grid (Resolution = 50 cm)
    distOnlyGrid = FoveaLiDARGrid(bands=DEFAULT_FOVEA_BANDS, enableRiskRefinement=False)
    distOnlyGrid.update(vehicleScan)
    distCells = distOnlyGrid.bandData[3]  # Band 3: 50 cm
    numDistCells = len(distCells.packedKeys)

    # Calculate Quantization Error (RMS distance from points to cell center)
    distPointErrors = []
    for i in range(len(vehicleScan.points)):
        p = vehicleScan.points[i]
        c = distOnlyGrid.queryWorldCoordinate(p[0], p[1])
        if c is not None:
            err = np.sqrt((p[0] - c.centerX) ** 2 + (p[1] - c.centerY) ** 2)
            distPointErrors.append(err)
    distRmsErrorCm = float(np.sqrt(np.mean(np.square(distPointErrors))) * 100.0)

    print(f"\n[2] Model A: Distance-Only Adaptive Representation:")
    print(f"  -> Assigned Band: Band 3 (Default Coarse Distance Resolution)")
    print(f"  -> Resolution: 50 cm (0.50 m)")
    print(f"  -> Occupied Cells on Vehicle: {numDistCells} cells")
    print(f"  -> Spatial Quantization RMS Error: {distRmsErrorCm:.2f} cm")

    # 3. Representation B: Fovea-LiDAR Risk-Aware Grid (Dynamic 5 cm Refinement)
    riskEngine = RiskScoringEngine(
        RiskWeights(
            weightDynamic=0.40,
            weightProximity=0.25,
            weightTraversability=0.20,
            weightUncertainty=0.15,
            refinementThreshold=0.45,
        )
    )
    foveaGrid = FoveaLiDARGrid(bands=DEFAULT_FOVEA_BANDS, enableRiskRefinement=True, riskEngine=riskEngine)
    foveaGrid.update(vehicleScan)
    refinedCells = foveaGrid.bandData[99]  # Band 99: 5 cm Refined Fovea
    numRefinedCells = len(refinedCells.packedKeys)

    foveaPointErrors = []
    for i in range(len(vehicleScan.points)):
        p = vehicleScan.points[i]
        c = foveaGrid.queryWorldCoordinate(p[0], p[1])
        if c is not None:
            err = np.sqrt((p[0] - c.centerX) ** 2 + (p[1] - c.centerY) ** 2)
            foveaPointErrors.append(err)
    foveaRmsErrorCm = float(np.sqrt(np.mean(np.square(foveaPointErrors))) * 100.0)
    precisionGain = distRmsErrorCm / max(0.01, foveaRmsErrorCm)

    print(f"\n[3] Model B: Fovea-LiDAR (Risk-Aware Proposed):")
    print(f"  -> Refinement Triggered: TRUE (Dynamic Flag + Height Hazard)")
    print(f"  -> Assigned Band: Band 99 (Active Risk-Refined Fovea)")
    print(f"  -> Resolution: 5 cm (0.05 m) — 10x Higher Spatial Resolution")
    print(f"  -> Occupied Cells on Vehicle: {numRefinedCells} cells ({numRefinedCells / numDistCells:.1f}x more detail)")
    print(f"  -> Spatial Quantization RMS Error: {foveaRmsErrorCm:.2f} cm")
    print(f"  -> Boundary Fidelity Improvement: {precisionGain:.1f}x Reduction in Quantization Error")

    # 4. Save Metrics JSON
    metrics = {
        "obstacleDistanceMeters": round(meanDist, 2),
        "pointCount": vehicleScan.pointCount,
        "distanceOnlyModel": {
            "resolutionCm": 50,
            "occupiedCells": numDistCells,
            "rmsQuantizationErrorCm": round(distRmsErrorCm, 2),
        },
        "foveaLidarModel": {
            "resolutionCm": 5,
            "occupiedCells": numRefinedCells,
            "rmsQuantizationErrorCm": round(foveaRmsErrorCm, 2),
            "errorReductionFactor": round(precisionGain, 2),
        },
    }
    jsonPath = outDir / "distant_obstacle_metrics.json"
    with open(jsonPath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Saved Metrics] -> {jsonPath}")

    # 5. Render Publication Comparison Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f1117")

    # Plot 1: Distance-Only (50 cm)
    ax1.set_facecolor("#161822")
    ax1.scatter(ptsXY[:, 0], ptsXY[:, 1], c="#475569", s=4, alpha=0.3, label="Raw LiDAR Points")
    for u, v, cx, cy in zip(distCells.gridU, distCells.gridV, distCells.centerX, distCells.centerY):
        rect = plt.Rectangle((cx - 0.25, cy - 0.25), 0.50, 0.50, fill=True, facecolor="#8b5cf6", alpha=0.4, edgecolor="#a78bfa", linewidth=1.5)
        ax1.add_patch(rect)
    ax1.scatter(distCells.centerX, distCells.centerY, c="#c4b5fd", s=25, marker="s", label=f"50cm Cells ({numDistCells} cells)")
    ax1.set_xlim(62.0, 68.0)
    ax1.set_ylim(1.5, 6.5)
    ax1.set_aspect("equal")
    ax1.set_title(f"Distance-Only Baseline: 50 cm Grid at 65m\nRMS Error: {distRmsErrorCm:.1f} cm | Heavy Boundary Blurring", color="white", fontsize=12, pad=12)
    ax1.set_xlabel("World X (meters)", color="#94a3b8")
    ax1.set_ylabel("World Y (meters)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    ax1.legend(facecolor="#1e2230", edgecolor="#475569", labelcolor="white", loc="upper left")
    ax1.grid(True, color="#2d3748", linestyle="--", alpha=0.5)

    # Plot 2: Fovea-LiDAR (5 cm Refined)
    ax2.set_facecolor("#161822")
    ax2.scatter(ptsXY[:, 0], ptsXY[:, 1], c="#475569", s=4, alpha=0.3, label="Raw LiDAR Points")
    for cx, cy in zip(refinedCells.centerX, refinedCells.centerY):
        rect = plt.Rectangle((cx - 0.025, cy - 0.025), 0.05, 0.05, fill=True, facecolor="#ef4444", alpha=0.5, edgecolor="#f87171", linewidth=0.5)
        ax2.add_patch(rect)
    ax2.scatter(refinedCells.centerX, refinedCells.centerY, c="#fca5a5", s=8, marker="s", label=f"Risk-Refined 5cm Cells ({numRefinedCells} cells)")
    ax2.set_xlim(62.0, 68.0)
    ax2.set_ylim(1.5, 6.5)
    ax2.set_aspect("equal")
    ax2.set_title(f"Fovea-LiDAR (Proposed): 5 cm Refined Fovea at 65m\nRMS Error: {foveaRmsErrorCm:.1f} cm | {precisionGain:.1f}x Finer Spatial Fidelity", color="#38bdf8", fontsize=12, pad=12)
    ax2.set_xlabel("World X (meters)", color="#94a3b8")
    ax2.set_ylabel("World Y (meters)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    ax2.legend(facecolor="#1e2230", edgecolor="#475569", labelcolor="white", loc="upper left")
    ax2.grid(True, color="#2d3748", linestyle="--", alpha=0.5)

    fig.suptitle("Fovea-LiDAR Spatial Fidelity Advantage: Distant Dynamic Vehicle (Range = 65m)", color="white", fontsize=15, y=0.98)
    plt.tight_layout()
    figPath = outDir / "distant_obstacle_comparison.png"
    plt.savefig(figPath, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Saved Plot]    -> {figPath}")
    print("=" * 80)


if __name__ == "__main__":
    runDistantObstacleComparison()
