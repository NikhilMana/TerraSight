"""
Verification and Benchmarking Script for Phase 6, 7 & 8:
Semantic Segmentation, Dynamic Actor Perception, and Risk-Aware Refinement.
Evaluates:
  - Quantitative segmentation metrics (mIoU, per-class IoU, distance breakdown)
  - Continuous multi-criteria spatial risk heatmap
  - Distance-Only vs Risk-Aware Fovea-LiDAR comparison
  - Generates publication-quality 4-panel analytical figure
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.point_cloud import PointCloud
from data.kitti_loader import loadSemanticKittiScan
from segmentation.geometric_ground import GeometricSegmenter
from segmentation.kitti_classes import TargetClass
from core.grid_uniform import UniformElevationGrid
from core.grid_fovea import FoveaLiDARGrid, DEFAULT_FOVEA_BANDS
from core.risk_engine import RiskScoringEngine, RiskWeights


def runPhase6_7_8() -> bool:
    print("=" * 80)
    print("  PHASE 6, 7 & 8: SEMANTIC SEGMENTATION, DYNAMIC OBJECTS & RISK-AWARE REFINEMENT")
    print("=" * 80)

    # 1. Load scan
    sampleBin = PROJECT_ROOT / "data" / "sample_scans" / "sample_000000.bin"
    sampleLabel = PROJECT_ROOT / "data" / "sample_scans" / "sample_000000.label"
    scan = loadSemanticKittiScan(sampleBin, sampleLabel)
    print(f"\n[Dataset] Loaded scan with {scan.pointCount:,} points.")

    # 2. Semantic Segmentation Pipeline & IoU Evaluation
    print("\n[Phase 6] Evaluating Semantic Segmentation Pipeline...")
    segmenter = GeometricSegmenter()
    predLabels, predProbs = segmenter.segment(scan)
    metrics = segmenter.evaluate(scan, predLabels)

    print(f"  -> Mean IoU (mIoU): {metrics.mIoU:.2f}%")
    print(f"  -> Overall Accuracy: {metrics.overallAccuracy:.2f}%")
    for cName, iouVal in metrics.perClassIoU.items():
        print(f"     * {cName:<16} IoU: {iouVal:.2f}% | Accuracy: {metrics.perClassAccuracy[cName]:.2f}%")

    print("\n  -> Distance-Stratified Accuracy:")
    for dBand, dAcc in metrics.distanceStratifiedIoU.items():
        print(f"     * {dBand:<10}: {dAcc:.2f}%")

    # 3. Risk Engine Scoring
    print("\n[Phase 7 & 8] Computing Scene Criticality & Spatial Risk Scores...")
    riskEngine = RiskScoringEngine(
        RiskWeights(
            weightDynamic=0.40,
            weightProximity=0.25,
            weightTraversability=0.20,
            weightUncertainty=0.15,
            refinementThreshold=0.45,
        )
    )

    dists = scan.calculateDistances2D()
    heightRel = np.maximum(0.0, scan.points[:, 2] + 1.5)
    # Entropy from segmentation probabilities
    entropy = -np.sum(predProbs * np.log(np.maximum(predProbs, 1e-6)), axis=1) / np.log(3.0)

    pointRiskScores = riskEngine.computeBatchRisk(
        distances=dists,
        isDynamic=scan.dynamicFlags,
        heightDiffs=heightRel,
        semanticEntropy=entropy,
    )

    highRiskCount = int(np.sum(pointRiskScores >= riskEngine.weights.refinementThreshold))
    print(f"  -> Analyzed {len(pointRiskScores):,} points.")
    print(f"  -> Points Flagged as High Risk (Criticality >= 0.45): {highRiskCount:,} ({highRiskCount / len(pointRiskScores) * 100:.1f}%)")

    # 4. Compare Representations: Distance-Only vs Risk-Aware
    print("\n[Comparison] Distance-Only Adaptive Grid vs Risk-Aware Fovea-LiDAR...")
    gridDistOnly = FoveaLiDARGrid(bands=DEFAULT_FOVEA_BANDS, enableRiskRefinement=False)
    gridDistOnly.update(scan)

    gridRiskAware = FoveaLiDARGrid(bands=DEFAULT_FOVEA_BANDS, enableRiskRefinement=True, refinedResolution=0.05, riskEngine=riskEngine)
    gridRiskAware.update(scan)

    refinedCount = gridRiskAware.bandOccupiedCounts.get(99, 0)
    print(f"  -> Distance-Only Active Cells:  {gridDistOnly.totalActiveCells:,} cells ({gridDistOnly.getMemoryUsageMb():.2f} MB)")
    print(f"  -> Risk-Aware Active Cells:     {gridRiskAware.totalActiveCells:,} cells ({gridRiskAware.getMemoryUsageMb():.2f} MB)")
    print(f"  -> High-Resolution Refined Cells: {refinedCount:,} cells (dynamically promoted to 5 cm)")
    print(f"  -> Latency: {gridRiskAware.lastUpdateLatencyMs:.2f} ms ({1000.0/gridRiskAware.lastUpdateLatencyMs:.1f} FPS)")

    # Save metrics JSON
    resultsDir = PROJECT_ROOT / "experiments" / "results"
    resultsDir.mkdir(parents=True, exist_ok=True)
    summaryData = {
        "segmentation": {
            "mIoU": metrics.mIoU,
            "overallAccuracy": metrics.overallAccuracy,
            "perClassIoU": metrics.perClassIoU,
            "distanceStratifiedAccuracy": metrics.distanceStratifiedIoU,
        },
        "riskEngine": {
            "highRiskPointsCount": highRiskCount,
            "refinementThreshold": riskEngine.weights.refinementThreshold,
            "refinedCellsCount": refinedCount,
        },
        "foveaLiDAR": gridRiskAware.getMetrics(),
    }
    with open(resultsDir / "phase8_risk_aware_metrics.json", "w") as f:
        json.dump(summaryData, f, indent=2)
    print(f"  -> Metrics JSON saved to: {resultsDir / 'phase8_risk_aware_metrics.json'}")

    # 5. Render 4-Panel Visualization
    print("\n[Visualization] Rendering 4-Panel Risk-Aware Mapping Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor="#101014")

    # Panel 1: Semantic Segmentation Predictions
    ax1 = axes[0, 0]
    ax1.set_facecolor("#16161c")
    semColors = {1: ("#10b981", "Terrain", 0.3, 1), 2: ("#3b82f6", "Static", 0.7, 4), 3: ("#ef4444", "Dynamic", 0.9, 6)}
    for cVal, (col, cName, alphaVal, ptSize) in semColors.items():
        cMask = predLabels == cVal
        if np.any(cMask):
            ax1.scatter(scan.points[cMask, 0], scan.points[cMask, 1], c=col, s=ptSize, alpha=alphaVal, label=f"{cName} (IoU: {metrics.perClassIoU[cName]:.1f}%)")
    ax1.set_title(f"A. Real-Time Semantic Segmentation (mIoU: {metrics.mIoU:.1f}%)\nGreen: Terrain | Blue: Static | Red: Dynamic", color="white", fontsize=11)
    ax1.set_xlim(-15, 65)
    ax1.set_ylim(-30, 30)
    ax1.set_aspect("equal")
    ax1.legend(facecolor="#22222a", edgecolor="#444", labelcolor="white", fontsize=8, loc="upper left")

    # Panel 2: Continuous Risk Heatmap
    ax2 = axes[0, 1]
    ax2.set_facecolor("#16161c")
    sc2 = ax2.scatter(scan.points[:, 0], scan.points[:, 1], c=pointRiskScores, cmap="inferno", s=3, alpha=0.8, vmin=0.0, vmax=1.0)
    cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r"Criticality Risk $\mathcal{R}$", color="white")
    cbar2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="white")
    ax2.set_title(r"B. Spatial Risk Score $\mathcal{R}(x,y)$" + "\nPeaks at Dynamic Vehicles, Pedestrians & Curbs", color="white", fontsize=11)
    ax2.set_xlim(-15, 65)
    ax2.set_ylim(-30, 30)
    ax2.set_aspect("equal")

    # Panel 3: Spatial Resolution Allocation (The Core Innovation!)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#16161c")
    # Plot base band cells
    bandColors = {0: "#10b981", 1: "#3b82f6", 2: "#f59e0b", 3: "#8b5cf6"}
    for b in gridRiskAware.bands:
        if b.bandId in gridRiskAware.bandData:
            bd = gridRiskAware.bandData[b.bandId]
            ax3.scatter(bd.centerX, bd.centerY, s=bd.resolution * 30, c=bandColors[b.bandId], alpha=0.5, label=f"Band {b.bandId}: {int(b.resolution*100)}cm")

    # Plot Refined Band 99 cells with distinctive highlighted red markers
    if 99 in gridRiskAware.bandData:
        bdRef = gridRiskAware.bandData[99]
        ax3.scatter(bdRef.centerX, bdRef.centerY, s=12, c="#ef4444", alpha=0.9, label=f"Risk-Refined: 5cm ({len(bdRef.centerX)} cells)")

    # Draw distance boundary rings
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    for r in [10, 30, 60]:
        ax3.plot(r * np.cos(theta), r * np.sin(theta), color="#666677", linestyle="--", alpha=0.6)

    ax3.plot(0, 0, marker="o", color="yellow", markersize=8, label="LiDAR Origin")
    ax3.set_title("C. Fovea-LiDAR Adaptive Resolution Allocation\nBase Distance Bands + 5cm Risk-Refined Foveas", color="white", fontsize=11)
    ax3.set_xlim(-15, 65)
    ax3.set_ylim(-30, 30)
    ax3.set_aspect("equal")
    ax3.legend(facecolor="#22222a", edgecolor="#444", labelcolor="white", fontsize=8, loc="upper left")

    # Panel 4: Refined 2.5D Elevation Map
    ax4 = axes[1, 1]
    ax4.set_facecolor("#16161c")
    allXs = []
    allYs = []
    allElevs = []
    for bd in gridRiskAware.bandData.values():
        allXs.extend(bd.centerX)
        allYs.extend(bd.centerY)
        allElevs.extend(bd.zMean)

    sc4 = ax4.scatter(allXs, allYs, c=allElevs, cmap="plasma", s=3, alpha=0.8, vmin=-2.0, vmax=2.5)
    cbar4 = plt.colorbar(sc4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label("Elevation $z_{mean}$ (m)", color="white")
    cbar4.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar4.ax.yaxis.get_ticklabels(), color="white")
    ax4.set_title("D. Refined 2.5D Elevation Grid\nFull Spatial Geometry Preserved for Distant Hazards", color="white", fontsize=11)
    ax4.set_xlim(-15, 65)
    ax4.set_ylim(-30, 30)
    ax4.set_aspect("equal")

    for ax in axes.flatten():
        ax.set_xlabel("X (Forward, meters)", color="white", fontsize=10)
        ax.set_ylabel("Y (Lateral, meters)", color="white", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333340")
        ax.grid(True, color="#252530", linestyle="--", alpha=0.4)

    plt.suptitle("Fovea-LiDAR: Risk-Aware Adaptive 2.5D Perception Engine (DRDO SIH26053)", color="white", fontsize=14, y=0.98)
    plt.tight_layout()
    outPlotPath = resultsDir / "phase8_risk_aware_mapping.png"
    plt.savefig(outPlotPath, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> Analytical verification figure saved to: {outPlotPath}")

    print("\n" + "=" * 80)
    print("  PHASE 6, 7 & 8 STATUS: SUCCESSFUL")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = runPhase6_7_8()
    sys.exit(0 if success else 1)
