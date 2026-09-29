"""
Verification Script for Phase 1: Data Ingestion and Point Cloud Loading.
Validates:
  1. Synthetic 64-beam LiDAR scan generation
  2. SemanticKITTI .bin and .label binary serialization and round-trip deserialization
  3. Class distribution and distance band stratification
  4. Generates a BEV validation plot saved to experiments/results/phase1_scan_bev.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.point_cloud import PointCloud
from data.synthetic_generator import generateSyntheticLiDARScan, saveScanAsSemanticKitti
from data.kitti_loader import loadSemanticKittiScan
from segmentation.kitti_classes import TargetClass, TARGET_CLASS_COLORS


def verifyPhase1() -> bool:
    print("=" * 60)
    print("  PHASE 1 VERIFICATION: LiDAR Data Ingestion & Serialization")
    print("=" * 60)

    # 1. Generate synthetic 64-channel scan
    print("\n[Step 1] Generating synthetic 64-beam LiDAR scan...")
    scan = generateSyntheticLiDARScan(numRings=64, horizontalResolutionDeg=0.25, maxRangeMeters=80.0)
    print(f"  -> Generated {scan.pointCount:,} LiDAR points.")

    minCoords, maxCoords = scan.getBoundingBox()
    print(f"  -> Bounding Box X: [{minCoords[0]:.2f}m, {maxCoords[0]:.2f}m]")
    print(f"  -> Bounding Box Y: [{minCoords[1]:.2f}m, {maxCoords[1]:.2f}m]")
    print(f"  -> Bounding Box Z: [{minCoords[2]:.2f}m, {maxCoords[2]:.2f}m]")

    # 2. Check class breakdown
    terrainCount = np.sum(scan.semanticLabels == TargetClass.TERRAIN.value)
    staticCount = np.sum(scan.semanticLabels == TargetClass.STATIC_OBSTACLE.value)
    dynamicCount = np.sum(scan.semanticLabels == TargetClass.DYNAMIC_OBSTACLE.value)

    print("\n[Step 2] Class Distribution:")
    print(f"  - Terrain (1):          {terrainCount:6,d} ({terrainCount / scan.pointCount * 100:.1f}%)")
    print(f"  - Static Obstacle (2):  {staticCount:6,d} ({staticCount / scan.pointCount * 100:.1f}%)")
    print(f"  - Dynamic Obstacle (3): {dynamicCount:6,d} ({dynamicCount / scan.pointCount * 100:.1f}%)")

    assert terrainCount > 0, "No terrain points generated!"
    assert staticCount > 0, "No static obstacle points generated!"
    assert dynamicCount > 0, "No dynamic obstacle points generated!"

    # 3. Distance band stratification
    dist2d = scan.calculateDistances2D()
    b0_10 = np.sum((dist2d >= 0) & (dist2d < 10))
    b10_30 = np.sum((dist2d >= 10) & (dist2d < 30))
    b30_60 = np.sum((dist2d >= 30) & (dist2d < 60))
    b60_100 = np.sum((dist2d >= 60) & (dist2d <= 100))

    print("\n[Step 3] Distance Band Stratification:")
    print(f"  - Band 0 (0-10m, Fine):   {b0_10:6,d} points")
    print(f"  - Band 1 (10-30m, Med):   {b10_30:6,d} points")
    print(f"  - Band 2 (30-60m, Coarse):{b30_60:6,d} points")
    print(f"  - Band 3 (60-100m, Far):  {b60_100:6,d} points")

    # 4. Serialize to SemanticKITTI format (.bin + .label)
    sampleDir = PROJECT_ROOT / "data" / "sample_scans"
    binPath = sampleDir / "sample_000000.bin"
    labelPath = sampleDir / "sample_000000.label"

    print(f"\n[Step 4] Serializing to SemanticKITTI binary format at {binPath.name}...")
    saveScanAsSemanticKitti(scan, binPath, labelPath)
    assert binPath.is_file(), "Binary file not created!"
    assert labelPath.is_file(), "Label file not created!"
    print(f"  -> File sizes: .bin = {binPath.stat().st_size / 1024:.1f} KB, .label = {labelPath.stat().st_size / 1024:.1f} KB")

    # 5. Round-trip test: Load back via kitti_loader
    print("\n[Step 5] Round-trip test: Deserializing via data.kitti_loader...")
    loadedScan = loadSemanticKittiScan(binPath, labelPath)
    assert loadedScan.pointCount == scan.pointCount, f"Point count mismatch: {loadedScan.pointCount} vs {scan.pointCount}"
    np.testing.assert_allclose(loadedScan.points, scan.points, atol=1e-5, err_msg="Coordinates altered during round-trip!")
    assert np.array_equal(loadedScan.semanticLabels, scan.semanticLabels), "Semantic labels altered during round-trip!"
    assert np.array_equal(loadedScan.dynamicFlags, scan.dynamicFlags), "Dynamic flags altered during round-trip!"
    print("  -> Round-trip integrity check: PASSED (100% exact match).")

    # 6. Generate BEV visualization image
    print("\n[Step 6] Rendering 2D Bird's-Eye-View (BEV) verification plot...")
    resultsDir = PROJECT_ROOT / "experiments" / "results"
    resultsDir.mkdir(parents=True, exist_ok=True)
    plotPath = resultsDir / "phase1_scan_bev.png"

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#121212")
    ax.set_facecolor("#181818")

    # Color map
    colorMap = {
        TargetClass.TERRAIN.value: ("#2ca02c", "Terrain / Drivable", 0.3, 1),
        TargetClass.STATIC_OBSTACLE.value: ("#1f77b4", "Static Obstacle", 0.7, 4),
        TargetClass.DYNAMIC_OBSTACLE.value: ("#d62728", "Dynamic Obstacle", 0.9, 6),
    }

    for classVal, (col, labelStr, alphaVal, ptSize) in colorMap.items():
        mask = loadedScan.semanticLabels == classVal
        if np.any(mask):
            ax.scatter(
                loadedScan.points[mask, 0],
                loadedScan.points[mask, 1],
                c=col,
                s=ptSize,
                alpha=alphaVal,
                label=labelStr,
                edgecolors="none",
            )

    # Draw sensor location and distance bands
    ax.plot(0, 0, marker="o", color="yellow", markersize=10, label="Ego LiDAR (Origin)")
    theta = np.linspace(0, 2 * np.pi, 200)
    for bandRadius, color, style in [(10, "cyan", ":"), (30, "cyan", "--"), (60, "cyan", "-.")]:
        ax.plot(bandRadius * np.cos(theta), bandRadius * np.sin(theta), color=color, linestyle=style, alpha=0.5)
        ax.text(bandRadius * 0.707 + 0.5, bandRadius * 0.707 + 0.5, f"{bandRadius}m band", color="cyan", fontsize=8)

    ax.set_xlim(-15, 65)
    ax.set_ylim(-30, 30)
    ax.set_aspect("equal")
    ax.set_xlabel("X (Forward, meters)", color="white", fontsize=11)
    ax.set_ylabel("Y (Left/Right, meters)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(True, color="#333333", linestyle="--", alpha=0.5)
    ax.legend(facecolor="#222222", edgecolor="#444444", labelcolor="white", loc="upper left")
    ax.set_title("Fovea-LiDAR Phase 1: Synthetic 64-Beam LiDAR Scan (Semantic BEV)", color="white", fontsize=13, pad=12)

    plt.tight_layout()
    plt.savefig(plotPath, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> Validation BEV plot saved to: {plotPath}")

    print("\n" + "=" * 60)
    print("  PHASE 1 STATUS: VERIFICATION SUCCESSFUL")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = verifyPhase1()
    sys.exit(0 if success else 1)
