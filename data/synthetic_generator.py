"""
Realistic Synthetic LiDAR Scan Generator.
Simulates a 64-beam automotive LiDAR (Velodyne HDL-64E) with:
  - Drivable undulating terrain (ground plane + road surface)
  - Static structures (buildings, curbs, poles)
  - Dynamic actors (vehicles and pedestrians at various distance bands)
  - SemanticKITTI-compatible labels and binary export
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from core.point_cloud import PointCloud
from segmentation.kitti_classes import TargetClass


def generateSyntheticLiDARScan(
    numRings: int = 64,
    horizontalResolutionDeg: float = 0.2,  # 1800 azimuth steps
    maxRangeMeters: float = 80.0,
    noiseStd: float = 0.02,
    randomSeed: int = 42,
) -> PointCloud:
    """
    Simulates a 64-channel LiDAR scan in a structured urban scene.
    Scene contents:
      - Ground road plane with slight undulation (z ~ -1.73m + slope)
      - Curbs at y = -4.0m and y = +4.0m
      - Static building facade at y = 8.0m to 12.0m
      - Static tree/pole at (x=15.0m, y=5.0m)
      - Dynamic vehicle 1 at near range (x=8.0m, y=-2.0m) [0-10m band]
      - Dynamic vehicle 2 at mid range (x=24.0m, y=1.5m) [10-30m band]
      - Dynamic pedestrian at near range (x=5.0m, y=3.0m) [0-10m band]
      - Static parked truck at far range (x=50.0m, y=-3.0m) [30-60m band]
    """
    rng = np.random.default_rng(randomSeed)

    # Vertical elevation angles for 64-beam (from +2.0 deg to -24.8 deg)
    verticalAngles = np.linspace(np.radians(2.0), np.radians(-24.8), numRings)
    # Azimuth angles (360 degrees)
    horizontalAngles = np.radians(np.arange(-180.0, 180.0, horizontalResolutionDeg))

    # Mesh grid of beam directions
    vGrid, hGrid = np.meshgrid(verticalAngles, horizontalAngles, indexing="ij")
    vFlat = vGrid.flatten()
    hFlat = hGrid.flatten()

    # Direction vectors for each beam
    dirX = np.cos(vFlat) * np.cos(hFlat)
    dirY = np.cos(vFlat) * np.sin(hFlat)
    dirZ = np.sin(vFlat)

    # Sensor origin at height +1.73m above nominal road
    sensorZ = 1.73
    pointsList = []
    labelsList = []
    dynamicList = []
    intensityList = []

    # 1. Ground intersection: plane at z = -sensorZ
    # Downward beams (dirZ < -0.01) intersect ground
    downwardMask = dirZ < -0.015
    groundDist = -sensorZ / dirZ[downwardMask]
    validGround = (groundDist > 0.5) & (groundDist <= maxRangeMeters)

    gx = dirX[downwardMask][validGround] * groundDist[validGround]
    gy = dirY[downwardMask][validGround] * groundDist[validGround]
    gz = dirZ[downwardMask][validGround] * groundDist[validGround]

    # Add gentle undulating slope to ground
    gz = gz + 0.03 * np.sin(gx / 5.0) + 0.01 * np.cos(gy / 2.0)
    # Add sensor noise
    gx += rng.normal(0, noiseStd, len(gx))
    gy += rng.normal(0, noiseStd, len(gy))
    gz += rng.normal(0, noiseStd, len(gz))

    gPoints = np.column_stack([gx, gy, gz])
    pointsList.append(gPoints)
    labelsList.append(np.full(len(gPoints), TargetClass.TERRAIN.value, dtype=np.uint32))
    dynamicList.append(np.zeros(len(gPoints), dtype=bool))
    intensityList.append(rng.uniform(0.15, 0.45, len(gPoints)).astype(np.float32))

    # Helper function to generate dense surface box points
    def generateBoxPoints(
        center: Tuple[float, float, float],
        dims: Tuple[float, float, float],
        targetClass: TargetClass,
        isDynamic: bool,
        pointCount: int,
    ) -> None:
        cx, cy, cz = center
        dx, dy, dz = dims
        # Points on 5 outer faces (front, back, left, right, top)
        facePoints = []
        # Front/Back (X faces)
        nFb = pointCount // 3
        fbY = rng.uniform(cy - dy / 2, cy + dy / 2, nFb)
        fbZ = rng.uniform(cz - dz / 2, cz + dz / 2, nFb)
        fbX = np.where(rng.random(nFb) > 0.5, cx + dx / 2, cx - dx / 2)
        facePoints.append(np.column_stack([fbX, fbY, fbZ]))

        # Left/Right (Y faces)
        nLr = pointCount // 3
        lrX = rng.uniform(cx - dx / 2, cx + dx / 2, nLr)
        lrZ = rng.uniform(cz - dz / 2, cz + dz / 2, nLr)
        lrY = np.where(rng.random(nLr) > 0.5, cy + dy / 2, cy - dy / 2)
        facePoints.append(np.column_stack([lrX, lrY, lrZ]))

        # Top (Z face)
        nTop = pointCount - 2 * (pointCount // 3)
        topX = rng.uniform(cx - dx / 2, cx + dx / 2, nTop)
        topY = rng.uniform(cy - dy / 2, cy + dy / 2, nTop)
        topZ = np.full(nTop, cz + dz / 2)
        facePoints.append(np.column_stack([topX, topY, topZ]))

        boxPts = np.vstack(facePoints)
        boxPts += rng.normal(0, noiseStd, boxPts.shape)

        pointsList.append(boxPts)
        labelsList.append(np.full(len(boxPts), targetClass.value, dtype=np.uint32))
        dynamicList.append(np.full(len(boxPts), isDynamic, dtype=bool))
        intensityList.append(rng.uniform(0.4, 0.9, len(boxPts)).astype(np.float32))

    # 2. Static Obstacles
    # Building wall along right side: (x=20 to 40, y=10.0, z=-1.73 to 3.0)
    generateBoxPoints(center=(30.0, 9.0, 0.5), dims=(25.0, 2.0, 4.5),
                      targetClass=TargetClass.STATIC_OBSTACLE, isDynamic=False, pointCount=3500)

    # Road curb / barriers along left side
    generateBoxPoints(center=(15.0, -5.0, -1.4), dims=(35.0, 0.6, 0.6),
                      targetClass=TargetClass.STATIC_OBSTACLE, isDynamic=False, pointCount=1200)

    # Static pole / tree trunk
    generateBoxPoints(center=(12.0, 4.5, -0.2), dims=(0.5, 0.5, 3.0),
                      targetClass=TargetClass.STATIC_OBSTACLE, isDynamic=False, pointCount=600)

    # 3. Dynamic Obstacles
    # Dynamic Vehicle 1 (Near: 8m ahead)
    generateBoxPoints(center=(8.5, -1.5, -0.9), dims=(4.2, 1.9, 1.5),
                      targetClass=TargetClass.DYNAMIC_OBSTACLE, isDynamic=True, pointCount=2200)

    # Dynamic Pedestrian (Near: 5.5m ahead, on the right)
    generateBoxPoints(center=(5.5, 2.5, -0.85), dims=(0.5, 0.5, 1.75),
                      targetClass=TargetClass.DYNAMIC_OBSTACLE, isDynamic=True, pointCount=500)

    # Dynamic Vehicle 2 (Mid-range: 22m ahead)
    generateBoxPoints(center=(22.0, 1.8, -0.9), dims=(4.5, 2.0, 1.6),
                      targetClass=TargetClass.DYNAMIC_OBSTACLE, isDynamic=True, pointCount=1400)

    # Dynamic Cyclist (Mid-range: 17m ahead, right side)
    generateBoxPoints(center=(17.0, 3.8, -0.9), dims=(1.8, 0.7, 1.6),
                      targetClass=TargetClass.DYNAMIC_OBSTACLE, isDynamic=True, pointCount=750)

    # Dynamic Vehicle 3 (Far-range: 48m ahead)
    generateBoxPoints(center=(48.0, -2.0, -0.8), dims=(6.0, 2.4, 2.2),
                      targetClass=TargetClass.DYNAMIC_OBSTACLE, isDynamic=True, pointCount=800)

    allPoints = np.vstack(pointsList).astype(np.float32)
    allLabels = np.concatenate(labelsList).astype(np.uint32)
    allDynamics = np.concatenate(dynamicList).astype(bool)
    allIntensity = np.concatenate(intensityList).astype(np.float32)

    # Clip points to maxRangeMeters
    ranges = np.linalg.norm(allPoints[:, :2], axis=1)
    validMask = (ranges >= 0.5) & (ranges <= maxRangeMeters)

    return PointCloud(
        points=allPoints[validMask],
        intensity=allIntensity[validMask],
        semanticLabels=allLabels[validMask],
        dynamicFlags=allDynamics[validMask],
        timestamp=0.0,
    )


def saveScanAsSemanticKitti(pointCloud: PointCloud, outBinPath: Path, outLabelPath: Path) -> None:
    """Export PointCloud in exact SemanticKITTI binary format (.bin + .label)."""
    outBinPath.parent.mkdir(parents=True, exist_ok=True)
    outLabelPath.parent.mkdir(parents=True, exist_ok=True)

    # .bin: [x, y, z, intensity] in float32
    scanData = np.column_stack([pointCloud.points, pointCloud.intensity]).astype(np.float32)
    scanData.tofile(str(outBinPath))

    # .label: lower 16 bits = semantic label, upper 16 bits = instance ID
    # For synthetic scans, convert targetClass to canonical SemanticKITTI IDs:
    # Terrain -> 40 (road), Static -> 50 (building), Dynamic -> 10 (car)
    canonicalIds = np.zeros(pointCloud.pointCount, dtype=np.uint32)
    labels = pointCloud.semanticLabels
    canonicalIds[labels == TargetClass.TERRAIN.value] = 40
    canonicalIds[labels == TargetClass.STATIC_OBSTACLE.value] = 50
    canonicalIds[labels == TargetClass.DYNAMIC_OBSTACLE.value] = 10

    rawLabels = (canonicalIds & 0xFFFF) | (pointCloud.instanceIds.astype(np.uint32) << 16)
    rawLabels.tofile(str(outLabelPath))
