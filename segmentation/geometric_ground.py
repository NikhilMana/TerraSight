"""
Fast Geometric Ground Plane and Obstacle Segmenter.
Provides real-time ground/obstacle classification and computes
standard semantic segmentation metrics:
  - Per-class IoU (Terrain, Static, Dynamic)
  - Mean Intersection-over-Union (mIoU)
  - Distance-stratified accuracy (0-10m, 10-30m, 30-60m, 60-100m)
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np

from core.point_cloud import PointCloud
from segmentation.kitti_classes import TargetClass


@dataclass
class SegmentationMetrics:
    mIoU: float
    perClassIoU: Dict[str, float]
    perClassAccuracy: Dict[str, float]
    overallAccuracy: float
    distanceStratifiedIoU: Dict[str, float]
    confusionMatrix: np.ndarray


class GeometricSegmenter:
    """
    Real-time geometric ground and obstacle segmenter.
    Uses concentric radial bins and lowest-point surface modeling to rapidly
    separate terrain from static and dynamic obstacles.
    """

    def __init__(
        self,
        groundHeightThreshold: float = 0.25,  # meters above estimated ground
        sensorHeightNominal: float = 1.73,     # meters above ground
        maxRangeMeters: float = 100.0,
    ) -> None:
        self.groundHeightThreshold = groundHeightThreshold
        self.sensorHeightNominal = sensorHeightNominal
        self.maxRangeMeters = maxRangeMeters

    def segment(self, pointCloud: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment point cloud into predicted classes:
          1: Terrain
          2: Static Obstacle
          3: Dynamic Obstacle
        Returns:
            predLabels: (N,) uint8
            predProbabilities: (N, 3) float32 class confidence
        """
        points = pointCloud.points
        N = len(points)
        if N == 0:
            return np.empty(0, dtype=np.uint8), np.empty((0, 3), dtype=np.float32)

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        dists2D = np.sqrt(xs * xs + ys * ys)

        # 1. Ground Surface Estimation using Radial Annular Slices
        # Points near or below -sensorHeight + groundHeightThreshold are candidate ground
        # Estimate radial ground profile z_ground(r) = -sensorHeight + slope * r
        numRadialBins = 20
        rBins = np.linspace(0.0, self.maxRangeMeters, numRadialBins + 1)
        binIndices = np.digitize(dists2D, rBins) - 1
        binIndices = np.clip(binIndices, 0, numRadialBins - 1)

        # Find 5th percentile elevation in each radial bin as ground estimate
        groundElevationProfile = np.full(numRadialBins, -self.sensorHeightNominal, dtype=np.float32)
        for b in range(numRadialBins):
            inBin = binIndices == b
            if np.sum(inBin) > 10:
                groundElevationProfile[b] = np.percentile(zs[inBin], 5)

        estimatedGroundZ = groundElevationProfile[binIndices]
        heightAboveGround = zs - estimatedGroundZ

        predLabels = np.full(N, TargetClass.STATIC_OBSTACLE.value, dtype=np.uint8)
        predProbs = np.zeros((N, 3), dtype=np.float32)

        # Terrain: points close to ground surface
        terrainMask = heightAboveGround < self.groundHeightThreshold
        predLabels[terrainMask] = TargetClass.TERRAIN.value

        # Non-ground points: differentiate Static vs Dynamic
        # Dynamic objects typically have compact bounding extent, higher intensity, or
        # specific height profiles (pedestrians 0.5-2.0m, cars 0.5-2.2m)
        nonGround = ~terrainMask
        # In this perception module, points in the road corridor (|y| < 5.0m) with vehicle height
        # are classified as dynamic candidates
        isVehicleCandidate = (
            nonGround &
            (np.abs(ys) < 6.0) &
            (heightAboveGround >= 0.25) &
            (heightAboveGround <= 2.2) &
            (dists2D < 55.0)
        )

        predLabels[isVehicleCandidate] = TargetClass.DYNAMIC_OBSTACLE.value

        # Class confidence scores
        # Terrain confidence: highest near ground
        predProbs[terrainMask, 0] = np.clip(1.0 - (heightAboveGround[terrainMask] / self.groundHeightThreshold), 0.5, 1.0)
        predProbs[~terrainMask, 0] = 0.05

        # Dynamic confidence
        predProbs[isVehicleCandidate, 2] = 0.85
        predProbs[~isVehicleCandidate, 2] = 0.10

        # Static confidence
        staticMask = nonGround & (~isVehicleCandidate)
        predProbs[staticMask, 1] = 0.80
        predProbs[~staticMask, 1] = 0.10

        # Normalize probabilities
        probSum = np.sum(predProbs, axis=1, keepdims=True)
        predProbs = predProbs / np.maximum(probSum, 1e-6)

        return predLabels, predProbs

    def evaluate(self, pointCloud: PointCloud, predLabels: np.ndarray) -> SegmentationMetrics:
        """
        Compute quantitative segmentation metrics against ground-truth labels:
          - IoU per class
          - mIoU
          - Accuracy by distance band
        """
        gt = pointCloud.semanticLabels
        classes = [
            (TargetClass.TERRAIN.value, "Terrain"),
            (TargetClass.STATIC_OBSTACLE.value, "Static"),
            (TargetClass.DYNAMIC_OBSTACLE.value, "Dynamic"),
        ]

        perClassIoU = {}
        perClassAcc = {}
        confMatrix = np.zeros((4, 4), dtype=np.int64)

        for cVal, cName in classes:
            tp = int(np.sum((gt == cVal) & (predLabels == cVal)))
            fp = int(np.sum((gt != cVal) & (predLabels == cVal)))
            fn = int(np.sum((gt == cVal) & (predLabels != cVal)))

            iou = tp / max(1, (tp + fp + fn))
            acc = tp / max(1, (tp + fn))
            perClassIoU[cName] = float(iou * 100.0)
            perClassAcc[cName] = float(acc * 100.0)

        for i in range(1, 4):
            for j in range(1, 4):
                confMatrix[i, j] = np.sum((gt == i) & (predLabels == j))

        mIoU = float(np.mean(list(perClassIoU.values())))
        overallAcc = float(np.sum(gt == predLabels) / max(1, len(gt)) * 100.0)

        # Distance-stratified IoU
        dist2D = pointCloud.calculateDistances2D()
        bands = [
            ("0-10m", (dist2D >= 0) & (dist2D < 10)),
            ("10-30m", (dist2D >= 10) & (dist2D < 30)),
            ("30-60m", (dist2D >= 30) & (dist2D < 60)),
            ("60-100m", (dist2D >= 60) & (dist2D <= 100)),
        ]

        distIoU = {}
        for bName, bMask in bands:
            if np.sum(bMask) > 0:
                bGt = gt[bMask]
                bPred = predLabels[bMask]
                bAcc = np.sum(bGt == bPred) / len(bGt) * 100.0
                distIoU[bName] = float(bAcc)
            else:
                distIoU[bName] = 0.0

        return SegmentationMetrics(
            mIoU=mIoU,
            perClassIoU=perClassIoU,
            perClassAccuracy=perClassAcc,
            overallAccuracy=overallAcc,
            distanceStratifiedIoU=distIoU,
            confusionMatrix=confMatrix,
        )
