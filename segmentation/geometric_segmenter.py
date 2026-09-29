"""
Geometric Ground & Obstacle Segmenter (Reference / Fallback Baseline).
Implements BaseSegmenter using radial ground elevation profile estimation and spatial heuristics.
"""

import time
from typing import Optional, Tuple
import numpy as np

from core.point_cloud import PointCloud
from segmentation.base_segmenter import BaseSegmenter, SegmentationResult
from segmentation.class_mapping import DRDOTargetClass


class GeometricSegmenter(BaseSegmenter):
    """
    Heuristic real-time geometric ground and obstacle segmenter.
    Serves as the non-neural baseline and fallback module.
    """

    def __init__(
        self,
        groundHeightThreshold: float = 0.25,
        sensorHeightNominal: float = 1.73,
        maxRangeMeters: float = 100.0,
    ) -> None:
        self.groundHeightThreshold = groundHeightThreshold
        self.sensorHeightNominal = sensorHeightNominal
        self.maxRangeMeters = maxRangeMeters

    def predict(self, pointCloud: PointCloud) -> SegmentationResult:
        startTime = time.perf_counter()
        points = pointCloud.points
        N = len(points)
        if N == 0:
            return SegmentationResult(
                labels=np.empty(0, dtype=np.uint8),
                confidence=np.empty(0, dtype=np.float32),
                uncertainty=np.empty(0, dtype=np.float32),
                probabilities=np.empty((0, 3), dtype=np.float32),
                latencyMs=0.0,
            )

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        dists2D = np.sqrt(xs * xs + ys * ys)

        # 1. Radial Ground Profile
        numRadialBins = 20
        rBins = np.linspace(0.0, self.maxRangeMeters, numRadialBins + 1)
        binIndices = np.clip(np.digitize(dists2D, rBins) - 1, 0, numRadialBins - 1)

        groundProfile = np.full(numRadialBins, -self.sensorHeightNominal, dtype=np.float32)
        for b in range(numRadialBins):
            inBin = binIndices == b
            if np.sum(inBin) > 10:
                groundProfile[b] = np.percentile(zs[inBin], 5)

        estimatedGroundZ = groundProfile[binIndices]
        heightAboveGround = zs - estimatedGroundZ

        predLabels = np.full(N, DRDOTargetClass.STATIC_OBSTACLE.value, dtype=np.uint8)
        predProbs = np.zeros((N, 3), dtype=np.float32)

        # Terrain Mask
        terrainMask = heightAboveGround < self.groundHeightThreshold
        predLabels[terrainMask] = DRDOTargetClass.TERRAIN.value

        # Dynamic Candidate Mask (road corridor)
        nonGround = ~terrainMask
        isDynamicCand = (
            nonGround &
            (np.abs(ys) < 6.0) &
            (heightAboveGround >= 0.25) &
            (heightAboveGround <= 2.2) &
            (dists2D < 55.0)
        )
        predLabels[isDynamicCand] = DRDOTargetClass.DYNAMIC_OBSTACLE.value

        # Confidence Estimation
        predProbs[terrainMask, 0] = np.clip(1.0 - (heightAboveGround[terrainMask] / self.groundHeightThreshold), 0.55, 0.98)
        predProbs[~terrainMask, 0] = 0.05

        predProbs[isDynamicCand, 2] = 0.88
        predProbs[~isDynamicCand, 2] = 0.08

        staticMask = nonGround & (~isDynamicCand)
        predProbs[staticMask, 1] = 0.82
        predProbs[~staticMask, 1] = 0.10

        probSum = np.maximum(np.sum(predProbs, axis=1, keepdims=True), 1e-6)
        predProbs = predProbs / probSum

        # Confidence and Uncertainty
        confidence = np.max(predProbs, axis=1).astype(np.float32)
        # Normalized Shannon entropy: H = -sum(p * log(p)) / log(3)
        entropy = -np.sum(predProbs * np.log(np.maximum(predProbs, 1e-6)), axis=1) / np.log(3.0)
        uncertainty = np.clip(entropy, 0.0, 1.0).astype(np.float32)

        latencyMs = (time.perf_counter() - startTime) * 1000.0

        return SegmentationResult(
            labels=predLabels,
            confidence=confidence,
            uncertainty=uncertainty,
            probabilities=predProbs,
            latencyMs=latencyMs,
        )
