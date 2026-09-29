"""
Abstract Base Class for LiDAR Semantic Segmenters.
Defines standard interface for both Geometric and Deep Learning segmentation stages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from core.point_cloud import PointCloud
from segmentation.class_mapping import DRDOTargetClass, DRDO_CLASS_NAMES


@dataclass
class SegmentationResult:
    """Standardized output container across all segmenter implementations."""
    labels: np.ndarray             # (N,) uint8: 0=Terrain, 1=Static, 2=Dynamic
    confidence: np.ndarray         # (N,) float32 in [0.0, 1.0]
    uncertainty: np.ndarray        # (N,) float32 in [0.0, 1.0] (1 - confidence or entropy)
    probabilities: Optional[np.ndarray] = None  # (N, 3) float32 class distribution
    latencyMs: float = 0.0         # Inference latency in milliseconds


class BaseSegmenter(ABC):
    """Abstract interface that all perception segmentation models must implement."""

    @abstractmethod
    def predict(self, pointCloud: PointCloud) -> SegmentationResult:
        """
        Takes raw PointCloud and returns point-level predictions, confidence, and uncertainty.
        """
        pass

    def evaluate(self, pointCloud: PointCloud, result: SegmentationResult) -> Dict[str, Any]:
        """
        Computes standard classification metrics (mIoU, per-class IoU, accuracy)
        against pointCloud.semanticLabels.
        """
        gt = pointCloud.semanticLabels
        pred = result.labels

        # Support both 0-indexed (0, 1, 2) and 1-indexed (1, 2, 3) ground truth
        if np.max(gt) == 3 and np.min(gt[gt > 0]) == 1:
            # Map legacy 1, 2, 3 -> 0, 1, 2
            evalGt = np.where(gt > 0, gt - 1, 255)
        else:
            evalGt = gt

        validMask = evalGt != DRDOTargetClass.IGNORED.value
        gtValid = evalGt[validMask]
        predValid = pred[validMask]

        classes = [
            (DRDOTargetClass.TERRAIN.value, "Terrain"),
            (DRDOTargetClass.STATIC_OBSTACLE.value, "Static"),
            (DRDOTargetClass.DYNAMIC_OBSTACLE.value, "Dynamic"),
        ]

        perClassIoU = {}
        perClassAcc = {}
        perClassRecall = {}

        for cVal, cName in classes:
            tp = int(np.sum((gtValid == cVal) & (predValid == cVal)))
            fp = int(np.sum((gtValid != cVal) & (predValid == cVal)))
            fn = int(np.sum((gtValid == cVal) & (predValid != cVal)))

            iou = tp / max(1, (tp + fp + fn))
            acc = tp / max(1, (tp + fn))  # Recall
            precision = tp / max(1, (tp + fp))

            perClassIoU[cName] = float(iou * 100.0)
            perClassRecall[cName] = float(acc * 100.0)
            perClassAcc[cName] = float(precision * 100.0)

        mIoU = float(np.mean(list(perClassIoU.values())))
        overallAcc = float(np.sum(gtValid == predValid) / max(1, len(gtValid)) * 100.0)

        # Distance-stratified accuracy
        dist2D = pointCloud.calculateDistances2D()[validMask]
        bands = [
            ("0-10m", (dist2D >= 0) & (dist2D < 10)),
            ("10-30m", (dist2D >= 10) & (dist2D < 30)),
            ("30-60m", (dist2D >= 30) & (dist2D < 60)),
            ("60-100m", (dist2D >= 60) & (dist2D <= 100)),
        ]

        distAcc = {}
        for bName, bMask in bands:
            if np.sum(bMask) > 0:
                accB = np.sum(gtValid[bMask] == predValid[bMask]) / np.sum(bMask) * 100.0
                distAcc[bName] = float(accB)
            else:
                distAcc[bName] = 0.0

        return {
            "mIoU": mIoU,
            "overallAccuracy": overallAcc,
            "perClassIoU": perClassIoU,
            "perClassRecall": perClassRecall,
            "perClassPrecision": perClassAcc,
            "distanceStratifiedAccuracy": distAcc,
            "inferenceLatencyMs": result.latencyMs,
            "throughputFps": 1000.0 / max(0.001, result.latencyMs),
        }
