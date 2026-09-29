"""
Fovea-LiDAR Point Cloud Data Container
Defines the core PointCloud class holding 3D spatial coordinates,
intensity, semantic labels, dynamic status, and bounding volumes.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class PointCloud:
    points: np.ndarray  # Shape: (N, 3), float32 (x, y, z)
    intensity: Optional[np.ndarray] = None  # Shape: (N,), float32
    semanticLabels: Optional[np.ndarray] = None  # Shape: (N,), uint32 (0: unlabeled)
    instanceIds: Optional[np.ndarray] = None  # Shape: (N,), uint32
    dynamicFlags: Optional[np.ndarray] = None  # Shape: (N,), bool (True: dynamic)
    confidence: Optional[np.ndarray] = None  # Shape: (N,), float32
    uncertainty: Optional[np.ndarray] = None  # Shape: (N,), float32 (entropy/1-conf)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float32)
        pointCount = len(self.points)

        if self.intensity is None:
            self.intensity = np.ones(pointCount, dtype=np.float32)
        else:
            self.intensity = np.ascontiguousarray(self.intensity, dtype=np.float32)

        if self.semanticLabels is None:
            self.semanticLabels = np.zeros(pointCount, dtype=np.uint32)
        else:
            self.semanticLabels = np.ascontiguousarray(self.semanticLabels, dtype=np.uint32)

        if self.instanceIds is None:
            self.instanceIds = np.zeros(pointCount, dtype=np.uint32)
        else:
            self.instanceIds = np.ascontiguousarray(self.instanceIds, dtype=np.uint32)

        if self.dynamicFlags is None:
            self.dynamicFlags = np.zeros(pointCount, dtype=bool)
        else:
            self.dynamicFlags = np.ascontiguousarray(self.dynamicFlags, dtype=bool)

        if self.confidence is not None:
            self.confidence = np.ascontiguousarray(self.confidence, dtype=np.float32)

        if self.uncertainty is not None:
            self.uncertainty = np.ascontiguousarray(self.uncertainty, dtype=np.float32)

    @property
    def pointCount(self) -> int:
        return len(self.points)

    def calculateDistances2D(self) -> np.ndarray:
        """Returns Euclidean distance in the XY ground plane: sqrt(x^2 + y^2)."""
        return np.linalg.norm(self.points[:, :2], axis=1)

    def calculateDistances3D(self) -> np.ndarray:
        """Returns 3D Euclidean distance from sensor origin: sqrt(x^2 + y^2 + z^2)."""
        return np.linalg.norm(self.points, axis=1)

    def filterByDistance(self, minDistance: float = 0.5, maxDistance: float = 100.0) -> "PointCloud":
        """Filter points based on 2D planar distance from sensor origin."""
        distances = self.calculateDistances2D()
        mask = (distances >= minDistance) & (distances <= maxDistance)
        return self._applyMask(mask)

    def filterByBoundingBox(
        self,
        minBound: Tuple[float, float, float] = (-50.0, -50.0, -5.0),
        maxBound: Tuple[float, float, float] = (50.0, 50.0, 5.0),
    ) -> "PointCloud":
        """Crop points inside a 3D Axis-Aligned Bounding Box (AABB)."""
        minB = np.array(minBound, dtype=np.float32)
        maxB = np.array(maxBound, dtype=np.float32)
        mask = np.all((self.points >= minB) & (self.points <= maxB), axis=1)
        return self._applyMask(mask)

    def _applyMask(self, mask: np.ndarray) -> "PointCloud":
        return PointCloud(
            points=self.points[mask],
            intensity=self.intensity[mask] if self.intensity is not None else None,
            semanticLabels=self.semanticLabels[mask] if self.semanticLabels is not None else None,
            instanceIds=self.instanceIds[mask] if self.instanceIds is not None else None,
            dynamicFlags=self.dynamicFlags[mask] if self.dynamicFlags is not None else None,
            confidence=self.confidence[mask] if self.confidence is not None else None,
            uncertainty=self.uncertainty[mask] if self.uncertainty is not None else None,
            timestamp=self.timestamp,
        )

    def getBoundingBox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (minCoords, maxCoords) across x, y, z."""
        if self.pointCount == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return self.points.min(axis=0), self.points.max(axis=0)
