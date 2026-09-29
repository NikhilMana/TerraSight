"""
Baseline Uniform-Resolution 2.5D Elevation Grid.
Maintains dense multi-layer arrays representing elevation (min/max/mean),
height difference (obstacle height), occupancy, point count, and semantic class
at a fixed uniform spatial resolution (e.g., 5 cm or 10 cm).
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np

from core.point_cloud import PointCloud
from core.projection import aggregatePointsVectorized, CellAggregationResult
from segmentation.kitti_classes import TargetClass, TARGET_CLASS_COLORS


class UniformElevationGrid:
    """
    Dense 2.5D Elevation Grid with uniform cell resolution across the entire ROI.
    Used as the rigorous baseline to benchmark against Fovea-LiDAR.
    """

    def __init__(
        self,
        resolution: float = 0.05,  # 5 cm cell size
        minX: float = -50.0,
        maxX: float = 50.0,
        minY: float = -50.0,
        maxY: float = 50.0,
    ) -> None:
        self.resolution = float(resolution)
        self.minX = float(minX)
        self.maxX = float(maxX)
        self.minY = float(minY)
        self.maxY = float(maxY)

        self.widthMeters = self.maxX - self.minX
        self.heightMeters = self.maxY - self.minY

        self.widthCells = int(np.ceil(self.widthMeters / self.resolution))
        self.heightCells = int(np.ceil(self.heightMeters / self.resolution))
        self.totalCells = self.widthCells * self.heightCells

        # Multi-layer grid arrays
        self.elevationMin = np.full((self.heightCells, self.widthCells), np.nan, dtype=np.float32)
        self.elevationMax = np.full((self.heightCells, self.widthCells), np.nan, dtype=np.float32)
        self.elevationMean = np.full((self.heightCells, self.widthCells), np.nan, dtype=np.float32)
        self.heightDiff = np.zeros((self.heightCells, self.widthCells), dtype=np.float32)
        self.pointCount = np.zeros((self.heightCells, self.widthCells), dtype=np.int32)
        self.semanticClass = np.zeros((self.heightCells, self.widthCells), dtype=np.uint8)
        self.isDynamic = np.zeros((self.heightCells, self.widthCells), dtype=bool)
        self.isOccupied = np.zeros((self.heightCells, self.widthCells), dtype=bool)

        # Performance tracking
        self.lastUpdateLatencyMs: float = 0.0
        self.lastPointsProcessed: int = 0
        self.lastPointsDropped: int = 0

    def clear(self) -> None:
        """Reset all grid layers to unoccupied states."""
        self.elevationMin.fill(np.nan)
        self.elevationMax.fill(np.nan)
        self.elevationMean.fill(np.nan)
        self.heightDiff.fill(0.0)
        self.pointCount.fill(0)
        self.semanticClass.fill(0)
        self.isDynamic.fill(False)
        self.isOccupied.fill(False)

    def update(self, pointCloud: PointCloud) -> CellAggregationResult:
        """
        Populate/update the 2.5D grid layers from a raw PointCloud.
        Measures exact mapping latency in milliseconds.
        """
        startTime = time.perf_counter()

        result = aggregatePointsVectorized(
            pointCloud=pointCloud,
            minX=self.minX,
            minY=self.minY,
            resolution=self.resolution,
            gridWidth=self.widthCells,
            gridHeight=self.heightCells,
        )

        if len(result.cellIndices1D) > 0:
            gx = result.cellIndices2D[:, 0]
            gy = result.cellIndices2D[:, 1]

            # Vectorized assignment into dense layers
            self.elevationMin[gy, gx] = result.zMin
            self.elevationMax[gy, gx] = result.zMax
            self.elevationMean[gy, gx] = result.zMean
            self.heightDiff[gy, gx] = result.heightDiff
            self.pointCount[gy, gx] = result.pointCount
            self.semanticClass[gy, gx] = result.majoritySemantic
            self.isDynamic[gy, gx] = result.isDynamic
            self.isOccupied[gy, gx] = True

        elapsedMs = (time.perf_counter() - startTime) * 1000.0
        self.lastUpdateLatencyMs = elapsedMs
        self.lastPointsProcessed = result.totalPointsProcessed
        self.lastPointsDropped = result.totalPointsDropped

        return result

    def getOccupiedCellCount(self) -> int:
        """Number of active cells with LiDAR points."""
        return int(np.sum(self.isOccupied))

    def getMemoryUsageBytes(self) -> int:
        """Calculate total memory consumed by dense layer buffers in bytes."""
        totalBytes = (
            self.elevationMin.nbytes +
            self.elevationMax.nbytes +
            self.elevationMean.nbytes +
            self.heightDiff.nbytes +
            self.pointCount.nbytes +
            self.semanticClass.nbytes +
            self.isDynamic.nbytes +
            self.isOccupied.nbytes
        )
        return totalBytes

    def getMemoryUsageMb(self) -> float:
        """Calculate memory in Megabytes (MB)."""
        return self.getMemoryUsageBytes() / (1024.0 * 1024.0)

    def queryWorldCoordinate(self, worldX: float, worldY: float) -> Optional[Dict[str, Any]]:
        """Query 2.5D cell attributes at a given world (X, Y) coordinate."""
        if not (self.minX <= worldX < self.maxX and self.minY <= worldY < self.maxY):
            return None

        gx = int(np.floor((worldX - self.minX) / self.resolution))
        gy = int(np.floor((worldY - self.minY) / self.resolution))

        if not self.isOccupied[gy, gx]:
            return {
                "occupied": False,
                "worldX": worldX,
                "worldY": worldY,
                "gridX": gx,
                "gridY": gy,
            }

        return {
            "occupied": True,
            "worldX": worldX,
            "worldY": worldY,
            "gridX": gx,
            "gridY": gy,
            "zMin": float(self.elevationMin[gy, gx]),
            "zMax": float(self.elevationMax[gy, gx]),
            "zMean": float(self.elevationMean[gy, gx]),
            "heightDiff": float(self.heightDiff[gy, gx]),
            "pointCount": int(self.pointCount[gy, gx]),
            "semanticClass": int(self.semanticClass[gy, gx]),
            "isDynamic": bool(self.isDynamic[gy, gx]),
        }

    def getMetrics(self) -> Dict[str, Any]:
        """Summary metrics dictionary for benchmarking."""
        occupiedCells = self.getOccupiedCellCount()
        sparsity = (1.0 - (occupiedCells / max(1, self.totalCells))) * 100.0
        fps = 1000.0 / max(0.001, self.lastUpdateLatencyMs)

        return {
            "gridType": str(f"Uniform_{int(self.resolution * 100)}cm"),
            "resolutionMeters": float(self.resolution),
            "dimensionsMeters": [float(self.widthMeters), float(self.heightMeters)],
            "gridShape": [int(self.heightCells), int(self.widthCells)],
            "totalAllocatedCells": int(self.totalCells),
            "occupiedCells": int(occupiedCells),
            "sparsityPercent": float(sparsity),
            "memoryUsageMb": float(self.getMemoryUsageMb()),
            "latencyMs": float(self.lastUpdateLatencyMs),
            "fps": float(fps),
            "pointsProcessed": int(self.lastPointsProcessed),
            "pointsDropped": int(self.lastPointsDropped),
            "retentionRatePercent": float(
                (self.lastPointsProcessed / max(1, self.lastPointsProcessed + self.lastPointsDropped)) * 100.0
            ),
        }
