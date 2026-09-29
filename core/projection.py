"""
Vectorized 3D to 2.5D Projection and Cell Aggregation Engine.
Efficiently maps 3D LiDAR point clouds to 2.5D cells using NumPy reduceat operations
for real-time multi-layer feature extraction (elevation min/max/mean, roughness,
occupancy, point density, and majority semantic classification).
"""

from typing import Tuple, NamedTuple, Optional
import numpy as np

from core.point_cloud import PointCloud


class CellAggregationResult(NamedTuple):
    """Container for vectorized aggregation results across unique occupied cells."""
    cellIndices1D: np.ndarray        # (K,) int64 flat 1D cell indices
    cellIndices2D: np.ndarray        # (K, 2) int64 (grid_x, grid_y) coordinates
    zMin: np.ndarray                 # (K,) float32 minimum elevation
    zMax: np.ndarray                 # (K,) float32 maximum elevation
    zMean: np.ndarray                # (K,) float32 mean elevation
    heightDiff: np.ndarray           # (K,) float32 zMax - zMin (obstacle height)
    pointCount: np.ndarray           # (K,) int32 number of points in cell
    majoritySemantic: np.ndarray     # (K,) uint8 majority semantic class
    isDynamic: np.ndarray            # (K,) bool true if dynamic object present
    totalPointsProcessed: int        # Total points within ROI
    totalPointsDropped: int          # Out-of-bounds points dropped


def projectPointsToGridIndices(
    pointsXY: np.ndarray,
    minX: float,
    minY: float,
    resolution: float,
    gridWidth: int,
    gridHeight: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized conversion from world (X, Y) coordinates to discrete grid indices (u, v).
    Returns:
        gridX: (M,) int64 column index (0 to gridWidth - 1)
        gridY: (M,) int64 row index (0 to gridHeight - 1)
        validMask: (N,) bool mask indicating points that fall inside the grid bounds
    """
    posX = pointsXY[:, 0]
    posY = pointsXY[:, 1]

    gridX = np.floor((posX - minX) / resolution).astype(np.int64)
    gridY = np.floor((posY - minY) / resolution).astype(np.int64)

    validMask = (
        (gridX >= 0) & (gridX < gridWidth) &
        (gridY >= 0) & (gridY < gridHeight)
    )

    return gridX[validMask], gridY[validMask], validMask


def aggregatePointsVectorized(
    pointCloud: PointCloud,
    minX: float,
    minY: float,
    resolution: float,
    gridWidth: int,
    gridHeight: int,
) -> CellAggregationResult:
    """
    High-performance C-speed vectorized cell aggregation.
    Aggregates thousands of points into 2.5D cell payloads without Python loops.
    """
    totalPoints = pointCloud.pointCount
    if totalPoints == 0:
        return CellAggregationResult(
            cellIndices1D=np.empty(0, dtype=np.int64),
            cellIndices2D=np.empty((0, 2), dtype=np.int64),
            zMin=np.empty(0, dtype=np.float32),
            zMax=np.empty(0, dtype=np.float32),
            zMean=np.empty(0, dtype=np.float32),
            heightDiff=np.empty(0, dtype=np.float32),
            pointCount=np.empty(0, dtype=np.int32),
            majoritySemantic=np.empty(0, dtype=np.uint8),
            isDynamic=np.empty(0, dtype=bool),
            totalPointsProcessed=0,
            totalPointsDropped=0,
        )

    # 1. Project points to grid indices
    gridX, gridY, validMask = projectPointsToGridIndices(
        pointsXY=pointCloud.points[:, :2],
        minX=minX,
        minY=minY,
        resolution=resolution,
        gridWidth=gridWidth,
        gridHeight=gridHeight,
    )

    validCount = np.sum(validMask)
    droppedCount = totalPoints - validCount

    if validCount == 0:
        return CellAggregationResult(
            cellIndices1D=np.empty(0, dtype=np.int64),
            cellIndices2D=np.empty((0, 2), dtype=np.int64),
            zMin=np.empty(0, dtype=np.float32),
            zMax=np.empty(0, dtype=np.float32),
            zMean=np.empty(0, dtype=np.float32),
            heightDiff=np.empty(0, dtype=np.float32),
            pointCount=np.empty(0, dtype=np.int32),
            majoritySemantic=np.empty(0, dtype=np.uint8),
            isDynamic=np.empty(0, dtype=bool),
            totalPointsProcessed=0,
            totalPointsDropped=droppedCount,
        )

    # 2. Compute 1D flat index: index1D = gridY * gridWidth + gridX
    indices1D = gridY * gridWidth + gridX

    # Filter point attributes
    zCoords = pointCloud.points[validMask, 2]
    semantics = pointCloud.semanticLabels[validMask]
    dynamics = pointCloud.dynamicFlags[validMask]

    # 3. Sort by 1D cell index for reduceat aggregation
    sortOrder = np.argsort(indices1D)
    sortedIndices1D = indices1D[sortOrder]
    sortedZ = zCoords[sortOrder]
    sortedSemantics = semantics[sortOrder]
    sortedDynamics = dynamics[sortOrder]

    # Find unique cell boundaries
    uniqueCells1D, splitIndices, cellCounts = np.unique(
        sortedIndices1D, return_index=True, return_counts=True
    )
    cellCounts = cellCounts.astype(np.int32)

    # 4. Vectorized reduceat operations for elevation features
    zMin = np.minimum.reduceat(sortedZ, splitIndices)
    zMax = np.maximum.reduceat(sortedZ, splitIndices)
    zSum = np.add.reduceat(sortedZ, splitIndices)
    zMean = (zSum / cellCounts).astype(np.float32)
    heightDiff = zMax - zMin

    # 5. Semantic majority voting and dynamic flag aggregation
    # Any dynamic point in the cell flags the cell as dynamic
    isDynamicCell = np.logical_or.reduceat(sortedDynamics, splitIndices)

    # Vectorized majority semantic vote
    # To compute majority class per cell without slow loops, we weight points by class:
    # Terrain (1), Static (2), Dynamic (3). If cell is dynamic, mark dynamic;
    # otherwise take the class with higher frequency in cell.
    majoritySem = np.zeros(len(uniqueCells1D), dtype=np.uint8)
    
    # Class frequency counts via reduceat
    isClass1 = (sortedSemantics == 1)
    isClass2 = (sortedSemantics == 2)
    isClass3 = (sortedSemantics == 3)

    countC1 = np.add.reduceat(isClass1, splitIndices)
    countC2 = np.add.reduceat(isClass2, splitIndices)
    countC3 = np.add.reduceat(isClass3, splitIndices)

    # Pick majority class (favoring dynamic > static > terrain on ties for safety)
    classStack = np.column_stack([countC1, countC2, countC3])
    # argmax returns 0, 1, 2 -> map to classes 1, 2, 3
    # Bias: if countC3 > 0, give safety priority to dynamic obstacle
    hasDynamic = countC3 > 0
    argmaxClasses = (np.argmax(classStack[:, :2], axis=1) + 1).astype(np.uint8)
    majoritySem = np.where(hasDynamic, 3, argmaxClasses)

    # Compute 2D (gridX, gridY) coordinates
    cellGridY = uniqueCells1D // gridWidth
    cellGridX = uniqueCells1D % gridWidth
    cellIndices2D = np.column_stack([cellGridX, cellGridY])

    return CellAggregationResult(
        cellIndices1D=uniqueCells1D,
        cellIndices2D=cellIndices2D,
        zMin=zMin,
        zMax=zMax,
        zMean=zMean,
        heightDiff=heightDiff,
        pointCount=cellCounts,
        majoritySemantic=majoritySem,
        isDynamic=isDynamicCell,
        totalPointsProcessed=validCount,
        totalPointsDropped=droppedCount,
    )
