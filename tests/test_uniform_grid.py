"""
Unit tests for 3D to 2.5D Projection and Baseline Uniform Elevation Grid.
"""

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from core.point_cloud import PointCloud
from core.projection import aggregatePointsVectorized, projectPointsToGridIndices
from core.grid_uniform import UniformElevationGrid


class TestUniformGrid(unittest.TestCase):
    def setUp(self) -> None:
        # Create deterministic synthetic test points
        points = np.array([
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],    # 3 points in same cell (1.0, 1.0)
            [5.0, 5.0, 2.5],    # 1 point in cell (5.0, 5.0)
            [200.0, 200.0, 0.0] # 1 out-of-bounds point
        ], dtype=np.float32)

        semantics = np.array([1, 1, 3, 2, 1], dtype=np.uint32)
        dynamics = np.array([False, False, True, False, False], dtype=bool)

        self.testCloud = PointCloud(
            points=points,
            semanticLabels=semantics,
            dynamicFlags=dynamics,
        )

    def testProjectionIndices(self) -> None:
        gx, gy, validMask = projectPointsToGridIndices(
            pointsXY=self.testCloud.points[:, :2],
            minX=-10.0,
            minY=-10.0,
            resolution=1.0,
            gridWidth=20,
            gridHeight=20,
        )
        self.assertEqual(np.sum(validMask), 4)
        self.assertFalse(validMask[4])  # 200, 200 is out-of-bounds

    def testCellAggregationAccuracy(self) -> None:
        grid = UniformElevationGrid(resolution=1.0, minX=-10.0, maxX=10.0, minY=-10.0, maxY=10.0)
        result = grid.update(self.testCloud)

        self.assertEqual(result.totalPointsProcessed, 4)
        self.assertEqual(result.totalPointsDropped, 1)

        # Cell at (1.0, 1.0) has 3 points with z = -1.0, 0.0, 1.0
        cellInfo = grid.queryWorldCoordinate(1.05, 1.05)
        self.assertIsNotNone(cellInfo)
        self.assertTrue(cellInfo["occupied"])
        self.assertEqual(cellInfo["pointCount"], 3)
        self.assertAlmostEqual(cellInfo["zMin"], -1.0, places=4)
        self.assertAlmostEqual(cellInfo["zMax"], 1.0, places=4)
        self.assertAlmostEqual(cellInfo["zMean"], 0.0, places=4)
        self.assertAlmostEqual(cellInfo["heightDiff"], 2.0, places=4)
        self.assertTrue(cellInfo["isDynamic"])
        self.assertEqual(cellInfo["semanticClass"], 3)

    def testElevationOrderingInvariant(self) -> None:
        grid = UniformElevationGrid(resolution=0.1, minX=-20.0, maxX=20.0, minY=-20.0, maxY=20.0)
        # Generate random points
        rng = np.random.default_rng(42)
        randPts = rng.uniform(-15.0, 15.0, (1000, 3)).astype(np.float32)
        cloud = PointCloud(points=randPts)

        grid.update(cloud)
        occMask = grid.isOccupied
        zMinVals = grid.elevationMin[occMask]
        zMaxVals = grid.elevationMax[occMask]
        zMeanVals = grid.elevationMean[occMask]

        # Assert mathematical invariant: zMin <= zMean <= zMax everywhere
        self.assertTrue(np.all(zMinVals <= zMeanVals + 1e-5))
        self.assertTrue(np.all(zMeanVals <= zMaxVals + 1e-5))


if __name__ == "__main__":
    unittest.main()
