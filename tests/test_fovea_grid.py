"""
Unit tests for Fovea-LiDAR Adaptive Variable-Resolution Representation.
Rigorous verification for:
  - Exact point conservation (zero dropped points, zero duplicate points)
  - Seamless boundary consistency (zero gaps, zero overlaps across 10m, 30m, 60m bands)
  - Correct multi-layer elevation aggregation (zMin <= zMean <= zMax)
  - O(1) world coordinate spatial query accuracy
"""

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from core.point_cloud import PointCloud
from core.grid_fovea import FoveaLiDARGrid, DEFAULT_FOVEA_BANDS


class TestFoveaLiDARGrid(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = FoveaLiDARGrid(bands=DEFAULT_FOVEA_BANDS, maxRangeMeters=100.0)

    def testPointConservationInvariant(self) -> None:
        """
        Verify that 100% of valid points within range are accounted for:
        sum(cell.pointCount) == totalPoints inside max range.
        Zero points dropped, zero points duplicated.
        """
        rng = np.random.default_rng(123)
        # Generate 5,000 points scattered between 0 and 95 meters
        angles = rng.uniform(-np.pi, np.pi, 5000)
        radii = rng.uniform(0.5, 95.0, 5000)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        zs = rng.uniform(-2.0, 3.0, 5000)

        pts = np.column_stack([xs, ys, zs]).astype(np.float32)
        cloud = PointCloud(points=pts)

        self.grid.update(cloud)

        totalPointsInCells = sum(c.pointCount for c in self.grid.activeCells.values())
        self.assertEqual(
            totalPointsInCells,
            5000,
            f"Point conservation failed: expected 5000 points in cells, got {totalPointsInCells}!"
        )
        self.assertEqual(self.grid.lastPointsProcessed, 5000)
        self.assertEqual(self.grid.lastPointsDropped, 0)

    def testExactBoundaryTransitions(self) -> None:
        """
        Test points placed right on the band boundaries:
          - 9.999m (Band 0) vs 10.000m (Band 1)
          - 29.999m (Band 1) vs 30.000m (Band 2)
          - 59.999m (Band 2) vs 60.000m (Band 3)
        Must map to exactly one cell, with correct bandId and zero gaps/overlaps.
        """
        boundaryPoints = np.array([
            [9.999, 0.0, 0.0],   # Band 0
            [10.000, 0.0, 0.0],  # Band 1
            [0.0, 29.999, 0.0],  # Band 1
            [0.0, 30.000, 0.0],  # Band 2
            [-59.999, 0.0, 0.0], # Band 2
            [-60.000, 0.0, 0.0], # Band 3
        ], dtype=np.float32)

        cloud = PointCloud(points=boundaryPoints)
        self.grid.update(cloud)

        # 6 points must yield 6 occupied cells (since resolutions and coordinates differ)
        totalPointsInCells = sum(c.pointCount for c in self.grid.activeCells.values())
        self.assertEqual(totalPointsInCells, 6, "Boundary points were dropped or duplicated!")

        # Verify Band IDs
        cell_9_999 = self.grid.queryWorldCoordinate(9.999, 0.0)
        self.assertIsNotNone(cell_9_999)
        self.assertEqual(cell_9_999.bandId, 0)
        self.assertAlmostEqual(cell_9_999.resolution, 0.05)

        cell_10_000 = self.grid.queryWorldCoordinate(10.000, 0.0)
        self.assertIsNotNone(cell_10_000)
        self.assertEqual(cell_10_000.bandId, 1)
        self.assertAlmostEqual(cell_10_000.resolution, 0.10)

        cell_29_999 = self.grid.queryWorldCoordinate(0.0, 29.999)
        self.assertIsNotNone(cell_29_999)
        self.assertEqual(cell_29_999.bandId, 1)

        cell_30_000 = self.grid.queryWorldCoordinate(0.0, 30.000)
        self.assertIsNotNone(cell_30_000)
        self.assertEqual(cell_30_000.bandId, 2)
        self.assertAlmostEqual(cell_30_000.resolution, 0.25)

        cell_59_999 = self.grid.queryWorldCoordinate(-59.999, 0.0)
        self.assertIsNotNone(cell_59_999)
        self.assertEqual(cell_59_999.bandId, 2)

        cell_60_000 = self.grid.queryWorldCoordinate(-60.000, 0.0)
        self.assertIsNotNone(cell_60_000)
        self.assertEqual(cell_60_000.bandId, 3)
        self.assertAlmostEqual(cell_60_000.resolution, 0.50)

    def testElevationAggregationInvariants(self) -> None:
        """Verify zMin <= zMean <= zMax and heightDiff >= 0 for all cells."""
        rng = np.random.default_rng(456)
        pts = rng.uniform(-40.0, 40.0, (2000, 3)).astype(np.float32)
        cloud = PointCloud(points=pts)
        self.grid.update(cloud)

        for cell in self.grid.activeCells.values():
            self.assertLessEqual(cell.zMin, cell.zMean + 1e-5)
            self.assertLessEqual(cell.zMean, cell.zMax + 1e-5)
            self.assertAlmostEqual(cell.heightDiff, cell.zMax - cell.zMin, places=5)
            self.assertGreaterEqual(cell.heightDiff, 0.0)

    def testSpatialContainment(self) -> None:
        """Verify that point is physically inside its queried cell boundaries."""
        testPt = np.array([[15.42, -8.73, 1.25]], dtype=np.float32)
        cloud = PointCloud(points=testPt)
        self.grid.update(cloud)

        cell = self.grid.queryWorldCoordinate(15.42, -8.73)
        self.assertIsNotNone(cell)
        halfRes = cell.resolution / 2.0
        self.assertTrue(cell.centerX - halfRes <= 15.42 <= cell.centerX + halfRes + 1e-5)
        self.assertTrue(cell.centerY - halfRes <= -8.73 <= cell.centerY + halfRes + 1e-5)


if __name__ == "__main__":
    unittest.main()
