"""
Precision Boundary Transition Tests for Fovea-LiDAR.
Verifies micro-meter transitions across all band boundaries:
  - 9.999m, 10.000m, 10.001m (Band 0 <-> Band 1)
  - 29.999m, 30.000m, 30.001m (Band 1 <-> Band 2)
  - 59.999m, 60.000m, 60.001m (Band 2 <-> Band 3)
  - 99.999m, 100.000m, 100.001m (Band 3 <-> Out-of-bounds)
Asserts zero gaps, zero overlaps, and exact cell resolution assignments.
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


class TestBoundaryTransitions(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = FoveaLiDARGrid(bands=DEFAULT_FOVEA_BANDS, maxRangeMeters=100.0, enableRiskRefinement=False)

    def testMicroBoundaryTransitions(self) -> None:
        testDistances = [
            (9.999, 0, 0.05),     # Inside Band 0 (5cm)
            (10.000, 1, 0.10),    # Exactly start of Band 1 (10cm)
            (10.001, 1, 0.10),    # Inside Band 1 (10cm)
            (29.999, 1, 0.10),    # Inside Band 1 (10cm)
            (30.000, 2, 0.25),    # Exactly start of Band 2 (25cm)
            (30.001, 2, 0.25),    # Inside Band 2 (25cm)
            (59.999, 2, 0.25),    # Inside Band 2 (25cm)
            (60.000, 3, 0.50),    # Exactly start of Band 3 (50cm)
            (60.001, 3, 0.50),    # Inside Band 3 (50cm)
            (99.999, 3, 0.50),    # Inside Band 3 (50cm)
            (100.000, 3, 0.50),   # Exactly outer perimeter of Band 3 (50cm)
        ]

        # Place points along the positive X axis
        points = np.array([[d, 0.0, 0.0] for d, _, _ in testDistances], dtype=np.float32)
        cloud = PointCloud(points=points)

        self.grid.update(cloud)

        # Verify exact point conservation: all 11 boundary points must be assigned
        totalPointsInCells = sum(bd.pointCount.sum() for bd in self.grid.bandData.values())
        self.assertEqual(totalPointsInCells, len(testDistances), "Boundary points were dropped or duplicated!")

        # Verify each point maps to its mathematically expected band and resolution
        for distVal, expectedBand, expectedRes in testDistances:
            cell = self.grid.queryWorldCoordinate(distVal, 0.0)
            self.assertIsNotNone(cell, f"Query at distance {distVal}m returned None!")
            self.assertEqual(
                cell.bandId, expectedBand,
                f"Distance {distVal}m assigned to Band {cell.bandId}, expected Band {expectedBand}!"
            )
            self.assertAlmostEqual(
                cell.resolution, expectedRes, places=4,
                msg=f"Distance {distVal}m resolution was {cell.resolution}m, expected {expectedRes}m!"
            )

    def testOutOfRangeDrop(self) -> None:
        """Points beyond 100.000m should be dropped with zero boundary violations."""
        outPt = np.array([[100.001, 0.0, 0.0]], dtype=np.float32)
        cloud = PointCloud(points=outPt)
        self.grid.update(cloud)

        self.assertEqual(self.grid.lastPointsProcessed, 0)
        self.assertEqual(self.grid.lastPointsDropped, 1)
        self.assertIsNone(self.grid.queryWorldCoordinate(100.001, 0.0))


if __name__ == "__main__":
    unittest.main()
