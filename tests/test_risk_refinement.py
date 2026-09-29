"""
Unit test for Risk-Aware Local Refinement in Fovea-LiDAR.
Verifies that:
  1. Distant dynamic objects are selectively promoted to 5 cm resolution.
  2. Non-risk background points remain at coarse base resolution.
  3. Total point conservation remains strictly 100.0%.
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
from data.synthetic_generator import generateSyntheticLiDARScan


class TestRiskRefinement(unittest.TestCase):
    def testDistantDynamicRefinement(self) -> None:
        scan = generateSyntheticLiDARScan(numRings=32, horizontalResolutionDeg=0.5, randomSeed=789)

        # 1. Distance-only grid (no risk refinement)
        gridDistanceOnly = FoveaLiDARGrid(
            bands=DEFAULT_FOVEA_BANDS,
            enableRiskRefinement=False,
        )
        gridDistanceOnly.update(scan)
        self.assertNotIn(99, gridDistanceOnly.bandData)

        # 2. Risk-aware grid (with risk refinement)
        gridRiskAware = FoveaLiDARGrid(
            bands=DEFAULT_FOVEA_BANDS,
            enableRiskRefinement=True,
            refinedResolution=0.05,
        )
        gridRiskAware.update(scan)

        # Must have refined cells in band 99
        self.assertIn(99, gridRiskAware.bandData)
        refinedCellsCount = len(gridRiskAware.bandData[99].packedKeys)
        self.assertGreater(refinedCellsCount, 0, "No cells were refined despite dynamic actors!")

        # Dynamic vehicle points in Band 1 or Band 2 (distance > 15m)
        dist2D = scan.calculateDistances2D()
        dynamicFarMask = scan.dynamicFlags & (dist2D >= 15.0)
        self.assertTrue(np.any(dynamicFarMask), "No distant dynamic points generated!")

        dynPt = scan.points[dynamicFarMask][0]
        cell = gridRiskAware.queryWorldCoordinate(dynPt[0], dynPt[1])
        self.assertIsNotNone(cell, f"Query at dynamic point ({dynPt[0]:.2f}, {dynPt[1]:.2f}) returned None!")
        self.assertEqual(cell.bandId, 99)
        self.assertAlmostEqual(cell.resolution, 0.05)  # Refined down to 5 cm!
        self.assertTrue(cell.isDynamic)

        # Ground terrain point at distance > 15m without dynamic flag
        groundFarMask = (~scan.dynamicFlags) & (scan.semanticLabels == 1) & (dist2D >= 15.0) & (dist2D < 30.0)
        if np.any(groundFarMask):
            gPt = scan.points[groundFarMask][0]
            groundCell = gridRiskAware.queryWorldCoordinate(gPt[0], gPt[1])
            if groundCell is not None:
                self.assertEqual(groundCell.bandId, 1)
                self.assertAlmostEqual(groundCell.resolution, 0.10)  # Coarse 10 cm!

        # 3. Point conservation invariant check
        totalPointsInCells = sum(bd.pointCount.sum() for bd in gridRiskAware.bandData.values())
        self.assertEqual(totalPointsInCells, gridRiskAware.lastPointsProcessed)


if __name__ == "__main__":
    unittest.main()
