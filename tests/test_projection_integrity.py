"""
Projection Integrity and Conservation Verification for Fovea-LiDAR.
Rigorous assertion that for any point cloud within valid sensor range:
  Input Points: N
  Assigned Points: N
  Dropped Points: 0
  Duplicate Assignments: 0
  Boundary Violations: 0
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


class TestProjectionIntegrity(unittest.TestCase):
    def testCompletePointConservation(self) -> None:
        """
        Tests point conservation across multiple random seeds and scan densities.
        """
        for seed in [111, 222, 333]:
            with self.subTest(seed=seed):
                scan = generateSyntheticLiDARScan(numRings=32, horizontalResolutionDeg=0.5, randomSeed=seed)
                # Filter points strictly to 0.5m - 99.0m to test inside-ROI conservation
                scanInside = scan.filterByDistance(minDistance=0.5, maxDistance=99.0)
                N = scanInside.pointCount

                grid = FoveaLiDARGrid(bands=DEFAULT_FOVEA_BANDS, maxRangeMeters=100.0, enableRiskRefinement=True)
                grid.update(scanInside)

                # Total points counted across all active cells
                assignedPoints = sum(bd.pointCount.sum() for bd in grid.bandData.values())
                droppedPoints = grid.lastPointsDropped

                # Globally unique cell key check: (bandId, packedKey)
                allKeys = []
                for bd in grid.bandData.values():
                    for k in bd.packedKeys:
                        allKeys.append((bd.bandId, int(k)))
                uniqueKeyCount = len(set(allKeys))

                # Assertions
                self.assertEqual(assignedPoints, N, f"Assigned {assignedPoints} != Input {N}")
                self.assertEqual(droppedPoints, 0, f"Dropped {droppedPoints} points unexpectedly!")
                self.assertEqual(len(allKeys), uniqueKeyCount, "Duplicate cell assignments detected!")

    def testDuplicateAssignmentAbsence(self) -> None:
        """
        Explicitly check that points lying close together or on band boundaries
        never result in double-counting.
        """
        # Clusters of points clustered tightly around band boundaries
        pts = np.array([
            [9.9999, 0.0, 0.0],
            [10.0001, 0.0, 0.0],
            [29.9999, 0.0, 0.0],
            [30.0001, 0.0, 0.0],
            [59.9999, 0.0, 0.0],
            [60.0001, 0.0, 0.0],
        ], dtype=np.float32)

        cloud = PointCloud(points=pts)
        grid = FoveaLiDARGrid(bands=DEFAULT_FOVEA_BANDS, maxRangeMeters=100.0)
        grid.update(cloud)

        assignedPoints = sum(bd.pointCount.sum() for bd in grid.bandData.values())
        self.assertEqual(assignedPoints, 6, f"Expected 6 assigned points, got {assignedPoints}!")
        self.assertEqual(grid.lastPointsDropped, 0)


if __name__ == "__main__":
    unittest.main()
