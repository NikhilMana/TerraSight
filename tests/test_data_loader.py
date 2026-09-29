"""
Unit tests for data loaders and point cloud representations.
"""

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.point_cloud import PointCloud
from data.synthetic_generator import generateSyntheticLiDARScan, saveScanAsSemanticKitti
from data.kitti_loader import loadSemanticKittiScan
from segmentation.kitti_classes import TargetClass


class TestDataLoader(unittest.TestCase):
    def testSyntheticScanGeneration(self) -> None:
        scan = generateSyntheticLiDARScan(numRings=32, horizontalResolutionDeg=0.5, randomSeed=101)
        self.assertGreater(scan.pointCount, 5000)
        self.assertIn(TargetClass.TERRAIN.value, scan.semanticLabels)
        self.assertIn(TargetClass.STATIC_OBSTACLE.value, scan.semanticLabels)
        self.assertIn(TargetClass.DYNAMIC_OBSTACLE.value, scan.semanticLabels)

    def testFiltering(self) -> None:
        scan = generateSyntheticLiDARScan(numRings=32, horizontalResolutionDeg=0.5, randomSeed=101)
        filtered = scan.filterByDistance(minDistance=5.0, maxDistance=20.0)
        dists = filtered.calculateDistances2D()
        self.assertTrue(np.all(dists >= 5.0))
        self.assertTrue(np.all(dists <= 20.0))

    def testBinaryRoundTrip(self) -> None:
        scan = generateSyntheticLiDARScan(numRings=32, horizontalResolutionDeg=0.5, randomSeed=101)
        tmpBin = Path("data/sample_scans/test_temp.bin")
        tmpLabel = Path("data/sample_scans/test_temp.label")
        try:
            saveScanAsSemanticKitti(scan, tmpBin, tmpLabel)
            loaded = loadSemanticKittiScan(tmpBin, tmpLabel)
            self.assertEqual(loaded.pointCount, scan.pointCount)
            np.testing.assert_allclose(loaded.points, scan.points, atol=1e-5)
            np.testing.assert_array_equal(loaded.semanticLabels, scan.semanticLabels)
        finally:
            if tmpBin.exists():
                tmpBin.unlink()
            if tmpLabel.exists():
                tmpLabel.unlink()


if __name__ == "__main__":
    unittest.main()
