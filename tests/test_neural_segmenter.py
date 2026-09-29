"""
Unit tests for NeuralSegmenter and FoveaRangeNet deep learning pipeline.
"""

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from core.point_cloud import PointCloud
from data.synthetic_generator import generateSyntheticLiDARScan
from segmentation.neural_segmenter import NeuralSegmenter
from segmentation.class_mapping import DRDOTargetClass


class TestNeuralSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.scan = generateSyntheticLiDARScan(numRings=32, horizontalResolutionDeg=0.5, randomSeed=555)
        weightsPath = str(PROJECT_ROOT / "segmentation" / "weights" / "fovea_rangenet_v1.pt")
        self.segmenter = NeuralSegmenter(weightsPath=weightsPath)
        # Warmup GPU context
        for _ in range(3):
            self.segmenter.predict(self.scan)

    def testNeuralInferenceShapeAndOutputs(self) -> None:
        result = self.segmenter.predict(self.scan)
        N = self.scan.pointCount

        # Shapes
        self.assertEqual(len(result.labels), N)
        self.assertEqual(len(result.confidence), N)
        self.assertEqual(len(result.uncertainty), N)
        self.assertEqual(result.probabilities.shape, (N, 3))

        # Value ranges
        self.assertTrue(np.all(result.confidence >= 0.0))
        self.assertTrue(np.all(result.confidence <= 1.0 + 1e-5))
        self.assertTrue(np.all(result.uncertainty >= 0.0))
        self.assertTrue(np.all(result.uncertainty <= 1.0 + 1e-5))

        # Probability sums
        probSums = np.sum(result.probabilities, axis=1)
        np.testing.assert_allclose(probSums, 1.0, atol=1e-4)

        # Classes in 0, 1, 2
        uniquePredClasses = np.unique(result.labels)
        for c in uniquePredClasses:
            self.assertIn(c, [DRDOTargetClass.TERRAIN.value, DRDOTargetClass.STATIC_OBSTACLE.value, DRDOTargetClass.DYNAMIC_OBSTACLE.value])

    def testEvaluationMetrics(self) -> None:
        for _ in range(2):
            self.segmenter.predict(self.scan)
        result = self.segmenter.predict(self.scan)
        metrics = self.segmenter.evaluate(self.scan, result)

        self.assertIn("mIoU", metrics)
        self.assertIn("overallAccuracy", metrics)
        self.assertIn("perClassIoU", metrics)
        self.assertGreater(metrics["mIoU"], 0.0)
        self.assertGreater(metrics["overallAccuracy"], 0.0)
        self.assertGreater(metrics["throughputFps"], 10.0)


if __name__ == "__main__":
    unittest.main()
