"""
Unit tests for TerraSight FastAPI Backend.
Verifies health, config, scenario listing, scenario execution, comparator, export, and file parsing.
"""

import unittest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.scenario_manager import ScenarioManager
from backend.comparator import runHeadToHeadComparison
from backend.point_cloud_parser import parsePointCloudBytes
from backend.app import app, _processPointCloudScan, RunOptions


class TestBackendAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.scenarioMgr = ScenarioManager()
        self.client = TestClient(app)

    def testHealthEndpoint(self) -> None:
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("hardware", data)
        self.assertIn("primaryDevice", data["hardware"])

    def testConfigGetAndUpdate(self) -> None:
        res = self.client.get("/api/config")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("bands", data)
        self.assertIn("riskRefinement", data)

        updateRes = self.client.post("/api/config", json={
            "weightDynamic": 0.45,
            "weightProximity": 0.25,
            "weightTraversability": 0.15,
            "weightUncertainty": 0.15,
            "refinementThreshold": 0.50,
        })
        self.assertEqual(updateRes.status_code, 200)
        self.assertEqual(updateRes.json()["status"], "success")

    def testScenarioManagerListing(self) -> None:
        res = self.client.get("/api/scenarios")
        self.assertEqual(res.status_code, 200)
        scenarios = res.json()
        self.assertGreaterEqual(len(scenarios), 4)
        scenarioIds = [s["id"] for s in scenarios]
        self.assertIn("urban_patrol", scenarioIds)
        self.assertIn("distant_threat", scenarioIds)
        self.assertIn("trench_obstacle", scenarioIds)

    def testScenarioExecutionEndpoint(self) -> None:
        res = self.client.post("/api/scenarios/distant_threat/run", json={"maxDisplayPoints": 5000, "enableRiskRefinement": True})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("summary", data)
        self.assertIn("points", data)
        self.assertIn("cells", data)
        self.assertIn("latency", data)
        self.assertGreater(data["summary"]["cellReductionPercent"], 90.0)

    def testSimulationEndpoint(self) -> None:
        res = self.client.post("/api/scenarios/distant_threat/simulate?numFrames=3")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["totalFrames"], 3)
        self.assertEqual(len(data["frames"]), 3)

    def testComparatorEndpoint(self) -> None:
        res = self.client.post("/api/comparator?scenarioId=distant_threat")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["models"]), 5)
        self.assertGreater(data["foveaCellReductionPercent"], 95.0)

    def testExportEndpoints(self) -> None:
        # Pre-populate state
        self.client.post("/api/scenarios/urban_patrol/run")

        jsonRes = self.client.get("/api/export/json")
        self.assertEqual(jsonRes.status_code, 200)
        self.assertIn("application/json", jsonRes.headers["content-type"])

        geoRes = self.client.get("/api/export/geojson")
        self.assertEqual(geoRes.status_code, 200)

        csvRes = self.client.get("/api/export/csv")
        self.assertEqual(csvRes.status_code, 200)
        self.assertIn("centerX,centerY", csvRes.text)

    def testFrontendSPARouting(self) -> None:
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("<!doctype html>", res.text)
        self.assertIn("TerraSight", res.text)

    def testParsePointCloudBytes(self) -> None:
        import numpy as np
        testPts = np.random.randn(100, 4).astype(np.float32)
        rawBytes = testPts.tobytes()

        parsed = parsePointCloudBytes("test.bin", rawBytes)
        self.assertEqual(parsed.pointCount, 100)
        self.assertEqual(parsed.points.shape, (100, 3))


if __name__ == "__main__":
    unittest.main()
