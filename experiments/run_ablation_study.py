"""
Systematic Ablation Study for Fovea-LiDAR (DRDO SIH26053).
Evaluates the incremental contribution of each component:
  1. Config A: Uniform 5cm Baseline (No Adaptivity, No Risk)
  2. Config B: Distance-Only Variable-Resolution (Concentric Bands only)
  3. Config C: Distance + Dynamic Actor Risk (w_dyn=0.40, w_prox=0.25)
  4. Config D: Distance + Dynamic + Traversability (w_dyn=0.40, w_prox=0.25, w_slope=0.20)
  5. Config E: Distance + Dynamic + Traversability + Neural Uncertainty (Full Proposed Fovea-LiDAR)
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.point_cloud import PointCloud
from data.kitti_loader import loadSemanticKittiScan
from data.synthetic_generator import generateSyntheticLiDARScan
from core.grid_uniform import UniformElevationGrid
from core.grid_fovea import FoveaLiDARGrid, DEFAULT_FOVEA_BANDS
from core.risk_engine import RiskScoringEngine, RiskWeights
from segmentation.neural_segmenter import NeuralSegmenter


def runAblationStudy(numRuns: int = 10) -> None:
    print("=" * 95)
    print("  FOVEA-LiDAR: SYSTEMATIC ABLATION STUDY & COMPONENT CONTRIBUTION")
    print("=" * 95)

    resultsDir = PROJECT_ROOT / "experiments" / "results"
    resultsDir.mkdir(parents=True, exist_ok=True)

    # Ingest scan
    sampleBin = PROJECT_ROOT / "data" / "sample_scans" / "sample_000000.bin"
    sampleLabel = PROJECT_ROOT / "data" / "sample_scans" / "sample_000000.label"
    if sampleBin.is_file():
        scan = loadSemanticKittiScan(sampleBin, sampleLabel)
    else:
        scan = generateSyntheticLiDARScan(numRings=64, horizontalResolutionDeg=0.25)

    # Neural segmentation front end
    weightsPath = PROJECT_ROOT / "segmentation" / "weights" / "fovea_rangenet_v1.pt"
    segmenter = NeuralSegmenter(weightsPath=str(weightsPath) if weightsPath.is_file() else None)
    # Warmup
    segmenter.predict(scan)
    segResult = segmenter.predict(scan)

    scan.confidence = segResult.confidence
    scan.uncertainty = segResult.uncertainty

    # Define Configurations
    configurations = [
        {
            "id": "A",
            "name": "Uniform 5cm Baseline",
            "type": "uniform",
            "grid": UniformElevationGrid(resolution=0.05, minX=-50.0, maxX=50.0, minY=-50.0, maxY=50.0),
            "description": "Dense uniform 5cm grid across entire scene (no adaptivity, no risk)",
        },
        {
            "id": "B",
            "name": "Distance-Only Adaptive",
            "type": "fovea",
            "grid": FoveaLiDARGrid(bands=DEFAULT_FOVEA_BANDS, enableRiskRefinement=False),
            "description": "Pure concentric distance bands (5cm, 10cm, 25cm, 50cm)",
        },
        {
            "id": "C",
            "name": "Distance + Dynamic Risk",
            "type": "fovea",
            "grid": FoveaLiDARGrid(
                bands=DEFAULT_FOVEA_BANDS,
                enableRiskRefinement=True,
                riskEngine=RiskScoringEngine(
                    RiskWeights(weightDynamic=0.60, weightProximity=0.40, weightTraversability=0.0, weightUncertainty=0.0, refinementThreshold=0.45)
                ),
            ),
            "description": "Concentric bands + dynamic actor proximity refinement",
        },
        {
            "id": "D",
            "name": "Distance + Dynamic + Traversability",
            "type": "fovea",
            "grid": FoveaLiDARGrid(
                bands=DEFAULT_FOVEA_BANDS,
                enableRiskRefinement=True,
                riskEngine=RiskScoringEngine(
                    RiskWeights(weightDynamic=0.45, weightProximity=0.30, weightTraversability=0.25, weightUncertainty=0.0, refinementThreshold=0.45)
                ),
            ),
            "description": "Adds obstacle height step hazard (curbs, barriers, debris)",
        },
        {
            "id": "E",
            "name": "Full Proposed Fovea-LiDAR (+ Uncertainty)",
            "type": "fovea",
            "grid": FoveaLiDARGrid(
                bands=DEFAULT_FOVEA_BANDS,
                enableRiskRefinement=True,
                riskEngine=RiskScoringEngine(
                    RiskWeights(weightDynamic=0.40, weightProximity=0.25, weightTraversability=0.20, weightUncertainty=0.15, refinementThreshold=0.45)
                ),
            ),
            "description": "Full multi-criteria risk (dynamicity, proximity, slope, Shannon entropy uncertainty)",
        },
    ]

    records = []
    baseCells = 4_000_000
    baseMem = 87.74

    for cfg in configurations:
        model = cfg["grid"]

        # Warmup
        model.update(scan)

        timings = []
        for _ in range(numRuns):
            if hasattr(model, "clear"):
                model.clear()
            t0 = time.perf_counter()
            model.update(scan)
            timings.append((time.perf_counter() - t0) * 1000.0)

        latencyMean = float(np.mean(timings))
        fps = 1000.0 / max(0.001, latencyMean)

        if cfg["type"] == "uniform":
            totCells = model.totalCells
            activeCells = model.getOccupiedCellCount()
            refinedCells = 0
            memMb = model.getMemoryUsageMb()
            ptsProc = model.lastPointsProcessed
            ptsDrop = model.lastPointsDropped
        else:
            totCells = model.totalActiveCells
            activeCells = model.totalActiveCells
            refinedCells = model.bandOccupiedCounts.get(99, 0)
            memMb = model.getMemoryUsageMb()
            ptsProc = model.lastPointsProcessed
            ptsDrop = model.lastPointsDropped

        cellReduction = ((baseCells - totCells) / baseCells) * 100.0
        memSaved = ((baseMem - memMb) / baseMem) * 100.0
        retention = (ptsProc / max(1, ptsProc + ptsDrop)) * 100.0

        records.append({
            "Config": cfg["id"],
            "Ablation Configuration": cfg["name"],
            "Total Cells": totCells,
            "Occupied Cells": activeCells,
            "Refined Cells (5cm Foveas)": refinedCells,
            "Memory (MB)": round(memMb, 2),
            "Memory Saved %": round(memSaved, 1),
            "Cell Reduction %": round(cellReduction, 1),
            "Mapping Latency (ms)": round(latencyMean, 2),
            "Mapping FPS": round(fps, 1),
            "Point Retention %": round(retention, 2),
            "Description": cfg["description"],
        })

    df = pd.DataFrame(records)

    print("\n" + "=" * 125)
    print(df[["Config", "Ablation Configuration", "Total Cells", "Refined Cells (5cm Foveas)", "Memory (MB)", "Memory Saved %", "Cell Reduction %", "Mapping Latency (ms)", "Mapping FPS"]].to_string(index=False))
    print("=" * 125)

    csvPath = resultsDir / "ablation_study.csv"
    jsonPath = resultsDir / "ablation_study.json"

    df.to_csv(csvPath, index=False)
    with open(jsonPath, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print(f"\n[Verification] Ablation Artifacts Saved:")
    print(f"  -> CSV:  {csvPath}")
    print(f"  -> JSON: {jsonPath}")
    print("=" * 95)


if __name__ == "__main__":
    runAblationStudy()
