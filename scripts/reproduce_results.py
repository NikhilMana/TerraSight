"""
Master One-Command Reproduction Script for Fovea-LiDAR (DRDO SIH26053).
Executes the entire validation and benchmarking pipeline from end to end:
  1. System and Hardware Audit
  2. Full 17-Test Unit Test Suite
  3. Authoritative Multi-Representation Benchmark
  4. Visual Artifact Generation
"""

import sys
import unittest
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.system_info import main as runSystemInfo
from experiments.run_final_experiments import runAuthoritativeSIHBenchmark
from experiments.run_ablation_study import runAblationStudy
from scripts.demo_distant_obstacle import runDistantObstacleComparison


def reproduceAll() -> bool:
    print("=" * 80)
    print("  FOVEA-LiDAR: MASTER REPRODUCIBILITY SUITE (DRDO SIH26053)")
    print("=" * 80)

    # Step 1: Hardware Audit
    print("\n[STEP 1/5] Running Hardware & Environment Audit...")
    runSystemInfo()

    # Step 2: Test Suite
    print("\n[STEP 2/5] Running Complete Unit Test Suite...")
    loader = unittest.TestLoader()
    suite = loader.discover(str(PROJECT_ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    testResult = runner.run(suite)

    if not testResult.wasSuccessful():
        print("  -> ERROR: Unit tests failed! Halting reproduction.")
        return False

    print("  -> All unit tests PASSED successfully.")

    # Step 3: Benchmarking and Artifact Generation
    print("\n[STEP 3/5] Running Authoritative Multi-Representation Benchmark...")
    runAuthoritativeSIHBenchmark(numWarmup=2, numRuns=10)

    # Step 4: Systematic Ablation Study
    print("\n[STEP 4/5] Running Systematic Component Ablation Study...")
    runAblationStudy(numRuns=10)

    # Step 5: Distant Dynamic Obstacle Demonstration
    print("\n[STEP 5/5] Running Distant Dynamic Obstacle Spatial Fidelity Benchmark...")
    runDistantObstacleComparison()

    print("\n" + "=" * 80)
    print("  ALL BENCHMARKS, ABLATIONS & ARTIFACTS REPRODUCED SUCCESSFULLY")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = reproduceAll()
    sys.exit(0 if success else 1)
