# Benchmark Results & Artifacts Index

This directory contains the authoritative, machine-measured benchmark results for **Fovea-LiDAR** evaluated on DRDO Problem Statement SIH26053.

---

## Output Files

1. **`final_benchmark.csv` & `final_benchmark.json`**:
   - Master comparative benchmark table across all 5 representations:
     - Uniform 5cm Baseline
     - Uniform 10cm Baseline
     - Uniform 25cm Baseline
     - Distance-Only Adaptive Grid
     - Fovea-LiDAR (Risk-Aware Proposed)
   - Metrics: Total cells, occupied cells, 5cm foveal refined cells, memory (MB), mapping latency (ms), mapping FPS, end-to-end FPS, points dropped, duplicate assignments, point retention %.

2. **`semantic_metrics.json`**:
   - Neural semantic segmentation metrics for `FoveaRangeNet`:
     - Mean IoU (mIoU: 68.56%)
     - Overall accuracy (93.22%)
     - Per-class IoU and Recall for Terrain, Static, Dynamic
     - Distance-stratified accuracy breakdown (0-10m, 10-30m, 30-60m, 60-100m)
     - Inference latency and throughput FPS.

3. **`projection_integrity.json`**:
   - Rigorous mathematical validation report:
     - `inputPoints`: 91,590
     - `assignedPoints`: 91,590
     - `droppedPoints`: 0
     - `duplicateAssignments`: 0
     - `boundaryViolations`: 0
     - `retentionPercent`: 100.0%
     - `conservationVerified`: true

4. **Visual Figures (High-Resolution PNG)**:
   - **`final_dashboard.png` / `final_sih_dashboard.png`**: Complete 4-panel visual dashboard with live HUD telemetry.
   - **`semantic_segmentation.png`**: Point cloud colored by deep-learning class predictions.
   - **`risk_heatmap.png`**: Continuous multi-criteria spatial hazard heatmap $\mathcal{R}(x, y)$.
   - **`foveated_resolution.png`**: Concentric distance bands with 5 cm dynamic foveas highlighted around high-risk dynamic actors.
   - **`elevation_map.png`**: 2.5D elevation surface map ($z_{\text{mean}}$).

---

## Reproduction Command

To reproduce all outputs from scratch:
```bash
python experiments/run_final_experiments.py
```
