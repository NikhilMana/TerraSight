# Project Audit: Fovea-LiDAR (SIH26053 Prototype)

**Date**: September 16, 2026  
**Project**: Fovea-LiDAR  
**Team**: FoveaX  
**Problem Statement**: SIH26053 — *Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception* (DRDO)  
**Author**: Lead AI / Robotics Engineer & Technical Lead  

---

## 1. Executive Summary

This audit establishes the ground truth of the current codebase at `C:\Coding\Python\TerraSight`. It separates verified working functionality from prototype placeholders, distinguishes synthetic validation from real-data validation, and outlines the precise engineering path to bridge the remaining critical gap: **integrating a genuine deep-learning LiDAR semantic segmentation stage**.

---

## 2. Hardware and Execution Environment

The host system was inspected directly via PyTorch and system APIs:
- **Operating System**: Windows 11 Home / Pro (Build 10.0.26200)
- **Python**: `3.12.10` (64-bit AMD64)
- **PyTorch**: `2.6.0+cu124`
- **CUDA Acceleration**: **Enabled** (`CUDA 12.4`)
- **GPU**: **NVIDIA GeForce RTX 4050 Laptop GPU**
- **GPU VRAM**: **6.44 GB**
- **CPU**: Intel Core (Family 6 Model 183 Stepping 1, GenuineIntel)
- **System Memory (RAM)**: **15.71 GB**

> [!NOTE]
> Hardware constraints: 6.44 GB VRAM and Python 3.12 on Windows require pure-PyTorch architectures (such as Range-View spherical U-Net / PointNet++) rather than heavy Linux-only custom C++ sparse-convolution dependencies (like `spconv` or `torchsparse` which lack Python 3.12 Windows wheels).

---

## 3. Existing Repository Architecture

```
TerraSight/
├── core/
│   ├── point_cloud.py             # PointCloud container (points, intensity, labels, dynamics)
│   ├── projection.py              # Vectorized 3D -> 2.5D NumPy reduceat projection engine
│   ├── grid_uniform.py            # Baseline Uniform 2.5D Grid (5cm, 10cm, 25cm)
│   ├── grid_fovea.py              # Fovea-LiDAR Variable-Resolution Grid with Risk Refinement
│   └── risk_engine.py             # Multi-criteria Risk & Criticality Scoring Engine
├── segmentation/
│   ├── kitti_classes.py           # 28-class SemanticKITTI to 3-class target mapping
│   └── geometric_ground.py        # Heuristic radial surface & corridor obstacle classifier
├── data/
│   ├── kitti_loader.py            # SemanticKITTI binary reader (.bin + .label)
│   ├── synthetic_generator.py     # Deterministic 64-beam LiDAR scan generator
│   └── sample_scans/
│       ├── sample_000000.bin      # Generated synthetic 64-beam binary scan (1.43 MB)
│       └── sample_000000.label    # Generated synthetic label file (357.8 KB)
├── visualization/
│   └── visualizer_2d.py           # 4-panel perception dashboard with live HUD telemetry
├── benchmarks/
│   └── run_benchmarks.py          # Comparative benchmarking script (Uniform vs Fovea)
├── experiments/
│   ├── run_final_experiments.py   # End-to-end evaluation suite
│   └── results/                   # Benchmark CSV, JSON reports, and rendered PNG figures
└── tests/
    ├── test_data_loader.py        # Point cloud serialization and filtering tests
    ├── test_uniform_grid.py       # Uniform projection and elevation invariant tests
    ├── test_fovea_grid.py         # Boundary transitions, point conservation, and fovea tests
    └── test_risk_refinement.py    # Risk-driven 5cm foveal refinement unit tests
```

---

## 4. Component-by-Component Audit

| Component | Status | Reality Check | Implementation Details |
| :--- | :--- | :--- | :--- |
| **PointCloud Data Model** | **Genuinely Complete** | Real / Functional | Full vectorization in NumPy. Bounding box cropping, 2D/3D Euclidean distance calculations, intensity/label/dynamic masking. |
| **Vectorized Projection Engine** | **Genuinely Complete** | Real / Functional | C-speed NumPy `reduceat` operations. Computes $z_{\min}, z_{\max}, z_{\text{mean}}, \Delta z$, occupancy, density, majority semantics in **< 14 ms**. |
| **Uniform Elevation Grid** | **Genuinely Complete** | Real / Functional | Dense 2.5D multi-layer grid. Measures exact byte allocation, cell counts, sparsity %, and mapping latency. |
| **Fovea-LiDAR Adaptive Grid** | **Genuinely Complete** | Real / Functional | Quadtree-aligned concentric distance bands ($5\text{cm}, 10\text{cm}, 25\text{cm}, 50\text{cm}$) + dynamic 5 cm foveal refinement. Columnar active-cell storage runs at **~52 FPS**. |
| **Boundary Consistency** | **Genuinely Complete** | Real / Functional | Strict mutually exclusive interval partitions. Tested at $9.999\text{m} \leftrightarrow 10.000\text{m}$, $29.999\text{m} \leftrightarrow 30.000\text{m}$, $59.999\text{m} \leftrightarrow 60.000\text{m}$. Zero dropped points, zero duplicate assignments. |
| **Multi-Criteria Risk Engine** | **Genuinely Complete** | Real / Functional | Computes $\mathcal{R}(x,y) = w_1 \mathcal{S}_{\text{dyn}} + w_2 \mathcal{S}_{\text{prox}} + w_3 \mathcal{S}_{\text{slope}} + w_4 \mathcal{S}_{\text{unc}}$. Exposes vectorized batch computation. |
| **Class Taxonomy Remapping** | **Genuinely Complete** | Real / Functional | Remaps 28 SemanticKITTI classes into 3 DRDO target classes: Terrain (1), Static (2), Dynamic (3). Vectorized lookup table. |
| **Semantic Segmentation** | **GEOMETRIC / HEURISTIC** | **GAP: NOT NEURAL** | Currently implemented as `GeometricSegmenter` (radial percentile ground plane + corridor height thresholding). **Lacks deep learning.** |
| **Sample Dataset** | **SYNTHETIC** | Synthetic Scan | `sample_000000.bin` is a synthetic 64-beam Velodyne scan generated by `synthetic_generator.py`. Not a real KITTI recording. |
| **Real-Time Dashboard** | **Genuinely Complete** | Real / Functional | 4-panel Matplotlib/OpenCV visualization with live HUD telemetry reporting actual measured metrics. |
| **Test Suite** | **Genuinely Complete** | Real / Functional | 11 unit tests running in `unittest` passing in 0.297s. |

---

## 5. Critical Distinction: Synthetic vs. Real Data

- **What is currently tested**:
  - The 91,590-point scan (`sample_000000.bin`) was created by `synthetic_generator.py`.
  - It models 64 vertical beams ($+2^\circ$ to $-24.8^\circ$), undulating terrain, curbs, buildings, trees, and dynamic cars/pedestrians at $8\text{m}, 22\text{m}, 48\text{m}$.
  - This synthetic scan was essential to verify mathematical invariants (point conservation, boundary transitions, exact spatial containment).
- **What is needed for SIH**:
  - Real SemanticKITTI sequence scans (e.g. sequence 00 or 08) must be ingested and evaluated to validate the pipeline on real physical LiDAR sensor noise, dropouts, and complex real-world scenes.

---

## 6. Critical Distinction: Geometric vs. Deep-Learning Segmentation

- **Existing Implementation (`segmentation/geometric_ground.py`)**:
  - Uses radial elevation profiles (5th percentile lowest points) + geometric height bounds + road corridor bounds.
  - Achieves **81.70% mIoU** on the synthetic scan because the synthetic scan was generated with known geometric properties.
  - **This is NOT a deep neural network.**
- **The SIH-Critical Requirement**:
  - The problem statement mandates: *"A deep-learning pipeline that processes raw LiDAR point clouds and performs semantic segmentation into terrain, static obstacles, and dynamic objects."*
  - We must implement a genuine deep-learning LiDAR segmentation model (`segmentation/neural_segmenter.py`) operating on point clouds / range projections, exposing true softmax class probabilities and classification uncertainty.
  - The existing `GeometricSegmenter` must be retained as a baseline / fallback / ablation comparator.

---

## 7. Analysis of the Latency Discrepancy

As noted in the prompt:
- `phase8_risk_aware_metrics.json` recorded: **19.15 ms** (52.21 FPS)
- `final_sih_benchmark_summary.csv` recorded: **19.25 ms** (51.9 FPS)
- **Root Cause**: `verify_phase6_7_8.py` timed a single end-to-end run, whereas `run_final_experiments.py` executed a 10-iteration statistical loop with `time.perf_counter()`. Minor CPU/GPU thermal and OS scheduling fluctuations on Windows naturally produce $\pm 0.1\text{ ms}$ variance.
- **Remedy**: We will standardize the benchmark runner so that both CSV, JSON, and dashboard snapshot are generated synchronously in one authoritative execution pass.

---

## 8. Remaining SIH-Critical Gaps & Recommended Next Steps

1. **Task 6 (Hardware Script)**: Create `scripts/system_info.py` generating `outputs/logs/system_info.json`.
2. **Task 7 & 8 (Neural Semantic Segmentation)**:
   - Implement `segmentation/base_segmenter.py` (abstract interface).
   - Implement `segmentation/neural_segmenter.py` using a Range-View Convolutional Neural Network (`RangeNetLight` / `FoveaSegNet`) or PointNet++.
   - Expose per-point predicted classes, softmax confidence, and entropy-based uncertainty.
   - Provide genuine weights, training/validation pipeline, and real measured IoU metrics.
3. **Task 9 & 10 (Class Taxonomy & Clean Abstraction)**:
   - Enhance `segmentation/class_mapping.py` to clearly document raw SemanticKITTI ID $\to$ DRDO target class mappings, distinguishing moving vs parked vehicles.
4. **Task 14 & 15 (Boundary & Projection Integrity Tests)**:
   - Create `tests/test_boundary_transitions.py` testing $9.999\text{m}, 10.000\text{m}, 10.001\text{m}$, etc.
   - Create `tests/test_projection_integrity.py` asserting $\text{Input} = N, \text{Assigned} = N, \text{Dropped} = 0, \text{Duplicates} = 0$.
5. **Task 18 (Temporal Update)**:
   - Implement lightweight temporal decay / timestamp updating in `grid_fovea.py`.
6. **Task 22-25 (Unified Re-Benchmarking)**:
   - Separate Segmentation Latency, Risk Scoring Latency, and Mapping Latency.
   - Re-run benchmarks and generate unified, authoritative results.
