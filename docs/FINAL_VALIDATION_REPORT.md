# Fovea-LiDAR: Final Validation Report & SIH Submission Document

**Problem Statement**: SIH26053 — *Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception* (DRDO)  
**Project**: Fovea-LiDAR  
**Team**: FoveaX  
**Version**: 1.0 (Official Release Candidate)  
**Date**: September 16, 2026  

---

## 1. Executive Summary

Team **FoveaX** has developed **Fovea-LiDAR**, a perception and elevation mapping system submitted for DRDO Problem Statement SIH26053. The prototype addresses the dual challenges of conventional high-resolution LiDAR mapping: **prohibitive memory/cell explosion** on one hand, and **dangerous loss of obstacle geometry at range** on the other.

By coupling a **range-view deep-learning semantic segmentation neural network** with a **multi-criteria criticality risk engine**, Fovea-LiDAR achieves:
- **98.5% reduction in total spatial cells** compared to a uniform 5 cm grid (59,875 vs. 4,000,000 cells).
- **92.8% memory savings** (6.28 MB vs. 87.74 MB).
- **100.0% point retention** (zero points dropped within the 100m sensor range).
- **Zero duplicate point assignments** across concentric band boundaries.
- **9.6x improvement in spatial precision** on distant dynamic actors (2.06 cm vs. 19.78 cm RMS quantization error at 65m range).
- **Real-time end-to-end perception throughput** (20.0 FPS end-to-end; 47.9 FPS segmentation, 35.4 FPS mapping).

---

## 2. Hardware and Environment Audit

All benchmarks and neural inferences were executed natively on the target host workstation:
- **Operating System**: Windows 11 Home / Pro (Build 10.0.26200, AMD64)
- **Processor**: Intel Core (Family 6 Model 183 Stepping 1, 20 physical / 28 logical cores)
- **Host RAM**: 15.71 GB
- **Dedicated GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM)
- **CUDA Acceleration**: CUDA 12.4 enabled (`torch.cuda.is_available() == True`)
- **Python / Frameworks**: Python 3.12.10, PyTorch 2.6.0+cu124, NumPy 2.2.6

---

## 3. Authoritative Benchmark Results

Authoritative metrics evaluated over 10 consecutive runs on a 91,590-point LiDAR scan (recorded in `experiments/results/final_benchmark.json` and `final_benchmark.csv`):

| Representation Model | Total Cells | Occupied Cells | Refined Cells (5cm Foveas) | Memory (MB) | Memory Saved % | Cell Reduction % | Mapping Latency (ms) | Mapping FPS | End-to-End FPS | Point Retention % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform 5cm (Baseline)** | 4,000,000 | 64,769 | 0 | 87.74 MB | 0.0% | 0.0% | 12.50 ms | 80.0 | 29.2 | 97.4% |
| **Uniform 10cm** | 1,000,000 | 41,099 | 0 | 21.93 MB | 75.0% | 75.0% | 16.23 ms | 61.6 | 26.3 | 97.4% |
| **Uniform 25cm** | 160,000 | 16,965 | 0 | 3.51 MB | 96.0% | 96.0% | 10.77 ms | 92.8 | 30.7 | 97.4% |
| **Distance-Only Adaptive** | 58,810 | 58,810 | 0 | 6.17 MB | 93.0% | 98.5% | 26.53 ms | 37.7 | 20.7 | 100.0% |
| **Fovea-LiDAR (Risk-Aware Proposed)** | **59,875** | **59,875** | **1,951** | **6.28 MB** | **92.8%** | **98.5%** | **28.23 ms** | **35.4** | **20.0** | **100.0%** |

---

## 4. Latency Stage Breakdown

Perception stages are explicitly measured and decoupled:

```
[Raw LiDAR Points: 91,590]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Deep Learning Semantic Segmentation (FoveaRangeNet) │  20.90 ms  (47.9 FPS)
└─────────────────────────────────────────────────────────────┘
       │  Predictions, Softmax Confidence, Shannon Entropy
       ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Multi-Criteria Scene Criticality Risk Engine       │   0.86 ms  (1,162 FPS)
└─────────────────────────────────────────────────────────────┘
       │  Continuous Risk Field R(x,y)
       ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Variable-Resolution 2.5D Elevation Grid Mapping    │  28.23 ms  (35.4 FPS)
└─────────────────────────────────────────────────────────────┘
       │
       ▼
[Total End-to-End Perception Loop Latency]                    │  49.99 ms  (20.0 FPS)
```

---

## 5. Systematic Component Ablation Study

Evaluated across all 5 progressive pipeline configurations (recorded in `experiments/results/ablation_study.json`):

| Config | Architecture Variant | Active Cells | Refined Cells (5cm) | Memory (MB) | Memory Saved % | Mapping Latency (ms) | Description |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | Uniform 5cm Baseline | 4,000,000 | 0 | 87.74 | 0.0% | 20.61 ms | Dense grid, no adaptivity, zero spatial reasoning |
| **B** | Distance-Only Adaptive | 58,810 | 0 | 6.17 | 93.0% | 26.06 ms | Pure concentric bands (5cm, 10cm, 25cm, 50cm) |
| **C** | Distance + Dynamic Risk | 59,874 | 2,009 | 6.28 | 92.8% | 28.01 ms | Dynamic vehicle / pedestrian proximity refinement |
| **D** | Distance + Dynamic + Traversability | 60,019 | 2,225 | 6.30 | 92.8% | 30.40 ms | Adds curb, road barrier, and terrain step hazards |
| **E** | **Full Proposed (+ Uncertainty)** | **60,058** | **2,266** | **6.30** | **92.8%** | **30.43 ms** | Full multi-criteria hazard + Shannon entropy |

---

## 6. Distant Obstacle Spatial Fidelity Benchmark

To prove the necessity of risk-aware foveation beyond pure distance-based adaptivity, a 4.5m x 1.9m vehicle at 65m range was evaluated under both architectures (`scripts/demo_distant_obstacle.py`):

| Metric | Distance-Only Adaptive Grid (50cm) | Fovea-LiDAR Risk-Refined Grid (5cm) | Operational Advantage |
| :--- | :---: | :---: | :--- |
| **Assigned Band** | Band 3 (Coarse Distance Band) | Band 99 (Dynamic Refined Fovea) | **Focal Awareness** |
| **Spatial Resolution** | $50\text{ cm} \times 50\text{ cm}$ | $5\text{ cm} \times 5\text{ cm}$ | **10x Finer Grid Cells** |
| **Occupied Cells on Vehicle** | 41 cells | 773 cells | **18.9x Surface Detail** |
| **RMS Quantization Error** | **19.78 cm** | **2.06 cm** | **9.6x Reduction in Spatial Error** |
| **Downstream Impact** | Boundary blurred by $\pm 25\text{ cm}$; risk of clipping during overtaking | Sharp vehicle contour; centimeter-accurate collision boundary | **Safe Autonomous Overtaking & Braking** |

---

## 7. Deep Learning Front End: FoveaRangeNet

- **Architecture**: Range-View Spherical Convolutional U-Net with Dilated Context Bottlenecks.
- **Model Parameters**: 1,418,243 parameters (~5.41 MB checkpoint).
- **Target Classes**:
  - Class 0: Terrain / Drivable
  - Class 1: Static Obstacle
  - Class 2: Dynamic Actor
- **Measured Accuracy Metrics**:
  - **Overall Accuracy**: **93.22%**
  - **Mean IoU (mIoU)**: **68.56%**
  - **Terrain IoU / Recall**: **93.85% / 94.36%**
  - **Static Obstacle IoU / Recall**: **57.75% / 74.28%**
  - **Dynamic Actor IoU / Recall**: **54.08% / 94.74%**
  - **Inference Latency**: **20.90 ms (47.9 FPS)** on RTX 4050 GPU.

---

## 8. 30-Item SIH Readiness Audit Checklist

| Item # | Verification Criteria | Status | Implementation Evidence |
| :---: | :--- | :---: | :--- |
| **1** | Genuine Deep-Learning Semantic Segmentation | **VERIFIED** | `segmentation/neural_segmenter.py` (`FoveaRangeNet` PyTorch model) |
| **2** | Official 3-Class DRDO Taxonomy | **VERIFIED** | `segmentation/class_mapping.py` (Terrain, Static, Dynamic) |
| **3** | Softmax Confidence Estimation | **VERIFIED** | Point-wise max probability $c = \max_k p_k \in [0, 1]$ |
| **4** | Shannon Entropy Uncertainty Quantification | **VERIFIED** | $\mathcal{H} = -\sum p \ln p / \ln 3$ computed in `NeuralSegmenter` |
| **5** | Multi-Criteria Risk Scoring Engine | **VERIFIED** | `core/risk_engine.py` ($\text{Dynamicity} + \text{Proximity} + \text{Slope} + \text{Entropy}$) |
| **6** | Variable-Resolution 2.5D Elevation Grid | **VERIFIED** | `core/grid_fovea.py` with concentric band partitions |
| **7** | Near-Field Resolution (0-10m @ 5cm) | **VERIFIED** | Band 0 configured at 0.05m cell size |
| **8** | Intermediate Band (10-30m @ 10cm) | **VERIFIED** | Band 1 configured at 0.10m cell size |
| **9** | Mid-Range Band (30-60m @ 25cm) | **VERIFIED** | Band 2 configured at 0.25m cell size |
| **10** | Far-Field Band (60-100m @ 50cm) | **VERIFIED** | Band 3 configured at 0.50m cell size |
| **11** | Local Risk-Refined Fovea (Band 99 @ 5cm) | **VERIFIED** | Cells exceeding risk $\ge 0.45$ refined to 5 cm |
| **12** | Baseline Representation Preserved | **VERIFIED** | 59,875 cells, 6.28 MB, 100% retention maintained |
| **13** | > 90% Spatial Cell Reduction | **VERIFIED** | **98.5% reduction** (59,875 vs 4,000,000 cells) |
| **14** | > 85% Memory Savings | **VERIFIED** | **92.8% savings** (6.28 MB vs 87.74 MB) |
| **15** | 100% Point Conservation | **VERIFIED** | 91,590 input points $\to$ 91,590 assigned points, 0 dropped |
| **16** | Zero Duplicate Cell Key Assignments | **VERIFIED** | Disjoint spatial keys verified in `test_projection_integrity.py` |
| **17** | Strict Boundary Transitions | **VERIFIED** | Micro-boundary tests ($9.999\text{m}, 10.000\text{m}, 10.001\text{m}$) passing |
| **18** | High-Speed Vectorized Projection | **VERIFIED** | C-speed NumPy `reduceat` spatial aggregation |
| **19** | Decoupled Stage Latency Timing | **VERIFIED** | Segmentation (20.9ms), Risk (0.9ms), Mapping (28.2ms) |
| **20** | Real-Time Perception Throughput | **VERIFIED** | 20.0 FPS end-to-end loop; 35.4 FPS mapping throughput |
| **21** | Native SemanticKITTI Binary Ingestion | **VERIFIED** | `data/kitti_loader.py` reads `.bin` and `.label` files |
| **22** | Synthetic Deterministic Benchmark Generator | **VERIFIED** | `data/synthetic_generator.py` for mathematical regression |
| **23** | Pure-PyTorch Windows 11 Execution | **VERIFIED** | No Linux-only sparse convolution dependencies |
| **24** | Automated Unit Test Suite (17/17 Passing) | **VERIFIED** | `python -m pytest tests/ -v` passes 17/17 tests in 2.6s |
| **25** | Declarative YAML Configuration | **VERIFIED** | `config/fovea_config.yaml` & `FoveaLiDARGrid.fromYamlConfig` |
| **26** | Temporal Confidence Decay & Pruning | **VERIFIED** | `applyTemporalDecay` implemented and verified |
| **27** | Real-Time Perception HUD Dashboard | **VERIFIED** | `visualization/visualizer_2d.py` 4-panel telemetry dashboard |
| **28** | Hardware Architecture Audit Logs | **VERIFIED** | `outputs/logs/system_info.json` generated |
| **29** | Systematic 5-Config Component Ablation | **VERIFIED** | `experiments/run_ablation_study.py` (Uniform $\to$ Full Fovea) |
| **30** | Distant Dynamic Obstacle Demonstration | **VERIFIED** | `scripts/demo_distant_obstacle.py` (9.6x precision gain at 65m) |

---

## 9. Conclusion

Fovea-LiDAR is ready for technical demonstration at SIH 2026. Every metric presented is empirically measured, reproducible via `python scripts/reproduce_results.py`, and backed by rigorous mathematical verification.
