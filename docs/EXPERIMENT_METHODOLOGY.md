# Fovea-LiDAR: Experiment Methodology & Validation Protocol

**Problem Statement**: SIH26053 — *Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception* (DRDO)  
**Project**: Fovea-LiDAR  
**Team**: FoveaX  
**Version**: 1.0 (Authoritative Protocol)  

---

## 1. Experimental Objectives

The purpose of this experimental protocol is to provide reproducible, mathematically rigorous, and hardware-verified evidence of:
1. **Memory Efficiency**: Quantifying reduction in total cells and memory footprint compared to uniform grids.
2. **Point Conservation and Projection Integrity**: Verifying that 3D to 2.5D discretization drops zero points within the 100m sensor envelope and produces zero duplicate assignments.
3. **Execution Latency & Throughput**: Measuring individual stage latencies (Segmentation, Risk Engine, Mapping) using high-precision hardware timers.
4. **Semantic Classification Accuracy**: Benchmarking `FoveaRangeNet` on overall accuracy, mean IoU (mIoU), and per-class recall.
5. **Spatial Fidelity**: Demonstrating the superior resolution of 5 cm risk-refined foveas over coarse 50 cm distance-only cells for distant dynamic obstacles.

---

## 2. Dataset Ingestion Protocol

### 2.1 Primary Evaluated Scan
- **Format**: Binary 32-bit floating-point coordinates `[x, y, z, intensity]` and unsigned 32-bit label file `[semantic_label, instance_id]` matching the official SemanticKITTI standard.
- **Physical Geometry**: 64-beam spinning LiDAR scan covering a full $360^\circ$ azimuth and $+2.0^\circ$ to $-24.8^\circ$ vertical field-of-view.
- **Total Points**: **91,590 points**.
- **Spatial Bounds**:
  - $X \in [-49.9\text{ m}, +49.9\text{ m}]$
  - $Y \in [-49.9\text{ m}, +49.9\text{ m}]$
  - $Z \in [-2.5\text{ m}, +5.1\text{ m}]$
  - Maximum Planar Distance: **70.6 meters** (points reach into Band 3: 60-100m).
- **Scene Composition**:
  - Rolling undulating terrain with realistic sensor return density ($1/r^2$ beam dispersion).
  - Raised curbs and road barriers ($15\text{ cm} - 30\text{ cm}$ vertical step).
  - Static structures (buildings, fences, vegetation).
  - Multiple dynamic actors (vehicles and pedestrians at 5.7m, 22m, and 48m range).

### 2.2 SemanticKITTI Compatibility
- The loader `data/kitti_loader.py` natively reads real SemanticKITTI `.bin` scans and `.label` files directly from disk via NumPy `fromfile` at memory-mapped C speed.

---

## 3. Timing and Latency Protocol

To eliminate timing artifacts, cold-start jitter, and Windows thread scheduling anomalies:
1. **High-Resolution Clock**: All timings use `time.perf_counter()` (sub-microsecond resolution on x86_64).
2. **Warmup Iterations**: Every model undergoes a minimum of 2 complete forward passes prior to measurement to ensure:
   - PyTorch CUDA kernel compilation and cuDNN autotuner selection.
   - GPU frequency stabilization.
   - CPU memory cache population.
3. **Statistical Averaging**: Each measurement is evaluated across **$N = 10$ consecutive iterations**. The report records:
   $$\mu = \frac{1}{N} \sum_{i=1}^N t_i, \quad \sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (t_i - \mu)^2}$$
4. **Stage Separation**: Timings are recorded individually for:
   - Segmentation Latency ($t_{\text{seg}}$)
   - Risk Engine Latency ($t_{\text{risk}}$)
   - Grid Mapping Latency ($t_{\text{map}}$)
   - End-to-End Latency ($t_{\text{e2e}} = t_{\text{seg}} + t_{\text{risk}} + t_{\text{map}}$)

---

## 4. Memory Footprint Measurement

Memory consumption is measured strictly from first-principles byte allocation rather than unstable OS process RSS (which includes Python interpreter overhead, loaded shared libraries, and CUDA driver context):

### 4.1 Uniform Elevation Grid
For a dense grid of dimensions $W \times H$:
$$\text{Memory}_{\text{uniform}} = W \times H \times (\text{bytes per cell})$$
- 7 core float32/int32 layers ($z_{\min}, z_{\max}, z_{\text{mean}}, \Delta z, \text{pointCount}, \text{semanticClass}, \text{isDynamic}$) = 23 bytes/cell.
- Uniform 5 cm ($2000 \times 2000 = 4,000,000$ cells):
  $$\text{Memory} = 4,000,000 \times 23\text{ bytes} = 92,000,000\text{ bytes} \approx 87.74\text{ MB}$$

### 4.2 Fovea-LiDAR Grid
Calculated by summing contiguous NumPy array byte sizes across all active bands plus the 64-bit dictionary index table:
$$\text{Memory}_{\text{fovea}} = \sum_{b \in \text{Bands}} \left( \sum_{a \in \text{Arrays}} a.\text{nbytes} + 64 \times |\text{keyToIndex}| \right)$$
For 59,875 active cells:
$$\text{Memory} = 6,585,088\text{ bytes} \approx 6.28\text{ MB}$$

---

## 5. Mathematical Invariants & Verification Suite

The repository enforces automated unit test assertions (`tests/`) verifying:
1. **Point Conservation**:
   $$\text{Points}_{\text{processed}} + \text{Points}_{\text{dropped}} = \text{Total Input Points}$$
   For Fovea-LiDAR: $91,590 + 0 = 91,590$ (100.0% retention).
2. **Zero Duplicate Assignments**:
   $$\forall (b_1, k_1), (b_2, k_2), \quad (b_1, k_1) = (b_2, k_2) \implies \text{Duplicates} = 0$$
3. **Boundary Transition Non-Overlap**:
   $$\text{Band}_i = [r_i^{\min}, r_i^{\max}), \quad \text{Band}_j = [r_j^{\min}, r_j^{\max}) \implies \text{Band}_i \cap \text{Band}_j = \emptyset$$
   Tested with micro-steps ($9.999\text{m}, 10.000\text{m}, 10.001\text{m}$).
4. **Spatial Containment**:
   $$\forall p \in \text{Cell}_k, \quad |p_x - c_x| \le \frac{\text{res}}{2}, \quad |p_y - c_y| \le \frac{\text{res}}{2}$$
