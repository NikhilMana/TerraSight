# Model Card: FoveaRangeNet (LiDAR Semantic Front End)

**Model Name**: `FoveaRangeNet`  
**Version**: 1.0  
**Model Type**: Range-View Spherical Convolutional Residual U-Net with Dilated Bottlenecks  
**Application**: Real-Time Semantic Segmentation for DRDO Autonomous Ground Vehicles (SIH26053)  
**Authors**: Team FoveaX  
**Date**: September 16, 2026  

---

## 1. Model Details

### 1.1 Architecture Specifications
- **Input Representation**: Spherical Range Image of dimensions $(B, 5, 64, 1024)$.
  - Channel 0: Normalized range $r / r_{\max} \in [0, 1]$
  - Channel 1: Cartesian $x$ coordinate (forward axis, in meters)
  - Channel 2: Cartesian $y$ coordinate (lateral axis, in meters)
  - Channel 3: Cartesian $z$ coordinate (vertical elevation, in meters)
  - Channel 4: Calibrated LiDAR return intensity $i \in [0, 1]$
- **Stem**: $3 \times 3$ Conv2D (in=5, out=32, stride=1, padding=1) + BatchNorm2D + LeakyReLU(0.1).
- **Encoder**: 3-stage Residual Convolutional blocks:
  - Stage 1: $(32 \to 64)$, stride $(1, 2) \to (64, 512)$
  - Stage 2: $(64 \to 128)$, stride $(2, 2) \to (32, 256)$
  - Stage 3: $(128 \to 256)$, stride $(2, 2) \to (16, 128)$
- **Context Bottleneck**: Dual dilated convolutional layers:
  - Dilation rate = 2 (receptive field $5 \times 5$, padding 2)
  - Dilation rate = 4 (receptive field $9 \times 9$, padding 4)
- **Decoder**: 3-stage Upsampling + Residual feature concatenation:
  - Stage 3: Bilinear upsampling $(2, 2) + \text{Conv2D}(256 \to 128) + \text{ResBlock}(128 + 128 \to 128)$
  - Stage 2: Bilinear upsampling $(2, 2) + \text{Conv2D}(128 \to 64) + \text{ResBlock}(64 + 64 \to 64)$
  - Stage 1: Bilinear upsampling $(1, 2) + \text{Conv2D}(64 \to 32) + \text{ResBlock}(32 + 32 \to 32)$
- **Output Head**: $1 \times 1$ Conv2D $(32 \to 3)$ producing class logits for $(64 \times 1024)$.
- **Total Trainable Parameters**: **1,418,243 parameters** (~5.41 MB).

---

## 2. Intended Use and Target Classes

The model classifies each 3D point into one of 3 DRDO-defined operational classes:
1. **Class 0 — Terrain / Drivable**: Flat and undulating road surfaces, dirt paths, grass, sidewalks.
2. **Class 1 — Static Obstacle**: Buildings, road barriers, curbs, tree trunks, poles, walls.
3. **Class 2 — Dynamic Actor**: Moving or movable vehicles, trucks, cyclists, pedestrians.

---

## 3. Training and Evaluation Pipeline

- **Loss Function**: Multi-Class Focal Cross-Entropy with inverse frequency class weighting:
  $$\mathcal{L} = -\sum_{c=0}^2 w_c (1 - p_c)^\gamma y_c \log(p_c + \epsilon)$$
  where $\gamma = 2.0$, and class weights are $w_{\text{terrain}} = 0.5$, $w_{\text{static}} = 1.0$, $w_{\text{dynamic}} = 2.5$ to heavily penalize missing dynamic objects.
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = $10^{-4}$, initial learning rate = $10^{-3}$).
- **Hardware Platform**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.4, PyTorch 2.6.0+cu124).

---

## 4. Empirical Evaluation Metrics

Authoritative measured metrics on 91,590-point LiDAR scan (recorded in `experiments/results/semantic_metrics.json`):

| Metric | Measured Value | Target Requirement | Status |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **93.22%** | $> 85.0\%$ | **PASSED** |
| **Mean IoU (mIoU)** | **68.56%** | $> 60.0\%$ | **PASSED** |
| **Terrain IoU** | **93.85%** | $> 80.0\%$ | **PASSED** |
| **Terrain Recall** | **94.36%** | $> 85.0\%$ | **PASSED** |
| **Static Obstacle IoU** | **57.75%** | $> 50.0\%$ | **PASSED** |
| **Static Obstacle Recall** | **74.28%** | $> 70.0\%$ | **PASSED** |
| **Dynamic Actor IoU** | **54.08%** | $> 50.0\%$ | **PASSED** |
| **Dynamic Actor Recall** | **94.74%** | $> 90.0\%$ | **PASSED** |
| **Inference Latency** | **20.90 ms** | $< 33.3\text{ ms}$ ($> 30\text{ FPS}$) | **PASSED (47.9 FPS)** |

### Distance-Stratified Accuracy Breakdown
- **0 – 10 meters**: **96.78%**
- **10 – 30 meters**: **89.37%**
- **30 – 60 meters**: **84.16%**
- **60 – 100 meters**: **85.97%**

---

## 5. Uncertainty Quantification & Downstream Propagation

Rather than merely outputting discrete class labels, `FoveaRangeNet` computes two continuous signals:
1. **Confidence Score**: $c_i = \max_{k \in \{0,1,2\}} P(y_i = k \mid x_i)$
2. **Shannon Entropy Uncertainty**:
   $$\mathcal{H}_i = -\frac{1}{\ln 3} \sum_{k=0}^2 P(y_i=k \mid x_i) \ln(P(y_i=k \mid x_i) + 10^{-6}) \in [0.0, 1.0]$$
This entropy signal directly enters Stage 2's risk scoring engine ($w_{\text{unc}} = 0.15$), guaranteeing that points with high classification ambiguity are automatically promoted to fine 5 cm resolution for closer physical inspection by the robot.
