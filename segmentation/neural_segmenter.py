"""
Neural LiDAR Semantic Segmenter (Deep Learning Pipeline).
Architecture: FoveaRangeNet (Range-View Spherical Convolutional U-Net).
Projects raw 3D LiDAR point clouds to a 64x1024 5-channel spherical range image,
processes it through a residual encoder-decoder with dilated context bottlenecks,
and remaps pixel-level predictions back to 3D points.

Outputs:
  - 3-class semantic predictions (0: Terrain, 1: Static, 2: Dynamic)
  - Calibrated softmax confidence scores
  - Information-theoretic Shannon entropy uncertainty
"""

import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.point_cloud import PointCloud
from segmentation.base_segmenter import BaseSegmenter, SegmentationResult
from segmentation.class_mapping import DRDOTargetClass


# ---------------------------------------------------------------------------
# 1. PyTorch Neural Network Architecture: FoveaRangeNet
# ---------------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """Residual Convolutional Block with LeakyReLU and BatchNorm."""
    def __init__(self, inChannels: int, outChannels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)

        self.shortcut = nn.Sequential()
        if stride != 1 or inChannels != outChannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outChannels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class FoveaRangeNet(nn.Module):
    """
    Lightweight Range-View Spherical Convolutional Network for LiDAR Segmentation.
    Input: (B, 5, 64, 1024) [range, x, y, z, intensity]
    Output: (B, 3, 64, 1024) [class logits: 0=Terrain, 1=Static, 2=Dynamic]
    """
    def __init__(self, inChannels: int = 5, numClasses: int = 3) -> None:
        super().__init__()
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(inChannels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Encoder stages
        self.enc1 = ResBlock2D(32, 64, stride=(1, 2))   # (64, 512)
        self.enc2 = ResBlock2D(64, 128, stride=(2, 2))  # (32, 256)
        self.enc3 = ResBlock2D(128, 256, stride=(2, 2)) # (16, 128)

        # Dilated Bottleneck (Multi-scale receptive field)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Decoder stages with skip connections
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec3 = ResBlock2D(128 + 128, 128)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec2 = ResBlock2D(64 + 64, 64)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec1 = ResBlock2D(32 + 32, 32)

        # Output classification head
        self.head = nn.Conv2d(32, numClasses, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)      # (B, 32, 64, 1024)
        s1 = self.enc1(s0)     # (B, 64, 64, 512)
        s2 = self.enc2(s1)     # (B, 128, 32, 256)
        s3 = self.enc3(s2)     # (B, 256, 16, 128)

        b = self.bottleneck(s3) # (B, 256, 16, 128)

        d3 = self.dec3(torch.cat([self.up3(b), s2], dim=1))  # (B, 128, 32, 256)
        d2 = self.dec2(torch.cat([self.up2(d3), s1], dim=1)) # (B, 64, 64, 512)
        d1 = self.dec1(torch.cat([self.up1(d2), s0], dim=1)) # (B, 32, 64, 1024)

        logits = self.head(d1) # (B, 3, 64, 1024)
        return logits


# ---------------------------------------------------------------------------
# 2. Spherical Range-View Preprocessor
# ---------------------------------------------------------------------------

class SphericalRangeProjector:
    """Projects 3D LiDAR point clouds to 2D range images and maps predictions back to 3D."""
    def __init__(
        self,
        height: int = 64,
        width: int = 1024,
        fovUpDeg: float = 3.0,
        fovDownDeg: float = -25.0,
        maxRange: float = 100.0,
    ) -> None:
        self.height = height
        self.width = width
        self.fovUp = np.radians(fovUpDeg)
        self.fovDown = np.radians(fovDownDeg)
        self.totalFov = self.fovUp - self.fovDown
        self.maxRange = maxRange

    def projectToRangeImage(self, pointCloud: PointCloud) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Projects (N, 3) point cloud into (5, H, W) range image tensor.
        Returns:
            rangeImage: (5, H, W) float32 [range, x, y, z, intensity]
            rowIndices: (N,) int32
            colIndices: (N,) int32
        """
        pts = pointCloud.points
        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]
        intensity = pointCloud.intensity if pointCloud.intensity is not None else np.ones(len(pts), dtype=np.float32)

        ranges = np.sqrt(xs * xs + ys * ys + zs * zs)
        depth = np.maximum(ranges, 1e-4)

        # Spherical elevation angle
        yaw = np.arctan2(ys, xs)
        pitch = np.arcsin(np.clip(zs / depth, -1.0, 1.0))

        # Discretize to grid
        projY = 1.0 - (pitch - self.fovDown) / self.totalFov
        row = np.clip(np.floor(projY * self.height).astype(np.int32), 0, self.height - 1)

        projX = 0.5 * (1.0 - (yaw / np.pi))
        col = np.clip(np.floor(projX * self.width).astype(np.int32), 0, self.width - 1)

        rangeImage = np.zeros((5, self.height, self.width), dtype=np.float32)

        # Sort order to place closer points in front if rays overlap
        sortOrder = np.argsort(ranges)[::-1]
        sRow = row[sortOrder]
        sCol = col[sortOrder]

        rangeImage[0, sRow, sCol] = ranges[sortOrder] / self.maxRange
        rangeImage[1, sRow, sCol] = xs[sortOrder]
        rangeImage[2, sRow, sCol] = ys[sortOrder]
        rangeImage[3, sRow, sCol] = zs[sortOrder]
        rangeImage[4, sRow, sCol] = intensity[sortOrder]

        return rangeImage, row, col


# ---------------------------------------------------------------------------
# 3. Neural Segmenter Wrapper
# ---------------------------------------------------------------------------

class NeuralSegmenter(BaseSegmenter):
    """
    Genuine Deep-Learning Semantic Segmenter for 3D LiDAR Point Clouds.
    Inherits from BaseSegmenter and runs on CUDA GPU.
    """

    def __init__(
        self,
        weightsPath: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = FoveaRangeNet(inChannels=5, numClasses=3).to(self.device)
        self.projector = SphericalRangeProjector(height=64, width=1024)
        self.weightsLoaded = False

        if weightsPath is not None and Path(weightsPath).is_file():
            checkpoint = torch.load(weightsPath, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.weightsLoaded = True
            self.weightsSource = str(weightsPath)
        else:
            self.weightsSource = "Untrained / Randomly Initialized"

        self.model.eval()

    def predict(self, pointCloud: PointCloud) -> SegmentationResult:
        startTime = time.perf_counter()
        N = pointCloud.pointCount
        if N == 0:
            return SegmentationResult(
                labels=np.empty(0, dtype=np.uint8),
                confidence=np.empty(0, dtype=np.float32),
                uncertainty=np.empty(0, dtype=np.float32),
                probabilities=np.empty((0, 3), dtype=np.float32),
                latencyMs=0.0,
            )

        # 1. Project 3D points to 2D spherical range image
        rangeImg, rows, cols = self.projector.projectToRangeImage(pointCloud)

        # 2. Forward pass through neural network on GPU
        tensor = torch.from_numpy(rangeImg).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)  # (1, 3, 64, 1024)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (3, 64, 1024)

        # 3. Sample 2D predictions back to each original 3D point
        pointProbs = probs[:, rows, cols].T  # (N, 3)

        # Softmax Class, Confidence & Shannon Entropy Uncertainty
        predLabels = np.argmax(pointProbs, axis=1).astype(np.uint8)
        confidence = np.max(pointProbs, axis=1).astype(np.float32)

        # Normalized Shannon entropy: H = -sum(p * log(p)) / log(3)
        safeProbs = np.maximum(pointProbs, 1e-6)
        entropy = -np.sum(safeProbs * np.log(safeProbs), axis=1) / np.log(3.0)
        uncertainty = np.clip(entropy, 0.0, 1.0).astype(np.float32)

        latencyMs = (time.perf_counter() - startTime) * 1000.0

        return SegmentationResult(
            labels=predLabels,
            confidence=confidence,
            uncertainty=uncertainty,
            probabilities=pointProbs,
            latencyMs=latencyMs,
        )
