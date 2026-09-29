"""
Training & Weight Optimization Script for FoveaRangeNet.
Trains the lightweight spherical LiDAR segmentation network on CUDA GPU.
Outputs:
  - Saved model checkpoint: segmentation/weights/fovea_rangenet_v1.pt
  - Recorded training metrics: segmentation/weights/training_metrics.json
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.point_cloud import PointCloud
from data.kitti_loader import loadSemanticKittiScan
from data.synthetic_generator import generateSyntheticLiDARScan
from segmentation.neural_segmenter import FoveaRangeNet, SphericalRangeProjector, NeuralSegmenter
from segmentation.class_mapping import DRDOTargetClass


def trainFoveaRangeNet(
    epochs: int = 15,
    learningRate: float = 0.002,
    saveWeightsPath: Optional[Path] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  TRAINING FOVEA-RANGENET LiDAR SEGMENTATION MODEL")
    print("=" * 65)
    print(f"  Training Device : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Prepare Training & Validation Scans
    print("\n[Step 1] Preparing LiDAR training data...")
    scans = [
        generateSyntheticLiDARScan(numRings=64, horizontalResolutionDeg=0.25, randomSeed=101),
        generateSyntheticLiDARScan(numRings=64, horizontalResolutionDeg=0.25, randomSeed=202),
        generateSyntheticLiDARScan(numRings=64, horizontalResolutionDeg=0.25, randomSeed=303),
    ]
    valScan = generateSyntheticLiDARScan(numRings=64, horizontalResolutionDeg=0.25, randomSeed=999)

    projector = SphericalRangeProjector(height=64, width=1024)
    model = FoveaRangeNet(inChannels=5, numClasses=3).to(device)

    # Class weights to balance sparse dynamic actors vs dense terrain
    # Class 0: Terrain (~85%), Class 1: Static (~7%), Class 2: Dynamic (~8%)
    classWeights = torch.tensor([0.4, 1.8, 1.8], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=classWeights, ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Prepare 2D Range Image Training Tensors
    trainTensors = []
    trainLabels = []

    for s in scans:
        rImg, rows, cols = projector.projectToRangeImage(s)
        # 2D Label Map initialized to 255 (ignore)
        labelImg = np.full((64, 1024), 255, dtype=np.int64)

        # Ground truth labels in scan (mapping legacy 1,2,3 -> 0,1,2)
        rawGt = s.semanticLabels
        drdoGt = np.where(rawGt > 0, rawGt - 1, 255)
        labelImg[rows, cols] = drdoGt

        trainTensors.append(torch.from_numpy(rImg).unsqueeze(0).to(device))
        trainLabels.append(torch.from_numpy(labelImg).unsqueeze(0).to(device))

    valImg, valRows, valCols = projector.projectToRangeImage(valScan)
    valLabelImg = np.full((64, 1024), 255, dtype=np.int64)
    valGt = np.where(valScan.semanticLabels > 0, valScan.semanticLabels - 1, 255)
    valLabelImg[valRows, valCols] = valGt
    valTensor = torch.from_numpy(valImg).unsqueeze(0).to(device)
    valTarget = torch.from_numpy(valLabelImg).unsqueeze(0).to(device)

    # 2. Training Loop
    print(f"\n[Step 2] Training for {epochs} epochs...")
    history = []
    startTime = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epochLosses = []

        for x, y in zip(trainTensors, trainLabels):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epochLosses.append(loss.item())

        scheduler.step()
        meanTrainLoss = float(np.mean(epochLosses))

        # Validation
        model.eval()
        with torch.no_grad():
            valLogits = model(valTensor)
            valLoss = criterion(valLogits, valTarget).item()
            valPred = torch.argmax(valLogits, dim=1).squeeze(0).cpu().numpy()

        # Compute Validation IoU
        validValMask = valLabelImg != 255
        yTrue = valLabelImg[validValMask]
        yPred = valPred[validValMask]

        ious = []
        for c in range(3):
            tp = np.sum((yTrue == c) & (yPred == c))
            fp = np.sum((yTrue != c) & (yPred == c))
            fn = np.sum((yTrue == c) & (yPred != c))
            iou = tp / max(1, tp + fp + fn)
            ious.append(iou)

        valmIoU = float(np.mean(ious) * 100.0)

        history.append({
            "epoch": epoch,
            "trainLoss": round(meanTrainLoss, 4),
            "valLoss": round(valLoss, 4),
            "val_mIoU": round(valmIoU, 2),
            "terrain_IoU": round(ious[0] * 100.0, 2),
            "static_IoU": round(ious[1] * 100.0, 2),
            "dynamic_IoU": round(ious[2] * 100.0, 2),
        })

        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs} | Train Loss: {meanTrainLoss:.4f} | Val Loss: {valLoss:.4f} | Val mIoU: {valmIoU:.1f}% (Dynamic IoU: {ious[2]*100:.1f}%)")

    totalTrainTime = time.perf_counter() - startTime
    print(f"\n[Step 3] Training Complete in {totalTrainTime:.1f}s.")

    # 3. Save Weights and Metrics
    weightsDir = PROJECT_ROOT / "segmentation" / "weights"
    weightsDir.mkdir(parents=True, exist_ok=True)
    if saveWeightsPath is None:
        saveWeightsPath = weightsDir / "fovea_rangenet_v1.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": "FoveaRangeNet (Range-View Spherical U-Net)",
        "inChannels": 5,
        "numClasses": 3,
        "classes": ["Terrain", "Static Obstacle", "Dynamic Obstacle"],
        "final_mIoU": history[-1]["val_mIoU"],
        "history": history,
        "trainTimeSeconds": totalTrainTime,
    }
    torch.save(checkpoint, saveWeightsPath)
    print(f"  -> Model weights saved to: {saveWeightsPath}")

    metricsData = {
        "architecture": "FoveaRangeNet (Range-View Spherical U-Net)",
        "inChannels": 5,
        "numClasses": 3,
        "classes": ["Terrain", "Static Obstacle", "Dynamic Obstacle"],
        "final_mIoU": history[-1]["val_mIoU"],
        "final_terrain_IoU": history[-1]["terrain_IoU"],
        "final_static_IoU": history[-1]["static_IoU"],
        "final_dynamic_IoU": history[-1]["dynamic_IoU"],
        "history": history,
        "trainTimeSeconds": round(totalTrainTime, 2),
        "device": str(device),
        "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    metricsPath = weightsDir / "training_metrics.json"
    with open(metricsPath, "w") as f:
        json.dump(metricsData, f, indent=2)
    print(f"  -> Training metrics saved to: {metricsPath}")

    return checkpoint


if __name__ == "__main__":
    trainFoveaRangeNet(epochs=12)
