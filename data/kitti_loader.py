"""
SemanticKITTI Binary Scan and Label Loader.
Reads Velodyne HDL-64E .bin files and associated .label files,
integrating with PointCloud and 3-Class Target Remapping.
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np

from core.point_cloud import PointCloud
from segmentation.kitti_classes import mapRawLabelsToTarget, isDynamicClass


def loadKittiBin(binPath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a raw KITTI / SemanticKITTI .bin file.
    Format: float32 elements arranged as [x, y, z, remission] per point.
    Returns:
        points: (N, 3) float32
        remission: (N,) float32
    """
    path = Path(binPath)
    if not path.is_file():
        raise FileNotFoundError(f"KITTI bin file not found: {path}")

    scan = np.fromfile(str(path), dtype=np.float32)
    if scan.size % 4 != 0:
        raise ValueError(f"Corrupt .bin file: size {scan.size} is not divisible by 4.")

    scan = scan.reshape((-1, 4))
    points = scan[:, :3]
    remission = scan[:, 3]
    return points, remission


def loadKittiLabels(labelPath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a raw SemanticKITTI .label file.
    Format: uint32 per point.
      - Lower 16 bits: semantic class ID
      - Upper 16 bits: instance ID
    Returns:
        semanticLabels: (N,) uint16
        instanceIds: (N,) uint16
    """
    path = Path(labelPath)
    if not path.is_file():
        raise FileNotFoundError(f"SemanticKITTI label file not found: {path}")

    rawLabels = np.fromfile(str(path), dtype=np.uint32)
    semanticLabels = (rawLabels & 0xFFFF).astype(np.uint16)
    instanceIds = (rawLabels >> 16).astype(np.uint16)
    return semanticLabels, instanceIds


def loadSemanticKittiScan(
    binPath: Union[str, Path],
    labelPath: Optional[Union[str, Path]] = None,
    timestamp: float = 0.0,
) -> PointCloud:
    """
    Load a full SemanticKITTI scan into a PointCloud instance.
    If labelPath is provided, remaps classes to 3-class target representation.
    """
    points, remission = loadKittiBin(binPath)

    if labelPath is not None and Path(labelPath).is_file():
        rawLabels, instanceIds = loadKittiLabels(labelPath)
        targetLabels = mapRawLabelsToTarget(rawLabels)
        dynamicFlags = isDynamicClass(targetLabels)
    else:
        targetLabels = np.zeros(len(points), dtype=np.uint32)
        instanceIds = np.zeros(len(points), dtype=np.uint32)
        dynamicFlags = np.zeros(len(points), dtype=bool)

    return PointCloud(
        points=points,
        intensity=remission,
        semanticLabels=targetLabels,
        instanceIds=instanceIds,
        dynamicFlags=dynamicFlags,
        timestamp=timestamp,
    )


class KittiSequenceDataset:
    """Helper to iterate sequentially through a SemanticKITTI sequence folder."""

    def __init__(self, sequenceDir: Union[str, Path]) -> None:
        self.sequenceDir = Path(sequenceDir)
        self.velodyneDir = self.sequenceDir / "velodyne"
        self.labelsDir = self.sequenceDir / "labels"

        if not self.velodyneDir.is_dir():
            raise FileNotFoundError(f"Velodyne directory missing: {self.velodyneDir}")

        self.binFiles: List[Path] = sorted(list(self.velodyneDir.glob("*.bin")))
        self.hasLabels = self.labelsDir.is_dir()

    def __len__(self) -> int:
        return len(self.binFiles)

    def __getitem__(self, index: int) -> PointCloud:
        binFile = self.binFiles[index]
        labelFile = None
        if self.hasLabels:
            expectedLabel = self.labelsDir / f"{binFile.stem}.label"
            if expectedLabel.is_file():
                labelFile = expectedLabel

        return loadSemanticKittiScan(binFile, labelFile, timestamp=float(index) * 0.1)
