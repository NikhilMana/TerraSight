"""
SemanticKITTI Label Definitions and 3-Class Target Remapping.
Maps 28 raw SemanticKITTI classes into the 3 core DRDO target classes:
  1: Terrain / Drivable
  2: Static Obstacle
  3: Dynamic Obstacle
  0: Unlabeled / Outlier
"""

from enum import IntEnum
from typing import Dict, Tuple
import numpy as np


class TargetClass(IntEnum):
    UNLABELED = 0
    TERRAIN = 1
    STATIC_OBSTACLE = 2
    DYNAMIC_OBSTACLE = 3


# Standard SemanticKITTI Raw IDs
SEMANTIC_KITTI_LABEL_MAP: Dict[int, str] = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle",
}

# Mapping from raw SemanticKITTI label to TargetClass
RAW_TO_TARGET_MAP: Dict[int, TargetClass] = {
    0: TargetClass.UNLABELED,
    1: TargetClass.UNLABELED,
    # Dynamic Objects (vehicles, cyclists, pedestrians, moving objects)
    10: TargetClass.DYNAMIC_OBSTACLE,
    11: TargetClass.DYNAMIC_OBSTACLE,
    13: TargetClass.DYNAMIC_OBSTACLE,
    15: TargetClass.DYNAMIC_OBSTACLE,
    16: TargetClass.DYNAMIC_OBSTACLE,
    18: TargetClass.DYNAMIC_OBSTACLE,
    20: TargetClass.DYNAMIC_OBSTACLE,
    30: TargetClass.DYNAMIC_OBSTACLE,
    31: TargetClass.DYNAMIC_OBSTACLE,
    32: TargetClass.DYNAMIC_OBSTACLE,
    252: TargetClass.DYNAMIC_OBSTACLE,
    253: TargetClass.DYNAMIC_OBSTACLE,
    254: TargetClass.DYNAMIC_OBSTACLE,
    255: TargetClass.DYNAMIC_OBSTACLE,
    256: TargetClass.DYNAMIC_OBSTACLE,
    257: TargetClass.DYNAMIC_OBSTACLE,
    258: TargetClass.DYNAMIC_OBSTACLE,
    259: TargetClass.DYNAMIC_OBSTACLE,
    # Terrain / Drivable
    40: TargetClass.TERRAIN,
    44: TargetClass.TERRAIN,
    48: TargetClass.TERRAIN,
    49: TargetClass.TERRAIN,
    60: TargetClass.TERRAIN,
    72: TargetClass.TERRAIN,
    # Static Obstacles
    50: TargetClass.STATIC_OBSTACLE,
    51: TargetClass.STATIC_OBSTACLE,
    52: TargetClass.STATIC_OBSTACLE,
    70: TargetClass.STATIC_OBSTACLE,
    71: TargetClass.STATIC_OBSTACLE,
    80: TargetClass.STATIC_OBSTACLE,
    81: TargetClass.STATIC_OBSTACLE,
    99: TargetClass.STATIC_OBSTACLE,
}

# RGB Color scheme for target classes (normalized 0.0 to 1.0)
TARGET_CLASS_COLORS: Dict[TargetClass, Tuple[float, float, float]] = {
    TargetClass.UNLABELED: (0.4, 0.4, 0.4),          # Gray
    TargetClass.TERRAIN: (0.18, 0.55, 0.34),          # Forest Green
    TargetClass.STATIC_OBSTACLE: (0.30, 0.50, 0.85),   # Soft Blue
    TargetClass.DYNAMIC_OBSTACLE: (0.95, 0.25, 0.20),  # Bright Red
}

# Build fast lookup table array (up to max raw id 260)
LOOKUP_TABLE = np.zeros(260, dtype=np.uint8)
for rawId, target in RAW_TO_TARGET_MAP.items():
    if rawId < len(LOOKUP_TABLE):
        LOOKUP_TABLE[rawId] = target.value


def mapRawLabelsToTarget(rawLabels: np.ndarray) -> np.ndarray:
    """Vectorized remapping of raw SemanticKITTI labels to TargetClass IDs (0, 1, 2, 3)."""
    # Clip any outlier ids above table bounds to 0
    safeLabels = np.where(rawLabels < len(LOOKUP_TABLE), rawLabels, 0)
    return LOOKUP_TABLE[safeLabels]


def isDynamicClass(targetLabels: np.ndarray) -> np.ndarray:
    """Returns boolean mask where points are classified as dynamic obstacles."""
    return targetLabels == TargetClass.DYNAMIC_OBSTACLE.value
