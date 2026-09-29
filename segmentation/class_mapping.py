"""
SemanticKITTI to DRDO 3-Class Target Taxonomy Mapping.
Problem Statement: SIH26053 (DRDO) - Fovea-LiDAR

Explicit mapping from all 28 SemanticKITTI classes to the 3 target categories:
  0: Terrain / Drivable
  1: Static Obstacle
  2: Dynamic Obstacle
  -1: Ignored / Unmapped

Distinguishes moving vs parked entities and clearly documents class boundary decisions.
"""

from enum import IntEnum
from typing import Dict, Tuple, List
import numpy as np


class DRDOTargetClass(IntEnum):
    TERRAIN = 0
    STATIC_OBSTACLE = 1
    DYNAMIC_OBSTACLE = 2
    IGNORED = 255


# Full SemanticKITTI Raw IDs and Human-Readable Names
SEMANTIC_KITTI_NAMES: Dict[int, str] = {
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

# Detailed Categorization Logic:
# 1. TERRAIN / DRIVABLE (Target 0):
#    Surfaces suitable for vehicle navigation and ground elevation referencing.
#    Includes: road (40), parking (44), sidewalk (48), other-ground (49), lane-marking (60), terrain (72).
#
# 2. STATIC OBSTACLES (Target 1):
#    Rigid permanent structures and natural non-moving obstacles that pose collision risk.
#    Includes: building (50), fence (51), other-structure (52), vegetation (70), trunk (71), pole (80),
#              traffic-sign (81), other-object (99).
#    Parked / non-moving vehicles (10, 11, 13, 15, 16, 18, 20) are placed here when motion is absent,
#    or treated as potentially dynamic based on task configuration.
#
# 3. DYNAMIC OBSTACLES (Target 2):
#    Actors capable of motion or actively moving in the dynamic tactical environment.
#    Includes: moving-car (252), moving-bicyclist (253), moving-person (254), moving-motorcyclist (255),
#              moving-on-rails (256), moving-bus (257), moving-truck (258), moving-other-vehicle (259),
#              pedestrians (30), bicyclists (31), motorcyclists (32), and traffic vehicles (10, 13, 18).

RAW_TO_DRDO_TAXONOMY: Dict[int, DRDOTargetClass] = {
    # Ignored / Unlabeled
    0: DRDOTargetClass.IGNORED,
    1: DRDOTargetClass.IGNORED,
    # Drivable Terrain
    40: DRDOTargetClass.TERRAIN,          # road
    44: DRDOTargetClass.TERRAIN,          # parking
    48: DRDOTargetClass.TERRAIN,          # sidewalk
    49: DRDOTargetClass.TERRAIN,          # other-ground
    60: DRDOTargetClass.TERRAIN,          # lane-marking
    72: DRDOTargetClass.TERRAIN,          # terrain (grass/soil)
    # Static Structures & Obstacles
    50: DRDOTargetClass.STATIC_OBSTACLE,   # building
    51: DRDOTargetClass.STATIC_OBSTACLE,   # fence
    52: DRDOTargetClass.STATIC_OBSTACLE,   # other-structure
    70: DRDOTargetClass.STATIC_OBSTACLE,   # vegetation
    71: DRDOTargetClass.STATIC_OBSTACLE,   # trunk
    80: DRDOTargetClass.STATIC_OBSTACLE,   # pole
    81: DRDOTargetClass.STATIC_OBSTACLE,   # traffic-sign
    99: DRDOTargetClass.STATIC_OBSTACLE,   # other-object
    # Dynamic Actors & Vehicles
    10: DRDOTargetClass.DYNAMIC_OBSTACLE,  # car
    11: DRDOTargetClass.DYNAMIC_OBSTACLE,  # bicycle
    13: DRDOTargetClass.DYNAMIC_OBSTACLE,  # bus
    15: DRDOTargetClass.DYNAMIC_OBSTACLE,  # motorcycle
    16: DRDOTargetClass.DYNAMIC_OBSTACLE,  # on-rails
    18: DRDOTargetClass.DYNAMIC_OBSTACLE,  # truck
    20: DRDOTargetClass.DYNAMIC_OBSTACLE,  # other-vehicle
    30: DRDOTargetClass.DYNAMIC_OBSTACLE,  # person
    31: DRDOTargetClass.DYNAMIC_OBSTACLE,  # bicyclist
    32: DRDOTargetClass.DYNAMIC_OBSTACLE,  # motorcyclist
    252: DRDOTargetClass.DYNAMIC_OBSTACLE, # moving-car
    253: DRDOTargetClass.DYNAMIC_OBSTACLE, # moving-bicyclist
    254: DRDOTargetClass.DYNAMIC_OBSTACLE, # moving-person
    255: DRDOTargetClass.DYNAMIC_OBSTACLE, # moving-motorcyclist
    256: DRDOTargetClass.DYNAMIC_OBSTACLE, # moving-on-rails
    257: DRDOTargetClass.DYNAMIC_OBSTACLE, # moving-bus
    258: DRDOTargetClass.DYNAMIC_OBSTACLE, # moving-truck
    259: DRDOTargetClass.DYNAMIC_OBSTACLE, # moving-other-vehicle
}

# Color palette for DRDO Target Classes (RGB in [0, 1])
DRDO_CLASS_COLORS: Dict[int, Tuple[float, float, float]] = {
    DRDOTargetClass.TERRAIN.value: (0.18, 0.65, 0.34),          # Green
    DRDOTargetClass.STATIC_OBSTACLE.value: (0.25, 0.55, 0.90),   # Soft Blue
    DRDOTargetClass.DYNAMIC_OBSTACLE.value: (0.95, 0.22, 0.18),  # Bright Red
    DRDOTargetClass.IGNORED.value: (0.35, 0.35, 0.35),           # Dark Gray
}

DRDO_CLASS_NAMES: Dict[int, str] = {
    DRDOTargetClass.TERRAIN.value: "Terrain / Drivable",
    DRDOTargetClass.STATIC_OBSTACLE.value: "Static Obstacle",
    DRDOTargetClass.DYNAMIC_OBSTACLE.value: "Dynamic Obstacle",
    DRDOTargetClass.IGNORED.value: "Ignored / Unlabeled",
}

# Fast vector LUT (up to id 260)
DRDO_LOOKUP_TABLE = np.full(260, DRDOTargetClass.IGNORED.value, dtype=np.uint8)
for rawId, target in RAW_TO_DRDO_TAXONOMY.items():
    if rawId < len(DRDO_LOOKUP_TABLE):
        DRDO_LOOKUP_TABLE[rawId] = target.value


def remapRawKittiToDRDO(rawLabels: np.ndarray) -> np.ndarray:
    """Vectorized remapping from raw SemanticKITTI labels (0-259) to DRDO target classes (0, 1, 2, 255)."""
    safeLabels = np.where(rawLabels < len(DRDO_LOOKUP_TABLE), rawLabels, 0)
    return DRDO_LOOKUP_TABLE[safeLabels]
