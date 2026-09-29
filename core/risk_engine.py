"""
Risk and Criticality Engine for Fovea-LiDAR.
Computes multi-criteria spatial risk scores:
  - Dynamic actor hazard
  - Ego-proximity hazard
  - Traversability obstacle height / slope hazard
  - Classification uncertainty / entropy
Determines which coarse cells should be dynamically refined to fine resolution.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class RiskWeights:
    weightDynamic: float = 0.40     # Dynamic object presence
    weightProximity: float = 0.25   # Proximity to ego-vehicle
    weightTraversability: float = 0.20  # Obstacle height clearance / slope
    weightUncertainty: float = 0.15     # Classification entropy / uncertainty
    refinementThreshold: float = 0.45   # Threshold to trigger local high-res refinement


class RiskScoringEngine:
    """
    Computes real-time risk scores for 2.5D cells and determines spatial refinement.
    """

    def __init__(self, weights: Optional[RiskWeights] = None) -> None:
        self.weights = weights if weights is not None else RiskWeights()

    def computeRisk(
        self,
        distanceMeters: float,
        isDynamic: bool,
        heightDiffMeters: float,
        semanticEntropy: float = 0.0,
        maxProximityRange: float = 50.0,
        criticalHeightStep: float = 0.30,  # 30 cm curb/obstacle threshold
    ) -> float:
        """
        Computes a normalized risk score in [0.0, 1.0] for a spatial cell.
        """
        w = self.weights

        # 1. Dynamicity Score: 1.0 if dynamic obstacle, else 0.0
        scoreDyn = 1.0 if isDynamic else 0.0

        # 2. Proximity Score: 1.0 at 0m, decaying to 0.0 at maxProximityRange
        scoreProx = max(0.0, 1.0 - (distanceMeters / maxProximityRange))

        # 3. Traversability Hazard: height diff relative to vehicle ground clearance
        scoreSlope = min(1.0, max(0.0, heightDiffMeters / criticalHeightStep))

        # 4. Uncertainty Score (normalized entropy)
        scoreUnc = float(np.clip(semanticEntropy, 0.0, 1.0))

        # Combined weighted risk
        risk = (
            w.weightDynamic * scoreDyn +
            w.weightProximity * scoreProx +
            w.weightTraversability * scoreSlope +
            w.weightUncertainty * scoreUnc
        )
        return float(np.clip(risk, 0.0, 1.0))

    def computeBatchRisk(
        self,
        distances: np.ndarray,
        isDynamic: np.ndarray,
        heightDiffs: np.ndarray,
        semanticEntropy: Optional[np.ndarray] = None,
        maxProximityRange: float = 50.0,
        criticalHeightStep: float = 0.30,
    ) -> np.ndarray:
        """
        Vectorized C-speed risk score calculation across thousands of cells.
        Returns:
            riskScores: (M,) float32 in [0.0, 1.0]
        """
        w = self.weights
        M = len(distances)
        if M == 0:
            return np.empty(0, dtype=np.float32)

        scoreDyn = np.where(isDynamic, 1.0, 0.0).astype(np.float32)
        scoreProx = np.clip(1.0 - (distances / maxProximityRange), 0.0, 1.0).astype(np.float32)
        scoreSlope = np.clip(heightDiffs / criticalHeightStep, 0.0, 1.0).astype(np.float32)

        if semanticEntropy is not None:
            scoreUnc = np.clip(semanticEntropy, 0.0, 1.0).astype(np.float32)
        else:
            scoreUnc = np.zeros(M, dtype=np.float32)

        risk = (
            w.weightDynamic * scoreDyn +
            w.weightProximity * scoreProx +
            w.weightTraversability * scoreSlope +
            w.weightUncertainty * scoreUnc
        )
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    def shouldRefine(self, riskScore: float) -> bool:
        """Returns True if the cell's risk exceeds the refinement threshold."""
        return riskScore >= self.weights.refinementThreshold
