"""
Fovea-LiDAR Real-Time 2.5D Perception & Telemetry Dashboard.
Provides real-time multi-view rendering:
  - Top-Left: Raw 3D LiDAR Point Cloud with Semantic Colors
  - Top-Right: Scene Criticality & Spatial Risk Heatmap
  - Bottom-Left: Adaptive Resolution Allocation & Refinement Foveas
  - Bottom-Right: 2.5D Semantic Elevation Surface Map
  - Live HUD Telemetry: FPS, Active Cells, Cell Reduction %, Memory Saved %
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from core.point_cloud import PointCloud
from core.grid_fovea import FoveaLiDARGrid, DEFAULT_FOVEA_BANDS
from core.risk_engine import RiskScoringEngine
from segmentation.base_segmenter import BaseSegmenter
from segmentation.neural_segmenter import NeuralSegmenter
from segmentation.geometric_segmenter import GeometricSegmenter
from segmentation.class_mapping import DRDOTargetClass, DRDO_CLASS_COLORS, DRDO_CLASS_NAMES


class FoveaDashboardVisualizer:
    """
    Real-time multi-view perception dashboard with HUD telemetry for DRDO SIH presentation.
    """

    def __init__(
        self,
        grid: Optional[FoveaLiDARGrid] = None,
        riskEngine: Optional[RiskScoringEngine] = None,
        segmenter: Optional[BaseSegmenter] = None,
        roiRange: Tuple[float, float, float, float] = (-15.0, 65.0, -30.0, 30.0),
    ) -> None:
        self.riskEngine = riskEngine if riskEngine is not None else RiskScoringEngine()
        self.grid = grid if grid is not None else FoveaLiDARGrid(
            bands=DEFAULT_FOVEA_BANDS, enableRiskRefinement=True, riskEngine=self.riskEngine
        )
        if segmenter is not None:
            self.segmenter = segmenter
        else:
            weightsPath = Path(__file__).resolve().parent.parent / "segmentation" / "weights" / "fovea_rangenet_v1.pt"
            if weightsPath.is_file():
                self.segmenter = NeuralSegmenter(weightsPath=str(weightsPath))
            else:
                self.segmenter = GeometricSegmenter()

        self.minX, self.maxX, self.minY, self.maxY = roiRange

    def renderFrame(
        self,
        pointCloud: PointCloud,
        outputPath: Optional[Path] = None,
        showInteractive: bool = False,
        uniformBaselineCells: int = 4_000_000,
        uniformBaselineMb: float = 87.74,
    ) -> plt.Figure:
        """
        Process scan and render the 4-panel dashboard with live telemetry HUD.
        """
        # 1. Pipeline Execution
        segResult = self.segmenter.predict(pointCloud)
        predLabels = segResult.labels
        entropy = segResult.uncertainty

        dists = pointCloud.calculateDistances2D()
        heightRel = np.maximum(0.0, pointCloud.points[:, 2] + 1.5)

        riskScores = self.riskEngine.computeBatchRisk(
            distances=dists,
            isDynamic=pointCloud.dynamicFlags,
            heightDiffs=heightRel,
            semanticEntropy=entropy,
        )

        self.grid.update(pointCloud)
        metrics = self.grid.getMetrics(uniformBaselineCells=uniformBaselineCells, uniformBaselineMb=uniformBaselineMb)

        # 2. Setup Canvas
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor="#0e0e12")
        fig.subplots_adjust(top=0.91, bottom=0.06, left=0.06, right=0.96, hspace=0.25, wspace=0.22)

        # ---------------- Panel 1: Semantic Point Cloud ----------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#15151c")
        if np.max(predLabels) == 3:
            normLabels = np.where(predLabels > 0, predLabels - 1, 255)
        else:
            normLabels = predLabels

        semColors = {
            DRDOTargetClass.TERRAIN.value: ("#10b981", "Terrain / Drivable", 0.3, 1),
            DRDOTargetClass.STATIC_OBSTACLE.value: ("#3b82f6", "Static Obstacle", 0.7, 4),
            DRDOTargetClass.DYNAMIC_OBSTACLE.value: ("#ef4444", "Dynamic Actor [High Risk]", 0.9, 6),
        }
        for cVal, (col, cName, alphaVal, ptSize) in semColors.items():
            mask = normLabels == cVal
            if np.any(mask):
                ax1.scatter(
                    pointCloud.points[mask, 0],
                    pointCloud.points[mask, 1],
                    c=col,
                    s=ptSize,
                    alpha=alphaVal,
                    label=cName,
                    edgecolors="none",
                )
        segType = "DEEP LEARNING (FoveaRangeNet)" if isinstance(self.segmenter, NeuralSegmenter) else "GEOMETRIC BASELINE"
        ax1.set_title(f"1. SEMANTIC SEGMENTATION [{segType}]", color="white", fontsize=11, pad=8)
        ax1.legend(facecolor="#1c1c24", edgecolor="#333", labelcolor="white", fontsize=8, loc="upper left")

        # ---------------- Panel 2: Continuous Risk Heatmap ----------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#15151c")
        sc2 = ax2.scatter(
            pointCloud.points[:, 0],
            pointCloud.points[:, 1],
            c=riskScores,
            cmap="inferno",
            s=3,
            alpha=0.8,
            vmin=0.0,
            vmax=1.0,
        )
        cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label(r"Criticality Risk $\mathcal{R} \in [0, 1]$", color="white", fontsize=10)
        cbar2.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="white")
        ax2.set_title(r"2. RISK HEATMAP & UNCERTAINTY $\mathcal{R}(x, y)$" + "\n(Dynamicity + Proximity + Slope + Uncertainty)", color="white", fontsize=11, pad=8)

        # ---------------- Panel 3: Adaptive Resolution Allocation ----------------
        ax3 = axes[1, 0]
        ax3.set_facecolor("#15151c")
        bandColors = {0: "#10b981", 1: "#3b82f6", 2: "#f59e0b", 3: "#8b5cf6"}
        for b in self.grid.bands:
            if b.bandId in self.grid.bandData:
                bd = self.grid.bandData[b.bandId]
                ax3.scatter(
                    bd.centerX,
                    bd.centerY,
                    s=bd.resolution * 35,
                    c=bandColors[b.bandId],
                    alpha=0.5,
                    label=f"Band {b.bandId}: {int(b.resolution * 100)}cm",
                )

        if 99 in self.grid.bandData:
            bdRef = self.grid.bandData[99]
            ax3.scatter(
                bdRef.centerX,
                bdRef.centerY,
                s=14,
                c="#ef4444",
                alpha=0.9,
                label=f"Risk-Refined Fovea: 5cm ({len(bdRef.centerX):,} cells)",
            )

        theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
        for r in [10, 30, 60]:
            ax3.plot(r * np.cos(theta), r * np.sin(theta), color="#666677", linestyle="--", alpha=0.5)

        ax3.plot(0, 0, marker="o", color="yellow", markersize=8, label="Ego LiDAR")
        ax3.set_title("3. BASE RESOLUTION BANDS & RISK-REFINED FOVEAS\n(5/10/25/50 cm Distance Bands + 5 cm Dynamic Fovea)", color="white", fontsize=11, pad=8)
        ax3.legend(facecolor="#1c1c24", edgecolor="#333", labelcolor="white", fontsize=8, loc="upper left")

        # ---------------- Panel 4: 2.5D Semantic Elevation Surface Map ----------------
        ax4 = axes[1, 1]
        ax4.set_facecolor("#15151c")
        allXs, allYs, allElevs = [], [], []
        for bd in self.grid.bandData.values():
            allXs.extend(bd.centerX)
            allYs.extend(bd.centerY)
            allElevs.extend(bd.zMean)

        sc4 = ax4.scatter(allXs, allYs, c=allElevs, cmap="plasma", s=3, alpha=0.8, vmin=-2.0, vmax=2.5)
        cbar4 = plt.colorbar(sc4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label(r"Elevation $z_{mean}$ (m)", color="white", fontsize=10)
        cbar4.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar4.ax.yaxis.get_ticklabels(), color="white")
        ax4.set_title("4. SEMANTIC 2.5D ELEVATION MAP [ACTIVE CELLS]\n(Exact Elevation + Semantic + Confidence Payload)", color="white", fontsize=11, pad=8)

        # Standardize axes limits and styling
        for ax in axes.flatten():
            ax.set_xlim(self.minX, self.maxX)
            ax.set_ylim(self.minY, self.maxY)
            ax.set_aspect("equal")
            ax.set_xlabel("X (Forward, meters)", color="white", fontsize=9)
            ax.set_ylabel("Y (Lateral, meters)", color="white", fontsize=9)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#333344")
            ax.grid(True, color="#222230", linestyle="--", alpha=0.4)

        # ---------------- Live HUD Telemetry Header ----------------
        hudText = (
            f" FOVEA-LiDAR TELEMETRY | Speed: {metrics['fps']:.1f} FPS ({metrics['latencyMs']:.1f} ms) | "
            f"Active Cells: {metrics['totalActiveCells']:,} ({metrics['cellReductionPercent']:.1f}% reduction) | "
            f"Memory: {metrics['memoryUsageMb']:.2f} MB ({metrics['memorySavedPercent']:.1f}% saved) | "
            f"Point Retention: {metrics['pointRetentionRatePercent']:.1f}% "
        )
        fig.suptitle(hudText, color="#38bdf8", fontsize=12, fontweight="bold", y=0.96)

        if outputPath is not None:
            outPathObj = Path(outputPath)
            outPathObj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outPathObj, dpi=180, facecolor=fig.get_facecolor())
            print(f"  -> Saved perception dashboard snapshot to: {outPathObj}")

        if showInteractive:
            plt.show()
        else:
            plt.close()

        return fig
