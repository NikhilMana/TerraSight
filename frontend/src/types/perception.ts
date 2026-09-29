/**
 * TypeScript contracts and types for TerraSight // Fovea-LiDAR.
 */

export interface SystemHardware {
  cudaAvailable: boolean;
  deviceCount: number;
  primaryDevice: string;
  cpuCount: number;
  cpuUsagePercent: number;
  ramTotalGb: number;
  ramAvailableGb: number;
}

export interface HealthInfo {
  status: string;
  system: string;
  version: string;
  competition: string;
  hardware: SystemHardware;
  model: {
    type: string;
    weightsLoaded: boolean;
    weightsPath: string;
  };
}

export interface ScenarioItem {
  id: string;
  name: string;
  category: string;
  description: string;
  distanceRange: string;
  keyFeature: string;
  beamCount: number;
}

export interface RiskWeights {
  weightDynamic: number;
  weightProximity: number;
  weightTraversability: number;
  weightUncertainty: number;
  refinementThreshold: number;
}

export interface SummaryMetrics {
  totalInputPoints: number;
  streamedPoints: number;
  totalActiveCells: number;
  refinedFoveaCells: number;
  cellReductionPercent: number;
  memoryUsageMb: number;
  memorySavedPercent: number;
  pointRetentionPercent: number;
  pointsDropped: number;
  conservationVerified: boolean;
}

export interface LatencyMetrics {
  segmentationMs: number;
  riskScoringMs: number;
  mappingMs: number;
  endToEndMs: number;
  fps: number;
}

export interface BandDistribution {
  band0_5cm: number;
  band1_10cm: number;
  band2_25cm: number;
  band3_50cm: number;
  band99_fovea5cm: number;
}

export interface TrenchObstacle {
  x: number;
  y: number;
  depth: number;
  stepHeight: number;
  type: string;
}

export interface SemanticMetrics {
  overallAccuracy: number;
  mIoU: number;
  perClassIoU: {
    Terrain: number;
    'Static Obstacle': number;
    'Dynamic Actor': number;
  };
}

export interface CellPayload {
  centerX: number;
  centerY: number;
  zMin: number;
  zMax: number;
  zMean: number;
  heightDiff: number;
  pointCount: number;
  semanticClass: number; // 0: Terrain, 1: Static, 2: Dynamic
  isDynamic: boolean;
  bandId: number;
  resolution: number;
}

// Point: [x, y, z, intensity, semanticClass, riskScore, isDynamic, assignedBand]
export type PointTuple = [number, number, number, number, number, number, number, number];

export interface PerceptionResponse {
  summary: SummaryMetrics;
  latency: LatencyMetrics;
  bandDistribution: BandDistribution;
  trenchObstacles: TrenchObstacle[];
  semanticMetrics: SemanticMetrics;
  points: PointTuple[];
  cells: CellPayload[];
}

export interface SimulationFrame {
  frameIndex: number;
  timestamp: number;
  activeCells: number;
  refinedFoveaCells: number;
  pointsCount: number;
  points: [number, number, number, number, number][]; // [x, y, z, label, risk]
}

export interface SimulationResponse {
  scenarioId: string;
  totalFrames: number;
  frames: SimulationFrame[];
}

export interface ComparatorModel {
  key: string;
  name: string;
  category: string;
  resolution: string;
  totalCells: number;
  occupiedCells: number;
  refinedCells: number;
  cellReductionPercent: number;
  memoryMb: number;
  memorySavedPercent: number;
  latencyMs: number;
  fps: number;
  pointRetentionPercent: number;
  pointsProcessed: number;
  pointsDropped: number;
}

export interface DistantObstacleComparison {
  evaluatedPointCount: number;
  meanDistanceMeters: number;
  distanceOnlyResolutionCm: number;
  distanceOnlyRmsErrorCm: number;
  foveaLidarResolutionCm: number;
  foveaLidarRmsErrorCm: number;
  precisionGainFactor: number;
}

export interface ComparatorResponse {
  models: ComparatorModel[];
  baselineCells: number;
  baselineMemoryMb: number;
  foveaCellReductionPercent: number;
  foveaMemorySavedPercent: number;
  foveaFps: number;
  distantObstacleComparison?: DistantObstacleComparison;
}
