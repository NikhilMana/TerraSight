/**
 * API service for TerraSight // Fovea-LiDAR backend.
 */

import type {
  HealthInfo,
  ScenarioItem,
  RiskWeights,
  PerceptionResponse,
  SimulationResponse,
  ComparatorResponse,
} from '../types/perception';

function getApiBaseUrl(): string {
  let base = (import.meta.env.VITE_API_BASE_URL as string) || '/api';
  base = base.trim().replace(/\/+$/, '');
  // If user passed root domain e.g. https://terrasight-backend.onrender.com
  if (base.startsWith('http') && !base.endsWith('/api')) {
    base = `${base}/api`;
  }
  return base;
}

const API_BASE = getApiBaseUrl();

export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchConfig(): Promise<any> {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) throw new Error(`Failed to fetch config: ${res.statusText}`);
  return res.json();
}

export async function updateConfig(weights: RiskWeights): Promise<any> {
  const res = await fetch(`${API_BASE}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(weights),
  });
  if (!res.ok) throw new Error(`Failed to update config: ${res.statusText}`);
  return res.json();
}

export async function fetchScenarios(): Promise<ScenarioItem[]> {
  const res = await fetch(`${API_BASE}/scenarios`);
  if (!res.ok) throw new Error(`Failed to fetch scenarios: ${res.statusText}`);
  return res.json();
}

export async function runScenario(
  scenarioId: string,
  options?: {
    maxDisplayPoints?: number;
    enableRiskRefinement?: boolean;
    riskWeights?: RiskWeights;
  }
): Promise<PerceptionResponse> {
  const res = await fetch(`${API_BASE}/scenarios/${scenarioId}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options || { maxDisplayPoints: 25000, enableRiskRefinement: true }),
  });
  if (!res.ok) throw new Error(`Failed to run scenario: ${res.statusText}`);
  return res.json();
}

export async function simulateScenario(
  scenarioId: string,
  numFrames: number = 8
): Promise<SimulationResponse> {
  const res = await fetch(`${API_BASE}/scenarios/${scenarioId}/simulate?numFrames=${numFrames}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Failed to simulate scenario: ${res.statusText}`);
  return res.json();
}

export async function uploadPointCloud(
  file: File,
  maxDisplayPoints: number = 25000,
  enableRiskRefinement: boolean = true
): Promise<PerceptionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('maxDisplayPoints', maxDisplayPoints.toString());
  formData.append('enableRiskRefinement', enableRiskRefinement.toString());

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errData.detail || 'Upload failed');
  }
  return res.json();
}

export async function runComparator(scenarioId: string = 'distant_threat'): Promise<ComparatorResponse> {
  const res = await fetch(`${API_BASE}/comparator?scenarioId=${encodeURIComponent(scenarioId)}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Comparator benchmark failed: ${res.statusText}`);
  return res.json();
}

export function getExportUrl(format: 'json' | 'geojson' | 'csv'): string {
  return `${API_BASE}/export/${format}`;
}
