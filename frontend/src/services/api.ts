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

export function getApiBaseUrl(): string {
  // 1. Check runtime localStorage override (allows immediate connection without redeploying frontend)
  const stored = typeof window !== 'undefined' ? localStorage.getItem('TERRASIGHT_API_BASE') : null;
  let base = stored || (import.meta.env.VITE_API_BASE_URL as string) || '/api';
  base = base.trim().replace(/\/+$/, '');
  // If user passed root domain e.g. https://terrasight-backend.onrender.com
  if (base.startsWith('http') && !base.endsWith('/api')) {
    base = `${base}/api`;
  }
  return base;
}

export function setApiBaseUrl(newUrl: string): void {
  let cleaned = newUrl.trim().replace(/\/+$/, '');
  if (cleaned.startsWith('http') && !cleaned.endsWith('/api')) {
    cleaned = `${cleaned}/api`;
  }
  if (typeof window !== 'undefined') {
    localStorage.setItem('TERRASIGHT_API_BASE', cleaned);
  }
}

async function handleResponse<T>(res: Response, endpointDesc: string): Promise<T> {
  const contentType = res.headers.get('content-type') || '';
  if (!res.ok) {
    if (contentType.includes('application/json')) {
      const errJson = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errJson.detail || `${endpointDesc} failed: ${res.statusText}`);
    }
    throw new Error(`${endpointDesc} failed with HTTP ${res.status} (${res.statusText})`);
  }
  if (!contentType.includes('application/json')) {
    throw new Error(
      `Received non-JSON response from backend. Your frontend may not be pointing to the Render backend service. Check Backend URL settings.`
    );
  }
  return res.json();
}

export async function fetchHealth(): Promise<HealthInfo> {
  const url = `${getApiBaseUrl()}/health`;
  const res = await fetch(url);
  return handleResponse<HealthInfo>(res, 'Health check');
}

export async function fetchConfig(): Promise<any> {
  const res = await fetch(`${getApiBaseUrl()}/config`);
  return handleResponse<any>(res, 'Config fetch');
}

export async function updateConfig(weights: RiskWeights): Promise<any> {
  const res = await fetch(`${getApiBaseUrl()}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(weights),
  });
  return handleResponse<any>(res, 'Config update');
}

export async function fetchScenarios(): Promise<ScenarioItem[]> {
  const res = await fetch(`${getApiBaseUrl()}/scenarios`);
  return handleResponse<ScenarioItem[]>(res, 'Scenarios fetch');
}

export async function runScenario(
  scenarioId: string,
  options?: {
    maxDisplayPoints?: number;
    enableRiskRefinement?: boolean;
    riskWeights?: RiskWeights;
  }
): Promise<PerceptionResponse> {
  const res = await fetch(`${getApiBaseUrl()}/scenarios/${scenarioId}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options || { maxDisplayPoints: 25000, enableRiskRefinement: true }),
  });
  return handleResponse<PerceptionResponse>(res, `Scenario ${scenarioId} run`);
}

export async function simulateScenario(
  scenarioId: string,
  numFrames: number = 8
): Promise<SimulationResponse> {
  const res = await fetch(`${getApiBaseUrl()}/scenarios/${scenarioId}/simulate?numFrames=${numFrames}`, {
    method: 'POST',
  });
  return handleResponse<SimulationResponse>(res, `Simulation for ${scenarioId}`);
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

  const res = await fetch(`${getApiBaseUrl()}/upload`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse<PerceptionResponse>(res, 'Point cloud upload');
}

export async function runComparator(scenarioId: string = 'distant_threat'): Promise<ComparatorResponse> {
  const res = await fetch(`${getApiBaseUrl()}/comparator?scenarioId=${encodeURIComponent(scenarioId)}`, {
    method: 'POST',
  });
  return handleResponse<ComparatorResponse>(res, 'Comparator benchmark');
}

export function getExportUrl(format: 'json' | 'geojson' | 'csv'): string {
  return `${getApiBaseUrl()}/export/${format}`;
}
