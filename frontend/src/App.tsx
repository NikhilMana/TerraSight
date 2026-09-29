import React, { useState, useEffect } from 'react';
import {
  Layers,
  Compass,
  TrendingUp,
  Activity,
  PlayCircle,
  AlertTriangle,
} from 'lucide-react';
import type {
  HealthInfo,
  ScenarioItem,
  PerceptionResponse,
  RiskWeights,
} from './types/perception';
import {
  fetchHealth,
  fetchScenarios,
  runScenario,
  uploadPointCloud,
  updateConfig,
  getExportUrl,
} from './services/api';

import { Header } from './components/Header';
import { LiveTelemetryHud } from './components/LiveTelemetryHud';
import { ControlSidebar } from './components/ControlSidebar';
import { PointCloudViewer3D } from './components/PointCloudViewer3D';
import { BevGridInspector } from './components/BevGridInspector';
import { ArchitectureComparator } from './components/ArchitectureComparator';
import { DynamicSimulationTracker } from './components/DynamicSimulationTracker';
import { BenchmarkAnalyticsLab } from './components/BenchmarkAnalyticsLab';

type ActiveTab = '3d-pointcloud' | 'bev-grid' | 'comparator' | 'simulation' | 'benchmarks';

export const App: React.FC = () => {
  // Global States
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioItem[]>([]);
  const [currentScenarioId, setCurrentScenarioId] = useState<string>('urban_patrol');
  const [activeTab, setActiveTab] = useState<ActiveTab>('3d-pointcloud');

  // Perception results
  const [perceptionData, setPerceptionData] = useState<PerceptionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Initial Load: Health & Scenarios
  useEffect(() => {
    fetchHealth()
      .then((h) => setHealth(h))
      .catch((err) => console.warn('Backend offline or health error:', err));

    fetchScenarios()
      .then((scList) => {
        setScenarios(scList);
        if (scList.length > 0) {
          executeScenario(scList[0].id);
        }
      })
      .catch((err) => {
        console.error('Failed to load scenarios:', err);
        setErrorMessage('Failed to connect to TerraSight Perception Engine. Ensure the backend is running on port 8000.');
        setIsLoading(false);
      });
  }, []);

  // Run Scenario
  const executeScenario = async (scenarioId: string, customWeights?: RiskWeights) => {
    setIsLoading(true);
    setErrorMessage(null);
    setCurrentScenarioId(scenarioId);

    try {
      const res = await runScenario(scenarioId, {
        maxDisplayPoints: 25000,
        enableRiskRefinement: true,
        riskWeights: customWeights,
      });
      setPerceptionData(res);
    } catch (err: any) {
      setErrorMessage(err.message || 'Perception execution failed');
    } finally {
      setIsLoading(false);
    }
  };

  // Upload Custom Point Cloud
  const handleUploadFile = async (file: File) => {
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const res = await uploadPointCloud(file, 25000, true);
      setPerceptionData(res);
      setCurrentScenarioId('custom_upload');
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to process uploaded point cloud');
    } finally {
      setIsLoading(false);
    }
  };

  // Update Parameters & Recalculate
  const handleUpdateWeights = async (weights: RiskWeights) => {
    try {
      await updateConfig(weights);
      executeScenario(currentScenarioId, weights);
    } catch (err: any) {
      console.error('Failed to update config:', err);
    }
  };

  // Export Files
  const handleExport = (format: 'json' | 'geojson' | 'csv') => {
    const url = getExportUrl(format);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `fovea_export.${format}`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: 'var(--bg-app)' }}>
      {/* 1. Header Bar */}
      <Header
        health={health}
        scenarios={scenarios}
        currentScenarioId={currentScenarioId}
        onSelectScenario={(id) => executeScenario(id)}
        onRefresh={() => executeScenario(currentScenarioId)}
        isLoading={isLoading}
        onExport={handleExport}
      />

      {/* 2. Real-Time Telemetry HUD */}
      <LiveTelemetryHud
        summary={perceptionData?.summary || null}
        latency={perceptionData?.latency || null}
        trenchCount={perceptionData?.trenchObstacles?.length || 0}
      />

      {/* 3. Error Banner (if any) */}
      {errorMessage && (
        <div
          style={{
            margin: '0 16px 12px 16px',
            padding: '12px 16px',
            borderRadius: '8px',
            background: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid var(--border-red)',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            color: '#fff',
            fontSize: '0.85rem',
          }}
        >
          <AlertTriangle size={18} color="var(--red)" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* 4. Main Body: Sidebar + Dynamic Workspace */}
      <div style={{ display: 'flex', flex: 1, padding: '0 16px 16px 16px', gap: '16px', minHeight: '620px' }}>
        {/* Left Sidebar */}
        <ControlSidebar
          scenarios={scenarios}
          currentScenarioId={currentScenarioId}
          onSelectScenario={(id) => executeScenario(id)}
          onUploadFile={handleUploadFile}
          onUpdateWeights={handleUpdateWeights}
          isLoading={isLoading}
          onExport={handleExport}
        />

        {/* Center Main Stage */}
        <main style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0, gap: '12px' }}>
          {/* Navigation Tabs */}
          <div className="glass-panel" style={{ padding: '6px 10px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button
              id="tab-3d"
              onClick={() => setActiveTab('3d-pointcloud')}
              className={activeTab === '3d-pointcloud' ? 'btn btn-primary' : 'btn btn-secondary'}
              style={{ fontSize: '0.8rem', padding: '6px 14px' }}
            >
              <Compass size={14} />
              Tactical 3D Perception
            </button>

            <button
              id="tab-bev"
              onClick={() => setActiveTab('bev-grid')}
              className={activeTab === 'bev-grid' ? 'btn btn-primary' : 'btn btn-secondary'}
              style={{ fontSize: '0.8rem', padding: '6px 14px' }}
            >
              <Layers size={14} />
              2.5D BEV & Elevation Inspector
            </button>

            <button
              id="tab-comparator"
              onClick={() => setActiveTab('comparator')}
              className={activeTab === 'comparator' ? 'btn btn-primary' : 'btn btn-secondary'}
              style={{ fontSize: '0.8rem', padding: '6px 14px' }}
            >
              <TrendingUp size={14} />
              Architecture Comparator (SIH Breakthrough)
            </button>

            <button
              id="tab-simulation"
              onClick={() => setActiveTab('simulation')}
              className={activeTab === 'simulation' ? 'btn btn-primary' : 'btn btn-secondary'}
              style={{ fontSize: '0.8rem', padding: '6px 14px' }}
            >
              <PlayCircle size={14} />
              Dynamic Tracking & Temporal Decay
            </button>

            <button
              id="tab-benchmarks"
              onClick={() => setActiveTab('benchmarks')}
              className={activeTab === 'benchmarks' ? 'btn btn-primary' : 'btn btn-secondary'}
              style={{ fontSize: '0.8rem', padding: '6px 14px' }}
            >
              <Activity size={14} />
              Defense Benchmarks & Compliance
            </button>
          </div>

          {/* Tab Views */}
          <div style={{ flex: 1, minHeight: '520px', display: 'flex', flexDirection: 'column' }}>
            {activeTab === '3d-pointcloud' && (
              <PointCloudViewer3D
                points={perceptionData?.points || []}
                isLoading={isLoading}
              />
            )}

            {activeTab === 'bev-grid' && (
              <BevGridInspector
                cells={perceptionData?.cells || []}
                trenches={perceptionData?.trenchObstacles || []}
              />
            )}

            {activeTab === 'comparator' && (
              <ArchitectureComparator
                currentScenarioId={currentScenarioId}
              />
            )}

            {activeTab === 'simulation' && (
              <DynamicSimulationTracker
                scenarioId={currentScenarioId}
              />
            )}

            {activeTab === 'benchmarks' && (
              <BenchmarkAnalyticsLab
                summary={perceptionData?.summary || null}
                latency={perceptionData?.latency || null}
                semanticMetrics={perceptionData?.semanticMetrics || null}
                bandDistribution={perceptionData?.bandDistribution || null}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
