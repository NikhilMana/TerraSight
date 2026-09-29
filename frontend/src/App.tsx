import React, { useState, useEffect } from 'react';
import {
  Layers,
  Compass,
  TrendingUp,
  Activity,
  PlayCircle,
  AlertTriangle,
  Settings,
  RefreshCw,
  Zap,
  CheckCircle2,
  X,
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
  getApiBaseUrl,
  setApiBaseUrl,
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

  // Backend Connection Config & Wakeup
  const [currentBackendUrl, setCurrentBackendUrl] = useState<string>(getApiBaseUrl());
  const [showConfigModal, setShowConfigModal] = useState<boolean>(false);
  const [inputUrl, setInputUrl] = useState<string>(getApiBaseUrl());
  const [isWakingUp, setIsWakingUp] = useState<boolean>(false);
  const [wakeUpTimer, setWakeUpTimer] = useState<number>(0);

  // Primary Initializer
  const initConnection = async () => {
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const h = await fetchHealth();
      setHealth(h);

      const scList = await fetchScenarios();
      setScenarios(scList);
      if (scList.length > 0) {
        await executeScenario(scList[0].id);
      }
    } catch (err: any) {
      console.error('Failed to initialize connection:', err);
      setHealth(null);
      const url = getApiBaseUrl();
      setErrorMessage(
        err.message ||
          `Failed to connect to backend at ${url}. If using Render free tier, the container may be waking up from sleep.`
      );
      setIsLoading(false);
    }
  };

  useEffect(() => {
    initConnection();
  }, []);

  // Save New Backend URL
  const handleSaveBackendUrl = async (newUrl: string) => {
    setApiBaseUrl(newUrl);
    const updated = getApiBaseUrl();
    setCurrentBackendUrl(updated);
    setShowConfigModal(false);
    await initConnection();
  };

  // Trigger Render Free Tier Wake-Up Loop
  const triggerRenderWakeup = async () => {
    setIsWakingUp(true);
    setWakeUpTimer(45);
    setErrorMessage('Pinging Render backend container (awakening cold start container)...');

    const interval = setInterval(() => {
      setWakeUpTimer((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    for (let i = 0; i < 15; i++) {
      try {
        const h = await fetchHealth();
        if (h.status === 'operational') {
          clearInterval(interval);
          setIsWakingUp(false);
          await initConnection();
          return;
        }
      } catch {
        // Wait 3s before next health probe
        await new Promise((resolve) => setTimeout(resolve, 3000));
      }
    }

    clearInterval(interval);
    setIsWakingUp(false);
    setErrorMessage('Render wake-up timed out. Verify your Render service status in dashboard.render.com');
  };

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
        onOpenBackendConfig={() => setShowConfigModal(true)}
        currentBackendUrl={currentBackendUrl}
      />

      {/* 2. Real-Time Telemetry HUD */}
      <LiveTelemetryHud
        summary={perceptionData?.summary || null}
        latency={perceptionData?.latency || null}
        trenchCount={perceptionData?.trenchObstacles?.length || 0}
      />

      {/* 3. Error / Connection Diagnostic Banner */}
      {errorMessage && (
        <div
          className="glass-panel"
          style={{
            margin: '0 16px 14px 16px',
            padding: '14px 18px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(12, 14, 21, 0.8))',
            border: '1px solid var(--border-red)',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
            color: '#fff',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <AlertTriangle size={20} color="var(--red)" style={{ flexShrink: 0 }} />
              <div>
                <span style={{ fontWeight: 600, fontSize: '0.9rem', color: '#ff8080' }}>Backend Disconnected / Error: </span>
                <span style={{ fontSize: '0.85rem', color: '#f3f4f6' }}>{errorMessage}</span>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <button
                onClick={() => setShowConfigModal(true)}
                className="btn btn-secondary"
                style={{ fontSize: '0.78rem', padding: '6px 12px', background: 'rgba(0, 240, 255, 0.1)', borderColor: 'var(--border-cyan)' }}
              >
                <Settings size={14} color="var(--cyan)" />
                Configure Backend URL
              </button>

              <button
                onClick={triggerRenderWakeup}
                disabled={isWakingUp}
                className="btn btn-primary"
                style={{ fontSize: '0.78rem', padding: '6px 12px' }}
                title="Render free-tier instances sleep when idle. Click to send awakening requests."
              >
                <Zap size={14} />
                {isWakingUp ? `Waking Render (${wakeUpTimer}s)...` : 'Wake Up Render'}
              </button>

              <button
                onClick={initConnection}
                disabled={isLoading}
                className="btn btn-secondary"
                style={{ fontSize: '0.78rem', padding: '6px 12px' }}
              >
                <RefreshCw size={14} className={isLoading ? 'pulse-indicator' : ''} />
                Retry
              </button>
            </div>
          </div>

          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '16px', flexWrap: 'wrap', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '6px' }}>
            <span>Target API Base: <strong style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>{currentBackendUrl}</strong></span>
            <span>Tip: Ensure your Render web service is finished deploying and status says "Live".</span>
          </div>
        </div>
      )}

      {/* Backend Configuration Modal */}
      {showConfigModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(6px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            padding: '16px',
          }}
        >
          <div
            className="glass-panel"
            style={{
              width: '100%',
              maxWidth: '560px',
              padding: '24px',
              borderRadius: '16px',
              border: '1px solid var(--border-cyan)',
              boxShadow: '0 20px 40px rgba(0,0,0,0.8), 0 0 25px -5px var(--cyan-glow)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Settings size={22} color="var(--cyan)" />
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#fff' }}>Configure Backend API Endpoint</h3>
              </div>
              <button
                onClick={() => setShowConfigModal(false)}
                className="btn-icon"
                style={{ color: 'var(--text-muted)' }}
              >
                <X size={18} />
              </button>
            </div>

            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '18px', lineHeight: 1.5 }}>
              Connect your Vercel frontend to your live Render backend container or a local FastAPI server.
            </p>

            {/* Quick Presets */}
            <div style={{ marginBottom: '16px' }}>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Quick Presets:</span>
              <div style={{ display: 'flex', gap: '8px', marginTop: '6px', flexWrap: 'wrap' }}>
                <button
                  type="button"
                  onClick={() => setInputUrl('https://terrasight-backend.onrender.com')}
                  className="btn btn-secondary"
                  style={{ fontSize: '0.75rem', padding: '5px 10px' }}
                >
                  Render Template (terrasight-backend)
                </button>
                <button
                  type="button"
                  onClick={() => setInputUrl('http://localhost:8000')}
                  className="btn btn-secondary"
                  style={{ fontSize: '0.75rem', padding: '5px 10px' }}
                >
                  Localhost (http://localhost:8000)
                </button>
              </div>
            </div>

            {/* URL Input */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: '#fff', marginBottom: '6px' }}>
                Backend Public URL:
              </label>
              <input
                type="text"
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                placeholder="https://terrasight-backend.onrender.com"
                style={{
                  width: '100%',
                  background: 'rgba(0,0,0,0.5)',
                  border: '1px solid var(--border-medium)',
                  borderRadius: '8px',
                  padding: '10px 14px',
                  color: '#fff',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.88rem',
                  outline: 'none',
                }}
              />
              <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '4px', display: 'block' }}>
                Appends <code>/api</code> automatically if omitted. Saves to browser <code>localStorage</code>.
              </span>
            </div>

            {/* Actions */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button
                type="button"
                onClick={() => setShowConfigModal(false)}
                className="btn btn-secondary"
                style={{ padding: '8px 16px' }}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => handleSaveBackendUrl(inputUrl)}
                className="btn btn-primary"
                style={{ padding: '8px 20px' }}
              >
                <CheckCircle2 size={16} />
                Save &amp; Connect
              </button>
            </div>
          </div>
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
