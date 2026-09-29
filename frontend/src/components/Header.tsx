import React from 'react';
import { Shield, Cpu, Activity, Download, Radar, RefreshCw } from 'lucide-react';
import type { HealthInfo, ScenarioItem } from '../types/perception';

interface HeaderProps {
  health: HealthInfo | null;
  scenarios: ScenarioItem[];
  currentScenarioId: string;
  onSelectScenario: (id: string) => void;
  onRefresh: () => void;
  isLoading: boolean;
  onExport: (format: 'json' | 'geojson' | 'csv') => void;
  onOpenBackendConfig?: () => void;
  currentBackendUrl: string;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  scenarios,
  currentScenarioId,
  onSelectScenario,
  onRefresh,
  isLoading,
  onExport,
  onOpenBackendConfig,
  currentBackendUrl,
}) => {
  const formatHost = (url: string) => {
    try {
      if (url.startsWith('http')) {
        return new URL(url).host;
      }
      return url;
    } catch {
      return url;
    }
  };
  return (
    <header className="glass-panel" style={{ margin: '12px 16px', padding: '12px 20px', borderRadius: '12px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        {/* Left: Branding & Competition */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div
            style={{
              width: '42px',
              height: '42px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, rgba(0, 240, 255, 0.25), rgba(16, 185, 129, 0.15))',
              border: '1px solid var(--border-cyan)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--cyan)',
              boxShadow: '0 0 15px -3px var(--cyan-glow)',
            }}
          >
            <Radar size={24} />
          </div>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '0.05em', color: '#fff', textTransform: 'uppercase' }}>
                TerraSight <span style={{ color: 'var(--cyan)' }}>// Fovea-LiDAR</span>
              </h1>
              <span className="badge badge-cyan">v1.0 PROD</span>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '8px', marginTop: '2px' }}>
              <Shield size={13} color="var(--cyan)" />
              DRDO SIH26053 &bull; Adaptive Variable-Resolution 2.5D Perception Engine &bull; Team FoveaX
            </p>
          </div>
        </div>

        {/* Center: Quick Scenario Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Tactical Preset:
          </span>
          <select
            id="scenario-select"
            value={currentScenarioId}
            onChange={(e) => onSelectScenario(e.target.value)}
            disabled={isLoading}
            style={{
              background: 'var(--bg-secondary)',
              color: '#fff',
              border: '1px solid var(--border-medium)',
              borderRadius: '8px',
              padding: '7px 12px',
              fontSize: '0.85rem',
              fontWeight: 600,
              cursor: 'pointer',
              outline: 'none',
              fontFamily: 'var(--font-sans)',
            }}
          >
            {scenarios.map((sc) => (
              <option key={sc.id} value={sc.id} style={{ background: '#0c0e15', color: '#fff' }}>
                {sc.name} ({sc.distanceRange})
              </option>
            ))}
          </select>

          <button
            id="refresh-btn"
            onClick={onRefresh}
            disabled={isLoading}
            className="btn btn-secondary"
            title="Re-run perception on scenario"
            style={{ padding: '7px 12px' }}
          >
            <RefreshCw size={15} className={isLoading ? 'pulse-indicator' : ''} />
            <span>{isLoading ? 'Processing...' : 'Run'}</span>
          </button>
        </div>

        {/* Right: Hardware Status & Quick Export */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Backend Connection Pill */}
          <button
            id="backend-status-pill"
            onClick={onOpenBackendConfig}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 12px',
              borderRadius: '8px',
              background: health ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.15)',
              border: `1px solid ${health ? 'rgba(16, 185, 129, 0.35)' : 'rgba(239, 68, 68, 0.4)'}`,
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)',
              cursor: 'pointer',
              color: '#fff',
            }}
            title="Click to view or update Backend API URL"
          >
            <span className={`pulse-indicator ${health ? 'pulse-green' : 'pulse-red'}`} />
            <span style={{ color: health ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
              {health ? 'API ONLINE' : 'API OFFLINE'}
            </span>
            <span style={{ color: 'var(--text-dim)', fontSize: '0.72rem' }}>
              [{formatHost(currentBackendUrl)}]
            </span>
          </button>

          {/* Hardware Device Pill */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 12px',
              borderRadius: '8px',
              background: 'rgba(0, 0, 0, 0.4)',
              border: '1px solid var(--border-subtle)',
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)',
            }}
          >
            <span className={health?.hardware.cudaAvailable ? 'pulse-indicator pulse-green' : 'pulse-indicator pulse-cyan'} />
            <Cpu size={14} color="var(--text-muted)" />
            <span style={{ color: health?.hardware.cudaAvailable ? 'var(--green)' : 'var(--cyan)' }}>
              {health?.hardware.cudaAvailable ? health.hardware.primaryDevice.split(' ')[0] : 'SIMD CPU'}
            </span>
            <span style={{ color: 'var(--text-dim)' }}>|</span>
            <Activity size={13} color="var(--text-muted)" />
            <span style={{ color: '#fff' }}>{health?.hardware.cpuUsagePercent ?? 0}% CPU</span>
          </div>

          {/* Export Dropdown / Button */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <button
              id="export-geojson-btn"
              onClick={() => onExport('geojson')}
              className="btn btn-secondary"
              style={{ fontSize: '0.75rem', padding: '6px 10px' }}
              title="Export 2.5D Elevation Grid as GeoJSON"
            >
              <Download size={13} />
              GeoJSON
            </button>
            <button
              id="export-csv-btn"
              onClick={() => onExport('csv')}
              className="btn btn-secondary"
              style={{ fontSize: '0.75rem', padding: '6px 10px' }}
              title="Export Cells as CSV"
            >
              CSV
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
