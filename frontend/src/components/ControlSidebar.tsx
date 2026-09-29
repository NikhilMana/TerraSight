import React, { useState, useRef } from 'react';
import { Sliders, Upload, Play, Crosshair, Check } from 'lucide-react';
import type { ScenarioItem, RiskWeights } from '../types/perception';

interface ControlSidebarProps {
  scenarios: ScenarioItem[];
  currentScenarioId: string;
  onSelectScenario: (id: string) => void;
  onUploadFile: (file: File) => void;
  onUpdateWeights: (weights: RiskWeights) => void;
  isLoading: boolean;
  onExport: (format: 'json' | 'geojson' | 'csv') => void;
}

export const ControlSidebar: React.FC<ControlSidebarProps> = ({
  scenarios,
  currentScenarioId,
  onSelectScenario,
  onUploadFile,
  onUpdateWeights,
  isLoading,
  onExport,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);

  // Live tuning sliders
  const [weights, setWeights] = useState<RiskWeights>({
    weightDynamic: 0.40,
    weightProximity: 0.25,
    weightTraversability: 0.20,
    weightUncertainty: 0.15,
    refinementThreshold: 0.45,
  });

  const [applied, setApplied] = useState(false);

  const handleApply = () => {
    onUpdateWeights(weights);
    setApplied(true);
    setTimeout(() => setApplied(false), 2000);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onUploadFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <aside style={{ display: 'flex', flexDirection: 'column', gap: '16px', width: '340px', flexShrink: 0 }}>
      {/* 1. Tactical Scenario Selector */}
      <div className="glass-panel" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
          <Crosshair size={16} color="var(--cyan)" />
          <h3 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#fff', textTransform: 'uppercase' }}>
            Tactical Scenarios
          </h3>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {scenarios.map((sc) => {
            const isSelected = sc.id === currentScenarioId;
            return (
              <button
                key={sc.id}
                onClick={() => onSelectScenario(sc.id)}
                disabled={isLoading}
                style={{
                  display: 'block',
                  width: '100%',
                  textAlign: 'left',
                  padding: '10px 12px',
                  borderRadius: '8px',
                  border: isSelected ? '1px solid var(--cyan)' : '1px solid var(--border-subtle)',
                  background: isSelected ? 'rgba(0, 240, 255, 0.1)' : 'rgba(255, 255, 255, 0.02)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  boxShadow: isSelected ? '0 0 15px -5px var(--cyan-glow)' : 'none',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '0.82rem', fontWeight: isSelected ? 700 : 600, color: isSelected ? 'var(--cyan)' : '#fff' }}>
                    {sc.name}
                  </span>
                  <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    {sc.distanceRange}
                  </span>
                </div>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px', lineHeight: '1.3' }}>
                  {sc.keyFeature}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* 2. Custom Point Cloud Ingestion */}
      <div className="glass-panel" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
          <Upload size={16} color="var(--green)" />
          <h3 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#fff', textTransform: 'uppercase' }}>
            Custom Scan Ingestion
          </h3>
        </div>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(false);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: dragOver ? '2px dashed var(--cyan)' : '2px dashed var(--border-medium)',
            borderRadius: '8px',
            padding: '16px',
            textAlign: 'center',
            cursor: 'pointer',
            background: dragOver ? 'rgba(0, 240, 255, 0.08)' : 'rgba(0,0,0,0.2)',
            transition: 'all 0.2s ease',
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".bin,.pcd,.ply,.xyz,.txt,.csv,.npy"
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                onUploadFile(e.target.files[0]);
              }
            }}
          />
          <Upload size={22} color="var(--cyan)" style={{ margin: '0 auto 6px auto' }} />
          <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#fff' }}>
            Drop LiDAR File or Click to Browse
          </div>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            .bin (KITTI), .pcd, .ply, .xyz, .npy
          </div>
        </div>
      </div>

      {/* 3. Real-Time Risk & Fovea Parameter Tuning Studio */}
      <div className="glass-panel" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sliders size={16} color="var(--amber)" />
            <h3 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#fff', textTransform: 'uppercase' }}>
              Foveal Tuning Studio
            </h3>
          </div>
          <span className="badge badge-amber">&tau; = {weights.refinementThreshold}</span>
        </div>

        {/* Sliders */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {/* Refinement Threshold */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '3px' }}>
              <span>Refinement Threshold (&tau;)</span>
              <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>{weights.refinementThreshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.80"
              step="0.05"
              value={weights.refinementThreshold}
              onChange={(e) => setWeights({ ...weights, refinementThreshold: parseFloat(e.target.value) })}
            />
          </div>

          {/* Dynamic Weight */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '3px' }}>
              <span>Dynamic Actor Weight (w_dyn)</span>
              <span style={{ color: '#ef4444', fontFamily: 'var(--font-mono)' }}>{weights.weightDynamic.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={weights.weightDynamic}
              onChange={(e) => setWeights({ ...weights, weightDynamic: parseFloat(e.target.value) })}
            />
          </div>

          {/* Proximity Weight */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '3px' }}>
              <span>Ego-Proximity Weight (w_prox)</span>
              <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>{weights.weightProximity.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={weights.weightProximity}
              onChange={(e) => setWeights({ ...weights, weightProximity: parseFloat(e.target.value) })}
            />
          </div>

          {/* Step/Slope Weight */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '3px' }}>
              <span>Traversability Slope Weight (w_slope)</span>
              <span style={{ color: 'var(--green)', fontFamily: 'var(--font-mono)' }}>{weights.weightTraversability.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={weights.weightTraversability}
              onChange={(e) => setWeights({ ...weights, weightTraversability: parseFloat(e.target.value) })}
            />
          </div>

          {/* Uncertainty Weight */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '3px' }}>
              <span>Shannon Entropy Weight (w_unc)</span>
              <span style={{ color: 'var(--purple)', fontFamily: 'var(--font-mono)' }}>{weights.weightUncertainty.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={weights.weightUncertainty}
              onChange={(e) => setWeights({ ...weights, weightUncertainty: parseFloat(e.target.value) })}
            />
          </div>
        </div>

        <button
          id="apply-weights-btn"
          onClick={handleApply}
          disabled={isLoading}
          className="btn btn-primary"
          style={{ width: '100%', marginTop: '14px', padding: '8px' }}
        >
          {applied ? <Check size={14} color="#000" /> : <Play size={14} />}
          <span>{applied ? 'Parameters Applied!' : 'Recompute Spatial Fovea'}</span>
        </button>
      </div>

      {/* 4. Export Center */}
      <div className="glass-panel" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
          <h3 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#fff', textTransform: 'uppercase' }}>
            Export Artifacts
          </h3>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          <button
            onClick={() => onExport('geojson')}
            className="btn btn-secondary"
            style={{ fontSize: '0.75rem', padding: '6px' }}
          >
            2.5D GeoJSON
          </button>
          <button
            onClick={() => onExport('csv')}
            className="btn btn-secondary"
            style={{ fontSize: '0.75rem', padding: '6px' }}
          >
            Cells CSV
          </button>
          <button
            onClick={() => onExport('json')}
            className="btn btn-secondary"
            style={{ fontSize: '0.75rem', padding: '6px', gridColumn: 'span 2' }}
          >
            Full Perception JSON Report
          </button>
        </div>
      </div>
    </aside>
  );
};
