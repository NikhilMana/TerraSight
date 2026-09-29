import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, SkipForward, SkipBack, Layers } from 'lucide-react';
import type { SimulationResponse, SimulationFrame } from '../types/perception';
import { simulateScenario } from '../services/api';

interface DynamicSimulationTrackerProps {
  scenarioId: string;
}

export const DynamicSimulationTracker: React.FC<DynamicSimulationTrackerProps> = ({ scenarioId }) => {
  const [data, setData] = useState<SimulationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [currentFrameIdx, setCurrentFrameIdx] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const timerRef = useRef<number | null>(null);

  const loadSimulation = (sid: string) => {
    setLoading(true);
    setError(null);
    setIsPlaying(false);

    simulateScenario(sid, 6)
      .then((res) => {
        setData(res);
        setCurrentFrameIdx(0);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Simulation error:', err);
        setError(err.message || 'Failed to generate temporal sequence.');
        setLoading(false);
      });
  };

  useEffect(() => {
    loadSimulation(scenarioId);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [scenarioId]);

  useEffect(() => {
    if (isPlaying && data && data.frames.length > 0) {
      timerRef.current = window.setInterval(() => {
        setCurrentFrameIdx((prev) => {
          if (prev >= data.frames.length - 1) {
            return 0;
          }
          return prev + 1;
        });
      }, 500 / playbackSpeed);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, data, playbackSpeed]);

  if (loading) {
    return (
      <div className="glass-panel" style={{ padding: '60px', textAlign: 'center' }}>
        <div className="pulse-indicator pulse-cyan" style={{ margin: '0 auto 16px auto', width: '24px', height: '24px' }} />
        <h3 style={{ color: 'var(--cyan)', fontSize: '1.1rem', fontWeight: 700 }}>
          GENERATING DYNAMIC TEMPORAL SEQUENCE...
        </h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '6px' }}>
          Simulating target motion, dynamic foveal tracking, and exponential temporal decay
        </p>
      </div>
    );
  }

  if (error || !data || data.frames.length === 0) {
    return (
      <div className="glass-panel" style={{ padding: '40px', textAlign: 'center' }}>
        <p style={{ color: '#ff8080', fontWeight: 600, marginBottom: '14px' }}>
          {error || 'No simulation frames available for this scenario.'}
        </p>
        <button
          onClick={() => loadSimulation(scenarioId)}
          className="btn btn-secondary"
          style={{ padding: '8px 18px', margin: '0 auto', fontSize: '0.82rem' }}
        >
          Retry Temporal Sequence
        </button>
      </div>
    );
  }

  const currentFrame: SimulationFrame = data.frames[currentFrameIdx] || data.frames[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Simulation Top Bar */}
      <div className="glass-panel" style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span className="badge badge-cyan">TEMPORAL SIMULATOR</span>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff' }}>
              Dynamic Actor Tracking & Temporal Decay Engine
            </h3>
          </div>

          {/* Playback Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={() => setCurrentFrameIdx((prev) => Math.max(0, prev - 1))}
              className="btn btn-secondary"
              style={{ padding: '6px 10px' }}
              title="Step Back"
            >
              <SkipBack size={14} />
            </button>
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="btn btn-primary"
              style={{ padding: '6px 14px' }}
            >
              {isPlaying ? <Pause size={15} /> : <Play size={15} />}
              <span>{isPlaying ? 'Pause' : 'Play Sequence'}</span>
            </button>
            <button
              onClick={() => setCurrentFrameIdx((prev) => Math.min(data.frames.length - 1, prev + 1))}
              className="btn btn-secondary"
              style={{ padding: '6px 10px' }}
              title="Step Forward"
            >
              <SkipForward size={14} />
            </button>

            <select
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
              style={{
                background: 'var(--bg-secondary)',
                color: '#fff',
                border: '1px solid var(--border-medium)',
                borderRadius: '6px',
                padding: '5px 8px',
                fontSize: '0.8rem',
              }}
            >
              <option value="0.5">0.5x</option>
              <option value="1">1.0x</option>
              <option value="2">2.0x</option>
            </select>
          </div>
        </div>

        {/* Timeline Scrubber */}
        <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '14px' }}>
          <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
            Frame {currentFrameIdx + 1} / {data.totalFrames} (t = {currentFrame.timestamp.toFixed(2)}s)
          </span>
          <input
            type="range"
            min="0"
            max={data.frames.length - 1}
            value={currentFrameIdx}
            onChange={(e) => setCurrentFrameIdx(parseInt(e.target.value))}
            style={{ flex: 1 }}
          />
        </div>
      </div>

      {/* Frame Telemetry Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
        <div className="glass-panel" style={{ padding: '14px 18px' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Time Timestamp</span>
          <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--cyan)', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            +{currentFrame.timestamp.toFixed(2)} s
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>10 Hz LiDAR sweep cycle</div>
        </div>

        <div className="glass-panel" style={{ padding: '14px 18px' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Active Grid Cells</span>
          <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#fff', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            {currentFrame.activeCells.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--green)', marginTop: '2px' }}>98.5% spatial reduction</div>
        </div>

        <div className="glass-panel" style={{ padding: '14px 18px' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Dynamic Fovea Cells</span>
          <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#ff0055', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            {currentFrame.refinedFoveaCells.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>5cm dynamic pursuit window</div>
        </div>

        <div className="glass-panel" style={{ padding: '14px 18px' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Decay Status</span>
          <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--amber)', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            ACTIVE (0.90^dt)
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>Stale cells pruned at 2.0s</div>
        </div>
      </div>

      {/* Frame Visual Simulation Canvas */}
      <div className="glass-panel" style={{ padding: '20px', minHeight: '360px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={15} color="var(--cyan)" />
            Top-Down Trajectory Radar: Target Tracking with Dynamic Refinement
          </span>
          <span className="badge badge-cyan">{currentFrame.pointsCount} Streamed Points</span>
        </div>

        <SimulationCanvas points={currentFrame.points} />
      </div>
    </div>
  );
};

const SimulationCanvas: React.FC<{ points: [number, number, number, number, number][] }> = ({ points }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#0a0d14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const scale = 5.5;
    const offsetX = 100;
    const offsetY = canvas.height / 2;

    [10, 30, 60, 100].forEach((r) => {
      ctx.beginPath();
      ctx.arc(offsetX, offsetY, r * scale, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    ctx.fillStyle = '#00f0ff';
    ctx.beginPath();
    ctx.arc(offsetX, offsetY, 5, 0, Math.PI * 2);
    ctx.fill();

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const cx = offsetX + p[0] * scale;
      const cy = offsetY - p[1] * scale;
      const label = p[3];
      const risk = p[4];

      if (label === 2 || risk >= 0.45) {
        ctx.fillStyle = '#ff0055';
        ctx.fillRect(cx - 2, cy - 2, 4, 4);
      } else if (label === 1) {
        ctx.fillStyle = '#3b82f6';
        ctx.fillRect(cx - 1, cy - 1, 2, 2);
      } else {
        ctx.fillStyle = 'rgba(16, 185, 129, 0.4)';
        ctx.fillRect(cx - 1, cy - 1, 2, 2);
      }
    }
  }, [points]);

  return (
    <canvas
      ref={canvasRef}
      width={900}
      height={320}
      style={{ width: '100%', height: '320px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}
    />
  );
};
