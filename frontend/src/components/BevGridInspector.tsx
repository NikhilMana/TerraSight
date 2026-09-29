import React, { useRef, useEffect, useState } from 'react';
import { Target, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';
import type { CellPayload, TrenchObstacle } from '../types/perception';

interface BevGridInspectorProps {
  cells: CellPayload[];
  trenches: TrenchObstacle[];
}

export const BevGridInspector: React.FC<BevGridInspectorProps> = ({ cells, trenches }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Viewport transformation: scale (pixels per meter) and offset
  const [scale, setScale] = useState<number>(6.5);
  const [offset, setOffset] = useState<{ x: number; y: number }>({ x: 200, y: 350 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const startPanRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Hovered Cell Inspection
  const [hoveredCell, setHoveredCell] = useState<CellPayload | null>(null);
  const [cursorWorld, setCursorWorld] = useState<{ x: number; y: number } | null>(null);

  // Layer filters
  const [showGround, setShowGround] = useState<boolean>(true);
  const [showObstacles, setShowObstacles] = useState<boolean>(true);
  const [showFoveas, setShowFoveas] = useState<boolean>(true);
  const [showTrenches, setShowTrenches] = useState<boolean>(true);

  // Draw BEV Grid
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear background
    ctx.fillStyle = '#0a0d14';
    ctx.fillRect(0, 0, width, height);

    const worldToCanvas = (wx: number, wy: number) => {
      const cx = offset.x + wx * scale;
      const cy = offset.y - wy * scale;
      return { cx, cy };
    };

    // Draw background grid lines (every 10m)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = -20; x <= 100; x += 10) {
      const p1 = worldToCanvas(x, -50);
      const p2 = worldToCanvas(x, 50);
      ctx.beginPath();
      ctx.moveTo(p1.cx, p1.cy);
      ctx.lineTo(p2.cx, p2.cy);
      ctx.stroke();
    }
    for (let y = -50; y <= 50; y += 10) {
      const p1 = worldToCanvas(-20, y);
      const p2 = worldToCanvas(100, y);
      ctx.beginPath();
      ctx.moveTo(p1.cx, p1.cy);
      ctx.lineTo(p2.cx, p2.cy);
      ctx.stroke();
    }

    // Draw Concentric Range Rings (10m, 30m, 60m, 100m)
    const rings = [
      { r: 10, stroke: 'rgba(16, 185, 129, 0.4)', label: '10m (Band 0: 5cm)' },
      { r: 30, stroke: 'rgba(59, 130, 246, 0.4)', label: '30m (Band 1: 10cm)' },
      { r: 60, stroke: 'rgba(245, 158, 11, 0.4)', label: '60m (Band 2: 25cm)' },
      { r: 100, stroke: 'rgba(168, 85, 247, 0.4)', label: '100m (Band 3: 50cm)' },
    ];

    rings.forEach(({ r, stroke, label }) => {
      const origin = worldToCanvas(0, 0);
      ctx.beginPath();
      ctx.arc(origin.cx, origin.cy, r * scale, 0, Math.PI * 2);
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = stroke;
      ctx.font = '10px JetBrains Mono';
      ctx.fillText(label, origin.cx + r * scale + 4, origin.cy - 4);
    });

    // Draw Active 2.5D Cells
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      if (c.semanticClass === 0 && !showGround) continue;
      if (c.semanticClass === 1 && !showObstacles) continue;
      if (c.bandId === 99 && !showFoveas) continue;

      const p = worldToCanvas(c.centerX, c.centerY);
      const sizePx = Math.max(1.5, c.resolution * scale);

      if (c.bandId === 99) {
        ctx.fillStyle = '#ff0055';
      } else if (c.isDynamic) {
        ctx.fillStyle = '#ef4444';
      } else if (c.semanticClass === 1) {
        ctx.fillStyle = '#3b82f6';
      } else {
        const zNorm = Math.min(1.0, Math.max(0.0, (c.zMean + 2.0) / 1.0));
        ctx.fillStyle = `rgba(16, 185, 129, ${0.35 + zNorm * 0.4})`;
      }

      ctx.fillRect(p.cx - sizePx / 2, p.cy - sizePx / 2, sizePx, sizePx);
    }

    // Highlight Negative Obstacles / Trenches
    if (showTrenches && trenches.length > 0) {
      trenches.forEach((t) => {
        const pt = worldToCanvas(t.x, t.y);
        ctx.strokeStyle = '#eab308';
        ctx.lineWidth = 2;
        ctx.strokeRect(pt.cx - 6, pt.cy - 6, 12, 12);

        ctx.fillStyle = 'rgba(234, 179, 8, 0.4)';
        ctx.fillRect(pt.cx - 6, pt.cy - 6, 12, 12);
      });
    }

    // Draw Sensor Origin (Ego Vehicle)
    const origin = worldToCanvas(0, 0);
    ctx.fillStyle = '#00f0ff';
    ctx.beginPath();
    ctx.arc(origin.cx, origin.cy, 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(origin.cx, origin.cy);
    ctx.lineTo(origin.cx + 20, origin.cy);
    ctx.stroke();

    ctx.fillStyle = '#00f0ff';
    ctx.font = '10px Inter';
    ctx.fillText('EGO SENSOR', origin.cx - 30, origin.cy + 16);
  }, [cells, trenches, scale, offset, showGround, showObstacles, showFoveas, showTrenches]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsPanning(true);
    startPanRef.current = { x: e.clientX - offset.x, y: e.clientY - offset.y };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseCanvasX = e.clientX - rect.left;
    const mouseCanvasY = e.clientY - rect.top;

    if (isPanning) {
      setOffset({
        x: e.clientX - startPanRef.current.x,
        y: e.clientY - startPanRef.current.y,
      });
    }

    const wx = (mouseCanvasX - offset.x) / scale;
    const wy = -(mouseCanvasY - offset.y) / scale;
    setCursorWorld({ x: round2(wx), y: round2(wy) });

    let closest: CellPayload | null = null;
    let minDistSq = 1.5 * 1.5;
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      const dx = c.centerX - wx;
      const dy = c.centerY - wy;
      const d2 = dx * dx + dy * dy;
      if (d2 < minDistSq) {
        minDistSq = d2;
        closest = c;
      }
    }
    setHoveredCell(closest);
  };

  const handleMouseUp = () => setIsPanning(false);

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    setScale((prev) => Math.max(2.0, Math.min(30.0, prev * zoomFactor)));
  };

  function round2(v: number) {
    return Math.round(v * 100) / 100;
  }

  return (
    <div className="glass-panel" style={{ position: 'relative', width: '100%', height: '100%', minHeight: '520px', overflow: 'hidden' }}>
      <canvas
        ref={canvasRef}
        width={980}
        height={560}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        style={{ width: '100%', height: '100%', display: 'block', cursor: isPanning ? 'grabbing' : 'crosshair' }}
      />

      {/* Top Layer & Zoom Controls */}
      <div
        style={{
          position: 'absolute',
          top: '14px',
          left: '16px',
          right: '16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          pointerEvents: 'none',
        }}
      >
        <div className="glass-panel" style={{ padding: '6px 12px', display: 'flex', gap: '12px', pointerEvents: 'auto', alignItems: 'center' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#fff', textTransform: 'uppercase' }}>Layers:</span>
          <label style={{ fontSize: '0.75rem', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
            <input type="checkbox" checked={showGround} onChange={(e) => setShowGround(e.target.checked)} />
            Terrain
          </label>
          <label style={{ fontSize: '0.75rem', color: '#3b82f6', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
            <input type="checkbox" checked={showObstacles} onChange={(e) => setShowObstacles(e.target.checked)} />
            Obstacles
          </label>
          <label style={{ fontSize: '0.75rem', color: '#ff0055', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontWeight: 600 }}>
            <input type="checkbox" checked={showFoveas} onChange={(e) => setShowFoveas(e.target.checked)} />
            5cm Risk Foveas
          </label>
          <label style={{ fontSize: '0.75rem', color: '#eab308', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
            <input type="checkbox" checked={showTrenches} onChange={(e) => setShowTrenches(e.target.checked)} />
            Trenches
          </label>
        </div>

        <div className="glass-panel" style={{ padding: '6px 10px', display: 'flex', gap: '6px', pointerEvents: 'auto' }}>
          <button
            onClick={() => setScale((s) => Math.min(30, s * 1.2))}
            className="btn btn-secondary"
            style={{ padding: '4px 8px' }}
            title="Zoom In"
          >
            <ZoomIn size={14} />
          </button>
          <button
            onClick={() => setScale((s) => Math.max(2, s * 0.8))}
            className="btn btn-secondary"
            style={{ padding: '4px 8px' }}
            title="Zoom Out"
          >
            <ZoomOut size={14} />
          </button>
          <button
            onClick={() => {
              setScale(6.5);
              setOffset({ x: 200, y: 350 });
            }}
            className="btn btn-secondary"
            style={{ padding: '4px 8px' }}
            title="Reset View"
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      {/* Cell Telemetry Inspection Card */}
      <div
        className="glass-panel"
        style={{
          position: 'absolute',
          bottom: '16px',
          right: '16px',
          padding: '12px 16px',
          fontSize: '0.78rem',
          minWidth: '280px',
        }}
      >
        <div style={{ fontWeight: 700, color: 'var(--cyan)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Target size={14} />
          2.5D Cell Telemetry Inspector
        </div>

        {hoveredCell ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', fontFamily: 'var(--font-mono)' }}>
            <div>
              <span style={{ color: 'var(--text-dim)', fontSize: '0.7rem' }}>POSITION (X, Y):</span>
              <div style={{ color: '#fff', fontWeight: 600 }}>
                {hoveredCell.centerX.toFixed(2)}m, {hoveredCell.centerY.toFixed(2)}m
              </div>
            </div>
            <div>
              <span style={{ color: 'var(--text-dim)', fontSize: '0.7rem' }}>RESOLUTION:</span>
              <div style={{ color: hoveredCell.bandId === 99 ? '#ff0055' : 'var(--cyan)', fontWeight: 700 }}>
                {intRes(hoveredCell.resolution)} cm {hoveredCell.bandId === 99 ? '(Refined)' : `(Band ${hoveredCell.bandId})`}
              </div>
            </div>
            <div>
              <span style={{ color: 'var(--text-dim)', fontSize: '0.7rem' }}>ELEVATION Z:</span>
              <div style={{ color: '#fff' }}>
                {hoveredCell.zMean.toFixed(2)}m (dz: {hoveredCell.heightDiff.toFixed(2)}m)
              </div>
            </div>
            <div>
              <span style={{ color: 'var(--text-dim)', fontSize: '0.7rem' }}>POINTS / CLASS:</span>
              <div style={{ color: hoveredCell.semanticClass === 2 ? '#ef4444' : '#fff' }}>
                {hoveredCell.pointCount} pts &bull; {className(hoveredCell.semanticClass)}
              </div>
            </div>
          </div>
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontStyle: 'italic' }}>
            Hover cursor over any cell in the 2.5D grid to inspect real-time elevation & resolution telemetry.
            {cursorWorld && (
              <div style={{ marginTop: '4px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                Cursor World: X={cursorWorld.x}m, Y={cursorWorld.y}m
              </div>
            )}
          </div>
        )}
      </div>

      {/* Legend */}
      <div
        className="glass-panel"
        style={{
          position: 'absolute',
          bottom: '16px',
          left: '16px',
          padding: '8px 12px',
          fontSize: '0.72rem',
          display: 'flex',
          gap: '12px',
          alignItems: 'center',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '10px', height: '10px', backgroundColor: '#10b981' }} />
          <span>Terrain (5-50cm)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '10px', height: '10px', backgroundColor: '#3b82f6' }} />
          <span>Static Obstacle</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '10px', height: '10px', backgroundColor: '#ff0055' }} />
          <span style={{ color: '#ff0055', fontWeight: 700 }}>5cm Risk Fovea</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '10px', height: '10px', border: '2px solid #eab308' }} />
          <span style={{ color: '#eab308' }}>Trench Hazard</span>
        </div>
      </div>
    </div>
  );

  function intRes(r: number) {
    return Math.round(r * 100);
  }

  function className(c: number) {
    if (c === 0) return 'Terrain';
    if (c === 1) return 'Static';
    return 'Dynamic';
  }
};
