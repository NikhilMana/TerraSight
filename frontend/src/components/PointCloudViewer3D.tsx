import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Eye, RotateCcw, Compass, Layers, ShieldAlert, Sparkles } from 'lucide-react';
import type { PointTuple } from '../types/perception';

interface PointCloudViewer3DProps {
  points: PointTuple[];
  isLoading: boolean;
}

type ColorMode = 'semantic' | 'risk' | 'bands' | 'elevation';

export const PointCloudViewer3D: React.FC<PointCloudViewer3DProps> = ({ points, isLoading }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const pointsMeshRef = useRef<THREE.Points | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const [colorMode, setColorMode] = useState<ColorMode>('semantic');
  const [pointSize, setPointSize] = useState<number>(3.0);
  const [showRings, setShowRings] = useState<boolean>(true);

  // Mouse interaction state
  const isDraggingRef = useRef(false);
  const isRightDraggingRef = useRef(false);
  const previousMousePositionRef = useRef({ x: 0, y: 0 });
  const cameraAngleRef = useRef({ theta: Math.PI / 4, phi: Math.PI / 3, radius: 45 });
  const targetRef = useRef(new THREE.Vector3(15, 0, -1));

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 550;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0c12);
    scene.fog = new THREE.FogExp2(0x0a0c12, 0.008);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 300);
    cameraRef.current = camera;
    updateCameraPosition();

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = false;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Ground Grid Helper
    const gridHelper = new THREE.GridHelper(120, 24, 0x1f293d, 0x141a29);
    gridHelper.position.y = -1.73;
    gridHelper.rotation.x = 0;
    scene.add(gridHelper);

    // Sensor origin marker (Ego Vehicle)
    const egoGroup = new THREE.Group();
    // Sensor pillar
    const pillarGeo = new THREE.CylinderGeometry(0.12, 0.12, 1.2, 16);
    const pillarMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, wireframe: true });
    const pillar = new THREE.Mesh(pillarGeo, pillarMat);
    pillar.position.y = -1.13;
    egoGroup.add(pillar);

    // Vehicle bounding wireframe box (4m x 2m x 1.5m)
    const boxGeo = new THREE.BoxGeometry(4.0, 1.8, 1.4);
    const boxEdges = new THREE.EdgesGeometry(boxGeo);
    const boxLine = new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.7 }));
    boxLine.position.set(0, 0, -1.0);
    egoGroup.add(boxLine);

    // Sensor pulse beacon at origin
    const beaconGeo = new THREE.SphereGeometry(0.3, 16, 16);
    const beaconMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
    const beacon = new THREE.Mesh(beaconGeo, beaconMat);
    beacon.position.set(0, 0, 0);
    egoGroup.add(beacon);

    scene.add(egoGroup);

    // Concentric Range Rings (10m, 30m, 60m, 100m)
    const ringsGroup = new THREE.Group();
    ringsGroup.name = 'rangeRings';
    const ringRanges = [
      { r: 10, color: 0x10b981 },
      { r: 30, color: 0x3b82f6 },
      { r: 60, color: 0xf59e0b },
      { r: 100, color: 0xa855f7 },
    ];

    ringRanges.forEach(({ r, color }) => {
      const ringGeo = new THREE.BufferGeometry();
      const segments = 128;
      const ringPts = [];
      for (let i = 0; i <= segments; i++) {
        const theta = (i / segments) * Math.PI * 2;
        ringPts.push(Math.cos(theta) * r, Math.sin(theta) * r, -1.73);
      }
      ringGeo.setAttribute('position', new THREE.Float32BufferAttribute(ringPts, 3));
      const ringLine = new THREE.Line(
        ringGeo,
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 })
      );
      ringLine.rotation.x = -Math.PI / 2;
      ringsGroup.add(ringLine);
    });
    scene.add(ringsGroup);

    // Animation Loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();

    // Resize Handler
    const handleResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      renderer.dispose();
    };
  }, []);

  function updateCameraPosition() {
    if (!cameraRef.current) return;
    const { theta, phi, radius } = cameraAngleRef.current;
    const target = targetRef.current;

    const x = target.x + radius * Math.sin(phi) * Math.sin(theta);
    const y = target.y + radius * Math.cos(phi);
    const z = target.z + radius * Math.sin(phi) * Math.cos(theta);

    cameraRef.current.position.set(x, y, z);
    cameraRef.current.lookAt(target);
  }

  // Update Point Cloud Geometry and Colors when points or colorMode change
  useEffect(() => {
    if (!sceneRef.current) return;

    if (pointsMeshRef.current) {
      sceneRef.current.remove(pointsMeshRef.current);
      pointsMeshRef.current.geometry.dispose();
      (pointsMeshRef.current.material as THREE.Material).dispose();
      pointsMeshRef.current = null;
    }

    if (!points || points.length === 0) return;

    const N = points.length;
    const positions = new Float32Array(N * 3);
    const colors = new Float32Array(N * 3);

    // Color palettes
    const cTerrain = new THREE.Color(0x10b981);
    const cStatic = new THREE.Color(0x3b82f6);
    const cDynamic = new THREE.Color(0xef4444);

    const cBand0 = new THREE.Color(0x10b981); // 5cm
    const cBand1 = new THREE.Color(0x3b82f6); // 10cm
    const cBand2 = new THREE.Color(0xf59e0b); // 25cm
    const cBand3 = new THREE.Color(0xa855f7); // 50cm
    const cBand99 = new THREE.Color(0xff0055); // 5cm Refined Fovea

    for (let i = 0; i < N; i++) {
      const p = points[i];
      // p: [x, y, z, intensity, semanticClass, riskScore, isDynamic, assignedBand]
      const px = p[0];
      const py = p[1];
      const pz = p[2];
      const sClass = p[4];
      const risk = p[5];
      const band = p[7];

      positions[i * 3 + 0] = -py;
      positions[i * 3 + 1] = pz;
      positions[i * 3 + 2] = -px;

      const c = new THREE.Color();
      if (colorMode === 'semantic') {
        if (sClass === 0) c.copy(cTerrain);
        else if (sClass === 1) c.copy(cStatic);
        else c.copy(cDynamic);
      } else if (colorMode === 'risk') {
        if (risk < 0.3) {
          c.setRGB(0.1, 0.2 + risk * 2, 0.8);
        } else if (risk < 0.6) {
          const t = (risk - 0.3) / 0.3;
          c.setRGB(0.2 + t * 0.8, 0.8 - t * 0.2, 0.1);
        } else {
          const t = (risk - 0.6) / 0.4;
          c.setRGB(1.0, 0.6 - t * 0.6, 0.1);
        }
      } else if (colorMode === 'bands') {
        if (band === 0) c.copy(cBand0);
        else if (band === 1) c.copy(cBand1);
        else if (band === 2) c.copy(cBand2);
        else if (band === 3) c.copy(cBand3);
        else if (band === 99) c.copy(cBand99);
        else c.setHex(0x94a3b8);
      } else {
        const normZ = Math.min(1.0, Math.max(0.0, (pz + 2.0) / 4.5));
        c.setHSL(0.7 - normZ * 0.7, 0.9, 0.5);
      }

      colors[i * 3 + 0] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.88,
    });

    const pointsMesh = new THREE.Points(geometry, material);
    sceneRef.current.add(pointsMesh);
    pointsMeshRef.current = pointsMesh;
  }, [points, colorMode, pointSize]);

  // Toggle Range Rings Visibility
  useEffect(() => {
    if (!sceneRef.current) return;
    const rings = sceneRef.current.getObjectByName('rangeRings');
    if (rings) rings.visible = showRings;
  }, [showRings]);

  // Mouse Handlers for Camera Orbit / Pan / Zoom
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) isDraggingRef.current = true;
    if (e.button === 2) isRightDraggingRef.current = true;
    previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const deltaX = e.clientX - previousMousePositionRef.current.x;
    const deltaY = e.clientY - previousMousePositionRef.current.y;
    previousMousePositionRef.current = { x: e.clientX, y: e.clientY };

    if (isDraggingRef.current) {
      cameraAngleRef.current.theta -= deltaX * 0.007;
      cameraAngleRef.current.phi = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, cameraAngleRef.current.phi - deltaY * 0.007));
      updateCameraPosition();
    } else if (isRightDraggingRef.current) {
      const panSpeed = 0.04;
      targetRef.current.x += -deltaY * panSpeed;
      targetRef.current.z += -deltaX * panSpeed;
      updateCameraPosition();
    }
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
    isRightDraggingRef.current = false;
  };

  const handleWheel = (e: React.WheelEvent) => {
    cameraAngleRef.current.radius = Math.max(5, Math.min(180, cameraAngleRef.current.radius + e.deltaY * 0.05));
    updateCameraPosition();
  };

  const resetCamera = (mode: 'orbit' | 'top') => {
    if (mode === 'orbit') {
      cameraAngleRef.current = { theta: Math.PI / 4, phi: Math.PI / 3, radius: 45 };
      targetRef.current.set(15, 0, -1);
    } else if (mode === 'top') {
      cameraAngleRef.current = { theta: 0, phi: 0.05, radius: 70 };
      targetRef.current.set(25, 0, 0);
    }
    updateCameraPosition();
  };

  return (
    <div className="glass-panel" style={{ position: 'relative', width: '100%', height: '100%', minHeight: '520px', overflow: 'hidden' }}>
      {/* 3D Canvas Container */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onWheel={handleWheel}
        onContextMenu={(e) => e.preventDefault()}
        style={{ width: '100%', height: '100%', minHeight: '520px', cursor: isDraggingRef.current ? 'grabbing' : 'grab' }}
      />

      {/* Top Floating Controls Bar */}
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
        {/* Color Mode Selector */}
        <div className="glass-panel" style={{ padding: '6px 10px', display: 'flex', gap: '6px', pointerEvents: 'auto' }}>
          <button
            id="color-mode-semantic"
            className={colorMode === 'semantic' ? 'btn btn-primary' : 'btn btn-secondary'}
            onClick={() => setColorMode('semantic')}
            style={{ padding: '5px 10px', fontSize: '0.75rem' }}
          >
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10b981' }} />
            Semantic Classes
          </button>
          <button
            id="color-mode-risk"
            className={colorMode === 'risk' ? 'btn btn-primary' : 'btn btn-secondary'}
            onClick={() => setColorMode('risk')}
            style={{ padding: '5px 10px', fontSize: '0.75rem' }}
          >
            <ShieldAlert size={13} color="#ef4444" />
            Criticality Risk
          </button>
          <button
            id="color-mode-bands"
            className={colorMode === 'bands' ? 'btn btn-primary' : 'btn btn-secondary'}
            onClick={() => setColorMode('bands')}
            style={{ padding: '5px 10px', fontSize: '0.75rem' }}
          >
            <Layers size={13} color="var(--cyan)" />
            Resolution Bands
          </button>
          <button
            id="color-mode-elevation"
            className={colorMode === 'elevation' ? 'btn btn-primary' : 'btn btn-secondary'}
            onClick={() => setColorMode('elevation')}
            style={{ padding: '5px 10px', fontSize: '0.75rem' }}
          >
            Elevation (Z)
          </button>
        </div>

        {/* Camera Presets & Point Size */}
        <div className="glass-panel" style={{ padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '10px', pointerEvents: 'auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Pt Size:</span>
            <input
              type="range"
              min="1"
              max="6"
              step="0.5"
              value={pointSize}
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
              style={{ width: '60px' }}
            />
          </div>

          <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showRings}
              onChange={(e) => setShowRings(e.target.checked)}
            />
            Rings
          </label>

          <button
            onClick={() => resetCamera('orbit')}
            className="btn btn-secondary"
            style={{ padding: '4px 8px', fontSize: '0.72rem' }}
            title="Isometric 3D View"
          >
            <Compass size={13} /> 3D
          </button>
          <button
            onClick={() => resetCamera('top')}
            className="btn btn-secondary"
            style={{ padding: '4px 8px', fontSize: '0.72rem' }}
            title="Top-Down Bird's Eye View"
          >
            <Eye size={13} /> BEV
          </button>
          <button
            onClick={() => resetCamera('orbit')}
            className="btn btn-secondary"
            style={{ padding: '4px 6px' }}
            title="Reset Camera"
          >
            <RotateCcw size={13} />
          </button>
        </div>
      </div>

      {/* Legend Overlay at Bottom-Left */}
      <div
        className="glass-panel"
        style={{
          position: 'absolute',
          bottom: '16px',
          left: '16px',
          padding: '10px 14px',
          fontSize: '0.75rem',
          maxWidth: '320px',
        }}
      >
        <div style={{ fontWeight: 700, color: '#fff', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Sparkles size={13} color="var(--cyan)" />
          {colorMode === 'semantic' && 'Semantic Classification (FoveaRangeNet)'}
          {colorMode === 'risk' && 'Multi-Criteria Scene Risk Hazard R(x,y)'}
          {colorMode === 'bands' && 'Adaptive Concentric Resolution Bands'}
          {colorMode === 'elevation' && '2.5D Elevation Surface Range'}
        </div>

        {colorMode === 'semantic' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#10b981' }} />
              <span style={{ color: 'var(--text-muted)' }}>Terrain / Drivable (Class 0) &bull; 99.1% IoU</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#3b82f6' }} />
              <span style={{ color: 'var(--text-muted)' }}>Static Obstacle (Class 1) &bull; Curbs & Buildings</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#ef4444' }} />
              <span style={{ color: '#fff', fontWeight: 600 }}>Dynamic Actor (Class 2) &bull; High Risk Target</span>
            </div>
          </div>
        )}

        {colorMode === 'bands' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#10b981' }} />
              <span style={{ color: 'var(--text-muted)' }}>Band 0 (0-10m): 5 cm Local Collision Grid</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#3b82f6' }} />
              <span style={{ color: 'var(--text-muted)' }}>Band 1 (10-30m): 10 cm Maneuvering Grid</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#f59e0b' }} />
              <span style={{ color: 'var(--text-muted)' }}>Band 2 (30-60m): 25 cm Horizon Grid</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#a855f7' }} />
              <span style={{ color: 'var(--text-muted)' }}>Band 3 (60-100m): 50 cm Far Awareness Grid</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#ff0055' }} />
              <span style={{ color: '#fff', fontWeight: 700 }}>Band 99 (Any Range): 5 cm Risk-Refined Fovea</span>
            </div>
          </div>
        )}

        {colorMode === 'risk' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
              <span>0.0 (Safe / Background)</span>
              <span>0.45 (Refinement Threshold)</span>
              <span>1.0 (Critical Threat)</span>
            </div>
            <div
              style={{
                width: '100%',
                height: '8px',
                borderRadius: '4px',
                background: 'linear-gradient(to right, #1d4ed8, #06b6d4, #10b981, #f59e0b, #ef4444)',
              }}
            />
          </div>
        )}

        {colorMode === 'elevation' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
              <span>-3.5m (Trench / Ditch)</span>
              <span>-1.73m (Nominal Road)</span>
              <span>+3.0m (Obstacle)</span>
            </div>
            <div
              style={{
                width: '100%',
                height: '8px',
                borderRadius: '4px',
                background: 'linear-gradient(to right, #4338ca, #3b82f6, #10b981, #eab308, #ef4444)',
              }}
            />
          </div>
        )}
      </div>

      {/* Point count & Instructions Pill at Bottom-Right */}
      <div
        className="glass-panel"
        style={{
          position: 'absolute',
          bottom: '16px',
          right: '16px',
          padding: '8px 12px',
          fontSize: '0.72rem',
          fontFamily: 'var(--font-mono)',
          color: 'var(--text-muted)',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
        }}
      >
        <span>
          Points: <strong style={{ color: 'var(--cyan)' }}>{points.length.toLocaleString()}</strong> (WebGL 60 FPS)
        </span>
        <span style={{ color: 'var(--text-dim)' }}>|</span>
        <span>Left Drag: Rotate &bull; Right Drag: Pan &bull; Scroll: Zoom</span>
      </div>

      {isLoading && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(8, 9, 13, 0.75)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column',
            gap: '12px',
          }}
        >
          <div className="pulse-indicator pulse-cyan" style={{ width: '20px', height: '20px' }} />
          <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--cyan)', letterSpacing: '0.05em' }}>
            INFERRING SPATIAL FOVEA & 2.5D ELEVATION...
          </span>
        </div>
      )}
    </div>
  );
};
