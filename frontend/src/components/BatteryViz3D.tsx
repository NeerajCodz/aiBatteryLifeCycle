import { useEffect, useState, useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, RoundedBox, Environment } from "@react-three/drei";
import * as THREE from "three";
import { fetchDashboard, BatteryVizData } from "../api";

// ── Single battery cell ─────────────────────────────────────────────────────
function BatteryCell({
  position,
  battery,
  selected,
  onClick,
}: {
  position: [number, number, number];
  battery: BatteryVizData;
  selected: boolean;
  onClick: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const color = useMemo(() => new THREE.Color(battery.color_hex), [battery.color_hex]);
  const [hovered, setHovered] = useState(false);

  useFrame((_, dt) => {
    if (!meshRef.current) return;
    const target = selected ? 1.15 : hovered ? 1.08 : 1.0;
    const s = meshRef.current.scale;
    s.lerp(new THREE.Vector3(target, target, target), dt * 5);
  });

  // Fill level based on SOH
  const fillHeight = (battery.soh_pct / 100) * 1.6;

  return (
    <group position={position}>
      {/* Cell body (transparent shell) */}
      <mesh
        ref={meshRef}
        onClick={(e) => { e.stopPropagation(); onClick(); }}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <cylinderGeometry args={[0.4, 0.4, 2, 32]} />
        <meshPhysicalMaterial
          color="#555"
          transparent
          opacity={0.25}
          roughness={0.3}
          metalness={0.7}
        />
      </mesh>

      {/* Fill level (inner cylinder) */}
      <mesh position={[0, -1 + fillHeight / 2, 0]}>
        <cylinderGeometry args={[0.35, 0.35, fillHeight, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={hovered ? 0.6 : 0.3}
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* Positive terminal */}
      <mesh position={[0, 1.1, 0]}>
        <cylinderGeometry args={[0.15, 0.15, 0.2, 16]} />
        <meshStandardMaterial color="#ccc" metalness={0.9} roughness={0.1} />
      </mesh>

      {/* Label */}
      <Text
        position={[0, -1.5, 0]}
        fontSize={0.2}
        color="white"
        anchorX="center"
        anchorY="top"
      >
        {battery.battery_id}
      </Text>
      <Text
        position={[0, -1.8, 0]}
        fontSize={0.18}
        color={battery.color_hex}
        anchorX="center"
        anchorY="top"
      >
        {battery.soh_pct.toFixed(0)}%
      </Text>
    </group>
  );
}

// ── Pack grid layout ────────────────────────────────────────────────────────
function BatteryPack({
  batteries,
  selected,
  onSelect,
}: {
  batteries: BatteryVizData[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const groupRef = useRef<THREE.Group>(null);

  // Auto-rotate
  useFrame((_, dt) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += dt * 0.1;
    }
  });

  // Grid layout: 6 columns
  const cols = 6;
  const spacing = 1.2;

  return (
    <group ref={groupRef}>
      {batteries.map((b, i) => {
        const row = Math.floor(i / cols);
        const col = i % cols;
        const x = (col - (cols - 1) / 2) * spacing;
        const z = (row - Math.floor(batteries.length / cols) / 2) * spacing;
        return (
          <BatteryCell
            key={b.battery_id}
            position={[x, 0, z]}
            battery={b}
            selected={selected === b.battery_id}
            onClick={() => onSelect(b.battery_id)}
          />
        );
      })}
    </group>
  );
}

// ── Main component ──────────────────────────────────────────────────────────
export default function BatteryViz3D() {
  const [batteries, setBatteries] = useState<BatteryVizData[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboard()
      .then((d) => setBatteries(d.batteries))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const selectedBattery = batteries.find((b) => b.battery_id === selected);

  if (loading)
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-400" />
      </div>
    );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
      {/* 3D Canvas */}
      <div className="lg:col-span-3 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden" style={{ height: "600px" }}>
        <Canvas
          camera={{ position: [8, 6, 8], fov: 50 }}
          gl={{ antialias: true, alpha: true }}
        >
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 15, 10]} intensity={1} castShadow />
          <pointLight position={[-5, 10, -5]} intensity={0.5} color="#22c55e" />

          <BatteryPack
            batteries={batteries}
            selected={selected}
            onSelect={setSelected}
          />

          <OrbitControls
            enablePan
            enableZoom
            enableRotate
            autoRotate={false}
            minDistance={5}
            maxDistance={20}
          />

          {/* Ground plane */}
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]}>
            <planeGeometry args={[20, 20]} />
            <meshStandardMaterial color="#111" transparent opacity={0.5} />
          </mesh>

          <gridHelper args={[20, 20, "#333", "#222"]} position={[0, -1.99, 0]} />
        </Canvas>
      </div>

      {/* Side panel */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 space-y-4 overflow-y-auto" style={{ maxHeight: "600px" }}>
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">Battery Pack</h3>

        {selectedBattery ? (
          <div className="space-y-3">
            <div className="text-xl font-bold text-white">{selectedBattery.battery_id}</div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-gray-800 rounded p-2">
                <div className="text-xs text-gray-400">SOH</div>
                <div className="font-bold" style={{ color: selectedBattery.color_hex }}>
                  {selectedBattery.soh_pct}%
                </div>
              </div>
              <div className="bg-gray-800 rounded p-2">
                <div className="text-xs text-gray-400">State</div>
                <div className="font-medium">{selectedBattery.degradation_state}</div>
              </div>
              <div className="bg-gray-800 rounded p-2">
                <div className="text-xs text-gray-400">Temperature</div>
                <div className="font-medium">{selectedBattery.temperature}°C</div>
              </div>
              <div className="bg-gray-800 rounded p-2">
                <div className="text-xs text-gray-400">Cycles</div>
                <div className="font-medium">{selectedBattery.cycle_number}</div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-500">Click a cell to inspect</div>
        )}

        {/* Legend */}
        <div className="pt-4 border-t border-gray-800">
          <h4 className="text-xs font-semibold text-gray-400 uppercase mb-2">Health Legend</h4>
          <div className="space-y-1 text-sm">
            {[
              { label: "Healthy (≥90%)", color: "#22c55e" },
              { label: "Moderate (80-90%)", color: "#eab308" },
              { label: "Degraded (70-80%)", color: "#f97316" },
              { label: "End-of-Life (<70%)", color: "#ef4444" },
            ].map((l) => (
              <div key={l.label} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: l.color }} />
                <span className="text-gray-300">{l.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Battery list */}
        <div className="pt-4 border-t border-gray-800">
          <h4 className="text-xs font-semibold text-gray-400 uppercase mb-2">All Cells</h4>
          <div className="space-y-1 max-h-60 overflow-y-auto">
            {batteries.map((b) => (
              <button
                key={b.battery_id}
                onClick={() => setSelected(b.battery_id)}
                className={`w-full text-left px-2 py-1.5 rounded text-sm transition-colors ${
                  selected === b.battery_id ? "bg-gray-700" : "hover:bg-gray-800"
                }`}
              >
                <span className="text-gray-300">{b.battery_id}</span>
                <span className="float-right font-medium" style={{ color: b.color_hex }}>
                  {b.soh_pct}%
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
