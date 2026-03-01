/**
 * SimulationPanel — Advanced Battery Lifecycle Simulation
 *
 * Workflow:
 *  1. User configures batteries & parameters
 *  2. Click "Run Simulation" → POST /api/v2/simulate (or local physics fallback)
 *  3. Full trajectories returned for ALL batteries at once
 *  4. Timer ticks advance playIndex through pre-computed data
 *  5. All charts + 3D view re-render from pre-computed histories[playIndex]
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import {
  Play, Pause, SkipBack, SkipForward, RotateCcw, FastForward,
  Plus, Trash2, Settings2, Pencil, ChevronRight,
  BatteryFull, BatteryLow, Activity, Zap,
  Thermometer, TrendingDown, AlertOctagon, Clock,
  BarChart3, GitBranch, Layers, Gauge, Cpu,
  TableProperties, ScrollText, CheckCircle2, WifiOff,
  Server, FlaskConical, Copy, Download,
} from "lucide-react";
import { simulateBatteries, BatterySimConfig, BatterySimResult } from "../api";
import { useToast } from "./Toast";

// ── Constants ─────────────────────────────────────────────────────────────
const CHART_COLORS = [
  "#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
  "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#a78bfa",
];

const TIME_UNITS = [
  { key: "cycle",  label: "Cycles"  },
  { key: "hour",   label: "Hours"   },
  { key: "day",    label: "Days"    },
  { key: "week",   label: "Weeks"   },
  { key: "month",  label: "Months"  },
  { key: "year",   label: "Years"   },
];

// 7 default batteries with varied real-world profiles
const DEFAULT_BATTERIES: BatterySimConfig[] = [
  {
    battery_id: "BAT-001", label: "Normal Lab",
    initial_soh: 100, start_cycle: 1, ambient_temperature: 24,
    avg_current: 1.82, peak_voltage: 4.19, min_voltage: 2.61,
    avg_temp: 32, temp_rise: 14, cycle_duration: 3690,
    Re: 0.045, Rct: 0.069, delta_capacity: -0.004,
  },
  {
    battery_id: "BAT-002", label: "Hot Climate",
    initial_soh: 98, start_cycle: 1, ambient_temperature: 38,
    avg_current: 1.82, peak_voltage: 4.19, min_voltage: 2.61,
    avg_temp: 42, temp_rise: 16, cycle_duration: 3690,
    Re: 0.048, Rct: 0.072, delta_capacity: -0.005,
  },
  {
    battery_id: "BAT-003", label: "High Current",
    initial_soh: 95, start_cycle: 1, ambient_temperature: 24,
    avg_current: 2.8, peak_voltage: 4.19, min_voltage: 2.61,
    avg_temp: 36, temp_rise: 20, cycle_duration: 3200,
    Re: 0.046, Rct: 0.070, delta_capacity: -0.006,
  },
  {
    battery_id: "BAT-004", label: "Aged Cell",
    initial_soh: 82, start_cycle: 1, ambient_temperature: 24,
    avg_current: 1.82, peak_voltage: 4.10, min_voltage: 2.70,
    avg_temp: 31, temp_rise: 12, cycle_duration: 3690,
    Re: 0.068, Rct: 0.105, delta_capacity: -0.005,
  },
  {
    battery_id: "BAT-005", label: "Cold Climate",
    initial_soh: 97, start_cycle: 1, ambient_temperature: 8,
    avg_current: 1.50, peak_voltage: 4.15, min_voltage: 2.61,
    avg_temp: 20, temp_rise: 10, cycle_duration: 4200,
    Re: 0.052, Rct: 0.085, delta_capacity: -0.003,
  },
  {
    battery_id: "BAT-006", label: "Overcharged",
    initial_soh: 90, start_cycle: 1, ambient_temperature: 28,
    avg_current: 1.82, peak_voltage: 4.32, min_voltage: 2.61,
    avg_temp: 34, temp_rise: 18, cycle_duration: 3800,
    Re: 0.050, Rct: 0.078, delta_capacity: -0.006,
  },
  {
    battery_id: "BAT-007", label: "Near EOL",
    initial_soh: 74, start_cycle: 1, ambient_temperature: 30,
    avg_current: 2.20, peak_voltage: 4.20, min_voltage: 2.71,
    avg_temp: 38, temp_rise: 22, cycle_duration: 3690,
    Re: 0.085, Rct: 0.130, delta_capacity: -0.008,
  },
];

// ── Local physics fallback (mirrors backend Arrhenius model exactly) ──────
const EA_OVER_R = 6200;
const Q_NOM = 2.0;
const T_REF_C = 24;
const I_REF = 1.82;
const V_REF = 4.19;

export function sohColor(soh: number): string {
  if (soh >= 90) return "#22c55e";
  if (soh >= 80) return "#eab308";
  if (soh >= 70) return "#f97316";
  return "#ef4444";
}

function degradeState(soh: number): string {
  if (soh >= 90) return "Healthy";
  if (soh >= 80) return "Moderate";
  if (soh >= 70) return "Degraded";
  return "End-of-Life";
}

function computeStressFactors(b: BatterySimConfig) {
  const Tk   = 273.15 + (b.ambient_temperature ?? T_REF_C);
  const TrK  = 273.15 + T_REF_C;
  const tempF = Math.max(0.15, Math.min(Math.exp(EA_OVER_R * (1 / TrK - 1 / Tk)), 25));
  const currF = 1 + Math.max(0, ((b.avg_current ?? I_REF) - I_REF) * 0.18);
  const voltF = 1 + Math.max(0, ((b.peak_voltage ?? V_REF) - V_REF) * 0.55);
  return { tempF: +tempF.toFixed(3), currF: +currF.toFixed(3), voltF: +voltF.toFixed(3), total: +(tempF * currF * voltF).toFixed(3) };
}

function runLocalSimulation(
  batteries: BatterySimConfig[],
  steps: number,
  timeUnit: string,
  eolThr = 70,
): BatterySimResult[] {
  const TU_SEC: Record<string, number | null> = {
    cycle: null, second: 1, minute: 60, hour: 3600,
    day: 86400, week: 604800, month: 2592000, year: 31536000,
  };
  const tuSec = TU_SEC[timeUnit] ?? 86400;

  return batteries.map((b) => {
    let soh = b.initial_soh ?? 100;
    let re  = b.Re  ?? 0.045;
    let rct = b.Rct ?? 0.069;
    const soh_h: number[] = [], rul_h: number[] = [], rul_t_h: number[] = [];
    const re_h: number[] = [], rct_h: number[] = [];
    const cyc_h: number[] = [], time_h: number[] = [];
    const deg_h: string[] = [], color_h: string[] = [];
    let eol_cycle: number | null = null;
    let eol_time: number | null = null;
    let totalDeg = 0;

    for (let step = 0; step < steps; step++) {
      const cycle = (b.start_cycle ?? 1) + step;
      const rateBase = Math.max(
        0.005,
        Math.min(Math.abs(b.delta_capacity ?? -0.005) / Q_NOM * 100, 1.5),
      );
      const Tk   = 273.15 + (b.ambient_temperature ?? T_REF_C);
      const TrK  = 273.15 + T_REF_C;
      const tempF  = Math.max(0.15, Math.min(Math.exp(EA_OVER_R * (1 / TrK - 1 / Tk)), 25));
      const currF  = 1 + Math.max(0, ((b.avg_current ?? I_REF) - I_REF) * 0.18);
      const voltF  = 1 + Math.max(0, ((b.peak_voltage ?? V_REF) - V_REF) * 0.55);
      const ageF   = 1 + (soh < 85 ? 0.08 : 0) + (soh < 75 ? 0.12 : 0);
      const degRate = Math.min(rateBase * tempF * currF * voltF * ageF, 2.0);

      soh = Math.max(0, soh - degRate);
      re  = Math.min(re  + 0.00012 * tempF * (1 + step * 5e-5), 2.0);
      rct = Math.min(rct + 0.00018 * tempF * (1 + step * 8e-5) * (soh < 80 ? 1.3 : 1), 3.0);
      totalDeg += degRate;

      const rulCycles = soh > eolThr && degRate > 0 ? (soh - eolThr) / degRate : 0;
      const cycleDur  = b.cycle_duration ?? 3690;
      const elapsedS  = cycle * cycleDur;
      const elapsedT  = tuSec ? elapsedS / tuSec : cycle;
      const rulT      = tuSec ? rulCycles * cycleDur / tuSec : rulCycles;

      if (soh <= eolThr && eol_cycle === null) {
        eol_cycle = cycle;
        eol_time  = +elapsedT.toFixed(3);
      }

      soh_h.push(+soh.toFixed(3));
      rul_h.push(+rulCycles.toFixed(1));
      rul_t_h.push(+rulT.toFixed(2));
      re_h.push(+re.toFixed(6));
      rct_h.push(+rct.toFixed(6));
      cyc_h.push(cycle);
      time_h.push(+elapsedT.toFixed(3));
      deg_h.push(degradeState(soh));
      color_h.push(sohColor(soh));
    }

    return {
      battery_id: b.battery_id,
      label: b.label ?? b.battery_id,
      soh_history:         soh_h,
      rul_history:         rul_h,
      rul_time_history:    rul_t_h,
      re_history:          re_h,
      rct_history:         rct_h,
      cycle_history:       cyc_h,
      time_history:        time_h,
      degradation_history: deg_h,
      color_history:       color_h,
      eol_cycle,
      eol_time,
      final_soh:    soh_h.length ? soh_h[soh_h.length - 1] : soh,
      final_rul:    rul_h.length ? rul_h[rul_h.length - 1] : 0,
      deg_rate_avg: +(totalDeg / steps).toFixed(6),
    };
  });
}

// ── 3D Battery Cell ────────────────────────────────────────────────────────
function BatteryCell({
  position, batteryId, label, soh, color, selected, isRunning, onClick, onDblClick,
}: {
  position: [number, number, number];
  batteryId: string;
  label: string;
  soh: number;
  color: string;
  selected: boolean;
  isRunning: boolean;
  onClick: () => void;
  onDblClick: () => void;
}) {
  const bodyRef  = useRef<THREE.Group>(null);
  const fillRef  = useRef<THREE.Mesh>(null);
  const ringRef  = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const fillColor = useMemo(() => new THREE.Color(color), [color]);
  const fillH = Math.max(0.06, (Math.min(100, Math.max(0, soh)) / 100) * 1.82);

  useFrame((state, dt) => {
    if (!bodyRef.current) return;
    const target = selected ? 1.14 : hovered ? 1.06 : 1.0;
    bodyRef.current.scale.lerp(new THREE.Vector3(target, target, target), dt * 9);

    if (fillRef.current) {
      const mat = fillRef.current.material as THREE.MeshStandardMaterial;
      const t = state.clock.elapsedTime;
      const pulse = isRunning ? 0.3 + Math.sin(t * 3.5) * 0.18 : 0.2 + Math.sin(t * 0.8) * 0.08;
      mat.emissiveIntensity = selected ? 0.9 : hovered ? 0.7 : pulse;
    }
    if (ringRef.current && selected) {
      ringRef.current.rotation.y += dt * 2;
    }
  });

  return (
    <group position={position}>
      <group
        ref={bodyRef}
        onClick={(e) => { e.stopPropagation(); onClick(); }}
        onDoubleClick={(e) => { e.stopPropagation(); onDblClick(); }}
        onPointerOver={() => { setHovered(true); document.body.style.cursor = "pointer"; }}
        onPointerOut={() => { setHovered(false); document.body.style.cursor = "auto"; }}
      >
        {/* Outer glass shell */}
        <mesh>
          <cylinderGeometry args={[0.42, 0.42, 2.12, 48]} />
          <meshPhysicalMaterial
            color="#9ca3af"
            transparent
            opacity={hovered ? 0.22 : 0.14}
            roughness={0}
            metalness={0.05}
            transmission={0.75}
            thickness={0.4}
            side={THREE.DoubleSide}
          />
        </mesh>
        {/* Metal band bottom */}
        <mesh position={[0, -0.99, 0]}>
          <cylinderGeometry args={[0.44, 0.44, 0.16, 40]} />
          <meshStandardMaterial color="#b0b8c8" metalness={0.96} roughness={0.12} />
        </mesh>
        {/* Metal band top */}
        <mesh position={[0, 0.99, 0]}>
          <cylinderGeometry args={[0.44, 0.44, 0.16, 40]} />
          <meshStandardMaterial color="#b0b8c8" metalness={0.96} roughness={0.12} />
        </mesh>
        {/* SOH fill */}
        <mesh ref={fillRef} position={[0, -1.05 + fillH / 2, 0]}>
          <cylinderGeometry args={[0.36, 0.36, fillH, 40]} />
          <meshStandardMaterial
            color={fillColor}
            emissive={fillColor}
            emissiveIntensity={0.3}
            roughness={0.35}
            metalness={0.25}
          />
        </mesh>
        {/* Positive terminal */}
        <mesh position={[0, 1.16, 0]}>
          <cylinderGeometry args={[0.15, 0.15, 0.28, 24]} />
          <meshStandardMaterial color="#e5e7eb" metalness={0.99} roughness={0.03} />
        </mesh>
        {/* Negative plate */}
        <mesh position={[0, -1.12, 0]}>
          <cylinderGeometry args={[0.42, 0.42, 0.06, 32]} />
          <meshStandardMaterial color="#c0c4cc" metalness={0.92} roughness={0.18} />
        </mesh>
        {/* Wrap stripe */}
        <mesh position={[0, 0, 0]}>
          <cylinderGeometry args={[0.43, 0.43, 0.28, 40]} />
          <meshStandardMaterial color={color} transparent opacity={0.35} roughness={0.6} />
        </mesh>
      </group>

      {/* Selection orbit ring */}
      {selected && (
        <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
          <torusGeometry args={[0.60, 0.035, 12, 48]} />
          <meshBasicMaterial color={color} transparent opacity={0.85} />
        </mesh>
      )}

      {/* Labels */}
      <Text position={[0, -1.56, 0]} fontSize={0.17} color="#e5e7eb" anchorX="center" anchorY="top">
        {batteryId}
      </Text>
      <Text position={[0, -1.80, 0]} fontSize={0.145} color={color} anchorX="center" anchorY="top">
        {soh.toFixed(1)}%
      </Text>
      {label !== batteryId && (
        <Text position={[0, 1.44, 0]} fontSize={0.11} color="#9ca3af" anchorX="center" anchorY="bottom">
          {label}
        </Text>
      )}

      {soh < 70 && (
        <pointLight position={[0, 0, 0]} intensity={0.55} color={color} distance={1.6} />
      )}
    </group>
  );
}

// ── 3D Battery Pack ────────────────────────────────────────────────────────
function BatteryPack({
  batteries, selected, onSelect, onOpenConfig, isRunning,
}: {
  batteries: { id: string; label: string; soh: number; color: string }[];
  selected: string | null;
  onSelect: (id: string) => void;
  onOpenConfig: (id: string) => void;
  isRunning: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((_, dt) => {
    if (groupRef.current && !isRunning) {
      groupRef.current.rotation.y += dt * 0.035;
    }
  });

  const cols = Math.min(4, Math.ceil(Math.sqrt(batteries.length)));
  const rows = Math.ceil(batteries.length / cols);
  const gap  = 1.32;

  return (
    <group ref={groupRef}>
      {/* Base plate */}
      <mesh position={[0, -1.42, 0]} receiveShadow>
        <boxGeometry args={[cols * gap + 0.5, 0.10, rows * gap + 0.5]} />
        <meshStandardMaterial color="#0f1c2e" metalness={0.72} roughness={0.38} />
      </mesh>
      {/* Bus bars */}
      {Array.from({ length: rows }, (_, r) => (
        <mesh key={`bus-${r}`} position={[0, -1.29, (r - (rows - 1) / 2) * gap]}>
          <boxGeometry args={[cols * gap, 0.04, 0.07]} />
          <meshStandardMaterial color="#6b7280" metalness={0.96} roughness={0.08} />
        </mesh>
      ))}
      {batteries.map((b, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = (col - (cols - 1) / 2) * gap;
        const z = (row - (rows - 1) / 2) * gap;
        return (
          <BatteryCell
            key={b.id}
            position={[x, 0, z]}
            batteryId={b.id}
            label={b.label}
            soh={b.soh}
            color={b.color}
            selected={selected === b.id}
            isRunning={isRunning}
            onClick={() => onSelect(b.id)}
            onDblClick={() => onOpenConfig(b.id)}
          />
        );
      })}
    </group>
  );
}

// ── Config Modal ───────────────────────────────────────────────────────────
const PARAM_FIELDS: { key: keyof BatterySimConfig; label: string; step: string; unit: string }[] = [
  { key: "initial_soh",         label: "Initial SOH",        step: "0.1",  unit: "%" },
  { key: "ambient_temperature", label: "Ambient Temp",       step: "0.5",  unit: "°C" },
  { key: "peak_voltage",        label: "Peak Voltage",       step: "0.01", unit: "V" },
  { key: "min_voltage",         label: "Min Voltage",        step: "0.01", unit: "V" },
  { key: "avg_current",         label: "Avg Current",        step: "0.1",  unit: "A" },
  { key: "avg_temp",            label: "Cell Temp",          step: "0.5",  unit: "°C" },
  { key: "temp_rise",           label: "Temp Rise/cycle",    step: "0.5",  unit: "°C" },
  { key: "cycle_duration",      label: "Cycle Duration",     step: "60",   unit: "s" },
  { key: "Re",                  label: "Re (Electrolyte Ω)", step: "0.001", unit: "Ω" },
  { key: "Rct",                 label: "Rct (Charge-Xfer Ω)", step: "0.001", unit: "Ω" },
  { key: "delta_capacity",      label: "ΔCapacity/cycle",    step: "0.001", unit: "Ah" },
];

function ConfigModal({
  battery, color, onSave, onClose,
}: {
  battery: BatterySimConfig;
  color: string;
  onSave: (b: BatterySimConfig) => void;
  onClose: () => void;
}) {
  const [form, setForm] = useState<BatterySimConfig>({ ...battery });
  const getNum = (k: keyof BatterySimConfig) => (form[k] as number) ?? 0;
  const setNum = (k: keyof BatterySimConfig, v: string) =>
    setForm(p => ({ ...p, [k]: parseFloat(v) || 0 }));

  const stress = computeStressFactors(form);

  return (
    <div
      className="fixed inset-0 z-200 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 rounded-2xl border border-gray-700 p-6 w-[600px] max-h-[90vh] overflow-y-auto shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ backgroundColor: color + "22", borderColor: color + "55", border: "1px solid" }}>
              <BatteryFull className="w-4 h-4" style={{ color }} />
            </div>
            <div>
              <div className="text-base font-bold text-white">{form.battery_id}</div>
              <div className="text-xs text-gray-400">Edit battery configuration</div>
            </div>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors p-1 rounded-lg hover:bg-gray-800">
            <Settings2 className="w-5 h-5" />
          </button>
        </div>

        {/* Live stress preview */}
        <div className="bg-gray-800/50 rounded-xl p-3 mb-5 grid grid-cols-4 gap-2 text-center">
          {[
            { label: "Temp Stress", value: stress.tempF, color: "#f59e0b" },
            { label: "Current Stress", value: stress.currF, color: "#ef4444" },
            { label: "Voltage Stress", value: stress.voltF, color: "#8b5cf6" },
            { label: "Total Stress", value: stress.total, color: "#ec4899" },
          ].map((s) => (
            <div key={s.label} className="bg-gray-800 rounded-lg p-2">
              <div className="text-[10px] text-gray-500 mb-0.5">{s.label}</div>
              <div className="font-bold text-sm font-mono" style={{ color: s.color }}>{s.value}×</div>
            </div>
          ))}
        </div>

        {/* Identity fields */}
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1 font-medium">Battery ID</label>
            <input
              type="text" value={form.battery_id}
              onChange={(e) => setForm(p => ({ ...p, battery_id: e.target.value }))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-green-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1 font-medium">Label</label>
            <input
              type="text" value={form.label ?? ""}
              onChange={(e) => setForm(p => ({ ...p, label: e.target.value }))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-green-500"
            />
          </div>
        </div>

        {/* Parameter grid */}
        <div className="grid grid-cols-2 gap-3">
          {PARAM_FIELDS.map((f) => (
            <div key={String(f.key)}>
              <label className="block text-xs text-gray-400 mb-1 font-medium">
                {f.label} <span className="text-gray-600">{f.unit}</span>
              </label>
              <input
                type="number" step={f.step} value={getNum(f.key)}
                onChange={(e) => setNum(f.key, e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-green-500"
              />
            </div>
          ))}
        </div>

        {/* Action buttons */}
        <div className="flex gap-3 mt-5">
          <button
            onClick={onClose}
            className="flex-1 flex items-center justify-center gap-2 bg-gray-700 hover:bg-gray-600 text-white py-2.5 rounded-xl text-sm transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onSave(form)}
            className="flex-1 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-500 text-white py-2.5 rounded-xl text-sm font-semibold transition-colors"
          >
            <CheckCircle2 className="w-4 h-4" />
            Save & Apply
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Stat Card ─────────────────────────────────────────────────────────────
function StatCard({
  label, value, color = "text-green-400", sub,
  icon: Icon,
}: {
  label: string;
  value: string | number;
  color?: string;
  sub?: string;
  icon?: React.ElementType;
}) {
  const IconEl = Icon as React.FC<{ className?: string }>;
  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-800 hover:border-gray-700 transition-colors">
      <div className="flex items-center gap-1.5 mb-2">
        {Icon && <IconEl className="w-3.5 h-3.5 text-gray-500" />}
        <span className="text-xs text-gray-400">{label}</span>
      </div>
      <div className={`text-xl font-bold ${color} truncate tabular-nums`}>{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-0.5 truncate">{sub}</div>}
    </div>
  );
}

// ── Custom recharts tooltip ────────────────────────────────────────────────
const DarkTooltip = ({ active, payload, label, unit = "" }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-3 text-xs shadow-2xl min-w-36">
      <p className="text-gray-400 mb-2 font-medium">{label}{unit ? ` ${unit}` : ""}</p>
      {payload.map((p: any) => (
        <div key={p.name} className="flex items-center justify-between gap-4 mb-0.5">
          <span className="flex items-center gap-1.5 text-gray-300">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }} />
            {p.name}
          </span>
          <span className="font-mono font-bold text-white">
            {typeof p.value === "number" ? p.value.toFixed(3) : p.value}
          </span>
        </div>
      ))}
    </div>
  );
};

// ── Chart metric config ────────────────────────────────────────────────────
const CHART_METRICS = [
  { key: "soh",   label: "SOH (%)",      color: "#22c55e" },
  { key: "rul",   label: "RUL (cycles)", color: "#3b82f6" },
  { key: "rul_t", label: "RUL (time)",   color: "#06b6d4" },
  { key: "re",    label: "Re (Ω)",       color: "#f59e0b" },
  { key: "rct",   label: "Rct (Ω)",      color: "#8b5cf6" },
];

const DEG_COLORS: Record<string, string> = {
  Healthy: "#22c55e",
  Moderate: "#eab308",
  Degraded: "#f97316",
  "End-of-Life": "#ef4444",
};

type ChartTab = "fleet" | "trajectories" | "stress" | "impedance" | "capacity" | "distribution" | "eol" | "log";

const CHART_TABS: { key: ChartTab; label: string; icon: React.ElementType }[] = [
  { key: "fleet",        label: "Fleet",        icon: Layers },
  { key: "trajectories", label: "Trajectories", icon: GitBranch },
  { key: "stress",       label: "Stress",       icon: Thermometer },
  { key: "impedance",    label: "Impedance",    icon: Activity },
  { key: "capacity",     label: "Capacity",     icon: BatteryFull },
  { key: "distribution", label: "Distribution", icon: BarChart3 },
  { key: "eol",          label: "EOL",          icon: AlertOctagon },
  { key: "log",          label: "Log",          icon: ScrollText },
];

// ── Main component ─────────────────────────────────────────────────────────
export default function SimulationPanel() {
  const { toast } = useToast();

  // --- Config state ---
  const [batteryConfigs, setBatteryConfigs] = useState<BatterySimConfig[]>(DEFAULT_BATTERIES);
  const [steps, setSteps]                   = useState(300);
  const [timeUnit, setTimeUnit]             = useState("day");
  const [eolThreshold, setEolThreshold]     = useState(70);

  // --- Simulation results ---
  const [results, setResults]             = useState<BatterySimResult[]>([]);
  const [timeUnitLabel, setTimeUnitLabel] = useState("Days");
  const [isSimulating, setIsSimulating]   = useState(false);
  const [modelUsed, setModelUsed]         = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("best_ensemble");

  // --- Playback ---
  const [playIndex, setPlayIndex]   = useState(0);
  const [isPlaying, setIsPlaying]   = useState(false);
  const [playSpeed, setPlaySpeed]   = useState(10);
  const playRef      = useRef<ReturnType<typeof setInterval> | null>(null);
  const playIdxRef   = useRef(0);

  // --- UI ---
  const [selected3D, setSelected3D]     = useState<string | null>(null);
  const [configTarget, setConfigTarget] = useState<string | null>(null);
  const [activeChart, setActiveChart]   = useState<ChartTab>("fleet");
  const [chartMetric, setChartMetric]   = useState("soh");
  const [simLog, setSimLog]             = useState<{ t: string; msg: string; type: "info" | "warn" | "ok" | "err" }[]>([]);

  const totalSteps = results.length > 0 ? results[0].soh_history.length : 0;

  // current per-battery snapshot at playIndex
  const currentSohs = useMemo(() =>
    results.map((r) => ({
      id:    r.battery_id,
      label: r.label ?? r.battery_id,
      soh:   r.soh_history[playIndex]   ?? 100,
      rul:   r.rul_history[playIndex]   ?? 0,
      re:    r.re_history[playIndex]    ?? 0,
      rct:   r.rct_history[playIndex]   ?? 0,
      color: r.color_history[playIndex] ?? sohColor(r.soh_history[playIndex] ?? 100),
      deg:   r.degradation_history[playIndex] ?? "Healthy",
    })),
    [results, playIndex],
  );

  const avgSoh   = currentSohs.length ? currentSohs.reduce((s, b) => s + b.soh, 0) / currentSohs.length : 0;
  const bestBat  = currentSohs.reduce<typeof currentSohs[0] | null>((a, b) => (!a || b.soh > a.soh) ? b : a, null);
  const worstBat = currentSohs.reduce<typeof currentSohs[0] | null>((a, b) => (!a || b.soh < a.soh) ? b : a, null);
  const eolCount = results.filter((r) => r.eol_cycle !== null).length;

  const elapsedTime = useMemo(() => {
    if (!results.length) return "—";
    const t = results[0]?.time_history?.[playIndex] ?? 0;
    return `${t.toFixed(1)} ${timeUnitLabel}`;
  }, [results, playIndex, timeUnitLabel]);

  // --- Playback loop ---
  useEffect(() => {
    if (isPlaying && totalSteps > 0) {
      const iv = Math.max(16, Math.round(1000 / playSpeed));
      playRef.current = setInterval(() => {
        playIdxRef.current += 1;
        if (playIdxRef.current >= totalSteps) {
          playIdxRef.current = totalSteps - 1;
          setIsPlaying(false);
        }
        setPlayIndex(playIdxRef.current);
      }, iv);
      return () => { if (playRef.current) clearInterval(playRef.current); };
    }
  }, [isPlaying, playSpeed, totalSteps]);

  // --- Log helpers ---
  const addLog = useCallback(
    (msg: string, type: "info" | "warn" | "ok" | "err" = "info") => {
      const t = new Date().toLocaleTimeString();
      setSimLog((prev) => [{ t, msg, type }, ...prev.slice(0, 199)]);
    },
    [],
  );

  // --- Run simulation ---
  const runSimulation = useCallback(async () => {
    if (!batteryConfigs.length) return;
    setIsSimulating(true);
    setIsPlaying(false);
    if (playRef.current) clearInterval(playRef.current);
    setPlayIndex(0);
    playIdxRef.current = 0;
    addLog(`Starting: ${batteryConfigs.length} batteries × ${steps} steps (${timeUnit})`, "info");

    try {
      const resp = await simulateBatteries({
        batteries: batteryConfigs,
        steps,
        time_unit: timeUnit,
        eol_threshold: eolThreshold,
        model_name: selectedModel,
        use_ml: true,
      });
      setResults(resp.results);
      setTimeUnitLabel(resp.time_unit_label);
      const usedModel = resp.model_used ?? selectedModel;
      setModelUsed(usedModel);
      addLog(`Backend OK — ${resp.results.length} trajectories | model: ${usedModel}`, "ok");
      toast({ type: "success", title: "Simulation complete", message: `${resp.results.length} batteries × ${steps} steps | ${usedModel}` });
      resp.results.forEach((r) => {
        if (r.eol_cycle !== null) {
          addLog(`${r.battery_id} EOL @ cycle ${r.eol_cycle} (${r.eol_time?.toFixed(1)} ${resp.time_unit_label})`, "warn");
        }
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Connection refused";
      addLog(`Backend unavailable (${msg}) — running local Arrhenius physics`, "warn");
      toast({
        type: "warning",
        title: "Backend unavailable",
        message: "Simulation is running entirely on local Arrhenius physics — API is offline or unreachable.",
        duration: 6000,
      });
      const local = runLocalSimulation(batteryConfigs, steps, timeUnit, eolThreshold);
      setResults(local);
      setTimeUnitLabel(TIME_UNITS.find((u) => u.key === timeUnit)?.label ?? "Days");
      setModelUsed(null);
      addLog(`Local physics OK — ${local.length} trajectories`, "ok");
      local.forEach((r) => {
        if (r.eol_cycle !== null) {
          addLog(`${r.battery_id} EOL @ cycle ${r.eol_cycle} (${r.eol_time?.toFixed(1)} ${timeUnit})`, "warn");
        }
      });
    } finally {
      setIsSimulating(false);
    }
  }, [batteryConfigs, steps, timeUnit, eolThreshold, selectedModel, addLog, toast]);

  // --- Playback controls ---
  const togglePlay = useCallback(() => {
    if (!results.length) return;
    if (playIdxRef.current >= totalSteps - 1) {
      setPlayIndex(0);
      playIdxRef.current = 0;
    }
    setIsPlaying((p) => !p);
  }, [results.length, totalSteps]);

  const stepFwd = useCallback(() => {
    const n = Math.min(playIndex + 1, totalSteps - 1);
    setPlayIndex(n); playIdxRef.current = n;
  }, [playIndex, totalSteps]);

  const stepBwd = useCallback(() => {
    const n = Math.max(playIndex - 1, 0);
    setPlayIndex(n); playIdxRef.current = n;
  }, [playIndex]);

  const resetPlay = useCallback(() => {
    setIsPlaying(false);
    if (playRef.current) clearInterval(playRef.current);
    setPlayIndex(0); playIdxRef.current = 0;
  }, []);

  // --- Battery management ---
  const addBattery = useCallback(() => {
    const id = `BAT-${String(batteryConfigs.length + 1).padStart(3, "0")}`;
    const newBat: BatterySimConfig = { ...DEFAULT_BATTERIES[0], battery_id: id, label: "New Battery", initial_soh: 100 };
    setBatteryConfigs((p) => [...p, newBat]);
    setConfigTarget(id);
  }, [batteryConfigs.length]);

  const duplicateBattery = useCallback((id: string) => {
    const src = batteryConfigs.find((b) => b.battery_id === id);
    if (!src) return;
    const newId = `BAT-${String(batteryConfigs.length + 1).padStart(3, "0")}`;
    setBatteryConfigs((p) => [...p, { ...src, battery_id: newId, label: `${src.label ?? id} (copy)` }]);
    toast({ type: "info", title: "Battery duplicated", message: `${src.label ?? id} → ${newId}` });
  }, [batteryConfigs, toast]);

  const removeBattery = useCallback((id: string) => {
    setBatteryConfigs((p) => p.filter((b) => b.battery_id !== id));
    if (selected3D === id) setSelected3D(null);
  }, [selected3D]);

  const saveConfig = useCallback((updated: BatterySimConfig) => {
    setBatteryConfigs((p) => p.map((b) => (b.battery_id === configTarget ? updated : b)));
    toast({ type: "success", title: "Configuration saved", message: `${updated.label ?? updated.battery_id} updated — re-run simulation to apply.` });
    setConfigTarget(null);
  }, [configTarget, toast]);

  const configBattery = useMemo(
    () => (configTarget ? batteryConfigs.find((b) => b.battery_id === configTarget) : null),
    [configTarget, batteryConfigs],
  );

  // --- 3D display data ---
  const display3D = useMemo(
    () => batteryConfigs.map((b) => {
      const cur = currentSohs.find((s) => s.id === b.battery_id);
      return {
        id:    b.battery_id,
        label: b.label ?? b.battery_id,
        soh:   cur?.soh   ?? (b.initial_soh ?? 100),
        color: cur?.color ?? sohColor(b.initial_soh ?? 100),
      };
    }),
    [batteryConfigs, currentSohs],
  );

  // --- Chart data helpers ---
  const sample = (n: number) => Math.max(1, Math.floor(n / 200));

  const fleetData = useMemo(() => {
    if (!results.length) return [];
    const r0 = results[0];
    const s  = sample(r0.soh_history.length);
    return r0.soh_history
      .map((_, i) => {
        if (i % s !== 0 && i !== r0.soh_history.length - 1) return null;
        const sohs = results.map((r) => r.soh_history[i] ?? 0);
        return {
          t:   +r0.time_history[i].toFixed(2),
          avg: +(sohs.reduce((a, b) => a + b, 0) / sohs.length).toFixed(2),
          min: +Math.min(...sohs).toFixed(2),
          max: +Math.max(...sohs).toFixed(2),
        };
      })
      .filter(Boolean) as { t: number; avg: number; min: number; max: number }[];
  }, [results]);

  const trajectoryData = useMemo(() => {
    if (!results.length) return { points: [] as any[], keys: [] as string[] };
    const r0 = results[0];
    const n  = r0.soh_history.length;
    const s  = Math.max(1, Math.floor(n / 150));
    const getHist = (r: BatterySimResult): number[] => {
      if (chartMetric === "soh")   return r.soh_history;
      if (chartMetric === "rul")   return r.rul_history;
      if (chartMetric === "rul_t") return r.rul_time_history;
      if (chartMetric === "re")    return r.re_history;
      return r.rct_history;
    };
    const points = r0.time_history
      .map((t, i) => {
        if (i % s !== 0 && i !== n - 1) return null;
        const row: any = { t: +t.toFixed(2) };
        results.forEach((r) => { row[r.battery_id] = +(getHist(r)[i] ?? 0).toFixed(4); });
        return row;
      })
      .filter(Boolean);
    return { points, keys: results.map((r) => r.battery_id) };
  }, [results, chartMetric]);

  const comparisonData = useMemo(
    () => currentSohs.map((b) => ({
      name:  b.id.replace("BAT-", "B"),
      soh:   +b.soh.toFixed(1),
      rul:   +b.rul.toFixed(0),
      re:    +b.re.toFixed(4),
      rct:   +b.rct.toFixed(4),
      color: b.color,
    })),
    [currentSohs],
  );

  const degDist = useMemo(() => {
    const counts: Record<string, number> = { Healthy: 0, Moderate: 0, Degraded: 0, "End-of-Life": 0 };
    currentSohs.forEach((b) => { counts[b.deg] = (counts[b.deg] ?? 0) + 1; });
    return Object.entries(counts).filter(([, v]) => v > 0).map(([name, value]) => ({ name, value }));
  }, [currentSohs]);

  const selectedResult = useMemo(
    () => results.find((r) => r.battery_id === selected3D),
    [results, selected3D],
  );

  const impedanceData = useMemo(() => {
    if (!selectedResult) return [];
    const n = selectedResult.re_history.length;
    const s = Math.max(1, Math.floor(n / 150));
    return selectedResult.re_history
      .map((re, i) => {
        if (i % s !== 0 && i !== n - 1) return null;
        return {
          t:   +(selectedResult.time_history[i] ?? i).toFixed(2),
          re:  +re.toFixed(5),
          rct: +(selectedResult.rct_history[i] ?? 0).toFixed(5),
        };
      })
      .filter(Boolean);
  }, [selectedResult]);

  const eolScatter = useMemo(
    () => results.filter((r) => r.eol_time !== null).map((r) => ({
      id:       r.battery_id,
      deg_rate: +(r.deg_rate_avg * 100).toFixed(4),
      eol_time: +(r.eol_time ?? 0).toFixed(2),
    })),
    [results],
  );

  // Stress analysis data — computed from config, not simulation results
  const stressData = useMemo(
    () => batteryConfigs.map((b, i) => {
      const sf = computeStressFactors(b);
      return {
        name:  b.battery_id.replace("BAT-", "B"),
        label: b.label ?? b.battery_id,
        temp:  sf.tempF,
        curr:  sf.currF,
        volt:  sf.voltF,
        total: sf.total,
        color: CHART_COLORS[i % CHART_COLORS.length],
      };
    }),
    [batteryConfigs],
  );

  // Stress radar (top-level factors across batteries)
  const stressRadar = useMemo(
    () => ["Temp", "Current", "Voltage", "Total"].map((factor) => {
      const row: any = { factor };
      batteryConfigs.forEach((b, i) => {
        const sf = computeStressFactors(b);
        const key = b.label ?? b.battery_id;
        row[key] = factor === "Temp" ? sf.tempF : factor === "Current" ? sf.currF : factor === "Voltage" ? sf.voltF : sf.total;
      });
      return row;
    }),
    [batteryConfigs],
  );

  // Capacity fade data (Ah over time)
  const capacityData = useMemo(() => {
    if (!results.length) return { points: [] as any[], keys: [] as string[] };
    const r0 = results[0];
    const n  = r0.soh_history.length;
    const s  = Math.max(1, Math.floor(n / 150));
    const points = r0.time_history
      .map((t, i) => {
        if (i % s !== 0 && i !== n - 1) return null;
        const row: any = { t: +t.toFixed(2) };
        results.forEach((r) => {
          // Capacity in Ah = Q_NOM × soh/100
          row[r.battery_id] = +(Q_NOM * (r.soh_history[i] ?? 0) / 100).toFixed(4);
        });
        return row;
      })
      .filter(Boolean);
    return { points, keys: results.map((r) => r.battery_id) };
  }, [results]);

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="space-y-4">
      {/* Config Modal */}
      {configBattery && (
        <ConfigModal
          battery={configBattery}
          color={sohColor(configBattery.initial_soh ?? 100)}
          onSave={saveConfig}
          onClose={() => setConfigTarget(null)}
        />
      )}

      {/* ── Setup bar ──────────────────────────────────────────────────── */}
      <div className="bg-gray-900 rounded-xl border border-gray-800">
        <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-800">
          <Cpu className="w-4 h-4 text-green-400" />
          <span className="text-sm font-semibold text-gray-200">Simulation Setup</span>
          {modelUsed !== null && (
            <div className="ml-auto flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full bg-green-900/40 text-green-400">
              <Server className="w-3 h-3" />
              {modelUsed}
            </div>
          )}
          {modelUsed === null && results.length > 0 && (
            <div className="ml-auto flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full bg-amber-900/40 text-amber-400">
              <WifiOff className="w-3 h-3" />
              Local Fallback
            </div>
          )}
        </div>
        <div className="flex flex-wrap gap-4 items-end px-4 py-3">
          {/* Steps */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 font-medium">Steps</label>
            <input
              type="number" min={10} max={5000} value={steps}
              onChange={(e) => setSteps(Math.max(10, Math.min(5000, +e.target.value)))}
              className="w-24 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-green-500"
            />
          </div>
          {/* Time Unit */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 font-medium">Time Unit</label>
            <select
              value={timeUnit}
              onChange={(e) => setTimeUnit(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-green-500"
            >
              {TIME_UNITS.map((u) => (
                <option key={u.key} value={u.key}>{u.label}</option>
              ))}
            </select>
          </div>
          {/* EOL threshold */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 font-medium">EOL Threshold</label>
            <div className="flex items-center gap-1.5">
              <input
                type="number" min={50} max={90} step={0.5} value={eolThreshold}
                onChange={(e) => setEolThreshold(+e.target.value)}
                className="w-20 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-green-500"
              />
              <span className="text-sm text-gray-400">%</span>
            </div>
          </div>

          {/* ML Model */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 font-medium">ML Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-green-500"
            >
              <option value="best_ensemble">Best Ensemble (RF+ET+LGB)</option>
              <option value="random_forest">Random Forest</option>
              <option value="extra_trees">Extra Trees</option>
              <option value="lightgbm">LightGBM</option>
              <option value="ridge">Ridge Regression</option>
              <option value="svr">SVR</option>
              <option value="batterygpt">BatteryGPT</option>
              <option value="tft">TFT</option>
              <option value="vae_lstm">VAE-LSTM</option>
            </select>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2 ml-auto">
            <button
              onClick={addBattery}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors border border-gray-600"
            >
              <Plus className="w-3.5 h-3.5" />
              Add Battery
            </button>
            <button
              onClick={runSimulation}
              disabled={isSimulating}
              className="flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-bold bg-green-600 hover:bg-green-500 text-white transition-all disabled:opacity-50 disabled:cursor-wait"
            >
              {isSimulating ? (
                <><Cpu className="w-4 h-4 animate-pulse" /> Computing…</>
              ) : (
                <><Zap className="w-4 h-4" /> Run Simulation</>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* ── Playback bar ───────────────────────────────────────────────── */}
      {results.length > 0 && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 space-y-3">
          <div className="flex flex-wrap gap-3 items-center">
            {/* Transport buttons */}
            <div className="flex items-center gap-1">
              <button onClick={resetPlay} title="Reset" className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-gray-700 transition-colors">
                <RotateCcw className="w-4 h-4" />
              </button>
              <button onClick={stepBwd} title="Step back" className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-gray-700 transition-colors">
                <SkipBack className="w-4 h-4" />
              </button>
              <button
                onClick={togglePlay}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-bold transition-all ${isPlaying ? "bg-red-600 hover:bg-red-500" : "bg-green-600 hover:bg-green-500"} text-white`}
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isPlaying ? "Pause" : "Play"}
              </button>
              <button onClick={stepFwd} title="Step forward" className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-gray-700 transition-colors">
                <SkipForward className="w-4 h-4" />
              </button>
            </div>

            {/* Speed */}
            <div className="flex items-center gap-1.5 bg-gray-800 rounded-lg px-3 py-1.5">
              <FastForward className="w-3.5 h-3.5 text-gray-400 mr-0.5" />
              {[0.5, 1, 2, 5, 10, 20, 50].map((s) => (
                <button
                  key={s}
                  onClick={() => setPlaySpeed(s)}
                  className={`px-2 py-0.5 rounded text-xs font-bold transition-colors ${playSpeed === s ? "bg-green-600 text-white" : "text-gray-500 hover:text-white"}`}
                >
                  {s}×
                </button>
              ))}
            </div>

            {/* Scrubber */}
            <div className="flex-1 min-w-48">
              <input
                type="range" min={0} max={Math.max(0, totalSteps - 1)} value={playIndex}
                onChange={(e) => { const v = +e.target.value; setPlayIndex(v); playIdxRef.current = v; }}
                className="w-full accent-green-500"
              />
            </div>

            {/* Time display */}
            <div className="flex gap-2">
              <div className="bg-gray-800 rounded-lg px-4 py-1.5 text-center min-w-32">
                <div className="text-[10px] text-gray-400 flex items-center gap-1 justify-center">
                  <Clock className="w-3 h-3" /> Elapsed
                </div>
                <div className="font-mono text-sm font-bold text-green-400">{elapsedTime}</div>
              </div>
              <div className="bg-gray-800 rounded-lg px-4 py-1.5 text-center min-w-20">
                <div className="text-[10px] text-gray-400">Step</div>
                <div className="font-mono text-sm font-bold text-white">{playIndex + 1}/{totalSteps}</div>
              </div>
            </div>
          </div>

          {/* Progress bar */}
          <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-linear-to-r from-green-500 to-teal-400 transition-all duration-75"
              style={{ width: `${totalSteps > 0 ? ((playIndex + 1) / totalSteps) * 100 : 0}%` }}
            />
          </div>
        </div>
      )}

      {/* ── Stats ──────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatCard icon={BatteryFull} label="Batteries"  value={batteryConfigs.length} />
        <StatCard
          icon={BarChart3} label="Avg SOH"
          value={results.length ? `${avgSoh.toFixed(1)}%` : "—"}
          color={avgSoh >= 90 ? "text-green-400" : avgSoh >= 80 ? "text-yellow-400" : avgSoh >= 70 ? "text-orange-400" : "text-red-400"}
        />
        <StatCard icon={Gauge}         label="Best Cell"    value={results.length ? `${bestBat?.soh.toFixed(1)}%` : "—"} sub={bestBat?.label} color="text-green-400" />
        <StatCard icon={TrendingDown}  label="Worst Cell"   value={results.length ? `${worstBat?.soh.toFixed(1)}%` : "—"} sub={worstBat?.label} color="text-orange-400" />
        <StatCard icon={AlertOctagon}  label="EOL Reached"  value={eolCount} color={eolCount > 0 ? "text-red-400" : "text-gray-500"} />
        <StatCard icon={Clock}         label="Elapsed"      value={elapsedTime} color="text-blue-400" />
      </div>

      {/* ── 3D view + fleet sidebar ─────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* 3D canvas */}
        <div
          className="lg:col-span-3 relative bg-gray-950 rounded-xl border border-gray-800 overflow-hidden"
          style={{ height: 560 }}
        >
          <Canvas
            camera={{ position: [6, 5, 9], fov: 48 }}
            shadows
            gl={{ antialias: true, alpha: false, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.1 }}
          >
            <color attach="background" args={["#050b16"]} />
            <fog attach="fog" args={["#050b16", 18, 42]} />
            <ambientLight intensity={0.38} />
            <directionalLight position={[8, 12, 8]} intensity={1.3} castShadow shadow-mapSize={[1024, 1024]} />
            <pointLight position={[-6, 8, -6]} intensity={0.65} color="#22c55e" />
            <pointLight position={[6,  4,  6]} intensity={0.45} color="#3b82f6" />
            <pointLight position={[0, 10,  0]} intensity={0.30} color="#ffffff" />

            <BatteryPack
              batteries={display3D}
              selected={selected3D}
              onSelect={(id) => setSelected3D((prev) => (prev === id ? null : id))}
              onOpenConfig={(id) => setConfigTarget(id)}
              isRunning={isPlaying}
            />

            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.56, 0]} receiveShadow>
              <planeGeometry args={[32, 32]} />
              <meshStandardMaterial color="#080f1a" roughness={1} />
            </mesh>
            <gridHelper args={[32, 32, "#1a2535", "#10192a"]} position={[0, -1.55, 0]} />

            <OrbitControls
              enablePan enableZoom enableRotate
              autoRotate={!isPlaying}
              autoRotateSpeed={0.4}
              minDistance={3} maxDistance={24}
              enableDamping dampingFactor={0.06}
            />
          </Canvas>

          {/* HUD */}
          <div className="absolute top-3 left-3 pointer-events-none">
            <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg px-3 py-1.5 text-xs text-gray-400 flex items-center gap-2">
              {isPlaying ? (
                <><span className="w-2 h-2 rounded-full bg-green-400 animate-pulse inline-block" />
                <span className="text-green-400">Animating</span></>
              ) : results.length > 0 ? (
                <><span className="text-blue-400">Click cell to inspect · Double-click to configure</span></>
              ) : (
                <span>Configure batteries and click Run Simulation</span>
              )}
            </div>
          </div>

          {/* SOH legend */}
          <div className="absolute bottom-3 right-3 bg-gray-900/85 backdrop-blur-sm rounded-xl p-3 text-xs space-y-1.5">
            {([["≥ 90%", "#22c55e", "Healthy"], ["80–90%", "#eab308", "Moderate"], ["70–80%", "#f97316", "Degraded"], ["< 70%", "#ef4444", "EOL"]] as const).map(([range, color, lbl]) => (
              <div key={lbl} className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-gray-300">{range} — {lbl}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Fleet Sidebar */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 flex flex-col" style={{ maxHeight: 560 }}>
          {/* Sidebar header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 shrink-0">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-gray-400" />
              <span className="text-xs font-semibold text-gray-300 uppercase tracking-wider">Fleet</span>
            </div>
            <span className="text-xs text-gray-600">{batteryConfigs.length} cells</span>
          </div>

          {/* Selected cell detail */}
          {selected3D && (() => {
            const b   = currentSohs.find((s) => s.id === selected3D);
            const cfg = batteryConfigs.find((c) => c.battery_id === selected3D);
            if (!b || !cfg) return null;
            return (
              <div className="px-3 py-3 border-b border-gray-800/60 shrink-0">
                <div className="bg-gray-800 rounded-xl p-3 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-bold text-white text-sm">{b.id}</div>
                      <div className="text-xs text-gray-400 italic">{b.label}</div>
                    </div>
                    <div className="flex gap-1">
                      <button
                        onClick={() => setConfigTarget(b.id)}
                        title="Edit configuration"
                        className="p-1.5 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition-colors"
                      >
                        <Pencil className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => duplicateBattery(b.id)}
                        title="Duplicate"
                        className="p-1.5 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition-colors"
                      >
                        <Copy className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => removeBattery(b.id)}
                        title="Remove battery"
                        className="p-1.5 rounded-lg bg-red-900/40 hover:bg-red-800/60 text-red-400 transition-colors"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-1.5 text-xs">
                    {([
                      ["SOH", `${b.soh.toFixed(1)}%`, b.color],
                      ["State", b.deg, DEG_COLORS[b.deg] ?? "#666"],
                      ["RUL", `${b.rul.toFixed(0)} cyc`, "#3b82f6"],
                      ["Re", `${b.re.toFixed(4)} Ω`, "#f59e0b"],
                      ["Rct", `${b.rct.toFixed(4)} Ω`, "#8b5cf6"],
                      ["Temp", `${cfg.ambient_temperature}°C`, "#06b6d4"],
                    ] as const).map(([lbl, val, clr]) => (
                      <div key={lbl} className="bg-gray-700/60 rounded-lg px-2 py-1.5">
                        <div className="text-gray-500 text-[10px] font-medium">{lbl}</div>
                        <div className="font-bold text-xs tabular-nums" style={{ color: clr }}>{val}</div>
                      </div>
                    ))}
                  </div>
                  {/* SOH sparkline */}
                  {selectedResult && selectedResult.soh_history.length > 1 && (
                    <div>
                      <div className="text-[10px] text-gray-500 mb-1 flex items-center gap-1">
                        <Activity className="w-3 h-3" /> SOH trend
                      </div>
                      <ResponsiveContainer width="100%" height={50}>
                        <AreaChart data={selectedResult.soh_history.filter((_, i) => i % Math.max(1, Math.floor(selectedResult.soh_history.length / 50)) === 0).map((s, i) => ({ i, s }))}>
                          <defs>
                            <linearGradient id="sparkG" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%"  stopColor={b.color} stopOpacity={0.45} />
                              <stop offset="95%" stopColor={b.color} stopOpacity={0}    />
                            </linearGradient>
                          </defs>
                          <Area type="monotone" dataKey="s" stroke={b.color} fill="url(#sparkG)" strokeWidth={1.5} dot={false} />
                          <YAxis domain={[0, 100]} hide />
                          <XAxis dataKey="i" hide />
                          <ReferenceLine y={eolThreshold} stroke="#ef4444" strokeDasharray="3 3" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              </div>
            );
          })()}

          {/* Battery list — scrollable */}
          <div className="flex-1 overflow-y-auto px-2 py-2 space-y-0.5">
            {batteryConfigs.map((cfg) => {
              const cur   = currentSohs.find((s) => s.id === cfg.battery_id);
              const soh   = cur?.soh   ?? cfg.initial_soh ?? 100;
              const color = cur?.color ?? sohColor(soh);
              const isSelected = selected3D === cfg.battery_id;
              return (
                <div
                  key={cfg.battery_id}
                  className={`group flex items-center justify-between px-2.5 py-2 rounded-lg text-sm transition-colors cursor-pointer ${isSelected ? "bg-gray-700/80" : "hover:bg-gray-800/70"}`}
                  onClick={() => setSelected3D((p) => (p === cfg.battery_id ? null : cfg.battery_id))}
                >
                  <span className="flex items-center gap-2 text-gray-300 truncate min-w-0">
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
                    <span className="truncate text-xs">{cfg.label ?? cfg.battery_id}</span>
                  </span>
                  <div className="flex items-center gap-1.5 shrink-0 ml-2">
                    <span className="font-mono text-xs font-bold" style={{ color }}>{soh.toFixed(1)}%</span>
                    {/* Edit button — visible on hover or when selected */}
                    <button
                      onClick={(e) => { e.stopPropagation(); setConfigTarget(cfg.battery_id); }}
                      title="Edit configuration"
                      className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded text-gray-500 hover:text-green-400"
                    >
                      <Pencil className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {/* EOL micro-table */}
          {results.length > 0 && (
            <div className="px-3 py-2 border-t border-gray-800 shrink-0">
              <div className="text-[10px] text-gray-500 font-semibold uppercase mb-1.5 flex items-center gap-1">
                <AlertOctagon className="w-3 h-3" /> EOL Status
              </div>
              <div className="space-y-0.5 max-h-28 overflow-y-auto">
                {results.map((r) => (
                  <div key={r.battery_id} className="flex items-center justify-between text-xs">
                    <span className="text-gray-400 text-[11px]">{r.battery_id}</span>
                    {r.eol_time !== null ? (
                      <span className="text-red-400 font-mono text-[11px]">{r.eol_time.toFixed(1)} {timeUnitLabel}</span>
                    ) : (
                      <span className="text-green-400 text-[11px]">Safe</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Analytics section ──────────────────────────────────────────── */}
      {results.length > 0 && (
        <div className="space-y-3">
          {/* Tab bar */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-2.5 flex flex-wrap gap-1.5 items-center justify-between">
            <div className="flex gap-1.5 flex-wrap">
              {CHART_TABS.map((tab) => {
                const Icon = tab.icon as React.FC<{ className?: string }>;
                return (
                  <button
                    key={tab.key}
                    onClick={() => setActiveChart(tab.key)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      activeChart === tab.key
                        ? "bg-green-700 text-white"
                        : "bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-750"
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    {tab.label}
                  </button>
                );
              })}
            </div>

            {activeChart === "trajectories" && (
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-gray-500">Metric:</span>
                {CHART_METRICS.map((m) => (
                  <button
                    key={m.key}
                    onClick={() => setChartMetric(m.key)}
                    className={`px-2.5 py-1 rounded text-xs font-medium border transition-colors ${chartMetric === m.key ? "text-white" : "border-transparent text-gray-500 hover:text-gray-300"}`}
                    style={chartMetric === m.key ? { backgroundColor: m.color + "22", color: m.color, borderColor: m.color + "55" } : {}}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* ── Fleet Overview ─── */}
          {activeChart === "fleet" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* SOH band */}
              <div className="lg:col-span-2 bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-gray-400" />
                    <h3 className="text-sm font-semibold text-gray-200">Fleet SOH Over Time</h3>
                  </div>
                  <span className="text-xs text-gray-500">X axis: {timeUnitLabel}</span>
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={fleetData} margin={{ top: 5, right: 20, bottom: 10, left: 0 }}>
                    <defs>
                      <linearGradient id="avgFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#22c55e" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#22c55e" stopOpacity={0}    />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                    <XAxis dataKey="t" stroke="#4b5563" tick={{ fontSize: 11 }}
                      label={{ value: timeUnitLabel, position: "insideBottomRight", offset: -5, fill: "#6b7280", fontSize: 11 }} />
                    <YAxis domain={[0, 105]} stroke="#4b5563" tick={{ fontSize: 11 }}
                      label={{ value: "SOH (%)", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
                    <Tooltip content={<DarkTooltip unit={timeUnitLabel} />} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <ReferenceLine y={eolThreshold} stroke="#ef4444" strokeDasharray="6 3"
                      label={{ value: `EOL ${eolThreshold}%`, fill: "#ef4444", fontSize: 10 }} />
                    <Area type="monotone" dataKey="max" name="Max SOH" stroke="#3b82f6" fill="none" dot={false} strokeWidth={1.5} strokeDasharray="4 3" />
                    <Area type="monotone" dataKey="avg" name="Avg SOH" stroke="#22c55e" fill="url(#avgFill)" strokeWidth={2.5} dot={false} />
                    <Area type="monotone" dataKey="min" name="Min SOH" stroke="#ef4444" fill="none" dot={false} strokeWidth={1.5} strokeDasharray="4 3" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Current SOH comparison */}
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <BarChart3 className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Current SOH Snapshot</h3>
                </div>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={comparisonData} layout="vertical" margin={{ left: 10, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                    <XAxis type="number" domain={[0, 100]} stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <YAxis dataKey="name" type="category" width={44} stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <Tooltip content={<DarkTooltip />} />
                    <ReferenceLine x={eolThreshold} stroke="#ef4444" strokeDasharray="4 2" />
                    <Bar dataKey="soh" name="SOH (%)" radius={[0, 5, 5, 0]}>
                      {comparisonData.map((d) => <Cell key={d.name} fill={d.color} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* RUL comparison */}
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <Clock className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Remaining Useful Life</h3>
                </div>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={comparisonData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                    <XAxis dataKey="name" stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <Tooltip content={<DarkTooltip />} />
                    <Bar dataKey="rul" name="RUL (cycles)" radius={[5, 5, 0, 0]}>
                      {comparisonData.map((d) => <Cell key={d.name} fill={d.color + "cc"} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── Trajectories ─── */}
          {activeChart === "trajectories" && (
            <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <GitBranch className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">
                    Individual Trajectories — {CHART_METRICS.find((m) => m.key === chartMetric)?.label}
                  </h3>
                </div>
                <span className="text-xs text-gray-500">X: {timeUnitLabel}</span>
              </div>
              <ResponsiveContainer width="100%" height={420}>
                <LineChart data={trajectoryData.points as any[]} margin={{ top: 5, right: 20, bottom: 10, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                  <XAxis dataKey="t" stroke="#4b5563" tick={{ fontSize: 11 }}
                    label={{ value: timeUnitLabel, position: "insideBottomRight", offset: -5, fill: "#6b7280", fontSize: 11 }} />
                  <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} />
                  <Tooltip content={<DarkTooltip unit={timeUnitLabel} />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {chartMetric === "soh" && (
                    <ReferenceLine y={eolThreshold} stroke="#ef4444" strokeDasharray="6 3"
                      label={{ value: "EOL", fill: "#ef4444", fontSize: 10 }} />
                  )}
                  {trajectoryData.keys.map((id, i) => (
                    <Line
                      key={id}
                      dataKey={id}
                      name={batteryConfigs.find((b) => b.battery_id === id)?.label ?? id}
                      stroke={CHART_COLORS[i % CHART_COLORS.length]}
                      dot={false}
                      strokeWidth={selected3D === id ? 3 : 1.5}
                      strokeOpacity={selected3D && selected3D !== id ? 0.22 : 1}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* ── Stress Analysis ─── */}
          {activeChart === "stress" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Grouped bar — stress factors */}
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <Thermometer className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Stress Factors per Battery</h3>
                  <span className="text-xs text-gray-500 ml-auto">1.0 = baseline</span>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={stressData} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                    <XAxis dataKey="name" stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <Tooltip content={<DarkTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <ReferenceLine y={1} stroke="#6b7280" strokeDasharray="4 2" />
                    <Bar dataKey="temp"  name="Temp (Arrhenius)" fill="#f59e0b" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="curr"  name="Current stress"   fill="#ef4444" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="volt"  name="Voltage stress"   fill="#8b5cf6" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Total stress radar */}
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <FlaskConical className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Multi-Factor Stress Radar</h3>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                  <RadarChart data={stressRadar}>
                    <PolarGrid stroke="#1f2a3a" />
                    <PolarAngleAxis dataKey="factor" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                    <PolarRadiusAxis stroke="#4b5563" tick={{ fontSize: 9 }} />
                    {batteryConfigs.slice(0, 7).map((b, i) => (
                      <Radar
                        key={b.battery_id}
                        name={b.label ?? b.battery_id}
                        dataKey={b.label ?? b.battery_id}
                        stroke={CHART_COLORS[i % CHART_COLORS.length]}
                        fill={CHART_COLORS[i % CHART_COLORS.length]}
                        fillOpacity={0.08}
                        strokeWidth={1.8}
                      />
                    ))}
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Total stress bar */}
              <div className="lg:col-span-2 bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <Zap className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Combined Stress Factor × Degradation Rate</h3>
                </div>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart
                    data={stressData.map((s) => {
                      const r = results.find((r) => r.battery_id === batteryConfigs.find((b) => b.battery_id.replace("BAT-", "B") === s.name || (b.label ?? b.battery_id) === s.label)?.battery_id);
                      return {
                        ...s,
                        deg: r ? +(r.deg_rate_avg * 100).toFixed(4) : 0,
                      };
                    })}
                    margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                    <XAxis dataKey="name" stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="left"  stroke="#ec4899" tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" tick={{ fontSize: 11 }} />
                    <Tooltip content={<DarkTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar yAxisId="left"  dataKey="total" name="Total Stress ×" fill="#ec4899" radius={[4, 4, 0, 0]} />
                    <Bar yAxisId="right" dataKey="deg"   name="Deg Rate %/cyc" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── Impedance ─── */}
          {activeChart === "impedance" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <Activity className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">
                    Impedance Growth — {selected3D ?? "select a cell"}
                  </h3>
                </div>
                {impedanceData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={impedanceData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                      <XAxis dataKey="t" stroke="#4b5563" tick={{ fontSize: 11 }}
                        label={{ value: timeUnitLabel, position: "insideBottomRight", offset: -5, fill: "#6b7280", fontSize: 11 }} />
                      <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} />
                      <Tooltip content={<DarkTooltip unit={timeUnitLabel} />} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      <Line type="monotone" dataKey="re"  name="Re (Ω)"  stroke="#f59e0b" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="rct" name="Rct (Ω)" stroke="#8b5cf6" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-40 text-gray-500 text-sm">
                    Select a battery cell in the 3D pack above
                  </div>
                )}
              </div>

              {/* Re / Rct current snapshot comparison */}
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <TableProperties className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Re / Rct Fleet Snapshot</h3>
                </div>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={comparisonData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                    <XAxis dataKey="name" stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <Tooltip content={<DarkTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="re"  name="Re (Ω)"  fill="#f59e0b" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="rct" name="Rct (Ω)" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Nyquist-style Re vs Rct scatter */}
              <div className="lg:col-span-2 bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <Gauge className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Nyquist-style: Re vs Rct (current step)</h3>
                  <span className="text-xs text-gray-500 ml-auto">Bubble size proportional to SOH</span>
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                    <XAxis dataKey="re"  name="Re (Ω)"  stroke="#4b5563" tick={{ fontSize: 11 }}
                      label={{ value: "Re (Ω)", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
                    <YAxis dataKey="rct" name="Rct (Ω)" stroke="#4b5563" tick={{ fontSize: 11 }}
                      label={{ value: "Rct (Ω)", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
                    <ZAxis dataKey="soh" range={[40, 280]} name="SOH %" />
                    <Tooltip
                      cursor={{ strokeDasharray: "3 3" }}
                      contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
                      formatter={(val: any) => typeof val === "number" ? val.toFixed(4) : val}
                      labelFormatter={(_, payload) => payload?.[0]?.payload?.name || ""}
                    />
                    <Scatter
                      data={comparisonData.map((d) => ({ ...d, name: d.name }))}
                      shape={(props: any) => {
                        const { cx, cy, r, fill } = props;
                        return <circle cx={cx} cy={cy} r={Math.max(5, r)} fill={fill} fillOpacity={0.75} stroke={fill} strokeWidth={1.5} />;
                      }}
                    >
                      {comparisonData.map((d, i) => <Cell key={d.name} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── Capacity Fade ─── */}
          {activeChart === "capacity" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Capacity over time */}
              <div className="lg:col-span-2 bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <BatteryFull className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Capacity Fade (Ah) Over Time</h3>
                  <span className="text-xs text-gray-500 ml-auto">Q = Q_nom × SOH/100</span>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={capacityData.points as any[]} margin={{ top: 5, right: 20, bottom: 10, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                    <XAxis dataKey="t" stroke="#4b5563" tick={{ fontSize: 11 }}
                      label={{ value: timeUnitLabel, position: "insideBottomRight", offset: -5, fill: "#6b7280", fontSize: 11 }} />
                    <YAxis stroke="#4b5563" tick={{ fontSize: 11 }}
                      label={{ value: "Capacity (Ah)", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
                    <Tooltip content={<DarkTooltip unit={timeUnitLabel} />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <ReferenceLine y={Q_NOM * eolThreshold / 100} stroke="#ef4444" strokeDasharray="6 3"
                      label={{ value: `EOL (${(Q_NOM * eolThreshold / 100).toFixed(2)} Ah)`, fill: "#ef4444", fontSize: 10 }} />
                    {capacityData.keys.map((id, i) => (
                      <Line
                        key={id}
                        dataKey={id}
                        name={batteryConfigs.find((b) => b.battery_id === id)?.label ?? id}
                        stroke={CHART_COLORS[i % CHART_COLORS.length]}
                        dot={false}
                        strokeWidth={selected3D === id ? 3 : 1.5}
                        strokeOpacity={selected3D && selected3D !== id ? 0.22 : 1}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Current capacity snapshot */}
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <Download className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Current Capacity (Ah)</h3>
                </div>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={currentSohs.map((b, i) => ({
                      name:  b.id.replace("BAT-", "B"),
                      cap:   +(Q_NOM * b.soh / 100).toFixed(3),
                      color: b.color,
                    }))}
                    margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                    <XAxis dataKey="name" stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} domain={[0, Q_NOM + 0.2]} />
                    <Tooltip content={<DarkTooltip />} />
                    <ReferenceLine y={Q_NOM * eolThreshold / 100} stroke="#ef4444" strokeDasharray="4 2" />
                    <Bar dataKey="cap" name="Capacity (Ah)" radius={[5, 5, 0, 0]}>
                      {currentSohs.map((b, i) => <Cell key={b.id} fill={b.color} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Degradation rate per battery */}
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <TrendingDown className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Avg Degradation Rate (%/cycle)</h3>
                </div>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={results.map((r) => ({ name: r.battery_id.replace("BAT-", "B"), rate: +(r.deg_rate_avg * 100).toFixed(4) }))}
                    margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                    <XAxis dataKey="name" stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <Tooltip content={<DarkTooltip />} />
                    <Bar dataKey="rate" name="Deg rate (%/cyc)" radius={[4, 4, 0, 0]}>
                      {results.map((r, i) => <Cell key={r.battery_id} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── Distribution ─── */}
          {activeChart === "distribution" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <BarChart3 className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Fleet Health Distribution</h3>
                </div>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie data={degDist} cx="50%" cy="50%" innerRadius={65} outerRadius={110} paddingAngle={4} dataKey="value"
                      label={({ name, value, percent }) => `${name}: ${value} (${((percent ?? 0) * 100).toFixed(0)}%)`}
                      labelLine={{ stroke: "#4b5563" }}>
                      {degDist.map((e) => <Cell key={e.name} fill={DEG_COLORS[e.name] ?? "#666"} />)}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 8 }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <TrendingDown className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Avg Degradation Rate (%/cycle)</h3>
                </div>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={results.map((r) => ({ name: r.battery_id.replace("BAT-", "B"), rate: +(r.deg_rate_avg * 100).toFixed(4) }))}
                    margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                    <XAxis dataKey="name" stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} />
                    <Tooltip content={<DarkTooltip />} />
                    <Bar dataKey="rate" name="Deg rate (%/cyc)" radius={[4, 4, 0, 0]}>
                      {results.map((r, i) => <Cell key={r.battery_id} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── EOL Analysis ─── */}
          {activeChart === "eol" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <AlertOctagon className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">EOL Time vs Degradation Rate</h3>
                </div>
                {eolScatter.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart margin={{ top: 5, right: 20, bottom: 30, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2a3a" />
                      <XAxis dataKey="deg_rate" name="Avg deg rate (%/cycle)" stroke="#4b5563" tick={{ fontSize: 11 }}
                        label={{ value: "Avg Deg Rate (%/cycle)", position: "insideBottom", offset: -15, fill: "#6b7280", fontSize: 11 }} />
                      <YAxis dataKey="eol_time" name={`EOL (${timeUnitLabel})`} stroke="#4b5563" tick={{ fontSize: 11 }}
                        label={{ value: `EOL (${timeUnitLabel})`, angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
                      <ZAxis range={[90, 90]} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
                        formatter={(val: any) => typeof val === "number" ? val.toFixed(3) : val} />
                      <Scatter data={eolScatter} fill="#ef4444">
                        {eolScatter.map((e, i) => <Cell key={e.id} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-40 text-sm text-gray-500 text-center">
                    No batteries reached EOL yet.<br />Increase steps or lower the threshold.
                  </div>
                )}
              </div>

              {/* Summary table */}
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-3">
                  <TableProperties className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Battery Summary</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-gray-700">
                        {["ID", "Label", "Final SOH", "RUL", "EOL Time", "Deg Rate"].map((h) => (
                          <th key={h} className="text-left py-1.5 px-2 text-gray-400 font-medium">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((r) => (
                        <tr
                          key={r.battery_id}
                          className="border-b border-gray-800 hover:bg-gray-800/40 transition-colors cursor-pointer"
                          onClick={() => setSelected3D(r.battery_id)}
                        >
                          <td className="py-1.5 px-2 text-gray-300 font-mono">{r.battery_id}</td>
                          <td className="py-1.5 px-2 text-gray-400">{r.label}</td>
                          <td className="py-1.5 px-2 font-bold" style={{ color: sohColor(r.final_soh) }}>{r.final_soh.toFixed(1)}%</td>
                          <td className="py-1.5 px-2 text-blue-400">{r.final_rul.toFixed(0)} cyc</td>
                          <td className="py-1.5 px-2 font-mono" style={{ color: r.eol_time !== null ? "#ef4444" : "#22c55e" }}>
                            {r.eol_time !== null ? `${r.eol_time.toFixed(1)} ${timeUnitLabel}` : "—"}
                          </td>
                          <td className="py-1.5 px-2 text-orange-400 font-mono">{(r.deg_rate_avg * 100).toFixed(4)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* ── Log ─── */}
          {activeChart === "log" && (
            <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <ScrollText className="w-4 h-4 text-gray-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Simulation Log</h3>
                  <span className="text-xs text-gray-600">{simLog.length} events</span>
                </div>
                <button
                  onClick={() => setSimLog([])}
                  className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors px-2 py-1 rounded hover:bg-gray-800"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear
                </button>
              </div>
              <div className="max-h-96 overflow-y-auto font-mono text-xs space-y-0.5 bg-gray-950 rounded-xl p-3">
                {simLog.length === 0 ? (
                  <span className="text-gray-600">No events. Configure batteries and click Run Simulation.</span>
                ) : (
                  simLog.map((e, i) => (
                    <div key={i} className={`flex gap-2 ${e.type === "err" ? "text-red-400" : e.type === "warn" ? "text-yellow-400" : e.type === "ok" ? "text-green-400" : "text-gray-400"}`}>
                      <span className="text-gray-600 shrink-0">{e.t}</span>
                      <span>{e.msg}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty state — before first simulation */}
      {!results.length && !isSimulating && (
        <div className="bg-gray-900/50 rounded-xl border border-dashed border-gray-700 py-14 flex flex-col items-center gap-4 text-gray-500">
          <BatteryLow className="w-12 h-12 text-gray-700" />
          <div className="text-center">
            <div className="text-base font-semibold text-gray-400">No simulation data yet</div>
            <div className="text-sm mt-1">Configure your battery fleet above, then click <strong className="text-green-400">Run Simulation</strong></div>
          </div>
        </div>
      )}
    </div>
  );
}
