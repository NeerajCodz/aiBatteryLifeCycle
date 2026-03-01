import { useEffect, useState, useMemo, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  CartesianGrid, ScatterChart, Scatter, ZAxis, AreaChart, Area,
  BarChart, Bar, Cell, ReferenceLine,
} from "recharts";
import {
  Activity, BarChart2, TrendingDown, Thermometer, Zap, GitCompare,
  Filter, RefreshCcw, Eye, EyeOff, Layers,
  AlertTriangle, CheckCircle2,
} from "lucide-react";
import { fetchBatteries, fetchBatteryCapacity, BatteryCapacity } from "../api";

const PALETTE = [
  "#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
  "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
  "#14b8a6", "#e879f9", "#fb923c", "#a3e635", "#38bdf8",
];

const TOOLTIP_STYLE = {
  backgroundColor: "#111827",
  border: "1px solid #374151",
  borderRadius: "8px",
  fontSize: 12,
};

type Section = "fleet" | "single" | "compare" | "temperature";

interface BatteryChartData {
  cycle: number;
  capacity: number;
  soh: number;
  degRate?: number;
}

function SectionBtn({ icon, label, active, onClick }: {
  icon: React.ReactNode; label: string; active: boolean; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
        active ? "bg-green-600 text-white shadow-lg shadow-green-900/40" : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white"
      }`}
    >
      <span className="w-4 h-4">{icon}</span>
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

function SohBadge({ soh }: { soh: number }) {
  const cls = soh >= 85 ? "text-green-400 bg-green-900/30" : soh >= 70 ? "text-yellow-400 bg-yellow-900/30" : "text-red-400 bg-red-900/30";
  const Icon = soh >= 85 ? CheckCircle2 : soh >= 70 ? AlertTriangle : TrendingDown;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold ${cls}`}>
      <Icon className="w-3 h-3" />
      {soh.toFixed(1)}%
    </span>
  );
}

export default function GraphPanel() {
  const [batteries, setBatteries] = useState<any[]>([]);
  const [section, setSection] = useState<Section>("fleet");

  // Single-battery state
  const [selectedBat, setSelectedBat] = useState<string>("B0005");
  const [capData, setCapData] = useState<BatteryCapacity | null>(null);
  const [loadingSingle, setLoadingSingle] = useState(false);

  // Compare state — multi-select up to 5
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<Record<string, BatteryCapacity>>({});
  const [loadingCompare, setLoadingCompare] = useState(false);

  // Filters
  const [showEol, setShowEol] = useState(true);
  const [tempMin, setTempMin] = useState(0);
  const [tempMax, setTempMax] = useState(100);
  const [sohMin, setSohMin] = useState(0);

  useEffect(() => {
    fetchBatteries().then((bs) => {
      setBatteries(bs);
      if (bs.length > 0) setSelectedBat(bs[0].battery_id);
    }).catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedBat) return;
    setLoadingSingle(true);
    fetchBatteryCapacity(selectedBat)
      .then(setCapData)
      .catch(console.error)
      .finally(() => setLoadingSingle(false));
  }, [selectedBat]);

  const loadCompare = useCallback(async (ids: string[]) => {
    setLoadingCompare(true);
    const missing = ids.filter((id) => !compareData[id]);
    await Promise.all(
      missing.map((id) =>
        fetchBatteryCapacity(id).then((d) => {
          setCompareData((prev) => ({ ...prev, [id]: d }));
        })
      )
    );
    setLoadingCompare(false);
  }, [compareData]);

  const toggleCompare = (id: string) => {
    setCompareIds((prev) => {
      const next = prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id].slice(-5);
      loadCompare(next);
      return next;
    });
  };

  const singleChart: BatteryChartData[] = useMemo(() => {
    if (!capData) return [];
    return capData.cycles.map((c, i) => ({
      cycle: c,
      capacity: capData.capacity_ah[i],
      soh: capData.soh_pct[i],
      degRate: i > 0 ? Math.abs(capData.soh_pct[i] - capData.soh_pct[i - 1]) : 0,
    }));
  }, [capData]);

  // Degradation rate smoothed
  const degRateChart = useMemo(() => {
    const window = 10;
    return singleChart.map((d, i) => {
      const slice = singleChart.slice(Math.max(0, i - window), i + 1);
      const avg = slice.reduce((s, r) => s + (r.degRate ?? 0), 0) / slice.length;
      return { cycle: d.cycle, rate: +avg.toFixed(4) };
    });
  }, [singleChart]);

  // RUL projection
  const rulProjection = useMemo(() => {
    if (singleChart.length < 10) return [];
    const last20 = singleChart.slice(-20);
    const n = last20.length;
    const xMean = last20.reduce((s, d) => s + d.cycle, 0) / n;
    const yMean = last20.reduce((s, d) => s + d.soh, 0) / n;
    const slope = last20.reduce((s, d) => s + (d.cycle - xMean) * (d.soh - yMean), 0) /
      last20.reduce((s, d) => s + (d.cycle - xMean) ** 2, 0);
    const intercept = yMean - slope * xMean;
    const lastCycle = singleChart[singleChart.length - 1].cycle;
    const eolCycle = (70 - intercept) / slope;
    const points: { cycle: number; projected: number }[] = [];
    for (let c = lastCycle; c <= eolCycle + 20; c += Math.ceil((eolCycle - lastCycle) / 30)) {
      const soh = slope * c + intercept;
      if (soh < 50) break;
      points.push({ cycle: Math.round(c), projected: +soh.toFixed(2) });
    }
    return points;
  }, [singleChart]);

  // Fleet stats
  const fleetStats = useMemo(() => {
    if (!batteries.length) return { healthy: 0, degraded: 0, eol: 0 };
    return {
      healthy: batteries.filter((b) => b.soh_pct >= 85).length,
      degraded: batteries.filter((b) => b.soh_pct >= 70 && b.soh_pct < 85).length,
      eol: batteries.filter((b) => b.soh_pct < 70).length,
    };
  }, [batteries]);

  const filteredBatteries = useMemo(() =>
    batteries.filter(
      (b) =>
        (b.ambient_temperature ?? b.avg_temperature ?? 25) >= tempMin &&
        (b.ambient_temperature ?? b.avg_temperature ?? 25) <= tempMax &&
        b.soh_pct >= sohMin
    ), [batteries, tempMin, tempMax, sohMin]);

  // Fleet bar data (sorted by SOH)
  const fleetBarData = useMemo(() =>
    [...filteredBatteries]
      .sort((a, b) => b.soh_pct - a.soh_pct)
      .slice(0, 25)
      .map((b) => ({
        id: b.battery_id,
        soh: +b.soh_pct.toFixed(1),
        temp: b.ambient_temperature ?? b.avg_temperature ?? 25,
      })), [filteredBatteries]);

  // Scatter: SOH vs cycles (temp as size)
  const scatterData = useMemo(() =>
    filteredBatteries.map((b) => ({
      x: b.n_cycles,
      y: b.soh_pct,
      z: b.ambient_temperature ?? b.avg_temperature ?? 25,
      name: b.battery_id,
    })), [filteredBatteries]);

  // Temp vs SOH scatter
  const tempScatter = useMemo(() =>
    batteries.map((b) => ({
      temp: b.ambient_temperature ?? b.avg_temperature ?? 25,
      soh: b.soh_pct,
      name: b.battery_id,
    })), [batteries]);

  // Compare overlay data
  const compareOverlay = useMemo(() => {
    if (!compareIds.length) return [];
    const maxLen = Math.max(...compareIds.map((id) => compareData[id]?.cycles?.length ?? 0));
    return Array.from({ length: maxLen }, (_, i) => {
      const row: any = { idx: i };
      compareIds.forEach((id) => {
        const d = compareData[id];
        if (d && i < d.cycles.length) {
          row[`cycle_${id}`] = d.cycles[i];
          row[`soh_${id}`] = +d.soh_pct[i].toFixed(2);
        }
      });
      return row;
    });
  }, [compareIds, compareData]);

  const sections: { key: Section; label: string; icon: React.ReactNode }[] = [
    { key: "fleet", label: "Fleet Overview", icon: <Layers className="w-4 h-4" /> },
    { key: "single", label: "Single Battery", icon: <Activity className="w-4 h-4" /> },
    { key: "compare", label: "Compare", icon: <GitCompare className="w-4 h-4" /> },
    { key: "temperature", label: "Temperature", icon: <Thermometer className="w-4 h-4" /> },
  ];

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2 bg-gray-950 p-2 rounded-xl border border-gray-800">
          {sections.map((s) => (
            <SectionBtn key={s.key} icon={s.icon} label={s.label} active={section === s.key} onClick={() => setSection(s.key)} />
          ))}
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <Zap className="w-3.5 h-3.5" />
          <span>{batteries.length} batteries loaded</span>
        </div>
      </div>

      {/* ── FLEET OVERVIEW ── */}
      {section === "fleet" && (
        <div className="space-y-5">
          {/* Fleet status cards */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-gray-900 rounded-xl p-4 border border-green-800/40">
              <div className="flex items-center gap-2 mb-1">
                <CheckCircle2 className="w-4 h-4 text-green-400" />
                <span className="text-xs text-gray-400">Healthy (≥85%)</span>
              </div>
              <div className="text-2xl font-bold text-green-400">{fleetStats.healthy}</div>
            </div>
            <div className="bg-gray-900 rounded-xl p-4 border border-yellow-800/40">
              <div className="flex items-center gap-2 mb-1">
                <AlertTriangle className="w-4 h-4 text-yellow-400" />
                <span className="text-xs text-gray-400">Degraded (70–85%)</span>
              </div>
              <div className="text-2xl font-bold text-yellow-400">{fleetStats.degraded}</div>
            </div>
            <div className="bg-gray-900 rounded-xl p-4 border border-red-800/40">
              <div className="flex items-center gap-2 mb-1">
                <TrendingDown className="w-4 h-4 text-red-400" />
                <span className="text-xs text-gray-400">Near EOL (&lt;70%)</span>
              </div>
              <div className="text-2xl font-bold text-red-400">{fleetStats.eol}</div>
            </div>
          </div>

          {/* Filters */}
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-gray-400" />
                <span className="text-xs text-gray-400">Filters:</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Min SOH:</span>
                <input
                  type="range" min={0} max={100} step={5} value={sohMin}
                  onChange={(e) => setSohMin(+e.target.value)}
                  className="w-24 accent-green-500"
                />
                <span className="text-xs text-green-400 font-mono w-8">{sohMin}%</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Temp range:</span>
                <input
                  type="number" value={tempMin} onChange={(e) => setTempMin(+e.target.value)}
                  className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white"
                />
                <span className="text-xs text-gray-500">–</span>
                <input
                  type="number" value={tempMax} onChange={(e) => setTempMax(+e.target.value)}
                  className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white"
                />
                <span className="text-xs text-gray-400">°C</span>
              </div>
              <button
                onClick={() => { setSohMin(0); setTempMin(0); setTempMax(100); }}
                className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors"
              >
                <RefreshCcw className="w-3 h-3" /> Reset
              </button>
              <span className="text-xs text-gray-500">{filteredBatteries.length} / {batteries.length} shown</span>
            </div>
          </div>

          {/* Fleet SOH bar chart */}
          <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
            <div className="flex items-center gap-2 mb-4">
              <BarChart2 className="w-4 h-4 text-green-400" />
              <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Fleet SOH — Sorted (Top 25)</h3>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={fleetBarData} margin={{ bottom: 30 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="id" tick={{ fontSize: 9 }} stroke="#6b7280" angle={-45} textAnchor="end" />
                <YAxis domain={[0, 110]} unit="%" stroke="#6b7280" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: any) => [`${v}%`, "SOH"]} />
                <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "EOL 70%", fill: "#ef4444", fontSize: 10 }} />
                <ReferenceLine y={85} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: "85%", fill: "#f59e0b", fontSize: 10 }} />
                <Bar dataKey="soh" radius={[3, 3, 0, 0]}>
                  {fleetBarData.map((d, i) => (
                    <Cell key={i} fill={d.soh >= 85 ? "#22c55e" : d.soh >= 70 ? "#f59e0b" : "#ef4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Fleet scatter */}
          <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="w-4 h-4 text-blue-400" />
              <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">SOH vs Cycles (bubble size = temperature)</h3>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="x" name="Cycles" stroke="#6b7280" label={{ value: "Cycles", position: "insideBottom", offset: -5, fill: "#9ca3af", fontSize: 11 }} tick={{ fontSize: 10 }} />
                <YAxis dataKey="y" name="SOH %" domain={[40, 110]} stroke="#6b7280" label={{ value: "SOH %", angle: -90, position: "insideLeft", fill: "#9ca3af", fontSize: 11 }} tick={{ fontSize: 10 }} />
                <ZAxis dataKey="z" range={[40, 400]} name="Temp °C" />
                <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="4 4" />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={TOOLTIP_STYLE}
                  content={({ payload }) => payload?.length ? (
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-2 text-xs">
                      <div className="font-semibold text-white">{payload[0].payload.name}</div>
                      <div className="text-green-400">SOH: {payload[0].payload.y}%</div>
                      <div className="text-blue-400">Cycles: {payload[0].payload.x}</div>
                      <div className="text-orange-400">Temp: {payload[0].payload.z}°C</div>
                    </div>
                  ) : null}
                />
                <Scatter data={scatterData} fill="#22c55e" fillOpacity={0.75} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Battery table */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
            <div className="p-4 border-b border-gray-800 flex items-center gap-2">
              <Layers className="w-4 h-4 text-gray-400" />
              <span className="text-sm font-semibold text-gray-300">Battery Roster</span>
            </div>
            <div className="overflow-x-auto max-h-72">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-gray-950">
                  <tr className="text-gray-500 border-b border-gray-800">
                    <th className="py-2 px-3 text-left">ID</th>
                    <th className="py-2 px-3 text-right">SOH</th>
                    <th className="py-2 px-3 text-right">Cycles</th>
                    <th className="py-2 px-3 text-right">Temp °C</th>
                    <th className="py-2 px-3 text-left">State</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredBatteries.map((b, i) => (
                    <tr
                      key={i}
                      className="border-b border-gray-800/40 hover:bg-gray-800/40 cursor-pointer transition-colors"
                      onClick={() => { setSelectedBat(b.battery_id); setSection("single"); }}
                    >
                      <td className="py-2 px-3 font-medium text-white">{b.battery_id}</td>
                      <td className="py-2 px-3 text-right"><SohBadge soh={b.soh_pct} /></td>
                      <td className="py-2 px-3 text-right text-gray-300">{b.n_cycles ?? "—"}</td>
                      <td className="py-2 px-3 text-right text-gray-300">{(b.ambient_temperature ?? b.avg_temperature ?? "—")}</td>
                      <td className="py-2 px-3">
                        <span className="text-xs text-gray-400 bg-gray-800 px-2 py-0.5 rounded-full">{b.degradation_state ?? "—"}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── SINGLE BATTERY ── */}
      {section === "single" && (
        <div className="space-y-5">
          <div className="flex flex-wrap items-center gap-3">
            <label className="text-sm text-gray-400">Battery:</label>
            <select
              value={selectedBat}
              onChange={(e) => setSelectedBat(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
            >
              {batteries.map((b) => (
                <option key={b.battery_id} value={b.battery_id}>
                  {b.battery_id} — {b.soh_pct?.toFixed(1)}% SOH
                </option>
              ))}
            </select>
            <button
              onClick={() => setShowEol((s) => !s)}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs transition-colors ${showEol ? "bg-red-900/40 text-red-400" : "bg-gray-800 text-gray-400"}`}
            >
              {showEol ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
              EOL Line
            </button>
            {loadingSingle && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-400" />}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* SOH + RUL projection */}
            <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
              <div className="flex items-center gap-2 mb-4">
                <Activity className="w-4 h-4 text-green-400" />
                <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">SOH Trajectory + RUL Projection</h3>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart margin={{ right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="cycle" type="number" domain={["dataMin", "dataMax"]} stroke="#6b7280" tick={{ fontSize: 10 }} />
                  <YAxis domain={[45, 110]} stroke="#6b7280" tick={{ fontSize: 10 }} unit="%" />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: any, name) => [`${v}%`, name]} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {showEol && <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "EOL", fill: "#ef4444", fontSize: 10 }} />}
                  <Line data={singleChart} type="monotone" dataKey="soh" stroke="#22c55e" dot={false} strokeWidth={2} name="Measured SOH" />
                  {rulProjection.length > 0 && (
                    <Line data={rulProjection} type="monotone" dataKey="projected" stroke="#f59e0b" dot={false} strokeWidth={1.5} strokeDasharray="6 3" name="Projected RUL" />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Capacity fade */}
            <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
              <div className="flex items-center gap-2 mb-4">
                <TrendingDown className="w-4 h-4 text-blue-400" />
                <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Capacity Fade (Ah)</h3>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={singleChart}>
                  <defs>
                    <linearGradient id="capGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="cycle" stroke="#6b7280" tick={{ fontSize: 10 }} />
                  <YAxis domain={[1.0, 2.2]} stroke="#6b7280" tick={{ fontSize: 10 }} unit="Ah" />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: any) => [`${v} Ah`, "Capacity"]} />
                  {showEol && <ReferenceLine y={1.4} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "EOL 1.4Ah", fill: "#ef4444", fontSize: 10 }} />}
                  <Area type="monotone" dataKey="capacity" stroke="#3b82f6" fill="url(#capGrad)" strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Degradation rate */}
            <div className="lg:col-span-2 bg-gray-900 rounded-xl p-5 border border-gray-800">
              <div className="flex items-center gap-2 mb-4">
                <BarChart2 className="w-4 h-4 text-orange-400" />
                <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Degradation Rate (SOH %/cycle, smoothed)</h3>
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={degRateChart}>
                  <defs>
                    <linearGradient id="rateGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f97316" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="cycle" stroke="#6b7280" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} unit="%" />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: any) => [`${v}%/cyc`, "Deg Rate"]} />
                  <Area type="monotone" dataKey="rate" stroke="#f97316" fill="url(#rateGrad)" strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* ── COMPARE ── */}
      {section === "compare" && (
        <div className="space-y-5">
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <div className="flex items-center gap-2 mb-3">
              <GitCompare className="w-4 h-4 text-purple-400" />
              <span className="text-sm font-semibold text-gray-300">Select up to 5 batteries to compare</span>
              {loadingCompare && <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-purple-400" />}
            </div>
            <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto">
              {batteries.map((b) => {
                const selected = compareIds.includes(b.battery_id);
                return (
                  <button
                    key={b.battery_id}
                    onClick={() => toggleCompare(b.battery_id)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                      selected
                        ? "text-white shadow-lg"
                        : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                    }`}
                    style={selected ? {
                      backgroundColor: PALETTE[compareIds.indexOf(b.battery_id)] + "33",
                      borderColor: PALETTE[compareIds.indexOf(b.battery_id)],
                      border: "1px solid",
                      color: PALETTE[compareIds.indexOf(b.battery_id)],
                    } : undefined}
                  >
                    {b.battery_id}
                  </button>
                );
              })}
            </div>
          </div>

          {compareIds.length === 0 ? (
            <div className="text-center py-16 text-gray-500">
              <GitCompare className="w-10 h-10 mx-auto mb-3 opacity-30" />
              <p>Select batteries above to compare</p>
            </div>
          ) : (
            <div className="space-y-5">
              {/* SOH overlay */}
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-4">
                  <Activity className="w-4 h-4 text-green-400" />
                  <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">SOH Comparison Overlay</h3>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="cycle" type="number" stroke="#6b7280" tick={{ fontSize: 10 }} />
                    <YAxis domain={[50, 110]} stroke="#6b7280" tick={{ fontSize: 10 }} unit="%" />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="4 4" />
                    {compareIds.map((id, i) => {
                      const d = compareData[id];
                      if (!d) return null;
                      const lineData = d.cycles.map((c, j) => ({ cycle: c, soh: d.soh_pct[j] }));
                      return (
                        <Line
                          key={id}
                          data={lineData}
                          type="monotone"
                          dataKey="soh"
                          stroke={PALETTE[i]}
                          dot={false}
                          strokeWidth={2}
                          name={id}
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Capacity overlay */}
              <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div className="flex items-center gap-2 mb-4">
                  <TrendingDown className="w-4 h-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Capacity Fade Comparison</h3>
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="cycle" type="number" stroke="#6b7280" tick={{ fontSize: 10 }} />
                    <YAxis domain={[1.0, 2.2]} stroke="#6b7280" tick={{ fontSize: 10 }} unit="Ah" />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <ReferenceLine y={1.4} stroke="#ef4444" strokeDasharray="4 4" />
                    {compareIds.map((id, i) => {
                      const d = compareData[id];
                      if (!d) return null;
                      const lineData = d.cycles.map((c, j) => ({ cycle: c, capacity: d.capacity_ah[j] }));
                      return (
                        <Line
                          key={id}
                          data={lineData}
                          type="monotone"
                          dataKey="capacity"
                          stroke={PALETTE[i]}
                          dot={false}
                          strokeWidth={2}
                          name={id}
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Summary comparison table */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
                <div className="p-4 border-b border-gray-800">
                  <span className="text-sm font-semibold text-gray-300">Comparison Summary</span>
                </div>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-500 border-b border-gray-800 bg-gray-950/50">
                      <th className="py-2 px-3 text-left">Battery</th>
                      <th className="py-2 px-3 text-right">Final SOH</th>
                      <th className="py-2 px-3 text-right">Cycles</th>
                      <th className="py-2 px-3 text-right">Min Capacity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {compareIds.map((id, i) => {
                      const d = compareData[id];
                      const lastSoh = d?.soh_pct[d.soh_pct.length - 1];
                      const minCap = d ? Math.min(...d.capacity_ah) : null;
                      return (
                        <tr key={id} className="border-b border-gray-800/40 hover:bg-gray-800/40">
                          <td className="py-2 px-3">
                            <span className="inline-flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: PALETTE[i] }} />
                              <span className="font-medium text-white">{id}</span>
                            </span>
                          </td>
                          <td className="py-2 px-3 text-right">
                            {lastSoh != null ? <SohBadge soh={lastSoh} /> : "—"}
                          </td>
                          <td className="py-2 px-3 text-right text-gray-300">{d?.cycles.length ?? "—"}</td>
                          <td className="py-2 px-3 text-right text-blue-400">{minCap != null ? `${minCap.toFixed(3)} Ah` : "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── TEMPERATURE ANALYSIS ── */}
      {section === "temperature" && (
        <div className="space-y-5">
          <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
            <div className="flex items-center gap-2 mb-4">
              <Thermometer className="w-4 h-4 text-orange-400" />
              <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Temperature vs Final SOH</h3>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="temp" name="Temp °C" stroke="#6b7280" label={{ value: "Temperature (°C)", position: "insideBottom", offset: -5, fill: "#9ca3af", fontSize: 11 }} tick={{ fontSize: 10 }} />
                <YAxis dataKey="soh" name="SOH %" stroke="#6b7280" label={{ value: "SOH %", angle: -90, position: "insideLeft", fill: "#9ca3af", fontSize: 11 }} tick={{ fontSize: 10 }} domain={[40, 110]} />
                <ZAxis range={[50, 50]} />
                <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="4 4" />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={TOOLTIP_STYLE}
                  content={({ payload }) => payload?.length ? (
                    <div className="bg-gray-900 border border-gray-700 rounded-lg p-2 text-xs">
                      <div className="font-semibold text-white">{payload[0].payload.name}</div>
                      <div className="text-orange-400">Temp: {payload[0].payload.temp}°C</div>
                      <div className="text-green-400">SOH: {payload[0].payload.soh}%</div>
                    </div>
                  ) : null}
                />
                <Scatter data={tempScatter} fill="#f97316" fillOpacity={0.75} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Temperature distribution histogram */}
          <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
            <div className="flex items-center gap-2 mb-4">
              <BarChart2 className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Temperature Distribution</h3>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={(() => {
                const bins: Record<string, number> = {};
                batteries.forEach((b) => {
                  const t = Math.round((b.ambient_temperature ?? b.avg_temperature ?? 25) / 5) * 5;
                  bins[t] = (bins[t] ?? 0) + 1;
                });
                return Object.entries(bins).sort(([a], [b]) => +a - +b).map(([t, count]) => ({ temp: `${t}°C`, count }));
              })()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="temp" stroke="#6b7280" tick={{ fontSize: 10 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} name="Batteries" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
