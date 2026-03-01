import { useState, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  CartesianGrid, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Cell, ReferenceLine,
} from "recharts";
import {
  Zap, Target, TrendingUp, TrendingDown, Thermometer, Activity,
  Trophy, Award, Medal, BarChart2, GitCompare, RefreshCcw, ChevronDown,
  ChevronUp, Info, AlertTriangle, CheckCircle2, Sliders,
} from "lucide-react";
import { fetchRecommendations, RecommendationResponse } from "../api";

const CHART_COLORS = [
  "#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
  "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
];

const TOOLTIP_STYLE = { backgroundColor: "#111827", border: "1px solid #374151", borderRadius: "8px", fontSize: 12 };

function RankIcon({ rank }: { rank: number }) {
  if (rank === 1) return <Trophy className="w-4 h-4 text-yellow-400" />;
  if (rank === 2) return <Award className="w-4 h-4 text-gray-300" />;
  if (rank === 3) return <Medal className="w-4 h-4 text-orange-400" />;
  return <span className="text-xs font-bold text-gray-400">#{rank}</span>;
}

function SliderInput({ label, value, min, max, step, unit, onChange }: {
  label: string; value: number; min: number; max: number; step: number; unit: string; onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between">
        <label className="text-xs text-gray-400">{label}</label>
        <span className="text-xs font-mono text-green-400">{value}{unit}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(+e.target.value)}
        className="w-full accent-green-500 h-1.5"
      />
      <div className="flex justify-between text-xs text-gray-600">
        <span>{min}{unit}</span><span>{max}{unit}</span>
      </div>
    </div>
  );
}

export default function RecommendationPanel() {
  const [batteryId, setBatteryId] = useState("B0005");
  const [currentCycle, setCurrentCycle] = useState(100);
  const [currentSoh, setCurrentSoh] = useState(85);
  const [ambientTemp, setAmbientTemp] = useState(24);
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const [chartTab, setChartTab] = useState<"rul" | "params" | "radar">("rul");

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchRecommendations({
        battery_id: batteryId,
        current_cycle: currentCycle,
        current_soh: currentSoh,
        ambient_temperature: ambientTemp,
        top_k: topK,
      });
      setResult(res);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  // Derived chart data
  const rulBarData = useMemo(() => {
    if (!result?.recommendations) return [];
    return result.recommendations.map((r) => ({
      rank: `#${r.rank}`,
      RUL: Math.round(r.predicted_rul),
      Improvement: Math.round(r.rul_improvement),
      fill: CHART_COLORS[(r.rank - 1) % CHART_COLORS.length],
    }));
  }, [result]);

  const paramData = useMemo(() => {
    if (!result?.recommendations) return [];
    return result.recommendations.map((r) => ({
      rank: `#${r.rank}`,
      "Temp (°C)": r.ambient_temperature,
      "Current (A)": r.discharge_current,
      "Cutoff (V)": r.cutoff_voltage * 10,
    }));
  }, [result]);

  const radarData = useMemo(() => {
    if (!result?.recommendations || result.recommendations.length < 2) return [];
    const top3 = result.recommendations.slice(0, 3);
    const maxRul = Math.max(...top3.map((r) => r.predicted_rul));
    const maxImp = Math.max(...top3.map((r) => Math.abs(r.rul_improvement)), 1);
    return [
      { metric: "RUL", ...Object.fromEntries(top3.map((r) => [`#${r.rank}`, +((r.predicted_rul / maxRul) * 100).toFixed(1)])) },
      { metric: "Improvement", ...Object.fromEntries(top3.map((r) => [`#${r.rank}`, +(Math.max(0, r.rul_improvement) / maxImp * 100).toFixed(1)])) },
      { metric: "Low Temp", ...Object.fromEntries(top3.map((r) => [`#${r.rank}`, +(Math.max(0, 45 - r.ambient_temperature) / 45 * 100).toFixed(1)])) },
      { metric: "Low Current", ...Object.fromEntries(top3.map((r) => [`#${r.rank}`, +(Math.max(0, 3 - r.discharge_current) / 3 * 100).toFixed(1)])) },
    ];
  }, [result]);

  const baseline = result?.recommendations[0];
  const bestOnly = result?.recommendations.find((r) => r.rank === 1);

  return (
    <div className="space-y-5">
      {/* Input panel */}
      <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
        <div className="flex items-center gap-2 mb-3">
          <Sliders className="w-4 h-4 text-green-400" />
          <h2 className="text-base font-semibold">Operating Condition Optimizer</h2>
        </div>
        <p className="text-xs text-gray-400 mb-4">
          Find optimal temperature, discharge current and cutoff voltage to maximize Remaining Useful Life.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Text inputs */}
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Battery ID</label>
              <input
                type="text" value={batteryId} onChange={(e) => setBatteryId(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Top K Results</label>
                <input
                  type="number" min={1} max={20} value={topK} onChange={(e) => setTopK(+e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Current Cycle</label>
                <input
                  type="number" value={currentCycle} onChange={(e) => setCurrentCycle(+e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
                />
              </div>
            </div>
          </div>
          {/* Sliders */}
          <div className="space-y-4">
            <SliderInput label="Current SOH" value={currentSoh} min={50} max={100} step={0.5} unit="%" onChange={setCurrentSoh} />
            <SliderInput label="Ambient Temperature" value={ambientTemp} min={0} max={60} step={1} unit="°C" onChange={setAmbientTemp} />
          </div>
        </div>

        <div className="flex items-center gap-3 mt-4">
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="flex items-center gap-2 bg-green-600 hover:bg-green-500 text-white font-medium px-5 py-2.5 rounded-lg transition-colors disabled:opacity-50"
          >
            {loading ? (
              <><div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" /> Optimizing…</>
            ) : (
              <><TrendingUp className="w-4 h-4" /> Get Recommendations</>
            )}
          </button>
          {result && (
            <button onClick={() => setResult(null)} className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors">
              <RefreshCcw className="w-3.5 h-3.5" /> Clear
            </button>
          )}
        </div>

        {error && (
          <div className="mt-3 flex items-start gap-2 text-sm text-red-400 bg-red-900/20 border border-red-800 p-3 rounded-lg">
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            {error}
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-5">
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-gray-900 rounded-xl p-4 border border-green-800/40">
              <div className="flex items-center gap-2 mb-1">
                <Zap className="w-4 h-4 text-green-400" />
                <span className="text-xs text-gray-400">Battery</span>
              </div>
              <div className="text-xl font-bold text-white">{result.battery_id}</div>
              <div className="text-xs text-gray-500">Current SOH: {result.current_soh}%</div>
            </div>
            <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
              <div className="flex items-center gap-2 mb-1">
                <Trophy className="w-4 h-4 text-yellow-400" />
                <span className="text-xs text-gray-400">Best RUL</span>
              </div>
              <div className="text-xl font-bold text-yellow-400">{bestOnly?.predicted_rul.toFixed(0)} cyc</div>
              <div className="text-xs text-gray-500">Top recommendation</div>
            </div>
            <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp className="w-4 h-4 text-blue-400" />
                <span className="text-xs text-gray-400">Best Improvement</span>
              </div>
              <div className="text-xl font-bold text-blue-400">
                {bestOnly && bestOnly.rul_improvement > 0 ? "+" : ""}{bestOnly?.rul_improvement.toFixed(0)} cyc
              </div>
              <div className="text-xs text-gray-500">{bestOnly?.rul_improvement_pct}% gain</div>
            </div>
            <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
              <div className="flex items-center gap-2 mb-1">
                <BarChart2 className="w-4 h-4 text-purple-400" />
                <span className="text-xs text-gray-400">Recommendations</span>
              </div>
              <div className="text-xl font-bold text-purple-400">{result.recommendations.length}</div>
              <div className="text-xs text-gray-500">configurations</div>
            </div>
          </div>

          {/* Chart tabs */}
          <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <GitCompare className="w-4 h-4 text-green-400" />
                <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Visual Analysis</h3>
              </div>
              <div className="flex gap-1">
                {(["rul", "params", "radar"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setChartTab(t)}
                    className={`px-2.5 py-1 rounded text-xs capitalize transition-colors ${chartTab === t ? "bg-green-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
                  >
                    {t === "rul" ? "RUL Comparison" : t === "params" ? "Parameters" : "Radar"}
                  </button>
                ))}
              </div>
            </div>

            {chartTab === "rul" && (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={rulBarData} margin={{ bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="rank" stroke="#6b7280" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} label={{ value: "Cycles", angle: -90, position: "insideLeft", fill: "#9ca3af", fontSize: 11 }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: any, name) => [`${v} cycles`, name]} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <ReferenceLine y={result.current_soh * 5} stroke="#6b7280" strokeDasharray="4 4" label={{ value: "Baseline", fill: "#6b7280", fontSize: 10 }} />
                  <Bar dataKey="RUL" name="Predicted RUL" radius={[4, 4, 0, 0]}>
                    {rulBarData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}

            {chartTab === "params" && (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={paramData} margin={{ bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="rank" stroke="#6b7280" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="Temp (°C)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Current (A)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Cutoff (V)" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}

            {chartTab === "radar" && radarData.length > 0 && (
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#374151" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                  <PolarRadiusAxis domain={[0, 100]} tick={{ fill: "#6b7280", fontSize: 9 }} />
                  {result.recommendations.slice(0, 3).map((r, i) => (
                    <Radar key={r.rank} name={`#${r.rank}`} dataKey={`#${r.rank}`}
                      stroke={CHART_COLORS[i]} fill={CHART_COLORS[i]} fillOpacity={0.15} />
                  ))}
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                </RadarChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Recommendations table */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
            <div className="p-4 border-b border-gray-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Trophy className="w-4 h-4 text-yellow-400" />
                <span className="text-sm font-semibold text-gray-300">
                  Recommendations for {result.battery_id} — SOH: {result.current_soh}%
                </span>
              </div>
              <span className="text-xs text-gray-500">{result.recommendations.length} configs</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-800 bg-gray-950/50">
                    <th className="py-2 px-3 text-left w-8">#</th>
                    <th className="py-2 px-3 text-right">Temp (°C)</th>
                    <th className="py-2 px-3 text-right">Current (A)</th>
                    <th className="py-2 px-3 text-right">Cutoff (V)</th>
                    <th className="py-2 px-3 text-right">Pred. RUL</th>
                    <th className="py-2 px-3 text-right">Improvement</th>
                    <th className="py-2 px-3 text-right">% Gain</th>
                    <th className="py-2 px-3 w-8" />
                  </tr>
                </thead>
                <tbody>
                  {result.recommendations.map((rec) => {
                    const expanded = expandedRow === rec.rank;
                    const impPositive = rec.rul_improvement > 0;
                    return (
                      <>
                        <tr
                          key={rec.rank}
                          className={`border-b border-gray-800/40 hover:bg-gray-800/40 transition-colors cursor-pointer ${
                            rec.rank === 1 ? "bg-yellow-900/10" : ""
                          }`}
                          onClick={() => setExpandedRow(expanded ? null : rec.rank)}
                        >
                          <td className="py-2.5 px-3">
                            <span className="flex items-center"><RankIcon rank={rec.rank} /></span>
                          </td>
                          <td className="py-2.5 px-3 text-right">
                            <span className="flex items-center justify-end gap-1">
                              <Thermometer className="w-3 h-3 text-orange-400" />{rec.ambient_temperature}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right text-blue-400">{rec.discharge_current}A</td>
                          <td className="py-2.5 px-3 text-right text-purple-400">{rec.cutoff_voltage}V</td>
                          <td className="py-2.5 px-3 text-right font-semibold text-green-400">{rec.predicted_rul.toFixed(0)}</td>
                          <td className="py-2.5 px-3 text-right">
                            <span className={impPositive ? "text-green-400" : "text-red-400"}>
                              {impPositive ? "+" : ""}{rec.rul_improvement.toFixed(0)}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right">
                            <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                              impPositive ? "bg-green-900/40 text-green-400" : "bg-red-900/40 text-red-400"
                            }`}>
                              {impPositive ? "+" : ""}{rec.rul_improvement_pct}%
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right">
                            {expanded ? <ChevronUp className="w-3.5 h-3.5 text-gray-400" /> : <ChevronDown className="w-3.5 h-3.5 text-gray-400" />}
                          </td>
                        </tr>
                        {expanded && (
                          <tr key={`${rec.rank}-exp`} className="border-b border-gray-800/40 bg-gray-950/50">
                            <td colSpan={8} className="px-4 py-3">
                              <div className="flex items-start gap-2">
                                <Info className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                                <p className="text-xs text-gray-400 leading-relaxed">{rec.explanation}</p>
                              </div>
                              <div className="mt-2 flex gap-4">
                                <div className="flex items-center gap-1.5 text-xs text-gray-400">
                                  <Thermometer className="w-3 h-3 text-orange-400" />
                                  Temp: <span className="text-white">{rec.ambient_temperature}°C</span>
                                </div>
                                <div className="flex items-center gap-1.5 text-xs text-gray-400">
                                  <Activity className="w-3 h-3 text-blue-400" />
                                  Current: <span className="text-white">{rec.discharge_current}A</span>
                                </div>
                                <div className="flex items-center gap-1.5 text-xs text-gray-400">
                                  <Zap className="w-3 h-3 text-purple-400" />
                                  Cutoff: <span className="text-white">{rec.cutoff_voltage}V</span>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
