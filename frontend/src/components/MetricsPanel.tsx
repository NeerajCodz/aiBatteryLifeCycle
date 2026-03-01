import { useEffect, useState, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  CartesianGrid, LineChart, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ScatterChart, Scatter, ZAxis,
  AreaChart, Area, PieChart, Pie, Cell,
} from "recharts";
import {
  Trophy, BarChart2, Target, Cpu, CheckCircle2, Database, Layers,
  TrendingUp, TrendingDown, Activity, Zap, Filter, ArrowUpDown,
  Award, GitCompare, Search, FlaskConical, BrainCircuit,
  GanttChart, ImageIcon, X, Gauge, Info,
} from "lucide-react";
import { fetchMetrics } from "../api";

interface MetricsData {
  unified_results: any[];
  classical_v2: any[];
  classical_soh: any[];
  lstm_results: any[];
  ensemble_results: any[];
  transformer_results: any[];
  validation: any[];
  rankings: any[];
  classical_rul: any[];
  training_summary: any;
  validation_summary: any;
  intra_battery: any;
  vae_lstm: any;
  dg_itransformer: any;
  v2_figures: string[];
  battery_stats: any;
}

const CHART_COLORS = [
  "#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
  "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
  "#14b8a6", "#e879f9", "#fb923c", "#a3e635", "#38bdf8",
];

type MetricKey = "R2" | "MAE" | "RMSE" | "MAPE";
const METRIC_KEYS: MetricKey[] = ["R2", "MAE", "RMSE", "MAPE"];

function getMetricVal(row: any, key: MetricKey): number {
  if (key === "R2") return row.R2 ?? row.r2 ?? 0;
  return row[key] ?? row[key.toLowerCase()] ?? 0;
}

function MetricBadge({ value, metric }: { value: number; metric: MetricKey }) {
  const isError = metric !== "R2";
  const good = isError ? value < 2 : value > 0.9;
  const ok = isError ? value < 5 : value > 0.7;
  const cls = good ? "text-green-400 bg-green-900/30" : ok ? "text-yellow-400 bg-yellow-900/30" : "text-red-400 bg-red-900/30";
  return <span className={`px-2 py-0.5 rounded text-xs font-mono font-semibold ${cls}`}>{value.toFixed(4)}</span>;
}

function SectionBadge({ icon, label, active, onClick }: {
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

function StatCard({ label, value, color = "text-green-400", subtitle, icon, trend }: {
  label: string; value: string | number; color?: string; subtitle?: string;
  icon?: React.ReactNode; trend?: "up" | "down";
}) {
  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-800 hover:border-gray-700 transition-all hover:bg-gray-900/80 group">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          {icon && (
            <div className="p-1.5 rounded-lg bg-gray-800 text-gray-400 group-hover:text-green-400 transition-colors">
              {icon}
            </div>
          )}
          <span className="text-xs text-gray-400 leading-tight">{label}</span>
        </div>
        {trend && (
          <span className={trend === "up" ? "text-green-400" : "text-red-400"}>
            {trend === "up" ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
          </span>
        )}
      </div>
      <div className={`text-2xl font-bold ${color} truncate`}>{value}</div>
      {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
    </div>
  );
}

export default function MetricsPanel() {
  const [data, setData] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<string>("overview");
  const [selectedFigure, setSelectedFigure] = useState<string | null>(null);
  const [figureSearch, setFigureSearch] = useState("");
  const [sortBy, setSortBy] = useState<MetricKey>("R2");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [familyFilter, setFamilyFilter] = useState<string>("all");
  const [compareMode, setCompareMode] = useState(false);
  const [compareSelected, setCompareSelected] = useState<string[]>([]);
  const [chartView, setChartView] = useState<"bar" | "radar" | "scatter">("bar");

  useEffect(() => {
    fetchMetrics()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const sections = [
    { key: "overview", label: "Overview", icon: <GanttChart className="w-4 h-4" /> },
    { key: "models", label: "Models", icon: <BarChart2 className="w-4 h-4" /> },
    { key: "validation", label: "Validation", icon: <CheckCircle2 className="w-4 h-4" /> },
    { key: "deep", label: "Deep Learning", icon: <BrainCircuit className="w-4 h-4" /> },
    { key: "dataset", label: "Dataset", icon: <Database className="w-4 h-4" /> },
    { key: "figures", label: "Figures", icon: <ImageIcon className="w-4 h-4" /> },
  ];

  // Process data — safe Array.isArray guards throughout
  const unifiedSorted = useMemo(() => {
    if (!Array.isArray(data?.unified_results)) return [];
    return [...data!.unified_results].sort((a, b) => (b.R2 ?? b.r2 ?? 0) - (a.R2 ?? a.r2 ?? 0));
  }, [data]);

  const classicalV2Sorted = useMemo(() => {
    if (!Array.isArray(data?.classical_v2)) return [];
    return [...data!.classical_v2].sort((a, b) => (b.r2 ?? b.R2 ?? 0) - (a.r2 ?? a.R2 ?? 0));
  }, [data]);

  const validationSorted = useMemo(() => {
    if (!Array.isArray(data?.validation)) return [];
    return [...data!.validation].sort((a, b) => (b.within_5pct ?? 0) - (a.within_5pct ?? 0));
  }, [data]);

  // Radar data for top models
  const radarData = useMemo(() => {
    if (!unifiedSorted.length) return [];
    const top5 = unifiedSorted.slice(0, 5);
    const metrics = ["R2", "MAE", "RMSE", "MAPE"];
    return metrics.map((m) => {
      const row: any = { metric: m };
      top5.forEach((model) => {
        let val = model[m] ?? 0;
        // Normalize: invert error metrics so higher = better
        if (m === "MAE" || m === "RMSE" || m === "MAPE") {
          const maxVal = Math.max(...unifiedSorted.map((x) => x[m] ?? 0));
          val = maxVal > 0 ? 1 - (val / maxVal) : 0;
        }
        row[model.model] = +val.toFixed(4);
      });
      return row;
    });
  }, [unifiedSorted]);

  // Model family distribution
  const familyDist = useMemo(() => {
    if (!unifiedSorted.length) return [];
    const families: Record<string, number> = {};
    unifiedSorted.forEach((m) => {
      const name = m.model.toLowerCase();
      let family = "Other";
      if (name.includes("lstm") || name.includes("gru")) family = "RNN/LSTM";
      else if (name.includes("transformer") || name.includes("batterygpt") || name.includes("tft")) family = "Transformer";
      else if (name.includes("ensemble") || name.includes("stacking")) family = "Ensemble";
      else if (name.includes("forest") || name.includes("xgboost") || name.includes("lightgbm") || name.includes("svr") || name.includes("knn") || name.includes("ridge") || name.includes("lasso") || name.includes("elastic")) family = "Classical ML";
      else if (name.includes("vae")) family = "VAE";
      families[family] = (families[family] || 0) + 1;
    });
    return Object.entries(families).map(([name, value]) => ({ name, value }));
  }, [unifiedSorted]);

  // R2 vs MAE scatter
  const scatterData = useMemo(() => {
    if (!unifiedSorted.length) return [];
    return unifiedSorted.map((m) => ({
      name: m.model,
      r2: m.R2 ?? 0,
      mae: m.MAE ?? 0,
      rmse: m.RMSE ?? 0,
    }));
  }, [unifiedSorted]);

  const filteredFigures = useMemo(() => {
    if (!data?.v2_figures) return [];
    if (!figureSearch) return data.v2_figures;
    const q = figureSearch.toLowerCase();
    return data.v2_figures.filter((f) => f.toLowerCase().includes(q));
  }, [data, figureSearch]);

  const filteredModels = useMemo(() => {
    const base = unifiedSorted.length ? unifiedSorted : classicalV2Sorted;
    const rows = familyFilter !== "all"
      ? base.filter((r) => (r.family ?? r.model_family ?? (r.model ?? "")).toLowerCase().includes(familyFilter))
      : base;
    return [...rows].sort((a, b) => {
      const av = getMetricVal(a, sortBy);
      const bv = getMetricVal(b, sortBy);
      const dir = sortDir === "desc" ? -1 : 1;
      return sortBy === "R2" ? dir * (bv - av) : -dir * (bv - av);
    });
  }, [unifiedSorted, classicalV2Sorted, sortBy, sortDir, familyFilter]);

  const families = useMemo(() => {
    const base = unifiedSorted.length ? unifiedSorted : classicalV2Sorted;
    const set = new Set(base.map((r) => (r.family ?? r.model_family ?? (r.model ?? "other")).toLowerCase()));
    return ["all", ...Array.from(set)];
  }, [unifiedSorted, classicalV2Sorted]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-400 mx-auto mb-4" />
          <p className="text-gray-400">Loading metrics dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-800 rounded-xl p-6 text-red-400">
        <div className="flex items-center gap-2 mb-2">
          <X className="w-5 h-5" />
          <span className="font-semibold">Failed to load metrics</span>
        </div>
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  if (!data) return null;

  // Normalise possibly-null array fields (API may return null instead of [])
  const lstmResults     = Array.isArray(data.lstm_results)     ? data.lstm_results     : [];
  const ensembleResults = Array.isArray(data.ensemble_results) ? data.ensemble_results : [];
  const transformerRes  = Array.isArray(data.transformer_results) ? data.transformer_results : [];

  const ts = data.training_summary;
  const vs = data.validation_summary;
  const bs = data.battery_stats;

  return (
    <div className="space-y-6">
      {/* Section Nav */}
      <div className="flex flex-wrap gap-2 bg-gray-950 p-2 rounded-xl border border-gray-800">
        {sections.map((s) => (
          <SectionBadge
            key={s.key}
            icon={(s as any).icon}
            label={s.label}
            active={activeSection === s.key}
            onClick={() => setActiveSection(s.key)}
          />
        ))}
      </div>

      {/* Figure lightbox */}
      {selectedFigure && (
        <div className="fixed inset-0 z-100 flex items-center justify-center bg-black/80 backdrop-blur-sm" onClick={() => setSelectedFigure(null)}>
          <div className="max-w-5xl max-h-[90vh] overflow-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-300">{selectedFigure}</span>
              <button onClick={() => setSelectedFigure(null)} className="text-gray-400 hover:text-white">
                <X className="w-5 h-5" />
              </button>
            </div>
            <img src={`/api/v2/figures/${selectedFigure}`} alt={selectedFigure} className="rounded-lg max-w-full" />
          </div>
        </div>
      )}

      {/* ═══════ OVERVIEW ═══════ */}
      {activeSection === "overview" && unifiedSorted.length === 0 && (
        <div className="bg-yellow-900/20 border border-yellow-800 rounded-xl p-6 text-center text-yellow-300 text-sm">
          ⚠ No model metrics found. Make sure the backend is running and artifacts are present in <code className="bg-yellow-900/40 px-1 rounded">artifacts/v2/</code>.
        </div>
      )}
      {activeSection === "overview" && (
        <>
          {/* KPI Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <StatCard icon={<Trophy className="w-4 h-4" />} label="Best Model" value={ts?.best_model ?? "—"} color="text-green-400" trend="up" />
            <StatCard icon={<BarChart2 className="w-4 h-4" />} label="Best R²" value={ts?.best_r2 != null ? ts.best_r2.toFixed(4) : "—"} color="text-green-400" />
            <StatCard icon={<Target className="w-4 h-4" />} label="Best Accuracy" value={ts?.best_within_5pct != null ? `${ts.best_within_5pct.toFixed(1)}%` : "—"} subtitle="within ±5% SOH" color="text-blue-400" />
            <StatCard icon={<Cpu className="w-4 h-4" />} label="Total Models" value={ts?.total_models ?? unifiedSorted.length} color="text-purple-400" />
            <StatCard icon={<CheckCircle2 className="w-4 h-4" />} label="Passed ≥95%" value={ts?.passed_models ?? 0} subtitle={`${ts?.pass_rate_pct?.toFixed(1) ?? 0}% pass rate`} color="text-green-400" />
            <StatCard icon={<Zap className="w-4 h-4" />} label="Batteries" value={ts?.batteries ?? bs?.batteries ?? 0} color="text-cyan-400" />
            <StatCard icon={<TrendingUp className="w-4 h-4" />} label="Train Samples" value={ts?.train_samples ?? "—"} color="text-yellow-400" />
            <StatCard icon={<FlaskConical className="w-4 h-4" />} label="Test Samples" value={ts?.test_samples ?? "—"} color="text-orange-400" />
            <StatCard icon={<Activity className="w-4 h-4" />} label="Avg Accuracy" value={ts?.mean_within_5pct != null ? `${ts.mean_within_5pct.toFixed(1)}%` : "—"} subtitle="mean ±5%" color="text-gray-300" />
            <StatCard icon={<BrainCircuit className="w-4 h-4" />} label="Val Models" value={vs?.total_models_tested ?? "—"} color="text-purple-400" />
            <StatCard icon={<CheckCircle2 className="w-4 h-4" />} label="Val Passed" value={vs?.models_passed_95pct ?? "—"} color="text-green-400" />
            <StatCard icon={<Award className="w-4 h-4" />} label="Val Best" value={vs?.best_model ?? "—"} subtitle={vs?.best_within_5pct != null ? `${vs.best_within_5pct.toFixed(1)}%` : ""} color="text-blue-400" />
          </div>

          {/* R² Comparison - All Models */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Model R² Score Ranking (All Models)</h3>
            <ResponsiveContainer width="100%" height={500}>
              <BarChart data={unifiedSorted} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis type="number" domain={[-0.5, 1]} stroke="#9ca3af" />
                <YAxis dataKey="model" type="category" width={160} stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
                  formatter={(val: any) => val.toFixed(4)}
                />
                <Bar dataKey="R2" name="R² Score" radius={[0, 4, 4, 0]}>
                  {unifiedSorted.map((entry, i) => (
                    <Cell key={entry.model} fill={(entry.R2 ?? 0) >= 0.9 ? "#22c55e" : (entry.R2 ?? 0) >= 0.7 ? "#3b82f6" : (entry.R2 ?? 0) >= 0.5 ? "#f59e0b" : "#ef4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Model Family Distribution */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Model Family Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie data={familyDist} cx="50%" cy="50%" innerRadius={50} outerRadius={100} paddingAngle={5} dataKey="value" label={({ name, value }) => `${name} (${value})`}>
                    {familyDist.map((_, i) => (
                      <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Radar - Top 5 */}
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Top 5 Models Radar (Normalized)</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#374151" />
                  <PolarAngleAxis dataKey="metric" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                  <PolarRadiusAxis domain={[0, 1]} stroke="#4b5563" tick={{ fontSize: 9 }} />
                  {unifiedSorted.slice(0, 5).map((m, i) => (
                    <Radar key={m.model} name={m.model} dataKey={m.model} stroke={CHART_COLORS[i]} fill={CHART_COLORS[i]} fillOpacity={0.15} strokeWidth={2} />
                  ))}
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* ═══════ MODEL COMPARISON ═══════ */}
      {activeSection === "models" && (
        <>
          {/* Interactive controls */}
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <ArrowUpDown className="w-4 h-4 text-gray-400" />
                <span className="text-xs text-gray-400">Sort:</span>
                <div className="flex gap-1">
                  {METRIC_KEYS.map((k) => (
                    <button
                      key={k}
                      onClick={() => { if (sortBy === k) setSortDir(d => d === "asc" ? "desc" : "asc"); else setSortBy(k); }}
                      className={`px-2 py-1 rounded text-xs font-medium transition-colors ${sortBy === k ? "bg-green-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
                    >
                      {k} {sortBy === k ? (sortDir === "desc" ? "↓" : "↑") : ""}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-gray-400" />
                <span className="text-xs text-gray-400">Chart:</span>
                {(["bar", "radar", "scatter"] as const).map((v) => (
                  <button key={v} onClick={() => setChartView(v)} className={`px-2 py-1 rounded text-xs capitalize transition-colors ${chartView === v ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}>{v}</button>
                ))}
              </div>
              <button
                onClick={() => { setCompareMode(m => !m); setCompareSelected([]); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs transition-colors ${compareMode ? "bg-purple-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
              >
                <GitCompare className="w-3 h-3" /> Compare Mode
              </button>
              <span className="text-xs text-gray-500">{filteredModels.length} models</span>
            </div>
          </div>
          {/* MAE Comparison */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">MAE Comparison (Lower = Better)</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={unifiedSorted} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis type="number" stroke="#9ca3af" />
                <YAxis dataKey="model" type="category" width={160} stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} formatter={(val: any) => val.toFixed(3)} />
                <Bar dataKey="MAE" name="MAE" radius={[0, 4, 4, 0]}>
                  {unifiedSorted.map((entry, i) => (
                    <Cell key={entry.model} fill={(entry.MAE ?? 100) < 5 ? "#22c55e" : (entry.MAE ?? 100) < 10 ? "#3b82f6" : "#ef4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* RMSE vs R² Scatter */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">R² vs MAE (Ideal: Top-Left)</h3>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="mae" name="MAE" stroke="#9ca3af" label={{ value: "MAE", position: "insideBottom", offset: -5 }} />
                <YAxis dataKey="r2" name="R²" stroke="#9ca3af" domain={[-0.5, 1]} label={{ value: "R²", angle: -90, position: "insideLeft" }} />
                <ZAxis dataKey="rmse" range={[40, 400]} name="RMSE" />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
                  formatter={(val: any) => val.toFixed(3)}
                  labelFormatter={(_, payload) => payload?.[0]?.payload?.name || ""}
                />
                <Scatter data={scatterData} fill="#22c55e" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* v2 Classical Results */}
          {classicalV2Sorted.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">v2 Classical Models (Intra-Battery Split)</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={classicalV2Sorted}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="model" stroke="#9ca3af" tick={{ fontSize: 10 }} height={60} />
                    <YAxis domain={[0, 1]} stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} formatter={(val: any) => val.toFixed(4)} />
                    <Bar dataKey="r2" name="R²" fill="#22c55e" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={classicalV2Sorted}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="model" stroke="#9ca3af" tick={{ fontSize: 10 }} height={60} />
                    <YAxis domain={[80, 100]} stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} formatter={(val: any) => val.toFixed(1) + "%"} />
                    <Bar dataKey="within_5pct" name="±5% Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Full Metrics Table */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Complete Model Metrics Table</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-400 border-b border-gray-800">
                    <th className="py-2 px-3 text-left">#</th>
                    <th className="py-2 px-3 text-left">Model</th>
                    <th className="py-2 px-3 text-right">R²</th>
                    <th className="py-2 px-3 text-right">MAE</th>
                    <th className="py-2 px-3 text-right">RMSE</th>
                    <th className="py-2 px-3 text-right">MAPE</th>
                    <th className="py-2 px-3 text-right">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {unifiedSorted.map((m, i) => (
                    <tr key={m.model} className="border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors">
                      <td className="py-2 px-3 text-gray-500">{i + 1}</td>
                      <td className="py-2 px-3 font-medium text-white">{m.model}</td>
                      <td className={`py-2 px-3 text-right font-mono ${(m.R2 ?? 0) >= 0.9 ? "text-green-400" : (m.R2 ?? 0) >= 0.7 ? "text-blue-400" : "text-red-400"}`}>
                        {m.R2 != null ? m.R2.toFixed(4) : "—"}
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-gray-300">{m.MAE != null ? m.MAE.toFixed(3) : "—"}</td>
                      <td className="py-2 px-3 text-right font-mono text-gray-300">{m.RMSE != null ? m.RMSE.toFixed(3) : "—"}</td>
                      <td className="py-2 px-3 text-right font-mono text-gray-300">{m.MAPE != null ? m.MAPE.toFixed(2) : "—"}</td>
                      <td className="py-2 px-3 text-right">
                        <span className={`px-2 py-0.5 rounded-full text-xs ${(m.R2 ?? 0) >= 0.9 ? "bg-green-900/50 text-green-300" : (m.R2 ?? 0) >= 0.7 ? "bg-blue-900/50 text-blue-300" : (m.R2 ?? 0) >= 0.5 ? "bg-yellow-900/50 text-yellow-300" : "bg-red-900/50 text-red-300"}`}>
                          {(m.R2 ?? 0) >= 0.9 ? "Excellent" : (m.R2 ?? 0) >= 0.7 ? "Good" : (m.R2 ?? 0) >= 0.5 ? "Fair" : "Poor"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ═══════ VALIDATION ═══════ */}
      {activeSection === "validation" && (
        <>
          {/* Validation Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard icon={<FlaskConical className="w-4 h-4" />} label="Test Samples" value={vs?.test_samples ?? "—"} color="text-blue-400" />
            <StatCard icon={<Zap className="w-4 h-4" />} label="Test Batteries" value={vs?.test_batteries ?? "—"} color="text-cyan-400" />
            <StatCard icon={<CheckCircle2 className="w-4 h-4" />} label="Passed ≥95%" value={vs?.models_passed_95pct ?? "—"} color="text-green-400" />
            <StatCard icon={<BarChart2 className="w-4 h-4" />} label="Avg ±5% Acc" value={vs?.mean_within_5pct != null ? `${vs.mean_within_5pct.toFixed(1)}%` : "—"} color="text-yellow-400" />
          </div>

          {/* Validation Results Chart */}
          {validationSorted.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Validation: ±5% Accuracy (Target ≥95%)</h3>
              <ResponsiveContainer width="100%" height={450}>
                <BarChart data={validationSorted} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" domain={[0, 100]} stroke="#9ca3af" />
                  <YAxis dataKey="model" type="category" width={180} stroke="#9ca3af" tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} formatter={(val: any) => `${val.toFixed(1)}%`} />
                  <Bar dataKey="within_5pct" name="±5% Accuracy" radius={[0, 4, 4, 0]}>
                    {validationSorted.map((entry: any) => (
                      <Cell key={entry.model} fill={(entry.within_5pct ?? 0) >= 95 ? "#22c55e" : (entry.within_5pct ?? 0) >= 80 ? "#3b82f6" : (entry.within_5pct ?? 0) >= 50 ? "#f59e0b" : "#ef4444"} />
                    ))}
                  </Bar>
                  {/* Target line at 95% */}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Validation R² vs ±2% Accuracy */}
          {validationSorted.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
                <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Validation R²</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={validationSorted} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" domain={[-3, 1]} stroke="#9ca3af" />
                    <YAxis dataKey="model" type="category" width={160} stroke="#9ca3af" tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} formatter={(val: any) => val.toFixed(4)} />
                    <Bar dataKey="r2" name="R²" radius={[0, 4, 4, 0]}>
                      {validationSorted.map((entry: any) => (
                        <Cell key={entry.model} fill={(entry.r2 ?? 0) >= 0.9 ? "#22c55e" : (entry.r2 ?? 0) >= 0 ? "#3b82f6" : "#ef4444"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
                <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Validation ±2% Accuracy</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={validationSorted.filter((v: any) => v.within_2pct != null)} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" domain={[0, 100]} stroke="#9ca3af" />
                    <YAxis dataKey="model" type="category" width={160} stroke="#9ca3af" tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} formatter={(val: any) => `${val.toFixed(1)}%`} />
                    <Bar dataKey="within_2pct" name="±2% Accuracy" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Validation Table */}
          {validationSorted.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Validation Results Table</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-800">
                      <th className="py-2 px-3 text-left">Model</th>
                      <th className="py-2 px-3 text-right">MAE</th>
                      <th className="py-2 px-3 text-right">RMSE</th>
                      <th className="py-2 px-3 text-right">R²</th>
                      <th className="py-2 px-3 text-right">±2% Acc</th>
                      <th className="py-2 px-3 text-right">±5% Acc</th>
                      <th className="py-2 px-3 text-right">Passed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {validationSorted.map((m: any) => (
                      <tr key={m.model} className="border-b border-gray-800/50 hover:bg-gray-800/50">
                        <td className="py-2 px-3 font-medium text-white">{m.model}</td>
                        <td className="py-2 px-3 text-right font-mono text-gray-300">{m.mae?.toFixed(3) ?? "—"}</td>
                        <td className="py-2 px-3 text-right font-mono text-gray-300">{m.rmse?.toFixed(3) ?? "—"}</td>
                        <td className={`py-2 px-3 text-right font-mono ${(m.r2 ?? 0) >= 0.9 ? "text-green-400" : (m.r2 ?? 0) >= 0 ? "text-blue-400" : "text-red-400"}`}>
                          {m.r2?.toFixed(4) ?? "—"}
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-gray-300">{m.within_2pct?.toFixed(1) ?? "—"}%</td>
                        <td className="py-2 px-3 text-right font-mono text-gray-300">{m.within_5pct?.toFixed(1) ?? "—"}%</td>
                        <td className="py-2 px-3 text-right">
                          {m.passed_95 ? (
                            <span className="text-green-400">✓ Pass</span>
                          ) : (
                            <span className="text-red-400">✗ Fail</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Intra-Battery Info */}
          {data.intra_battery && Object.keys(data.intra_battery).length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Intra-Battery Evaluation</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard icon={<Target className="w-4 h-4" />} label="Target" value={data.intra_battery.target ?? "—"} color="text-blue-400" />
                <StatCard icon={<CheckCircle2 className="w-4 h-4" />} label="Passed" value={`${data.intra_battery.passed_models ?? 0} / ${data.intra_battery.total_models ?? 0}`} color="text-green-400" />
                <StatCard icon={<Trophy className="w-4 h-4" />} label="Best Model" value={data.intra_battery.best_model ?? "—"} color="text-green-400" />
                <StatCard icon={<Award className="w-4 h-4" />} label="Best ±5%" value={data.intra_battery.best_within_5pct != null ? `${data.intra_battery.best_within_5pct.toFixed(1)}%` : "—"} color="text-blue-400" />
              </div>
              {data.intra_battery.notes && (
                <p className="mt-3 text-sm text-gray-400">{data.intra_battery.notes}</p>
              )}
            </div>
          )}
        </>
      )}

      {/* ═══════ DEEP LEARNING ═══════ */}
      {activeSection === "deep" && (
        <>
          {/* LSTM Results */}
          {lstmResults.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">LSTM / RNN Models</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={lstmResults}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="model" stroke="#9ca3af" tick={{ fontSize: 10 }} />
                    <YAxis domain={[0, 1]} stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
                    <Bar dataKey="R2" name="R²" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={lstmResults}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="model" stroke="#9ca3af" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
                    <Bar dataKey="MAE" name="MAE" fill="#ec4899" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Ensemble Results */}
          {ensembleResults.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Ensemble & Advanced Models</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={ensembleResults}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="model" stroke="#9ca3af" tick={{ fontSize: 9 }} height={60} />
                    <YAxis domain={[0, 1]} stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
                    <Legend />
                    <Bar dataKey="R2" name="R²" fill="#22c55e" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={ensembleResults}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="model" stroke="#9ca3af" tick={{ fontSize: 9 }} height={60} />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
                    <Legend />
                    <Bar dataKey="MAE" name="MAE" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="RMSE" name="RMSE" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Special Model Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* VAE-LSTM */}
            {data.vae_lstm && Object.keys(data.vae_lstm).length > 0 && (
              <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
                <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">VAE-LSTM</h3>
                <div className="grid grid-cols-2 gap-3">
                  <StatCard label="R²" value={data.vae_lstm.R2?.toFixed(4) ?? "—"} color={(data.vae_lstm.R2 ?? 0) >= 0.7 ? "text-green-400" : "text-yellow-400"} />
                  <StatCard label="MAE" value={data.vae_lstm.MAE?.toFixed(3) ?? "—"} color="text-blue-400" />
                  <StatCard label="RMSE" value={data.vae_lstm.RMSE?.toFixed(3) ?? "—"} color="text-orange-400" />
                  <StatCard label="MAPE" value={data.vae_lstm.MAPE != null ? `${data.vae_lstm.MAPE.toFixed(2)}%` : "—"} color="text-purple-400" />
                </div>
              </div>
            )}

            {/* DG-iTransformer */}
            {data.dg_itransformer && Object.keys(data.dg_itransformer).length > 0 && (
              <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
                <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">DG-iTransformer</h3>
                <div className="grid grid-cols-2 gap-3">
                  <StatCard label="R²" value={data.dg_itransformer.R2?.toFixed(4) ?? "—"} color={(data.dg_itransformer.R2 ?? 0) >= 0.5 ? "text-green-400" : "text-red-400"} />
                  <StatCard label="MAE" value={data.dg_itransformer.MAE?.toFixed(3) ?? "—"} color="text-blue-400" />
                  <StatCard label="RMSE" value={data.dg_itransformer.RMSE?.toFixed(3) ?? "—"} color="text-orange-400" />
                  <StatCard label="MAPE" value={data.dg_itransformer.MAPE != null ? `${data.dg_itransformer.MAPE.toFixed(2)}%` : "—"} color="text-purple-400" />
                </div>
              </div>
            )}
          </div>

          {/* Ensemble Metrics Table */}
          {ensembleResults.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Ensemble & Deep Model Metrics</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-800">
                      <th className="py-2 px-3 text-left">Model</th>
                      <th className="py-2 px-3 text-right">R²</th>
                      <th className="py-2 px-3 text-right">MAE</th>
                      <th className="py-2 px-3 text-right">RMSE</th>
                      <th className="py-2 px-3 text-right">MAPE</th>
                      <th className="py-2 px-3 text-right">±2% Acc</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ensembleResults.map((m: any) => (
                      <tr key={m.model} className="border-b border-gray-800/50 hover:bg-gray-800/50">
                        <td className="py-2 px-3 font-medium text-white">{m.model}</td>
                        <td className={`py-2 px-3 text-right font-mono ${(m.R2 ?? 0) >= 0.8 ? "text-green-400" : "text-yellow-400"}`}>
                          {m.R2?.toFixed(4) ?? "—"}
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-gray-300">{m.MAE?.toFixed(3) ?? "—"}</td>
                        <td className="py-2 px-3 text-right font-mono text-gray-300">{m.RMSE?.toFixed(3) ?? "—"}</td>
                        <td className="py-2 px-3 text-right font-mono text-gray-300">{m.MAPE?.toFixed(2) ?? "—"}</td>
                        <td className="py-2 px-3 text-right font-mono text-gray-300">{m.tol_2pct != null ? `${(m.tol_2pct * 100).toFixed(1)}%` : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* ═══════ DATASET ═══════ */}
      {activeSection === "dataset" && bs && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard icon={<Database className="w-4 h-4" />} label="Total Samples" value={bs.total_samples?.toLocaleString() ?? "—"} color="text-blue-400" />
            <StatCard icon={<Zap className="w-4 h-4" />} label="Batteries" value={bs.batteries ?? "—"} color="text-green-400" />
            <StatCard icon={<TrendingUp className="w-4 h-4" />} label="Avg SOH" value={bs.avg_soh != null ? `${bs.avg_soh}%` : "—"} color="text-green-400" />
            <StatCard icon={<TrendingDown className="w-4 h-4" />} label="Min SOH" value={bs.min_soh != null ? `${bs.min_soh}%` : "—"} color="text-red-400" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Feature List */}
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Engineered Features ({bs.feature_columns?.length ?? 0})</h3>
              <div className="grid grid-cols-2 gap-2">
                {(bs.feature_columns || []).map((col: string) => (
                  <div key={col} className="bg-gray-800 rounded px-3 py-1.5 text-sm text-gray-300 font-mono">{col}</div>
                ))}
              </div>
            </div>

            {/* Temperature Groups */}
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Temperature Groups</h3>
              <div className="flex flex-wrap gap-3">
                {(bs.temp_groups || []).map((t: number) => (
                  <div key={t} className="bg-gray-800 rounded-lg px-4 py-3 text-center">
                    <div className="text-2xl font-bold text-blue-400">{t}°C</div>
                    <div className="text-xs text-gray-400">Group</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Degradation Distribution */}
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">Degradation State Distribution</h3>
              {bs.degradation_distribution && (
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={Object.entries(bs.degradation_distribution).map(([name, value]) => ({ name: name === "0" ? "Healthy" : name === "1" ? "Moderate" : name === "2" ? "Degraded" : `State ${name}`, value }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="name" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
                    <Bar dataKey="value" name="Samples" fill="#22c55e" radius={[4, 4, 0, 0]}>
                      {Object.entries(bs.degradation_distribution).map((_, i) => (
                        <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>

            {/* Dataset Stats */}
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-400 uppercase mb-4">SOH Range</h3>
              <div className="space-y-4">
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="flex justify-between text-sm text-gray-400 mb-2">
                    <span>Min: {bs.min_soh}%</span>
                    <span>Avg: {bs.avg_soh}%</span>
                    <span>Max: {bs.max_soh}%</span>
                  </div>
                  <div className="h-4 bg-gray-700 rounded-full overflow-hidden relative">
                    <div
                      className="h-full bg-linear-to-r from-red-500 via-yellow-500 to-green-500 rounded-full"
                      style={{ width: `${bs.max_soh}%` }}
                    />
                    <div
                      className="absolute top-0 h-full w-0.5 bg-white"
                      style={{ left: `${bs.avg_soh}%` }}
                    />
                  </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Average RUL</div>
                  <div className="text-2xl font-bold text-blue-400">{bs.avg_rul} cycles</div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ═══════ FIGURES ═══════ */}
      {activeSection === "figures" && (
        <>
          <div className="flex items-center gap-3 bg-gray-900 p-3 rounded-xl border border-gray-800 mb-4">
            <Search className="w-4 h-4 text-gray-400 flex-shrink-0" />
            <input
              type="text"
              value={figureSearch}
              onChange={(e) => setFigureSearch(e.target.value)}
              placeholder="Search figures…"
              className="bg-transparent text-sm text-white placeholder-gray-500 flex-1 outline-none"
            />
            <span className="text-xs text-gray-500">{filteredFigures.length} figures</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredFigures.map((fig) => (
              <div
                key={fig}
                className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden cursor-pointer hover:border-gray-600 transition-colors group"
                onClick={() => setSelectedFigure(fig)}
              >
                <div className="aspect-4/3 bg-gray-800 overflow-hidden">
                  <img
                    src={`/api/v2/figures/${fig}`}
                    alt={fig}
                    className="w-full h-full object-contain group-hover:scale-105 transition-transform"
                    loading="lazy"
                  />
                </div>
                <div className="p-3">
                  <div className="text-sm text-gray-300 truncate">{fig.replace(/_/g, " ").replace(".png", "")}</div>
                </div>
              </div>
            ))}
          </div>

          {filteredFigures.length === 0 && (
            <div className="text-center py-12 text-gray-500">
              <ImageIcon className="w-8 h-8 mx-auto mb-2 opacity-40" />
              <p className="text-sm">No figures found</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
