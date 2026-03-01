import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, CartesianGrid,
} from "recharts";
import { fetchDashboard, DashboardData } from "../api";

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDashboard()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading)
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-400" />
      </div>
    );

  if (error)
    return (
      <div className="bg-red-900/30 border border-red-500 rounded-lg p-4 text-red-300">
        Error loading dashboard: {error}
      </div>
    );

  if (!data) return null;

  // Prepare capacity fade chart data (first 6 batteries)
  const fadeEntries = Object.entries(data.capacity_fade).slice(0, 6);
  const maxLen = Math.max(...fadeEntries.map(([, v]) => v.length));
  const fadeData = Array.from({ length: maxLen }, (_, i) => {
    const row: Record<string, number> = { cycle: i + 1 };
    fadeEntries.forEach(([bid, caps]) => {
      if (i < caps.length) row[bid] = +(caps[i] / 2 * 100).toFixed(1);
    });
    return row;
  });

  // Model metrics
  const metricsList = Object.entries(data.model_metrics)
    .map(([name, m]) => ({ name, r2: m.R2 ?? m.r2 ?? 0, mae: m.MAE ?? m.mae ?? 0 }))
    .sort((a, b) => b.r2 - a.r2)
    .slice(0, 10);

  const COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard label="Batteries" value={data.batteries.length} />
        <StatCard label="Models Trained" value={Object.keys(data.model_metrics).length} />
        <StatCard label="Best Model" value={data.best_model} />
        <StatCard
          label="Best R²"
          value={
            data.model_metrics[data.best_model]
              ? (data.model_metrics[data.best_model].R2 ?? data.model_metrics[data.best_model].r2 ?? 0).toFixed(4)
              : "—"
          }
        />
      </div>

      {/* Battery Grid */}
      <section>
        <h2 className="text-lg font-semibold mb-3">Battery Fleet Overview</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2">
          {data.batteries.map((b) => (
            <div
              key={b.battery_id}
              className="rounded-lg p-3 border border-gray-800 bg-gray-900 text-center"
              style={{ borderLeftColor: b.color_hex, borderLeftWidth: "4px" }}
            >
              <div className="text-xs text-gray-400">{b.battery_id}</div>
              <div className="text-xl font-bold" style={{ color: b.color_hex }}>
                {b.soh_pct}%
              </div>
              <div className="text-xs text-gray-500">{b.degradation_state}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Capacity Fade Chart */}
      <section className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h2 className="text-lg font-semibold mb-4">SOH Capacity Fade</h2>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={fadeData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="cycle" stroke="#9ca3af" />
            <YAxis domain={[50, 100]} stroke="#9ca3af" />
            <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
            <Legend />
            {fadeEntries.map(([bid], i) => (
              <Line
                key={bid}
                type="monotone"
                dataKey={bid}
                stroke={COLORS[i % COLORS.length]}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </section>

      {/* Model Comparison */}
      <section className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h2 className="text-lg font-semibold mb-4">Model R² Comparison</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={metricsList} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis type="number" domain={[0, 1]} stroke="#9ca3af" />
            <YAxis dataKey="name" type="category" width={150} stroke="#9ca3af" tick={{ fontSize: 12 }} />
            <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }} />
            <Bar dataKey="r2" fill="#22c55e" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </section>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
      <div className="text-sm text-gray-400">{label}</div>
      <div className="text-2xl font-bold text-green-400 mt-1 truncate">{value}</div>
    </div>
  );
}
