import { useState, useEffect } from "react";
import { predictSoh, fetchModels, PredictRequest, PredictResponse, ModelInfo } from "../api";

const DEFAULTS: PredictRequest = {
  battery_id: "B0005",
  cycle_number: 100,
  ambient_temperature: 24,
  peak_voltage: 4.2,
  min_voltage: 2.7,
  avg_current: 2.0,
  avg_temp: 25.0,
  temp_rise: 3.0,
  cycle_duration: 3600,
  Re: 0.04,
  Rct: 0.02,
  delta_capacity: -0.005,
  model_name: null,
};

/** Colour coding for model family badges */
const familyColour = (family: string) => {
  switch (family) {
    case "classical":    return "bg-blue-900/50 text-blue-300";
    case "deep_pytorch": return "bg-purple-900/50 text-purple-300";
    case "deep_keras":   return "bg-pink-900/50 text-pink-300";
    case "ensemble":     return "bg-green-900/50 text-green-300";
    default:             return "bg-gray-700 text-gray-300";
  }
};

export default function PredictionForm() {
  const [form, setForm] = useState<PredictRequest>(DEFAULTS);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load available models on mount
  useEffect(() => {
    fetchModels()
      .then((ms) => {
        // Prefer loaded models; sort by r2 desc
        const sorted = [...ms].sort((a, b) => {
          if (a.loaded !== b.loaded) return a.loaded ? -1 : 1;
          return (b.r2 ?? 0) - (a.r2 ?? 0);
        });
        setModels(sorted);
      })
      .catch(() => {/* silently ignore — model list is optional */});
  }, []);

  const handleChange = (key: keyof PredictRequest, val: string) => {
    setForm((prev) => ({
      ...prev,
      [key]: key === "battery_id" || key === "model_name" ? val || null : parseFloat(val) || 0,
    }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await predictSoh(form);
      setResult(res);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const stateColor = (state: string) => {
    switch (state) {
      case "Healthy":    return "text-green-400";
      case "Moderate":   return "text-yellow-400";
      case "Degraded":   return "text-orange-400";
      default:           return "text-red-400";
    }
  };

  const numericFields: { key: keyof PredictRequest; label: string; step?: string }[] = [
    { key: "cycle_number",        label: "Cycle Number",       step: "1" },
    { key: "ambient_temperature", label: "Ambient Temp (°C)", step: "0.1" },
    { key: "peak_voltage",        label: "Peak Voltage (V)",  step: "0.01" },
    { key: "min_voltage",         label: "Min Voltage (V)",   step: "0.01" },
    { key: "avg_current",         label: "Avg Current (A)",   step: "0.1" },
    { key: "avg_temp",            label: "Avg Cell Temp (°C)", step: "0.1" },
    { key: "temp_rise",           label: "Temp Rise (°C)",    step: "0.1" },
    { key: "cycle_duration",      label: "Duration (s)",       step: "1" },
    { key: "Re",                  label: "Re (Ω)",             step: "0.001" },
    { key: "Rct",                 label: "Rct (Ω)",            step: "0.001" },
    { key: "delta_capacity",      label: "ΔCapacity (Ah)",    step: "0.001" },
  ];

  // Find info for the currently selected model
  const selectedModel = models.find(
    (m) => m.name === (form.model_name ?? models.find((x) => x.is_default)?.name)
  ) ?? models.find((m) => m.is_default);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Form */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 flex flex-col gap-4">
        <h2 className="text-lg font-semibold">SOH Prediction</h2>

        {/* Battery ID */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">Battery ID</label>
          <input
            type="text"
            value={form.battery_id}
            onChange={(e) => handleChange("battery_id", e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-green-500"
          />
        </div>

        {/* Model selector */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">
            Model&nbsp;
            {selectedModel && (
              <span className={`text-xs px-2 py-0.5 rounded-full ml-1 ${familyColour(selectedModel.family)}`}>
                {selectedModel.family}
              </span>
            )}
            {selectedModel?.version && (
              <span className="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full ml-1">
                v{selectedModel.version}
              </span>
            )}
          </label>
          <select
            value={form.model_name ?? ""}
            onChange={(e) => setForm((p) => ({ ...p, model_name: e.target.value || null }))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-green-500"
          >
            <option value="">— Registry Default —</option>
            {models.map((m) => (
              <option key={m.name} value={m.name} disabled={!m.loaded}>
                {m.display_name ?? m.name}
                {m.r2 != null ? ` (R²=${m.r2.toFixed(3)})` : ""}
                {!m.loaded ? " [unavailable]" : ""}
              </option>
            ))}
          </select>
          {selectedModel && (
            <p className="mt-1 text-xs text-gray-500 leading-snug">
              {selectedModel.algorithm}
              {selectedModel.r2 != null && ` · R²=${selectedModel.r2.toFixed(3)}`}
            </p>
          )}
        </div>

        {/* Numeric feature inputs */}
        <div className="grid grid-cols-2 gap-3">
          {numericFields.map((f) => (
            <div key={f.key}>
              <label className="block text-xs text-gray-400 mb-1">{f.label}</label>
              <input
                type="number"
                step={f.step}
                value={form[f.key] as number}
                onChange={(e) => handleChange(f.key, e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
          ))}
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="mt-2 w-full bg-green-600 hover:bg-green-500 text-white font-medium py-2.5 rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? "Predicting…" : "Predict SOH"}
        </button>

        {error && (
          <div className="text-sm text-red-400 bg-red-900/30 p-3 rounded-lg">{error}</div>
        )}
      </div>

      {/* Result panel */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h2 className="text-lg font-semibold mb-4">Result</h2>
        {result ? (
          <div className="space-y-4">
            {/* SOH gauge */}
            <div className="relative w-48 h-48 mx-auto">
              <svg viewBox="0 0 120 120" className="w-full h-full">
                <circle cx="60" cy="60" r="52" fill="none" stroke="#374151" strokeWidth="12" />
                <circle
                  cx="60" cy="60" r="52"
                  fill="none"
                  stroke={
                    result.soh_pct >= 90 ? "#22c55e" :
                    result.soh_pct >= 80 ? "#eab308" :
                    result.soh_pct >= 70 ? "#f97316" : "#ef4444"
                  }
                  strokeWidth="12"
                  strokeDasharray={`${(result.soh_pct / 100) * 327} 327`}
                  strokeLinecap="round"
                  transform="rotate(-90 60 60)"
                />
                <text x="60" y="55" textAnchor="middle" className="fill-white text-2xl font-bold" fontSize="22">
                  {result.soh_pct.toFixed(1)}%
                </text>
                <text x="60" y="72" textAnchor="middle" className="fill-gray-400" fontSize="10">
                  SOH
                </text>
              </svg>
            </div>

            {/* Details grid */}
            <div className="grid grid-cols-2 gap-3">
              <InfoBox label="Battery"  value={result.battery_id} />
              <InfoBox label="Cycle"    value={`#${result.cycle_number}`} />
              <InfoBox label="State"    value={result.degradation_state} className={stateColor(result.degradation_state)} />
              <InfoBox label="RUL"      value={result.rul_cycles ? `${result.rul_cycles} cycles` : "—"} />
              <InfoBox
                label="95% CI"
                value={
                  result.confidence_lower != null && result.confidence_upper != null
                    ? `[${result.confidence_lower.toFixed(1)}, ${result.confidence_upper.toFixed(1)}]`
                    : "—"
                }
              />
              <InfoBox label="Model"   value={result.model_used} />
            </div>

            {/* Version badge */}
            {result.model_version && (
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <span className="bg-gray-700 rounded-full px-2 py-0.5">v{result.model_version}</span>
                <span>model version</span>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-64 text-gray-500">
            Enter parameters and click Predict
          </div>
        )}
      </div>
    </div>
  );
}

function InfoBox({
  label,
  value,
  className = "text-white",
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <div className="text-xs text-gray-400">{label}</div>
      <div className={`text-sm font-medium mt-0.5 truncate ${className}`}>{value}</div>
    </div>
  );
}
