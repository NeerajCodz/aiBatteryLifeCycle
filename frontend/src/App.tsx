import { useState } from "react";
import SimulationPanel from "./components/SimulationPanel";
import PredictionForm from "./components/PredictionForm";
import GraphPanel from "./components/GraphPanel";
import RecommendationPanel from "./components/RecommendationPanel";
import MetricsPanel from "./components/MetricsPanel";
import ResearchPaper from "./components/ResearchPaper";
import { ToastProvider } from "./components/Toast";
import { getApiVersion, setApiVersion } from "./api";
import { BatteryCharging } from "lucide-react";

type Tab = "simulation" | "predict" | "graphs" | "recommend" | "metrics" | "paper";

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("simulation");
  const [apiVersion, setVersion] = useState<"v1" | "v2">(getApiVersion());

  const handleVersionChange = (v: "v1" | "v2") => {
    setApiVersion(v);
    setVersion(v);
  };

  const tabs: { key: Tab; label: string }[] = [
    { key: "simulation", label: "Simulation" },
    { key: "predict", label: "Predict" },
    { key: "graphs", label: "Analytics" },
    { key: "recommend", label: "Recommendations" },
    { key: "metrics", label: "Metrics" },
    { key: "paper", label: "Research Paper" },
  ];

  return (
    <ToastProvider>
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-green-500/20 flex items-center justify-center">
              <BatteryCharging className="w-5 h-5 text-green-400" />
            </div>
            <h1 className="text-lg font-bold">AI Battery Lifecycle Predictor</h1>
          </div>
          <div className="flex items-center gap-4">
            {/* Version toggle */}
            <div className="flex items-center gap-2 bg-gray-800 rounded-lg p-1">
              {(["v1", "v2"] as const).map((v) => (
                <button
                  key={v}
                  onClick={() => handleVersionChange(v)}
                  className={`px-3 py-1 rounded text-xs font-bold transition-colors ${
                    apiVersion === v
                      ? v === "v2"
                        ? "bg-green-600 text-white"
                        : "bg-blue-600 text-white"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  {v.toUpperCase()}
                </button>
              ))}
            </div>
            <nav className="flex gap-1">
            {tabs.map((t) => (
              <button
                key={t.key}
                onClick={() => setActiveTab(t.key)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === t.key
                    ? "bg-green-600 text-white"
                    : "text-gray-400 hover:text-white hover:bg-gray-800"
                }`}
              >
                {t.label}
              </button>
            ))}
            </nav>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === "simulation" && <SimulationPanel />}
        {activeTab === "predict" && <PredictionForm />}
        {activeTab === "graphs" && <GraphPanel />}
        {activeTab === "recommend" && <RecommendationPanel />}
        {activeTab === "metrics" && <MetricsPanel />}
        {activeTab === "paper" && <ResearchPaper />}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-4 mt-8">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
          NASA PCoE Li-ion Battery Dataset &middot; IEEE Research-Grade Analysis &middot;{" "}
          <a href="/gradio" className="text-green-400 hover:underline">
            Gradio UI
          </a>{" "}
          &middot;{" "}
          <a href="/docs" className="text-green-400 hover:underline">
            API Docs
          </a>
        </div>
      </footer>
    </div>
    </ToastProvider>
  );
}
