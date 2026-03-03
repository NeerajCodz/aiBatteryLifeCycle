import axios from "axios";

/** Active API version — toggle between v1 (legacy) and v2 (latest). */
let _apiVersion: "v1" | "v2" = "v2";

export const getApiVersion = () => _apiVersion;
export const setApiVersion = (v: "v1" | "v2") => {
  _apiVersion = v;
};

/** Base axios instance — prefix depends on active version. */
const baseApi = axios.create({ baseURL: "/api" });

/** Version-aware axios instance. */
const api = () => axios.create({ baseURL: `/api/${_apiVersion}` });

// ── Types ───────────────────────────────────────────────────────────────────
export interface PredictRequest {
  battery_id: string;
  cycle_number: number;
  ambient_temperature: number;
  peak_voltage: number;
  min_voltage: number;
  avg_current: number;
  avg_temp: number;
  temp_rise: number;
  cycle_duration: number;
  Re: number;
  Rct: number;
  delta_capacity: number;
  /** Optional model key, e.g. 'random_forest', 'tft', 'best_ensemble' */
  model_name?: string | null;
}

export interface PredictResponse {
  battery_id: string;
  cycle_number: number;
  soh_pct: number;
  rul_cycles: number | null;
  degradation_state: string;
  confidence_lower: number | null;
  confidence_upper: number | null;
  model_used: string;
  model_version: string | null;
}

export interface ModelInfo {
  name: string;
  version: string | null;
  display_name: string | null;
  family: string;
  algorithm: string | null;
  target: string;
  r2: number | null;
  metrics: Record<string, number>;
  is_default: boolean;
  loaded: boolean;
  load_error: string | null;
}

export interface ModelVersionGroups {
  v1_classical: ModelInfo[];
  v2_deep: ModelInfo[];
  v2_ensemble: ModelInfo[];
  other: ModelInfo[];
  default_model: string | null;
}

// ── Version management ───────────────────────────────────────────────────────
export interface VersionInfo {
  id: string;            // "v1" | "v2"
  display: string;       // "Version 1" | "Version 2"
  loaded: boolean;
  model_count: number;
  status: "ready" | "not_downloaded" | "downloading" | "error";
}

export const fetchVersions = () =>
  baseApi.get<VersionInfo[]>("/versions").then((r) => r.data);

export const loadVersion = (version: string) =>
  baseApi.post<{ status: string; version: string }>(`/versions/${version}/load`).then((r) => r.data);

export interface BatteryVizData {
  battery_id: string;
  soh_pct: number;
  temperature: number;
  cycle_number: number;
  degradation_state: string;
  color_hex: string;
}

export interface DashboardData {
  batteries: BatteryVizData[];
  capacity_fade: Record<string, number[]>;
  model_metrics: Record<string, Record<string, number>>;
  best_model: string;
}

export interface BatteryCapacity {
  battery_id: string;
  cycles: number[];
  capacity_ah: number[];
  soh_pct: number[];
}

export interface RecommendationResponse {
  battery_id: string;
  current_soh: number;
  recommendations: {
    rank: number;
    ambient_temperature: number;
    discharge_current: number;
    cutoff_voltage: number;
    predicted_rul: number;
    rul_improvement: number;
    rul_improvement_pct: number;
    explanation: string;
  }[];
}

// ── API calls ───────────────────────────────────────────────────────────────
export const fetchDashboard = () =>
  baseApi.get<DashboardData>("/dashboard").then((r) => r.data);

export const fetchBatteries = () =>
  baseApi.get<any[]>("/batteries").then((r) => r.data);

export const fetchBatteryCapacity = (id: string) =>
  baseApi.get<BatteryCapacity>(`/battery/${id}/capacity`).then((r) => r.data);

export const predictSoh = (req: PredictRequest) =>
  api().post<PredictResponse>("/predict", req).then((r) => r.data);

/** Best-ensemble prediction (RF + XGB + LGB weighted average, R²=0.957). */
export const predictEnsemble = (req: PredictRequest) =>
  api().post<PredictResponse>("/predict/ensemble", req).then((r) => r.data);

export const fetchRecommendations = (body: any) =>
  api().post<RecommendationResponse>("/recommend", body).then((r) => r.data);

/** All models with version, metrics, load status. */
export const fetchModels = () =>
  api().get<ModelInfo[]>("/models").then((r) => r.data);

/** Models grouped by semantic version family (v1/v2). */
export const fetchModelVersions = () =>
  baseApi.get<ModelVersionGroups>("/models/versions").then((r) => r.data);

export const fetchFigures = () =>
  baseApi.get<string[]>("/figures").then((r) => r.data);

/** Comprehensive model metrics from v2 artifacts. */
export const fetchMetrics = () =>
  baseApi.get<any>("/metrics").then((r) => r.data);

// ── Simulation types ────────────────────────────────────────────────────────
export interface BatterySimConfig {
  battery_id: string;
  label?: string;
  initial_soh?: number;
  start_cycle?: number;
  ambient_temperature?: number;
  peak_voltage?: number;
  min_voltage?: number;
  avg_current?: number;
  avg_temp?: number;
  temp_rise?: number;
  cycle_duration?: number;
  Re?: number;
  Rct?: number;
  delta_capacity?: number;
}

export interface SimulateRequest {
  batteries: BatterySimConfig[];
  steps: number;
  time_unit: string;
  eol_threshold?: number;
  model_name?: string | null;
  use_ml?: boolean;
}

export interface BatterySimResult {
  battery_id: string;
  label: string | null;
  soh_history: number[];
  rul_history: number[];
  rul_time_history: number[];
  re_history: number[];
  rct_history: number[];
  cycle_history: number[];
  time_history: number[];
  degradation_history: string[];
  color_history: string[];
  eol_cycle: number | null;
  eol_time: number | null;
  final_soh: number;
  final_rul: number;
  deg_rate_avg: number;
  model_used?: string;
}

export interface SimulateResponse {
  results: BatterySimResult[];
  time_unit: string;
  time_unit_label: string;
  steps: number;
  model_used?: string;
}

export const simulateBatteries = (req: SimulateRequest) =>
  axios.post<SimulateResponse>("/api/v2/simulate", req).then((r) => r.data);
