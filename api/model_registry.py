"""
api.model_registry
==================
Singleton model registry providing unified loading, versioning, and inference
for all trained battery lifecycle models.

Model versioning
----------------
* v1.x  — Classical (tree-based / linear) models trained in NB03.
* v2.x  — Deep sequence models trained in NB04 – NB07.
* v3.x  — Ensemble / meta-models trained in NB08.

Usage
-----
    from api.model_registry import registry
    registry.load_all()           # FastAPI lifespan startup
    result = registry.predict(
        features={"cycle_number": 150, ...},
        model_name="best_ensemble",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Architecture constants (must match NB04 – NB07 training) ─────────────────
_N_FEAT: int = 12       # len(FEATURE_COLS_SCALAR)
_SEQ_LEN: int = 32      # WINDOW_SIZE
_HIDDEN: int = 128      # LSTM_HIDDEN
_LSTM_LAYERS: int = 2   # LSTM_LAYERS
_ATTN_LAYERS: int = 3   # AttentionLSTM trained with n_layers=3
_D_MODEL: int = 64      # TRANSFORMER_D_MODEL
_N_HEADS: int = 4       # TRANSFORMER_NHEAD
_TF_LAYERS: int = 2     # TRANSFORMER_NLAYERS
_DROPOUT: float = 0.2   # DROPOUT

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_MODELS_DIR = _PROJECT / "artifacts" / "models"
_ARTIFACTS = _PROJECT / "artifacts"


def _versioned_paths(version: str = "v1") -> dict[str, Path]:
    """Return artifact paths for a specific model version (v1 or v2)."""
    root = _PROJECT / "artifacts" / version
    return {
        "models_dir": root / "models",
        "artifacts":  root,
        "scalers":    root / "scalers",
        "results":    root / "results",
    }

FEATURE_COLS_SCALAR: list[str] = [
    "cycle_number", "ambient_temperature",
    "peak_voltage", "min_voltage", "voltage_range",
    "avg_current", "avg_temp", "temp_rise",
    "cycle_duration", "Re", "Rct", "delta_capacity",
]

# Columns present in the features CSV that are NOT model inputs
# (targets, identifiers, or derived columns excluded from training)
_NON_FEATURE_COLS: frozenset[str] = frozenset({
    "battery_id", "Capacity", "datetime", "SoH", "RUL",
    "degradation_state", "soh_rolling_mean",
})

# ── Model catalog (single source of truth for versions & metadata) ────────────
MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "random_forest":          {"version": "3.0.0", "display_name": "Random Forest",                  "family": "classical",    "algorithm": "RandomForestRegressor",       "target": "soh", "r2": 0.9814},
    "xgboost":                {"version": "3.0.0", "display_name": "XGBoost",                        "family": "classical",    "algorithm": "XGBRegressor",                "target": "soh", "r2": 0.9866},
    "lightgbm":               {"version": "3.0.0", "display_name": "LightGBM",                       "family": "classical",    "algorithm": "LGBMRegressor",               "target": "soh", "r2": 0.9826},
    "ridge":                  {"version": "1.0.0", "display_name": "Ridge Regression",               "family": "classical",    "algorithm": "Ridge",                      "target": "soh", "r2": 0.72},
    "svr":                    {"version": "1.0.0", "display_name": "SVR (RBF)",                      "family": "classical",    "algorithm": "SVR",                        "target": "soh", "r2": 0.805},
    "lasso":                  {"version": "1.0.0", "display_name": "Lasso",                          "family": "classical",    "algorithm": "Lasso",                      "target": "soh", "r2": 0.52},
    "elasticnet":             {"version": "1.0.0", "display_name": "ElasticNet",                     "family": "classical",    "algorithm": "ElasticNet",                 "target": "soh", "r2": 0.52},
    "knn_k5":                 {"version": "1.0.0", "display_name": "KNN (k=5)",                     "family": "classical",    "algorithm": "KNeighborsRegressor",        "target": "soh", "r2": 0.72},
    "knn_k10":                {"version": "1.0.0", "display_name": "KNN (k=10)",                    "family": "classical",    "algorithm": "KNeighborsRegressor",        "target": "soh", "r2": 0.724},
    "knn_k20":                {"version": "1.0.0", "display_name": "KNN (k=20)",                    "family": "classical",    "algorithm": "KNeighborsRegressor",        "target": "soh", "r2": 0.717},
    "extra_trees":            {"version": "3.0.0", "display_name": "ExtraTrees",                     "family": "classical",    "algorithm": "ExtraTreesRegressor",        "target": "soh", "r2": 0.9701},
    "gradient_boosting":      {"version": "3.0.0", "display_name": "GradientBoosting",               "family": "classical",    "algorithm": "GradientBoostingRegressor",  "target": "soh", "r2": 0.9860},
    "vanilla_lstm":           {"version": "2.0.0", "display_name": "Vanilla LSTM",                   "family": "deep_pytorch", "algorithm": "VanillaLSTM",                "target": "soh", "r2": 0.507},
    "bidirectional_lstm":     {"version": "2.0.0", "display_name": "Bidirectional LSTM",             "family": "deep_pytorch", "algorithm": "BidirectionalLSTM",          "target": "soh", "r2": 0.520},
    "gru":                    {"version": "2.0.0", "display_name": "GRU",                            "family": "deep_pytorch", "algorithm": "GRUModel",                   "target": "soh", "r2": 0.510},
    "attention_lstm":         {"version": "2.0.0", "display_name": "Attention LSTM",                 "family": "deep_pytorch", "algorithm": "AttentionLSTM",              "target": "soh", "r2": 0.540},
    "batterygpt":             {"version": "2.1.0", "display_name": "BatteryGPT",                     "family": "deep_pytorch", "algorithm": "BatteryGPT",                "target": "soh", "r2": 0.881},
    "tft":                    {"version": "2.2.0", "display_name": "Temporal Fusion Transformer",    "family": "deep_pytorch", "algorithm": "TemporalFusionTransformer",    "target": "soh", "r2": 0.881},
    "vae_lstm":               {"version": "2.3.0", "display_name": "VAE-LSTM",                       "family": "deep_pytorch", "algorithm": "VAE_LSTM",               "target": "soh", "r2": 0.730},
    "itransformer":           {"version": "2.4.0", "display_name": "iTransformer",                   "family": "deep_keras",   "algorithm": "iTransformer",               "target": "soh", "r2": 0.595},
    "physics_itransformer":   {"version": "2.4.1", "display_name": "Physics iTransformer",           "family": "deep_keras",   "algorithm": "PhysicsITransformer",        "target": "soh", "r2": 0.600},
    "dynamic_graph_itransformer": {"version": "2.5.0", "display_name": "DG-iTransformer",           "family": "deep_keras",   "algorithm": "DynamicGraphITransformer",   "target": "soh", "r2": 0.595},
    "best_ensemble":          {"version": "3.0.0", "display_name": "Best Ensemble (RF+XGB+LGB)",     "family": "ensemble",     "algorithm": "WeightedAverage",            "target": "soh", "r2": 0.9810},
}

# R²-proportional weights for BestEnsemble (v3 values)
_ENSEMBLE_WEIGHTS: dict[str, float] = {
    "random_forest":     0.9814,
    "xgboost":           0.9866,
    "lightgbm":          0.9826,
    "extra_trees":       0.9701,
    "gradient_boosting": 0.9860,
}


# ── Degradation state ───────────────────────────────────────────────────────
def classify_degradation(soh: float) -> str:
    if soh >= 90:
        return "Healthy"
    elif soh >= 80:
        return "Moderate"
    elif soh >= 70:
        return "Degraded"
    else:
        return "End-of-Life"


def soh_to_color(soh: float) -> str:
    """Map SOH percentage to a hex colour (green→yellow→red)."""
    if soh >= 90:
        return "#22c55e"   # green
    elif soh >= 80:
        return "#eab308"   # yellow
    elif soh >= 70:
        return "#f97316"   # orange
    else:
        return "#ef4444"   # red


# ── Registry ─────────────────────────────────────────────────────────────────
class ModelRegistry:
    """Thread-safe singleton that owns all model objects and inference logic.

    Attributes
    ----------
    models:
        Mapping from name to loaded model object (sklearn/XGBoost/LightGBM
        or PyTorch ``nn.Module`` or Keras model).
    default_model:
        Name of the best available model (set by :meth:`_choose_default`).
    device:
        PyTorch device string — ``"cuda"`` when a GPU is available, else ``"cpu"``.
    """

    # Model families that need the linear StandardScaler at inference
    _LINEAR_FAMILIES = {"ridge", "lasso", "elasticnet", "svr",
                        "knn_k5", "knn_k10", "knn_k20"}
    # Tree families that are scale-invariant (no scaler needed)
    _TREE_FAMILIES = {"random_forest", "xgboost", "lightgbm", "best_ensemble",
                      "extra_trees", "gradient_boosting"}

    def __init__(self, version: str = "v1"):
        self.models: dict[str, Any] = {}
        self.model_meta: dict[str, dict] = {}
        self.default_model: str | None = None
        self.scaler = None          # kept for backward compat
        self.linear_scaler = None   # StandardScaler for Ridge/Lasso/SVR/KNN
        self.sequence_scaler = None # StandardScaler for sequence deep models
        self.feature_cols: list[str] = list(FEATURE_COLS_SCALAR)  # updated by load_all
        self.device = "cpu"
        self.version = version
        # Set version-aware paths
        vp = _versioned_paths(version)
        self._models_dir = vp["models_dir"]
        self._artifacts = vp["artifacts"]
        self._scalers_dir = vp["scalers"]

    # ── Loading ──────────────────────────────────────────────────────────
    def load_all(self) -> None:
        """Scan artifacts/models and load all available model artefacts.

        Safe to call multiple times — subsequent calls are no-ops when the
        registry is already populated.
        """
        if self.models:
            log.debug("Registry already populated — skipping load_all()")
            return
        self._detect_device()
        self._load_scaler()
        self.feature_cols = self._load_feature_cols()
        self._load_classical()
        self._load_deep_pytorch()
        self._load_deep_keras()
        self._register_ensemble()
        self._choose_default()
        log.info(
            "Registry ready — %d models active, default='%s', device=%s",
            len(self.models), self.default_model, self.device,
        )

    def _detect_device(self) -> None:
        """Detect PyTorch compute device (CUDA > CPU)."""
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("PyTorch device: %s", self.device)
        except ImportError:
            log.info("torch not installed — deep PyTorch models unavailable")

    def _load_classical(self) -> None:
        """Eagerly load all sklearn/XGBoost/LightGBM joblib artefacts."""
        cdir = self._models_dir / "classical"
        if not cdir.exists():
            log.warning("Classical models dir not found: %s", cdir)
            return
        for p in sorted(cdir.glob("*.joblib")):
            name = p.stem
            # Skip non-model dumps (param search results, classifiers)
            if "best_params" in name or "classifier" in name:
                continue
            try:
                self.models[name] = joblib.load(p)
                catalog = MODEL_CATALOG.get(name, {})
                self.model_meta[name] = {
                    **catalog,
                    "family": "classical",
                    "loaded": True,
                    "path": str(p),
                }
                log.info("Loaded classical: %-22s  v%s", name, catalog.get("version", "?"))
            except Exception as exc:
                log.warning("Failed to load %s: %s", p.name, exc)

    def _detect_n_feat(self, p: Path) -> int:
        """Infer input feature dimension from a PyTorch checkpoint.

        Checks common first-layer weight keys to determine n_features,
        handling models trained with a different feature count (e.g. v3=18).
        """
        try:
            import torch
            state = torch.load(p, map_location="cpu", weights_only=True)
            # TFT-specific check MUST come first: TFT also has lstm.weight_ih_l0
            # but its LSTM takes d_model (not n_feat) as input, causing wrong detection.
            # softmax_proj.bias shape is (n_features,) — the true feature count.
            if "var_selection.softmax_proj.bias" in state:
                return int(state["var_selection.softmax_proj.bias"].shape[0])
            # LSTM / BiLSTM / GRU: weight_ih_l0 shape is (gates*hidden, n_feat)
            for key in ("lstm.weight_ih_l0", "encoder_lstm.weight_ih_l0", "gru.weight_ih_l0"):
                if key in state:
                    return int(state[key].shape[-1])
            # BatteryGPT: input_proj.weight shape is (d_model, n_feat)
            if "input_proj.weight" in state:
                return int(state["input_proj.weight"].shape[-1])
        except Exception:
            pass
        return _N_FEAT

    def _build_pytorch_model(self, name: str, n_feat: int = _N_FEAT) -> Any | None:
        """Instantiate a PyTorch module with the architecture used during training."""
        try:
            if name == "vanilla_lstm":
                from src.models.deep.lstm import VanillaLSTM
                return VanillaLSTM(n_feat, _HIDDEN, _LSTM_LAYERS, _DROPOUT)
            if name == "bidirectional_lstm":
                from src.models.deep.lstm import BidirectionalLSTM
                return BidirectionalLSTM(n_feat, _HIDDEN, _LSTM_LAYERS, _DROPOUT)
            if name == "gru":
                from src.models.deep.lstm import GRUModel
                return GRUModel(n_feat, _HIDDEN, _LSTM_LAYERS, _DROPOUT)
            if name == "attention_lstm":
                from src.models.deep.lstm import AttentionLSTM
                return AttentionLSTM(n_feat, _HIDDEN, _ATTN_LAYERS, _DROPOUT)
            if name == "batterygpt":
                from src.models.deep.transformer import BatteryGPT
                return BatteryGPT(
                    input_dim=n_feat, d_model=_D_MODEL, n_heads=_N_HEADS,
                    n_layers=_TF_LAYERS, dropout=_DROPOUT, max_len=64,
                )
            if name == "tft":
                from src.models.deep.transformer import TemporalFusionTransformer
                return TemporalFusionTransformer(
                    n_features=n_feat, d_model=_D_MODEL, n_heads=_N_HEADS,
                    n_layers=_TF_LAYERS, dropout=_DROPOUT,
                )
            if name == "vae_lstm":
                from src.models.deep.vae_lstm import VAE_LSTM
                return VAE_LSTM(
                    input_dim=n_feat, seq_len=_SEQ_LEN,
                    hidden_dim=_HIDDEN, latent_dim=16,
                    n_layers=_LSTM_LAYERS, dropout=_DROPOUT,
                )
        except Exception as exc:
            log.warning("Cannot build PyTorch model '%s': %s", name, exc)
        return None

    def _load_deep_pytorch(self) -> None:
        """Load PyTorch .pt state-dict files into reconstructed model instances."""
        ddir = self._models_dir / "deep"
        if not ddir.exists():
            return
        try:
            import torch
        except ImportError:
            log.info("torch not installed — skipping deep PyTorch model loading")
            return
        for p in sorted(ddir.glob("*.pt")):
            name = p.stem
            n_feat = self._detect_n_feat(p)
            model = self._build_pytorch_model(name, n_feat=n_feat)
            if model is None:
                self.model_meta[name] = {
                    **MODEL_CATALOG.get(name, {}),
                    "family": "deep_pytorch", "loaded": False,
                    "path": str(p), "load_error": "architecture unavailable",
                }
                continue
            try:
                state = torch.load(p, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                model.to(self.device)
                model.eval()
                self.models[name] = model
                catalog = MODEL_CATALOG.get(name, {})
                self.model_meta[name] = {
                    **catalog, "family": "deep_pytorch",
                    "loaded": True, "path": str(p), "n_feat": n_feat,
                }
                log.info("Loaded PyTorch:   %-22s  v%s", name, catalog.get("version", "?"))
            except Exception as exc:
                log.warning("Could not load PyTorch '%s': %s", name, exc)
                self.model_meta[name] = {
                    **MODEL_CATALOG.get(name, {}),
                    "family": "deep_pytorch", "loaded": False,
                    "path": str(p), "load_error": str(exc),
                }

    def _load_deep_keras(self) -> None:
        """Load TensorFlow/Keras .keras model files."""
        ddir = self._models_dir / "deep"
        if not ddir.exists():
            return
        try:
            import tensorflow as tf
        except ImportError:
            log.info("TensorFlow not installed — skipping Keras model loading")
            return
        # Import the custom Keras classes so they are registered before load
        try:
            from src.models.deep.itransformer import (
                FeatureWiseMHA,
                TokenWiseMHA,
                Conv1DFeedForward,
                DynamicGraphConv,
                PhysicsInformedLoss,
                AbsCumCurrentLayer,
            )
            _custom_objects: dict = {
                "FeatureWiseMHA": FeatureWiseMHA,
                "TokenWiseMHA": TokenWiseMHA,
                "Conv1DFeedForward": Conv1DFeedForward,
                "DynamicGraphConv": DynamicGraphConv,
                "PhysicsInformedLoss": PhysicsInformedLoss,
                "AbsCumCurrentLayer": AbsCumCurrentLayer,
            }
        except Exception as imp_err:
            log.warning("Could not import iTransformer custom classes: %s", imp_err)
            _custom_objects = {}
        for p in sorted(ddir.glob("*.keras")):
            name = p.stem
            try:
                model = tf.keras.models.load_model(str(p), custom_objects=_custom_objects, safe_mode=False)
                self.models[name] = model
                catalog = MODEL_CATALOG.get(name, {})
                self.model_meta[name] = {
                    **catalog, "family": "deep_keras",
                    "loaded": True, "path": str(p),
                }
                log.info("Loaded Keras:     %-22s  v%s", name, catalog.get("version", "?"))
            except Exception as exc:
                log.warning("Could not load Keras '%s': %s", name, exc)
                self.model_meta[name] = {
                    **MODEL_CATALOG.get(name, {}),
                    "family": "deep_keras", "loaded": False,
                    "path": str(p), "load_error": str(exc),
                }

    def _register_ensemble(self) -> None:
        """Register the BestEnsemble virtual model when components are loaded."""
        available = [m for m in _ENSEMBLE_WEIGHTS if m in self.models]
        if not available:
            log.warning("BestEnsemble: no component models loaded")
            return
        self.models["best_ensemble"] = "virtual_ensemble"
        self.model_meta["best_ensemble"] = {
            **MODEL_CATALOG["best_ensemble"],
            "components": available, "loaded": True,
        }
        log.info("BestEnsemble registered — components: %s", ", ".join(available))

    def _load_scaler(self) -> None:
        # Scaler mapping (from notebooks/03_classical_ml.ipynb):
        #   standard_scaler.joblib  — StandardScaler fitted on X_train
        #                             Used for: SVR, Ridge, Lasso, ElasticNet, KNN
        #   sequence_scaler.joblib  — StandardScaler for deep-model sequences
        #   Tree models (RF, ET, GB, XGB, LGB) were fitted on raw numpy X_train
        #                             → NO scaler applied, passed as-is
        #
        # v3 scalers use a version-prefixed naming scheme:
        #   {version}_features_standard.joblib  — StandardScaler
        #   {version}_features_minmax.joblib    — MinMaxScaler (fallback)
        scalers_dir = self._scalers_dir
        version_prefix = self.version  # e.g. "v3"
        candidate_linear = (
            f"{version_prefix}_features_standard.joblib",
            "standard_scaler.joblib",
            "linear_scaler.joblib",
        )
        for fname in candidate_linear:
            sp = scalers_dir / fname
            if sp.exists():
                try:
                    self.linear_scaler = joblib.load(sp)
                    log.info("Linear scaler loaded from %s", sp)
                    break
                except Exception as exc:
                    log.warning("Could not load %s: %s", fname, exc)
        else:
            # Try minmax as last resort (v3 fallback)
            sp_mm = scalers_dir / f"{version_prefix}_features_minmax.joblib"
            if sp_mm.exists():
                try:
                    self.linear_scaler = joblib.load(sp_mm)
                    log.info("Linear scaler (minmax fallback) loaded from %s", sp_mm)
                except Exception as exc:
                    log.warning("Could not load minmax scaler: %s", exc)
            else:
                log.warning("No linear scaler found — Ridge/Lasso/SVR/KNN will use raw features")

        sp_seq = scalers_dir / "sequence_scaler.joblib"
        if sp_seq.exists():
            try:
                self.sequence_scaler = joblib.load(sp_seq)
                log.info("Sequence scaler loaded from %s", sp_seq)
            except Exception as exc:
                log.warning("Could not load sequence_scaler.joblib: %s", exc)
        else:
            log.warning("sequence_scaler.joblib not found — deep models will use raw features")

    def _load_feature_cols(self) -> list[str]:
        """Discover feature column names from artifacts features CSV.

        Reads the features CSV for this version (if present), drops known
        non-feature columns (targets, identifiers, derived labels), and
        validates the count against the loaded scaler's ``n_features_in_``.
        Falls back to the module-level ``FEATURE_COLS_SCALAR`` list.
        """
        features_dir = self._artifacts / "features"
        for fname in ("battery_features.csv", "train_split.csv"):
            fpath = features_dir / fname
            if not fpath.exists():
                continue
            try:
                all_cols = pd.read_csv(fpath, nrows=0).columns.tolist()
                feat_cols = [c for c in all_cols if c not in _NON_FEATURE_COLS]
                n_expected = getattr(self.linear_scaler, "n_features_in_", None)
                if n_expected and len(feat_cols) != n_expected:
                    log.warning(
                        "Feature col count mismatch: CSV=%d, scaler=%d — using scaler count",
                        len(feat_cols), n_expected,
                    )
                    feat_cols = feat_cols[:n_expected]
                if feat_cols:
                    log.info(
                        "Feature columns loaded from artifacts (%d cols, source: %s)",
                        len(feat_cols), fname,
                    )
                    return feat_cols
            except Exception as exc:
                log.warning("Could not load feature cols from %s: %s", fname, exc)
        log.info("Using default FEATURE_COLS_SCALAR (%d features)", len(FEATURE_COLS_SCALAR))
        return list(FEATURE_COLS_SCALAR)

    def _choose_default(self) -> None:
        """Select the highest-quality loaded model as the registry default."""
        priority = [
            "best_ensemble",
            "extra_trees",
            "random_forest",
            "xgboost",
            "lightgbm",
            "gradient_boosting",
            "tft",
            "batterygpt",
            "attention_lstm",
            "ridge",
        ]
        for name in priority:
            if name in self.models:
                self.default_model = name
                log.info("Default model: %s", name)
                return
        if self.models:
            self.default_model = next(iter(self.models))
            log.info("Default model (fallback): %s", self.default_model)

    # ── Metrics retrieval ────────────────────────────────────────────────
    def get_metrics(self) -> dict[str, dict[str, float]]:
        """Return unified evaluation metrics from results CSV/JSON artefacts.

        CSV model name headers are normalised to lower-case underscore keys.
        Entries missing from result files fall back to the ``r2`` field in
        :data:`MODEL_CATALOG`.
        """
        _normalise = {
            "RandomForest": "random_forest", "LightGBM": "lightgbm",
            "XGBoost": "xgboost", "SVR": "svr", "Ridge": "ridge",
            "Lasso": "lasso", "ElasticNet": "elasticnet",
            "KNN-5": "knn_k5", "KNN-10": "knn_k10", "KNN-20": "knn_k20",
        }
        results: dict[str, dict[str, float]] = {}
        for csv_name in (
            "classical_soh_results.csv", "lstm_soh_results.csv",
            "transformer_soh_results.csv", "ensemble_results.csv",
            "unified_results.csv",
        ):
            path = self._artifacts / csv_name
            if not path.exists():
                # Fall back to root-level results (backward compat)
                path = _ARTIFACTS / csv_name
                if not path.exists():
                    continue
            try:
                df = pd.read_csv(path, index_col=0)
                for raw in df.index:
                    key = _normalise.get(str(raw), str(raw).lower().replace(" ", "_"))
                    results[key] = df.loc[raw].dropna().to_dict()
            except Exception as exc:
                log.warning("Could not read %s: %s", csv_name, exc)
        for json_name in ("dg_itransformer_results.json", "vae_lstm_results.json"):
            path = self._artifacts / json_name
            if not path.exists():
                path = _ARTIFACTS / json_name
                if not path.exists():
                    continue
            try:
                with open(path) as fh:
                    data = json.load(fh)
                key = json_name.replace("_results.json", "")
                results[key] = {k: float(v) for k, v in data.items()
                                if isinstance(v, (int, float))}
            except Exception as exc:
                log.warning("Could not read %s: %s", json_name, exc)
        # Fill from catalog for anything not in result files
        for name, info in MODEL_CATALOG.items():
            if name not in results and "r2" in info:
                results[name] = {"R2": info["r2"]}
        return results

    # ── Prediction helpers ────────────────────────────────────────────────
    def _build_x(self, features: dict[str, float]) -> np.ndarray:
        """Build raw (1, F) feature numpy array — NO scaling applied here.

        Uses ``self.feature_cols`` which is populated from the artifacts features
        CSV at startup, falling back to ``FEATURE_COLS_SCALAR``.  Unknown feature
        keys (e.g. v3-only engineered features not sent by the frontend) default
        to 0.0 and are zero-padded by the deep-sequence builders as needed.
        """
        return np.array([[features.get(c, 0.0) for c in self.feature_cols]])

    @staticmethod
    def _x_for_model(model: Any, x: np.ndarray) -> Any:
        """Return x in the format the model was fitted with.

        * If the model has ``feature_names_in_`` → pass a DataFrame whose
          columns match those exact names (handles LGB trained with Column_0…).
        * Otherwise → pass the raw numpy array (RF, ET trained without names).
        """
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            return x   # numpy — model was fitted without feature names
        # Build DataFrame with the same column names the model was trained with
        return pd.DataFrame(x, columns=list(names))

    def _scale_for_linear(self, x: np.ndarray) -> np.ndarray:
        """Apply StandardScaler for linear / SVR / KNN models."""
        if self.linear_scaler is not None:
            try:
                return self.linear_scaler.transform(x)
            except Exception as exc:
                log.warning("Linear scaler transform failed: %s", exc)
        return x

    def _build_sequence_array(
        self, x: np.ndarray, seq_len: int = _SEQ_LEN, n_feat: int | None = None
    ) -> np.ndarray:
        """Convert single-cycle feature row → scaled (1, seq_len, n_feat) numpy array.

        Tile the current feature vector across *seq_len* timesteps and apply
        the sequence scaler so values match the training distribution.
        If *n_feat* > x.shape[-1] the input is zero-padded to match the model.
        """
        if self.sequence_scaler is not None:
            try:
                x_sc = self.sequence_scaler.transform(x)   # (1, F)
            except Exception:
                x_sc = x
        else:
            x_sc = x
        # Pad to model's expected feature count if necessary (e.g. v3 deep = 18)
        if n_feat and n_feat > x_sc.shape[-1]:
            pad_width = n_feat - x_sc.shape[-1]
            x_sc = np.pad(x_sc, ((0, 0), (0, pad_width)))
        # Tile to (1, seq_len, F)
        return np.tile(x_sc[:, np.newaxis, :], (1, seq_len, 1)).astype(np.float32)

    def _build_sequence_tensor(
        self, x: np.ndarray, seq_len: int = _SEQ_LEN, n_feat: int | None = None
    ) -> Any:
        """Same as :meth:`_build_sequence_array` but returns a PyTorch tensor."""
        import torch
        return torch.tensor(self._build_sequence_array(x, seq_len, n_feat), dtype=torch.float32)

    def _predict_ensemble(self, x: np.ndarray) -> tuple[float, str]:
        """Weighted-average SOH prediction from BestEnsemble component models.

        Each component model receives input in the format it was trained with:
        - RF, ET, GB, XGB: raw numpy (trained on X_train.values, no feature names)
        - LGB: DataFrame with Column_0…Column_11 (LightGBM auto-assigned during training)
        Both cases handled by :meth:`_x_for_model`.
        """
        components = self.model_meta.get("best_ensemble", {}).get(
            "components", list(_ENSEMBLE_WEIGHTS.keys())
        )
        total_w, weighted_sum = 0.0, 0.0
        used: list[str] = []
        for cname in components:
            if cname not in self.models:
                continue
            w = _ENSEMBLE_WEIGHTS.get(cname, 1.0)
            xi = self._x_for_model(self.models[cname], x)
            soh = float(self.models[cname].predict(xi)[0])
            weighted_sum += w * soh
            total_w += w
            used.append(cname)
        if total_w == 0:
            raise ValueError("No BestEnsemble components available")
        return weighted_sum / total_w, f"best_ensemble({', '.join(used)})"

    # ── Prediction ────────────────────────────────────────────────────────
    def predict(
        self,
        features: dict[str, float],
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Predict SOH for a single battery cycle.

        Parameters
        ----------
        features:
            Dict of cycle features; keys from :data:`FEATURE_COLS_SCALAR`.
            Missing keys are filled with 0.0.
        model_name:
            Registry model key (e.g. ``"best_ensemble"``, ``"random_forest"``,
            ``"tft"``).  Defaults to :attr:`default_model`.

        Returns
        -------
        dict
            ``soh_pct``, ``degradation_state``, ``rul_cycles``,
            ``confidence_lower``, ``confidence_upper``,
            ``model_used``, ``model_version``.
        """
        name = model_name or self.default_model
        if name is None:
            raise ValueError("No models loaded in registry")

        x = self._build_x(features)

        # ── Dispatch by model type ──────────────────────────────────────
        if name == "best_ensemble":
            soh, label = self._predict_ensemble(x)
        elif name in self.models:
            model = self.models[name]
            family = self.model_meta.get(name, {}).get("family", "classical")
            if family == "deep_pytorch":
                try:
                    import torch
                    with torch.no_grad():
                        # Build scaled (1, seq_len, n_feat) sequence tensor
                        # n_feat may be > 12 for models trained with extra features (e.g. v3)
                        model_n_feat = self.model_meta.get(name, {}).get("n_feat", _N_FEAT)
                        t = self._build_sequence_tensor(x, n_feat=model_n_feat).to(self.device)
                        out = model(t)
                        # VAE-LSTM returns a dict; all others return a tensor
                        if isinstance(out, dict):
                            out = out["health_pred"]
                        soh = float(out.cpu().numpy().ravel()[0])
                except Exception as exc:
                    log.error("PyTorch inference error for '%s': %s", name, exc)
                    raise
            elif family == "deep_keras":
                try:
                    # Build scaled (1, seq_len, n_feat) numpy array for Keras
                    # Derive n_feat from model input shape: (None, seq_len, n_feat)
                    try:
                        keras_n_feat = int(model.input_shape[-1])
                    except Exception:
                        keras_n_feat = None
                    seq_np = self._build_sequence_array(x, n_feat=keras_n_feat)
                    out = model.predict(seq_np, verbose=0)
                    # Physics-Informed model returns a dict with multiple heads
                    if isinstance(out, dict):
                        out = out.get("soh_ml", next(iter(out.values())))
                    soh = float(np.asarray(out).ravel()[0])
                except Exception as exc:
                    log.error("Keras inference error for '%s': %s", name, exc)
                    raise
            elif name in self._LINEAR_FAMILIES:
                # Ridge/Lasso/ElasticNet/SVR/KNN need StandardScaler
                x_lin = self._scale_for_linear(x)
                soh = float(model.predict(x_lin)[0])
            else:
                # RF/XGB/LGB — scale-invariant; use per-model input format
                xi = self._x_for_model(model, x)
                soh = float(model.predict(xi)[0])
            label = name
        else:
            fallback = self.default_model
            if fallback and fallback != name and fallback in self.models:
                log.warning("Model '%s' not loaded — falling back to '%s'", name, fallback)
                return self.predict(features, fallback)
            raise ValueError(
                f"Model '{name}' is not available. "
                f"Loaded: {list(self.models.keys())}"
            )

        soh = float(np.clip(soh, 0.0, 100.0))

        # ── RUL estimate ────────────────────────────────────────────────
        # Data-driven estimate: linear degradation from current SOH to 70%
        # (EOL threshold), calibrated to NASA dataset's ~0.2-0.4 %/cycle rate.
        EOL_THRESHOLD = 70.0
        if soh > EOL_THRESHOLD:
            # Degradation rate: use delta_capacity as a proxy (Ah/cycle)
            # NASA nominal: ~2.0 Ah, so %/cycle = delta_cap / 2.0 * 100
            cap_loss_per_cycle_pct = abs(features.get("delta_capacity", -0.005)) / 2.0 * 100
            # Clamp to realistic range: 0.05 – 2.0 %/cycle
            rate = max(0.05, min(cap_loss_per_cycle_pct, 2.0))
            rul = (soh - EOL_THRESHOLD) / rate
        else:
            rul = 0.0

        version = self.model_meta.get(name, MODEL_CATALOG.get(name, {})).get("version", "?")

        return {
            "soh_pct": round(soh, 2),
            "degradation_state": classify_degradation(soh),
            "rul_cycles": round(rul, 1),
            "confidence_lower": round(soh - 2.0, 2),
            "confidence_upper": round(soh + 2.0, 2),
            "model_used": label,
            "model_version": version,
        }

    def predict_batch(
        self,
        battery_id: str,
        cycles: list[dict[str, float]],
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Predict SOH for multiple cycles of the same battery."""
        return [
            {**self.predict(c, model_name),
             "battery_id": battery_id,
             "cycle_number": c.get("cycle_number", i + 1)}
            for i, c in enumerate(cycles)
        ]

    def predict_array(
        self,
        X: np.ndarray,
        model_name: str | None = None,
    ) -> tuple[np.ndarray, str]:
        """Vectorized batch SOH prediction on an (N, F) feature matrix.

        Performs a **single** ``model.predict()`` call for the whole array,
        giving O(1) Python overhead regardless of how many rows N is.
        Used by the simulation endpoint to avoid per-step loop overhead.

        Parameters
        ----------
        X:
            Shape ``(N, len(FEATURE_COLS_SCALAR))`` — rows are ordered by
            ``FEATURE_COLS_SCALAR``, no scaling applied yet.
        model_name:
            Model key. Defaults to :attr:`default_model`.

        Returns
        -------
        tuple[np.ndarray, str]
            ``(soh_array, model_label)`` — ``soh_array`` has shape ``(N,)``,
            values clipped to ``[0, 100]``.

        Notes
        -----
        Deep sequence models (PyTorch / Keras) are not batchable here because
        they require multi-timestep tensors.  Callers that request a deep model
        will get a ``ValueError``; the simulate endpoint falls back to physics.
        """
        name = model_name or self.default_model
        if name is None:
            raise ValueError("No models loaded in registry")

        # Pad X to the model's expected feature count if needed.
        # This handles v3 models (18 features) when simulate builds X with 12.
        def _pad_X(X: np.ndarray, model: Any) -> np.ndarray:
            n_expected = getattr(model, "n_features_in_", None)
            if n_expected and X.shape[1] < n_expected:
                return np.pad(X, ((0, 0), (0, n_expected - X.shape[1])))
            return X

        if name == "best_ensemble":
            components = self.model_meta.get("best_ensemble", {}).get(
                "components", list(_ENSEMBLE_WEIGHTS.keys())
            )
            total_w: float = 0.0
            weighted_sum: np.ndarray | None = None
            used: list[str] = []
            for cname in components:
                if cname not in self.models:
                    continue
                w   = _ENSEMBLE_WEIGHTS.get(cname, 1.0)
                xi  = self._x_for_model(self.models[cname], _pad_X(X, self.models[cname]))
                preds = np.asarray(self.models[cname].predict(xi), dtype=float)
                weighted_sum = preds * w if weighted_sum is None else weighted_sum + preds * w
                total_w += w
                used.append(cname)
            if total_w == 0 or weighted_sum is None:
                raise ValueError("No BestEnsemble components available")
            return np.clip(weighted_sum / total_w, 0.0, 100.0), f"best_ensemble({', '.join(used)})"

        elif name in self.models:
            model  = self.models[name]
            family = self.model_meta.get(name, {}).get("family", "classical")
            if family in ("deep_pytorch", "deep_keras"):
                raise ValueError(
                    f"Model '{name}' is a deep sequence model and cannot be "
                    "batch-predicted. Use predict() per sample instead."
                )
            Xp = _pad_X(X, model)
            if name in self._LINEAR_FAMILIES:
                xi = self._scale_for_linear(Xp)
            else:
                xi = self._x_for_model(model, Xp)
            return np.clip(np.asarray(model.predict(xi), dtype=float), 0.0, 100.0), name

        else:
            fallback = self.default_model
            if fallback and fallback != name and fallback in self.models:
                log.warning("predict_array: '%s' not loaded — falling back to '%s'", name, fallback)
                return self.predict_array(X, fallback)
            raise ValueError(f"Model '{name}' is not available. Loaded: {list(self.models.keys())}")

    # ── Info helpers ──────────────────────────────────────────────────────
    @property
    def model_count(self) -> int:
        """Number of successfully loaded models."""
        return len(self.models)

    def list_models(self) -> list[dict[str, Any]]:
        """Return full model listing with versioning, metrics, and load status."""
        all_metrics = self.get_metrics()
        # Registry version prefix: "v1" -> "1", "v2" -> "2", "v3" -> "3"
        reg_major = self.version.lstrip("v")
        out: list[dict[str, Any]] = []
        for name in MODEL_CATALOG:
            catalog = MODEL_CATALOG[name]
            meta = self.model_meta.get(name, {})
            # Normalize version to registry major (e.g. v3 deep model shows 3.0.0, not 2.4.0)
            raw_ver = meta.get("version") or catalog.get("version", "?")
            if raw_ver and raw_ver != "?":
                ver_major = raw_ver.split(".")[0]
                if ver_major != reg_major:
                    raw_ver = f"{reg_major}.0.0"
            out.append({
                "name":         name,
                "version":      raw_ver,
                "display_name": catalog.get("display_name", name),
                "family":       catalog.get("family", "unknown"),
                "algorithm":    catalog.get("algorithm", ""),
                "target":       catalog.get("target", "soh"),
                "r2":           catalog.get("r2"),
                "metrics":      all_metrics.get(name, {}),
                "is_default":   name == self.default_model,
                "loaded":       name in self.models,
                "load_error":   meta.get("load_error"),
            })
        return out


# ── Singletons ───────────────────────────────────────────────────────────────
registry_v1 = ModelRegistry(version="v1")
registry_v2 = ModelRegistry(version="v2")
registry_v3 = ModelRegistry(version="v3")

# Default registry — v3 (best models, highest R²)
registry = registry_v3
