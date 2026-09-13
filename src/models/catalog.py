"""Authoritative 20-model registry used by notebooks, training, and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    family: str
    input_kind: str
    framework: str


MODEL_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("extra_trees", "Extra Trees", "classical", "tabular", "scikit-learn"),
    ModelSpec("gradient_boosting", "Gradient Boosting", "classical", "tabular", "scikit-learn"),
    ModelSpec("random_forest", "Random Forest", "classical", "tabular", "scikit-learn"),
    ModelSpec("xgboost", "XGBoost", "classical", "tabular", "xgboost"),
    ModelSpec("lightgbm", "LightGBM", "classical", "tabular", "lightgbm"),
    ModelSpec("svr", "Support Vector Regression", "classical", "tabular", "scikit-learn"),
    ModelSpec("ridge", "Ridge Regression", "classical", "tabular", "scikit-learn"),
    ModelSpec("knn", "K-Nearest Neighbours", "classical", "tabular", "scikit-learn"),
    ModelSpec("vanilla_lstm", "Vanilla LSTM", "recurrent", "sequence", "pytorch"),
    ModelSpec("bidirectional_lstm", "Bidirectional LSTM", "recurrent", "sequence", "pytorch"),
    ModelSpec("gru", "GRU", "recurrent", "sequence", "pytorch"),
    ModelSpec("attention_lstm", "Attention LSTM", "recurrent", "sequence", "pytorch"),
    ModelSpec("battery_gpt", "BatteryGPT", "transformer", "sequence", "pytorch"),
    ModelSpec("temporal_fusion_transformer", "Temporal Fusion Transformer", "transformer", "sequence", "pytorch"),
    ModelSpec("itransformer", "iTransformer", "transformer", "sequence", "tensorflow"),
    ModelSpec("physics_itransformer", "Physics-Informed iTransformer", "transformer", "sequence", "tensorflow"),
    ModelSpec("dynamic_graph_itransformer", "Dynamic-Graph iTransformer", "graph", "sequence", "tensorflow"),
    ModelSpec("vae_lstm", "VAE-LSTM", "variational", "sequence", "pytorch"),
    ModelSpec("stacking_ensemble", "Stacking Ensemble", "ensemble", "tabular", "scikit-learn"),
    ModelSpec("weighted_ensemble", "Weighted Average Ensemble", "ensemble", "tabular", "scipy"),
)


def model_catalog_records() -> list[dict[str, str]]:
    return [asdict(spec) for spec in MODEL_CATALOG]


def get_model_spec(model_id: str) -> ModelSpec:
    for spec in MODEL_CATALOG:
        if spec.model_id == model_id:
            return spec
    raise KeyError(f"Unknown model_id: {model_id}")
