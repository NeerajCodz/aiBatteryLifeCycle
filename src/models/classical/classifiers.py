"""
src.models.classical.classifiers
================================
Classification models for battery degradation state prediction.

4-class classification:
    0 – Healthy   (SOH ≥ 90%)
    1 – Aging     (80% ≤ SOH < 90%)
    2 – Near-EOL  (70% ≤ SOH < 80%)
    3 – EOL       (SOH < 70%)
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.evaluation.metrics import classification_metrics
from src.utils.config import MODELS_DIR, RANDOM_STATE

DEGRADATION_LABELS = ["Healthy", "Aging", "Near-EOL", "EOL"]


def _save_model(model: Any, name: str) -> None:
    path = MODELS_DIR / "classical" / f"{name}.joblib"
    joblib.dump(model, path)


def train_rf_classifier(
    X: np.ndarray, y: np.ndarray,
    n_estimators: int = 500,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1,
    )
    model.fit(X, y)
    _save_model(model, "rf_classifier")
    return model


def train_xgb_classifier(
    X: np.ndarray, y: np.ndarray,
    n_estimators: int = 500,
) -> Any:
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=6,
        learning_rate=0.1, tree_method="hist",
        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(X, y)
    _save_model(model, "xgb_classifier")
    return model


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "",
) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    metrics = classification_metrics(y_test, y_pred, labels=[0, 1, 2, 3])
    metrics["classification_report"] = classification_report(
        y_test, y_pred, target_names=DEGRADATION_LABELS, zero_division=0,
    )
    return metrics
