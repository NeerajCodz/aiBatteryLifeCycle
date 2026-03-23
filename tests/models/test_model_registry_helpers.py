from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from api import model_registry as mr


def test_classify_degradation_thresholds():
    assert mr.classify_degradation(95) == "Healthy"
    assert mr.classify_degradation(85) == "Moderate"
    assert mr.classify_degradation(75) == "Degraded"
    assert mr.classify_degradation(60) == "End-of-Life"


def test_soh_to_color_thresholds():
    assert mr.soh_to_color(92) == "#22c55e"
    assert mr.soh_to_color(85) == "#eab308"
    assert mr.soh_to_color(75) == "#f97316"
    assert mr.soh_to_color(60) == "#ef4444"


def test_versioned_paths():
    p = mr._versioned_paths("v9")
    assert p["models_dir"].parts[-2:] == ("v9", "models")
    assert p["scalers"].parts[-2:] == ("v9", "scalers")


def test_load_version_meta_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(mr, "_ARTIFACTS", tmp_path)
    data = mr._load_version_meta("v0")
    assert data == {}


def test_load_version_meta_valid(monkeypatch, tmp_path):
    vdir = tmp_path / "v1"
    vdir.mkdir(parents=True)
    payload = {"version": "v1", "models": {"rf": {"r2": 0.9}}}
    (vdir / "models.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(mr, "_ARTIFACTS", tmp_path)
    data = mr._load_version_meta("v1")
    assert data["version"] == "v1"
    assert "rf" in data["models"]


def test_registry_build_x_and_scaling():
    reg = mr.ModelRegistry(version="v3")
    features = {c: 1.0 for c in reg.feature_cols}
    x = reg._build_x(features)
    assert x.shape[0] == 1
    assert x.shape[1] == len(reg.feature_cols)
    # no scaler loaded in test env should be passthrough
    out = reg._scale_for_linear(x)
    assert np.allclose(out, x)


def test_registry_x_for_model_with_feature_names():
    class NamedModel:
        feature_names_in_ = np.array(["a", "b"])

    x = np.array([[1.0, 2.0]])
    df = mr.ModelRegistry._x_for_model(NamedModel(), x)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]


def test_registry_predict_batch_falls_back(monkeypatch):
    reg = mr.ModelRegistry(version="v3")

    def _fake_predict(features, model_name=None):
        return {"soh_pct": 90.0, "rul_cycles": 100.0, "degradation_state": "Healthy", "model_used": "fake"}

    monkeypatch.setattr(reg, "predict", _fake_predict)
    rows = reg.predict_batch("B1", [{"cycle_number": 1}, {"cycle_number": 2}])
    assert len(rows) == 2
    assert rows[0]["battery_id"] == "B1"


def test_registry_load_model_ensemble_loads_components(monkeypatch):
    reg = mr.ModelRegistry(version="v3")
    reg._ensemble_components = ["xgboost", "lightgbm"]
    captured = {}

    def _fake_load_all(*, only_models=None):
        captured["only_models"] = only_models

    monkeypatch.setattr(reg, "load_all", _fake_load_all)
    reg.load_model("best_ensemble")
    assert captured["only_models"] == {"best_ensemble", "xgboost", "lightgbm"}


def test_registry_predict_falls_back_to_default():
    class _DummyModel:
        def predict(self, x):
            return np.array([88.0] * len(x))

    reg = mr.ModelRegistry(version="v3")
    reg.models = {"xgboost": _DummyModel()}
    reg.default_model = "xgboost"
    reg._catalog = {"xgboost": {"version": "3.0.0"}}
    reg.model_meta = {"xgboost": {"family": "classical", "version": "3.0.0"}}

    out = reg.predict({"cycle_number": 10}, model_name="missing")
    assert out["model_used"] == "xgboost"
    assert out["model_version"] == "3.0.0"


def test_registry_predict_array_best_ensemble_weighted():
    class _DummyModel:
        def __init__(self, value):
            self.value = value

        def predict(self, x):
            return np.array([self.value] * len(x))

    reg = mr.ModelRegistry(version="v3")
    reg.models = {"xgboost": _DummyModel(80.0), "lightgbm": _DummyModel(90.0)}
    reg.model_meta = {"best_ensemble": {"components": ["xgboost", "lightgbm"]}}
    reg._ensemble_components = ["xgboost", "lightgbm"]
    reg._ensemble_weights = {"xgboost": 1.0, "lightgbm": 3.0}
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y, used = reg.predict_array(X, model_name="best_ensemble")
    assert used.startswith("best_ensemble(")
    assert np.allclose(y, np.array([87.5, 87.5]))


def test_registry_predict_array_rejects_deep_models():
    class _DummyModel:
        def predict(self, x):
            return np.array([90.0] * len(x))

    reg = mr.ModelRegistry(version="v3")
    reg.models = {"bilstm": _DummyModel()}
    reg.model_meta = {"bilstm": {"family": "deep_pytorch"}}
    reg.default_model = "bilstm"
    with pytest.raises(ValueError):
        reg.predict_array(np.array([[1.0, 2.0]]))


def test_registry_list_models_normalizes_version(monkeypatch):
    reg = mr.ModelRegistry(version="v3")
    reg._catalog = {"legacy": {"version": "2.4.0", "family": "classical", "target": "soh"}}
    reg.model_meta = {"legacy": {"version": "2.4.0"}}
    reg.default_model = "legacy"
    monkeypatch.setattr(reg, "get_metrics", lambda: {})
    rows = reg.list_models()
    assert rows[0]["version"] == "3.0.0"
    assert rows[0]["is_default"] is True
