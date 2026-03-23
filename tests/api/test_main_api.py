from __future__ import annotations

import pytest
from fastapi import BackgroundTasks, HTTPException

import api.main as api_main


class _StubRegistry:
    def __init__(self, model_count=0, meta=None, catalog=None, models=None):
        self.model_count = model_count
        self._version_meta = meta or {}
        self._catalog = catalog or {}
        self.models = models or {}
        self.loaded_models: list[str] = []

    def ensure_metadata_loaded(self):
        return None

    def refresh_metadata(self):
        return None

    def load_all(self, only_models=None):
        if only_models:
            for name in only_models:
                self.models[name] = object()
                self.loaded_models.append(name)
        self.model_count = max(self.model_count, 1)

    def model_on_disk(self, model_name):
        return model_name in self._catalog and self._catalog[model_name].get("on_disk", False)

    def load_model(self, model_name):
        if model_name in self._catalog:
            self.models[model_name] = object()
            self.loaded_models.append(model_name)
            return True
        return False


def test_model_status_key():
    assert api_main._model_status_key("v3", "xgboost") == "v3:xgboost"


def test_version_loaded_false_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "_artifacts_dir", lambda: tmp_path)
    assert api_main._version_loaded("v3") is False


def test_version_loaded_true_with_joblib(monkeypatch, tmp_path):
    models_dir = tmp_path / "v3" / "models" / "classical"
    models_dir.mkdir(parents=True)
    (models_dir / "x.joblib").write_bytes(b"x")
    monkeypatch.setattr(api_main, "_artifacts_dir", lambda: tmp_path)
    assert api_main._version_loaded("v3") is True


def test_list_versions(run_async, monkeypatch):
    regs = {
        "v1": _StubRegistry(1, {"display": "v1.0", "description": "d1", "features": 12, "champion": "rf"}, {"rf": {}}),
        "v2": _StubRegistry(0, {"display": "v2.0", "description": "d2", "features": 12, "champion": "et"}, {"et": {}}),
        "v3": _StubRegistry(2, {"display": "v3.0", "description": "d3", "features": 18, "champion": "xgb"}, {"xgb": {}}),
    }
    monkeypatch.setattr(api_main, "_REGISTRIES", regs)
    monkeypatch.setattr(api_main, "_version_status", {"v3": "ready"})
    monkeypatch.setattr(api_main, "_version_loaded", lambda v: v != "v2")
    out = run_async(api_main.list_versions())
    assert len(out) == 3
    assert out[0]["id"] == "v3"
    assert out[0]["status"] == "ready"


def test_health(run_async, monkeypatch):
    monkeypatch.setattr(api_main, "registry_v1", _StubRegistry(model_count=1))
    monkeypatch.setattr(api_main, "registry_v2", _StubRegistry(model_count=2))
    monkeypatch.setattr(api_main, "registry_v3", _StubRegistry(model_count=3))
    monkeypatch.setattr(api_main, "registry", type("R", (), {"device": "cpu"})())
    resp = run_async(api_main.health())
    assert resp.models_loaded == 6
    assert resp.status == "ok"


def test_load_version_unknown(run_async):
    with pytest.raises(HTTPException) as ex:
        run_async(api_main.load_version("v9", None))
    assert ex.value.status_code == 400


def test_load_version_returns_downloading_when_already_running(run_async, monkeypatch):
    reg = _StubRegistry()
    monkeypatch.setattr(api_main, "_REGISTRIES", {"v1": reg, "v2": reg, "v3": reg})
    monkeypatch.setattr(api_main, "_version_status", {"v1": "downloading"})
    monkeypatch.setattr(api_main, "ensure_metadata_first", lambda _v: None)
    out = run_async(api_main.load_version("v1", BackgroundTasks()))
    assert out["status"] == "downloading"


def test_load_version_loads_from_disk_when_available(run_async, monkeypatch):
    reg = _StubRegistry(model_count=0)
    monkeypatch.setattr(api_main, "_REGISTRIES", {"v1": reg, "v2": reg, "v3": reg})
    monkeypatch.setattr(api_main, "_version_status", {})
    monkeypatch.setattr(api_main, "_version_loaded", lambda _v: True)
    monkeypatch.setattr(api_main, "ensure_metadata_first", lambda _v: None)
    out = run_async(api_main.load_version("v1", BackgroundTasks()))
    assert out["status"] == "ready"
    assert reg.model_count >= 1


def test_get_version_models_meta_and_datamap(run_async, monkeypatch, tmp_path):
    v1 = tmp_path / "v1"
    v1.mkdir(parents=True)
    (v1 / "models.json").write_text('{"models":{"rf":{"family":"classical"}}}', encoding="utf-8")
    (v1 / "datamap.json").write_text('{"files":[{"path":"figures/a.png"}]}', encoding="utf-8")

    reg = _StubRegistry()
    monkeypatch.setattr(api_main, "_REGISTRIES", {"v1": reg, "v2": reg, "v3": reg})
    monkeypatch.setattr(api_main, "_artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(api_main, "ensure_metadata_first", lambda _v: None)

    out = run_async(api_main.get_version_models_meta("v1"))
    assert "models_meta" in out and "datamap" in out
    assert "rf" in out["models_meta"]["models"]
    assert out["datamap"]["files"][0]["path"] == "figures/a.png"


def test_get_version_datamap_generates_when_missing(run_async, monkeypatch, tmp_path):
    v1 = tmp_path / "v1"
    v1.mkdir(parents=True)

    reg = _StubRegistry()

    def _fake_write_datamap(version):
        (tmp_path / version / "datamap.json").write_text('{"files":[]}', encoding="utf-8")

    monkeypatch.setattr(api_main, "_REGISTRIES", {"v1": reg, "v2": reg, "v3": reg})
    monkeypatch.setattr(api_main, "_artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(api_main, "ensure_metadata_first", lambda _v: None)
    monkeypatch.setattr(api_main, "write_datamap", _fake_write_datamap)

    out = run_async(api_main.get_version_datamap("v1"))
    assert out["files"] == []


def test_list_version_models_and_load_single_model(run_async, monkeypatch):
    catalog = {
        "xgboost": {"family": "classical", "r2": 0.9, "file": "models/classical/xgboost.joblib", "on_disk": True},
        "best_ensemble": {"family": "ensemble", "r2": 0.93, "file": None},
    }
    reg = _StubRegistry(catalog=catalog)
    monkeypatch.setattr(api_main, "_REGISTRIES", {"v1": reg, "v2": reg, "v3": reg})
    monkeypatch.setattr(api_main, "_model_status", {})
    monkeypatch.setattr(api_main, "ensure_metadata_first", lambda _v: None)

    rows = run_async(api_main.list_version_models("v1"))
    assert {r["name"] for r in rows} == {"xgboost", "best_ensemble"}
    assert any(r["status"] in ("on_disk", "ready", "not_downloaded") for r in rows)

    out = run_async(api_main.load_single_model("v1", "xgboost", BackgroundTasks()))
    assert out["status"] == "ready"

    out_virtual = run_async(api_main.load_single_model("v1", "best_ensemble", BackgroundTasks()))
    assert out_virtual["status"] == "ready"
