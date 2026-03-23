from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from api.routers import visualize


def test_hf_url_helpers(monkeypatch):
    monkeypatch.setattr(visualize, "_HF_RAW_BASE", "https://hf.example/repo")
    assert visualize._hf_url("dataset.json") == "https://hf.example/repo/dataset.json"
    assert visualize._hf_version_url("v3", "results/x.csv") == "https://hf.example/repo/v3/results/x.csv"


def test_safe_read_json_local(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"x": 1}), encoding="utf-8")
    out = visualize._safe_read_json(p)
    assert out == {"x": 1}


def test_safe_read_csv_local(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    out = visualize._safe_read_csv(p)
    assert out == [{"a": 1, "b": 2}]


def test_version_figures_from_datamap_remote(monkeypatch, tmp_path):
    monkeypatch.setattr(visualize, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(visualize, "_read_remote_text", lambda _url: json.dumps(
        {
            "files": [
                {"path": "figures/a.png"},
                {"path": "figures/b.svg"},
                {"path": "results/unified_results.csv"},
            ]
        }
    ))
    figs = visualize._version_figures("v3")
    assert figs == ["a.png", "b.svg"]


def test_version_figures_prefers_remote_when_local_datamap_has_no_figures(monkeypatch, tmp_path):
    monkeypatch.setattr(visualize, "_ARTIFACTS", tmp_path)
    v3 = tmp_path / "v3"
    v3.mkdir(parents=True)
    (v3 / "datamap.json").write_text(json.dumps({"files": [{"path": "results/x.csv"}]}), encoding="utf-8")

    def _remote_text(url: str):
        if url.endswith("/v3/datamap.json"):
            return json.dumps({"files": [{"path": "figures/remote_a.png"}]})
        return None

    monkeypatch.setattr(visualize, "_read_remote_text", _remote_text)
    figs = visualize._version_figures("v3")
    assert figs == ["remote_a.png"]


def test_version_figures_fallback_to_tree_api(monkeypatch, tmp_path):
    monkeypatch.setattr(visualize, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(visualize, "_read_remote_text", lambda _url: json.dumps([
        {"path": "v3/figures/tree_a.png"},
        {"path": "v3/figures/tree_b.svg"},
    ]))
    monkeypatch.setattr(visualize, "_hf_tree_api_url", lambda rel="": f"https://hf/api/{rel}")
    monkeypatch.setattr(visualize, "_safe_read_json", lambda *_args, **_kwargs: {})
    figs = visualize._version_figures("v3")
    assert figs == ["tree_a.png", "tree_b.svg"]


def test_build_metrics_payload_from_local_files(monkeypatch, tmp_path):
    monkeypatch.setattr(visualize, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(visualize, "_read_remote_text", lambda _url: None)
    v3 = tmp_path / "v3"
    (v3 / "results").mkdir(parents=True)
    (v3 / "features").mkdir(parents=True)
    (v3 / "figures").mkdir(parents=True)
    (v3 / "models.json").write_text(json.dumps({"models": {"rf": {"r2": 0.9, "family": "classical"}}}), encoding="utf-8")
    (v3 / "datamap.json").write_text(json.dumps({"files": []}), encoding="utf-8")
    (v3 / "results" / "unified_results.csv").write_text("model,R2,MAE\nrf,0.9,1.2\n", encoding="utf-8")
    pd.DataFrame(
        {"battery_id": ["B1", "B1"], "SoH": [95, 94], "RUL": [100, 99], "ambient_temperature": [24, 24]}
    ).to_csv(v3 / "features" / "battery_features.csv", index=False)
    (tmp_path / "dataset.json").write_text(json.dumps({"summary": {"records_total": 2}}), encoding="utf-8")

    payload = visualize._build_metrics_payload("v3")
    assert payload["version"] == "v3"
    assert payload["unified_results"][0]["model"] == "rf"
    assert payload["battery_stats"]["batteries"] == 1


def test_get_version_figure_local(monkeypatch, tmp_path, run_async):
    monkeypatch.setattr(visualize, "_ARTIFACTS", tmp_path)
    path = tmp_path / "v3" / "figures"
    path.mkdir(parents=True)
    (path / "demo.png").write_bytes(b"png")
    resp = run_async(visualize.get_version_figure("v3", "demo.png"))
    assert isinstance(resp, FileResponse)


def test_get_version_figure_redirect_remote(monkeypatch, tmp_path, run_async):
    monkeypatch.setattr(visualize, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(visualize, "_version_figures", lambda _v: ["remote.png"])
    monkeypatch.setattr(visualize, "_hf_version_url", lambda v, rel: f"https://hf/{v}/{rel}")
    resp = run_async(visualize.get_version_figure("v3", "remote.png"))
    assert isinstance(resp, RedirectResponse)
    assert "https://hf/v3/figures/remote.png" in resp.headers["location"]


def test_get_version_figure_not_found(monkeypatch, tmp_path, run_async):
    monkeypatch.setattr(visualize, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(visualize, "_version_figures", lambda _v: [])
    with pytest.raises(HTTPException) as ex:
        run_async(visualize.get_version_figure("v3", "missing.png"))
    assert ex.value.status_code == 404


def test_ensure_version_validation():
    with pytest.raises(HTTPException):
        visualize._ensure_version("v9")


def test_figures_manifest_from_figures_json(monkeypatch, tmp_path):
    monkeypatch.setattr(visualize, "_ARTIFACTS", tmp_path)
    v3 = tmp_path / "v3"
    v3.mkdir(parents=True)
    (v3 / "figures.json").write_text(
        json.dumps(
            {
                "figures": [
                    {"name": "SOH Trend", "tags": ["soh", "trend"], "location": "soh_degradation_trends.png"},
                    {"name": "Remote", "tags": "remote,external", "location": "https://cdn.example/img.png"},
                    "capacity_and_rul.png",
                ]
            }
        ),
        encoding="utf-8",
    )
    out = visualize._version_figures_manifest("v3")
    assert len(out) == 3
    assert out[0]["name"] == "SOH Trend"
    assert out[0]["url"] == "/api/v3/figures/soh_degradation_trends.png"
    assert out[1]["url"] == "https://cdn.example/img.png"
    assert out[2]["location"] == "capacity_and_rul.png"


def test_figures_manifest_fallback_to_discovered_figures(monkeypatch, tmp_path):
    monkeypatch.setattr(visualize, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(visualize, "_version_figures", lambda _v: ["a.png", "b.svg"])
    out = visualize._version_figures_manifest("v3")
    assert [x["location"] for x in out] == ["a.png", "b.svg"]


def test_list_version_figures_and_figures_json_endpoint(monkeypatch, tmp_path, run_async):
    monkeypatch.setattr(visualize, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(
        visualize,
        "_version_figures_manifest",
        lambda _v: [
            {"name": "A", "tags": ["one"], "location": "a.png", "url": "/api/v3/figures/a.png"},
            {"name": "B", "tags": ["two"], "location": "https://cdn/x.png", "url": "https://cdn/x.png"},
        ],
    )
    names = run_async(visualize.list_version_figures("v3"))
    assert names == ["a.png", "x.png"]
    manifest = run_async(visualize.get_version_figures_manifest("v3"))
    assert manifest[0]["name"] == "A"


def test_dashboard_and_battery_endpoints(monkeypatch, tmp_path, run_async):
    meta_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "battery_id": ["B1", "B1", "B2"],
            "start_time": [1, 2, 1],
            "Capacity": [2.0, 1.9, 1.8],
            "ambient_temperature": [24, 24, 43],
        }
    ).to_csv(meta_path, index=False)

    class _Reg:
        def get_metrics(self):
            return {"rf": {"R2": 0.9}, "xgb": {"R2": 0.95}}

    monkeypatch.setattr(visualize, "_DATASET", tmp_path)
    monkeypatch.setattr(visualize, "registry", _Reg())

    dash = run_async(visualize.dashboard())
    assert dash.best_model == "xgb"
    assert len(dash.batteries) == 2

    cap = run_async(visualize.battery_capacity("B1"))
    assert cap["battery_id"] == "B1"
    assert len(cap["cycles"]) == 2

    batteries = run_async(visualize.list_batteries())
    assert len(batteries) == 2


def test_figure_listing_and_get_figure(monkeypatch, tmp_path, run_async):
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(parents=True)
    (fig_dir / "a.png").write_bytes(b"x")
    (fig_dir / "b.svg").write_text("<svg/>", encoding="utf-8")
    monkeypatch.setattr(visualize, "_FIGURES", fig_dir)

    items = run_async(visualize.list_figures())
    assert items == ["a.png", "b.svg"]
    resp = run_async(visualize.get_figure("a.png"))
    assert isinstance(resp, FileResponse)

    with pytest.raises(HTTPException):
        run_async(visualize.get_figure("missing.png"))
