from __future__ import annotations

from pathlib import Path

from src.utils import config


def test_get_version_paths_contains_expected_keys():
    paths = config.get_version_paths("v_test")
    expected = {
        "root",
        "models_classical",
        "models_deep",
        "models_ensemble",
        "scalers",
        "figures",
        "results",
        "logs",
    }
    assert set(paths.keys()) == expected
    assert all(isinstance(v, Path) for v in paths.values())
    assert paths["root"].name == "v_test"


def test_ensure_version_dirs_creates_dirs():
    paths = config.ensure_version_dirs("v_test_suite")
    for p in paths.values():
        assert p.exists()
        assert p.is_dir()
