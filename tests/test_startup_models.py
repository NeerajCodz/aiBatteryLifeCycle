from scripts import download_models


def test_startup_always_includes_manifest_default_within_budget(monkeypatch):
    monkeypatch.setattr(download_models, '_read_models_meta', lambda version: {
        'default_model': 'svr',
        'models': {
            'svr': {'file': 'models/svr.joblib', 'r2': .3},
            'xgboost': {'file': 'models/xgb.joblib', 'r2': .9},
            'gradient_boosting': {'file': 'models/gb.joblib', 'r2': .8},
            'lightgbm': {'file': 'models/lgb.joblib', 'r2': .7},
        },
    })
    assert download_models.select_top_models('v3', 3) == ['svr', 'xgboost', 'gradient_boosting']
    assert download_models.select_top_models('v3', 1) == ['svr']
    assert download_models.select_top_models('v3', 0) == []


def test_startup_ignores_virtual_default_and_keeps_ranked_fallback(monkeypatch):
    monkeypatch.setattr(download_models, '_read_models_meta', lambda version: {
        'default_model': 'ensemble',
        'models': {'ensemble': {'r2': 1.}, 'svr': {'file': 'svr.joblib', 'r2': .4}},
    })
    assert download_models.select_top_models('v3', 3) == ['svr']
