from __future__ import annotations

import numpy as np

from src.models.classical import classifiers, regressors


def test_regressors_and_classifiers_evaluate():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_reg = np.array([0.0, 1.0, 2.0, 3.0])
    m = regressors.train_ridge(X, y_reg, alpha=1.0)
    met = regressors.evaluate_model(m, X, y_reg, model_name="ridge")
    assert "ridge_MAE" in met

    y_cls = np.array([0, 1, 2, 3])
    cls = classifiers.train_rf_classifier(X, y_cls, n_estimators=10)
    out = classifiers.evaluate_classifier(cls, X, y_cls, model_name="rf")
    assert "accuracy" in out
