from __future__ import annotations

import numpy as np

from src.models.ensemble.stacking import StackingEnsemble, WeightedAverageEnsemble


def test_stacking_and_weighted_ensemble():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])

    learners = [
        ("m1", lambda z: z[:, 0] * 2.0),
        ("m2", lambda z: z[:, 0] * 1.8),
    ]

    stack = StackingEnsemble(learners)
    stack.fit(X, y, n_folds=2)
    preds = stack.predict(X)
    assert preds.shape == y.shape

    wa = WeightedAverageEnsemble(learners)
    wa.fit(X, y)
    w = wa.get_weights_dict()
    assert np.isclose(sum(w.values()), 1.0)
