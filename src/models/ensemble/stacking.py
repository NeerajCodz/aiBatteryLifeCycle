"""
src.models.ensemble.stacking
=============================
Ensemble methods for combining multiple model predictions.

1. Stacking ensemble — Level-0 base models + Level-1 meta-learner
2. Weighted averaging — Optimize weights via L-BFGS on held-out MAE
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.evaluation.metrics import regression_metrics


class StackingEnsemble:
    """Stacking ensemble with out-of-fold predictions for level-1 training.

    Level-0: list of (name, predict_fn) base learners that are already trained
    Level-1: Ridge regression meta-learner
    """

    def __init__(self, base_learners: list[tuple[str, Callable]], alpha: float = 1.0):
        """
        Parameters
        ----------
        base_learners : list of (name, predict_fn)
            Each predict_fn accepts X and returns predictions array.
        alpha : float
            Ridge regression regularization.
        """
        self.base_learners = base_learners
        self.meta_learner = Ridge(alpha=alpha)
        self.is_fitted = False

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base learner predictions."""
        preds = []
        for name, pred_fn in self.base_learners:
            p = pred_fn(X)
            if p.ndim == 1:
                p = p.reshape(-1, 1)
            preds.append(p)
        return np.hstack(preds)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_folds: int = 5) -> None:
        """Fit the meta-learner using out-of-fold predictions."""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_preds = np.zeros((len(X_train), len(self.base_learners)))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_val_fold = X_train[val_idx]
            meta = self._get_meta_features(X_val_fold)
            oof_preds[val_idx] = meta

        self.meta_learner.fit(oof_preds, y_train)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the stacking ensemble."""
        meta = self._get_meta_features(X)
        return self.meta_learner.predict(meta)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        preds = self.predict(X_test)
        return regression_metrics(y_test, preds, prefix="ensemble")


class WeightedAverageEnsemble:
    """Optimized weighted averaging of base model predictions.

    Weights are found by minimizing MAE on a validation set via L-BFGS-B.
    """

    def __init__(self, base_learners: list[tuple[str, Callable]]):
        self.base_learners = base_learners
        self.weights: np.ndarray | None = None

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Find optimal weights on validation set."""
        n = len(self.base_learners)
        preds_list = []
        for name, pred_fn in self.base_learners:
            preds_list.append(pred_fn(X_val))
        preds_matrix = np.column_stack(preds_list)  # (N, n_models)

        def objective(w):
            w_norm = w / w.sum()
            combined = preds_matrix @ w_norm
            return np.mean(np.abs(combined - y_val))

        # Constrain weights to be non-negative
        result = minimize(
            objective,
            x0=np.ones(n) / n,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * n,
        )
        self.weights = result.x / result.x.sum()

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds_list = []
        for name, pred_fn in self.base_learners:
            preds_list.append(pred_fn(X))
        preds_matrix = np.column_stack(preds_list)
        return preds_matrix @ self.weights

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        preds = self.predict(X_test)
        return regression_metrics(y_test, preds, prefix="weighted_avg")

    def get_weights_dict(self) -> dict[str, float]:
        return {name: float(w) for (name, _), w in zip(self.base_learners, self.weights)}
