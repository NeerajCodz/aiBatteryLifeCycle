"""
src.evaluation.recommendations
==============================
Recommendation engine for battery lifecycle optimization.

Given a trained model and current battery state, this module finds
operational parameter configurations that maximize the predicted
Remaining Useful Life (RUL).

Two approaches:
1. **Grid search** — exhaustive sweep over discrete parameter space
2. **Gradient-based inverse optimization** — treat operational params
   as differentiable inputs and minimize −RUL via PyTorch autograd

Both return ranked recommendations with:
- Recommended configuration (temperature, current, cutoff voltage)
- Predicted RUL improvement (absolute cycles and percentage)
- Human-readable explanation
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class Recommendation:
    """A single operational recommendation."""
    rank: int
    ambient_temperature: float
    discharge_current: float
    cutoff_voltage: float
    predicted_rul: float
    rul_improvement: float
    rul_improvement_pct: float
    explanation: str


# ── Operational parameter search space ───────────────────────────────────────
DEFAULT_TEMP_VALUES = [4.0, 24.0, 43.0]           # °C
DEFAULT_CURRENT_VALUES = [0.5, 1.0, 2.0, 4.0]     # A
DEFAULT_CUTOFF_VALUES = [2.0, 2.2, 2.5, 2.7]      # V


def grid_search_recommendations(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    base_features: dict[str, float],
    *,
    temp_values: list[float] | None = None,
    current_values: list[float] | None = None,
    cutoff_values: list[float] | None = None,
    top_k: int = 5,
) -> list[Recommendation]:
    """Exhaustive grid search over operational parameters.

    Parameters
    ----------
    predict_fn : callable
        Accepts a DataFrame of features (one row per config) and returns
        predicted RUL as 1D array.
    base_features : dict
        Current battery state features (everything except the 3 operational
        params). Keys must include all features the model expects minus
        ``ambient_temperature``, ``avg_current``, ``min_voltage``.
    temp_values, current_values, cutoff_values : list of float
        Discrete search grids (defaults if None).
    top_k : int
        Number of top recommendations to return.

    Returns
    -------
    list[Recommendation]
    """
    temps = temp_values or DEFAULT_TEMP_VALUES
    currents = current_values or DEFAULT_CURRENT_VALUES
    cutoffs = cutoff_values or DEFAULT_CUTOFF_VALUES

    configs = list(itertools.product(temps, currents, cutoffs))
    rows = []
    for temp, cur, cut in configs:
        row = dict(base_features)
        row["ambient_temperature"] = temp
        row["avg_current"] = cur
        row["min_voltage"] = cut
        rows.append(row)

    df = pd.DataFrame(rows)
    preds = predict_fn(df)

    # Baseline: find prediction for the current operational params
    baseline_temp = base_features.get("ambient_temperature", 24.0)
    baseline_cur = base_features.get("avg_current", 2.0)
    baseline_cut = base_features.get("min_voltage", 2.5)
    baseline_row = dict(base_features)
    baseline_row["ambient_temperature"] = baseline_temp
    baseline_row["avg_current"] = baseline_cur
    baseline_row["min_voltage"] = baseline_cut
    baseline_rul = float(predict_fn(pd.DataFrame([baseline_row]))[0])

    # Rank by predicted RUL (descending)
    ranked_idx = np.argsort(-preds)
    recommendations = []
    seen = set()
    for rank_i, idx in enumerate(ranked_idx):
        if len(recommendations) >= top_k:
            break
        temp, cur, cut = configs[idx]
        key = (temp, cur, cut)
        if key in seen:
            continue
        seen.add(key)

        pred_rul = float(preds[idx])
        improvement = pred_rul - baseline_rul
        improvement_pct = (improvement / max(baseline_rul, 1)) * 100

        explanation = _generate_explanation(
            temp, cur, cut, pred_rul, baseline_rul, improvement_pct
        )

        recommendations.append(Recommendation(
            rank=len(recommendations) + 1,
            ambient_temperature=temp,
            discharge_current=cur,
            cutoff_voltage=cut,
            predicted_rul=pred_rul,
            rul_improvement=improvement,
            rul_improvement_pct=improvement_pct,
            explanation=explanation,
        ))

    return recommendations


def gradient_based_recommendations(
    model,
    base_features_tensor,
    *,
    lr: float = 0.01,
    steps: int = 500,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Gradient-based inverse optimization using PyTorch autograd.

    Treat operational parameters (temp, current, cutoff) as differentiable
    inputs and minimize −predicted_RUL.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model (must accept full feature vector).
    base_features_tensor : torch.Tensor
        Shape ``(1, n_features)``. The 3 operational param indices will be
        optimized while the rest are frozen.

    Returns list of optimized configurations.
    """
    import torch

    # Clone and set operational params as learnable
    x = base_features_tensor.clone().detach().requires_grad_(False).float()
    # Indices for operational params — must match feature ordering
    # ambient_temperature=1, avg_current=5, min_voltage=3
    OP_INDICES = [1, 5, 3]

    results = []
    for trial in range(top_k):
        x_opt = x.clone()
        # Initialize with random perturbation
        for idx in OP_INDICES:
            x_opt[0, idx] = x[0, idx] + torch.randn(1).item() * 0.2
        x_opt = x_opt.detach().requires_grad_(True)

        optimizer = torch.optim.Adam([x_opt], lr=lr)
        for step in range(steps):
            optimizer.zero_grad()
            pred = model(x_opt)
            loss = -pred.mean()  # maximize RUL
            loss.backward()
            # Only update operational params
            with torch.no_grad():
                for i in range(x_opt.shape[1]):
                    if i not in OP_INDICES:
                        x_opt.grad[0, i] = 0.0
            optimizer.step()

        with torch.no_grad():
            final_rul = model(x_opt).item()
        results.append({
            "ambient_temperature": x_opt[0, 1].item(),
            "discharge_current": x_opt[0, 5].item(),
            "cutoff_voltage": x_opt[0, 3].item(),
            "predicted_rul": final_rul,
        })

    return results


def _generate_explanation(
    temp: float, current: float, cutoff: float,
    pred_rul: float, baseline_rul: float, improvement_pct: float,
) -> str:
    """Generate a human-readable explanation for a recommendation."""
    parts = []
    if improvement_pct > 0:
        parts.append(f"This configuration is predicted to improve RUL by {improvement_pct:.1f}% "
                      f"({pred_rul:.0f} vs {baseline_rul:.0f} cycles).")
    else:
        parts.append(f"Predicted RUL: {pred_rul:.0f} cycles (baseline: {baseline_rul:.0f}).")

    if temp <= 10:
        parts.append("Cold ambient (≤10°C) slows chemical degradation but increases internal resistance.")
    elif temp >= 40:
        parts.append("Elevated temperature (≥40°C) accelerates SEI growth and capacity fade.")
    else:
        parts.append("Room temperature (20–30°C) provides the best balance for cycle life.")

    if current <= 1.0:
        parts.append("Low discharge current (≤1A) reduces thermal stress and extends life.")
    elif current >= 4.0:
        parts.append("High discharge current (≥4A) increases heat generation and lithium plating risk.")

    if cutoff <= 2.2:
        parts.append("Lower cutoff voltage extracts more capacity per cycle but accelerates degradation.")
    elif cutoff >= 2.5:
        parts.append("Higher cutoff voltage (≥2.5V) preserves cell health by avoiding deep discharge.")

    return " ".join(parts)


def recommendations_to_dataframe(recs: list[Recommendation]) -> pd.DataFrame:
    """Convert recommendation list to a presentable DataFrame."""
    return pd.DataFrame([
        {
            "Rank": r.rank,
            "Temperature (°C)": r.ambient_temperature,
            "Current (A)": r.discharge_current,
            "Cutoff (V)": r.cutoff_voltage,
            "Predicted RUL": f"{r.predicted_rul:.0f}",
            "Δ RUL (cycles)": f"{r.rul_improvement:+.0f}",
            "Δ RUL (%)": f"{r.rul_improvement_pct:+.1f}%",
            "Explanation": r.explanation,
        }
        for r in recs
    ])
