from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.recommendations import (
    Recommendation,
    _generate_explanation,
    grid_search_recommendations,
    recommendations_to_dataframe,
)


def _predict_fn(df: pd.DataFrame) -> np.ndarray:
    # Higher RUL for lower current and room temperature
    return (
        200
        - (df["avg_current"].to_numpy() * 20)
        - np.abs(df["ambient_temperature"].to_numpy() - 24.0) * 2
        + (df["min_voltage"].to_numpy() * 5)
    )


def test_grid_search_recommendations_sorted():
    base = {
        "ambient_temperature": 24.0,
        "avg_current": 2.0,
        "min_voltage": 2.5,
        "cycle_number": 100,
    }
    recs = grid_search_recommendations(_predict_fn, base, top_k=3)
    assert len(recs) == 3
    assert recs[0].predicted_rul >= recs[1].predicted_rul
    assert isinstance(recs[0], Recommendation)


def test_generate_explanation_contains_keywords():
    text = _generate_explanation(
        temp=4.0,
        current=0.5,
        cutoff=2.7,
        pred_rul=150,
        baseline_rul=100,
        improvement_pct=50.0,
    )
    assert "improve RUL" in text
    assert "Cold ambient" in text


def test_recommendations_to_dataframe():
    recs = [
        Recommendation(
            rank=1,
            ambient_temperature=24.0,
            discharge_current=1.0,
            cutoff_voltage=2.7,
            predicted_rul=200.0,
            rul_improvement=20.0,
            rul_improvement_pct=10.0,
            explanation="ok",
        )
    ]
    df = recommendations_to_dataframe(recs)
    assert df.shape[0] == 1
    assert "Predicted RUL" in df.columns
