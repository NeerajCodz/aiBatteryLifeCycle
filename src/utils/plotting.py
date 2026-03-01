"""
src.utils.plotting
==================
Research-grade visualization helpers for battery lifecycle analysis.

All functions produce Matplotlib figures and optionally save them to
``artifacts/figures/``. Plotly interactive versions are generated
where noted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import (
    CMAP_DIVERGING,
    CMAP_SEQUENTIAL,
    FIG_DPI,
    FIG_SIZE,
    FIGURES_DIR,
    MATPLOTLIB_STYLE,
)

# Apply research-grade style
try:
    plt.style.use(MATPLOTLIB_STYLE)
except OSError:
    plt.style.use("seaborn-v0_8")
sns.set_context("paper", font_scale=1.3)


def save_fig(fig: plt.Figure, name: str, tight: bool = True) -> Path:
    """Save figure as PNG to artifacts/figures/.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    name:
        Base filename (without extension).
    tight:
        Whether to call ``tight_layout()`` before saving.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.
    """
    if tight:
        fig.tight_layout()
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    return path


# ── Capacity fade curves ────────────────────────────────────────────────────
def plot_capacity_fade(
    cap_df: pd.DataFrame,
    battery_ids: list[str] | None = None,
    eol_threshold: float | None = 1.4,
    title: str = "Capacity Fade Curves",
    save_name: str | None = "capacity_fade",
) -> plt.Figure:
    """MATLAB-style capacity degradation: Capacity (Ah) vs cycle number, per battery."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bats = battery_ids or sorted(cap_df["battery_id"].unique())
    cmap = plt.cm.get_cmap("tab20", len(bats))
    for i, bid in enumerate(bats):
        sub = cap_df[cap_df["battery_id"] == bid]
        ax.plot(sub["cycle_number"], sub["Capacity"], label=bid, color=cmap(i), linewidth=1.2)
    if eol_threshold:
        ax.axhline(y=eol_threshold, color="crimson", linestyle="--", linewidth=1.5, label=f"EOL = {eol_threshold} Ah")
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Discharge Capacity (Ah)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3)
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── SOH degradation with trend ──────────────────────────────────────────────
def plot_soh_degradation(
    cap_df: pd.DataFrame,
    battery_id: str,
    save_name: str | None = None,
) -> plt.Figure:
    """SOH (%) vs cycle with linear + exponential trend lines for one battery."""
    sub = cap_df[cap_df["battery_id"] == battery_id].copy()
    x = sub["cycle_number"].values.astype(float)
    y = sub["SoH"].values if "SoH" in sub.columns else (sub["Capacity"].values / 2.0) * 100

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.scatter(x, y, s=12, alpha=0.6, label="Measured SOH")

    # Linear fit
    if len(x) > 2:
        coeffs_lin = np.polyfit(x, y, 1)
        ax.plot(x, np.polyval(coeffs_lin, x), "r--", linewidth=1.5,
                label=f"Linear: slope={coeffs_lin[0]:.4f}%/cycle")
        # Exponential fit y = a * exp(b * x) → log(y) = log(a) + b*x
        try:
            valid = y > 0
            coeffs_exp = np.polyfit(x[valid], np.log(y[valid]), 1)
            y_exp = np.exp(coeffs_exp[1]) * np.exp(coeffs_exp[0] * x)
            ax.plot(x, y_exp, "g-.", linewidth=1.5,
                    label=f"Exponential: λ={coeffs_exp[0]:.6f}")
        except Exception:
            pass

    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("State of Health (%)")
    ax.set_title(f"SOH Degradation — {battery_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Correlation heatmap ─────────────────────────────────────────────────────
def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    save_name: str | None = "correlation_heatmap",
) -> plt.Figure:
    """Seaborn heatmap of feature correlations."""
    if columns:
        df = df[columns].dropna()
    else:
        df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix")
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Box / violin plots ──────────────────────────────────────────────────────
def plot_capacity_by_temperature(
    cap_df: pd.DataFrame,
    save_name: str | None = "capacity_by_temp",
) -> plt.Figure:
    """Violin plot of discharge capacity by ambient temperature group."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.violinplot(data=cap_df, x="ambient_temperature", y="Capacity", ax=ax,
                   inner="quartile", palette="coolwarm", cut=0)
    ax.set_xlabel("Ambient Temperature (°C)")
    ax.set_ylabel("Discharge Capacity (Ah)")
    ax.set_title("Capacity Distribution by Temperature")
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Training loss curves ────────────────────────────────────────────────────
def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    title: str = "Training Loss",
    save_name: str | None = None,
) -> plt.Figure:
    """Standard training/validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", linewidth=1.5, label="Train Loss")
    if val_losses:
        ax.plot(epochs, val_losses, "r-", linewidth=1.5, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Actual vs Predicted scatter ─────────────────────────────────────────────
def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "SOH (%)",
    model_name: str = "",
    save_name: str | None = None,
) -> plt.Figure:
    """Scatter with identity line and R² annotation."""
    from sklearn.metrics import r2_score
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, s=15, alpha=0.5, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    r2 = r2_score(y_true, y_pred)
    ax.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=14, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.set_xlabel(f"Actual {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"Actual vs Predicted — {model_name}" if model_name else "Actual vs Predicted")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Residual analysis ───────────────────────────────────────────────────────
def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    save_name: str | None = None,
) -> plt.Figure:
    """Histogram + KDE of prediction residuals and Q-Q plot."""
    import scipy.stats as stats

    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram + KDE
    sns.histplot(residuals, kde=True, ax=axes[0], color="steelblue", bins=40)
    axes[0].axvline(x=0, color="red", linestyle="--")
    axes[0].set_xlabel("Residual")
    axes[0].set_title(f"Residual Distribution — {model_name}")

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q Plot — {model_name}")

    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Radar / Spider chart ────────────────────────────────────────────────────
def plot_radar_chart(
    metrics_dict: dict[str, dict[str, float]],
    title: str = "Model Comparison (Normalized Metrics)",
    save_name: str | None = "radar_chart",
) -> plt.Figure:
    """Radar chart comparing multiple models across multiple metrics.

    Parameters
    ----------
    metrics_dict : dict
        ``{model_name: {metric_name: value, ...}, ...}``
    """
    models = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    N = len(metric_names)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    cmap = plt.cm.get_cmap("Set2", len(models))

    for i, model in enumerate(models):
        values = [metrics_dict[model][m] for m in metric_names]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=cmap(i))
        ax.fill(angles, values, alpha=0.1, color=cmap(i))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_title(title, size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Cumulative Error Distribution ────────────────────────────────────────────
def plot_ced(
    errors_dict: dict[str, np.ndarray],
    title: str = "Cumulative Error Distribution",
    save_name: str | None = "ced_curve",
) -> plt.Figure:
    """CED curves: P(|error| < threshold) vs threshold, for multiple models."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for name, errors in errors_dict.items():
        sorted_err = np.sort(np.abs(errors))
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax.plot(sorted_err, cdf, linewidth=1.8, label=name)
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Error heatmap (models × batteries) ──────────────────────────────────────
def plot_error_heatmap(
    error_df: pd.DataFrame,
    title: str = "Per-Battery Error Heatmap",
    save_name: str | None = "error_heatmap",
) -> plt.Figure:
    """Heatmap of MAE per model per battery.

    Parameters
    ----------
    error_df : pd.DataFrame
        Rows = models, Columns = batteries, Values = MAE.
    """
    fig, ax = plt.subplots(figsize=(14, max(6, len(error_df) * 0.8)))
    sns.heatmap(error_df, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "MAE"})
    ax.set_title(title)
    ax.set_xlabel("Battery ID")
    ax.set_ylabel("Model")
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Model comparison bar chart ───────────────────────────────────────────────
def plot_model_comparison_bars(
    summary_df: pd.DataFrame,
    metric_cols: list[str],
    model_col: str = "model",
    title: str = "Model Comparison",
    save_name: str | None = "model_comparison_bars",
) -> plt.Figure:
    """Grouped bar chart comparing models on multiple metrics."""
    n_metrics = len(metric_cols)
    n_models = len(summary_df)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(12, n_models * 1.5), 7))
    cmap = plt.cm.get_cmap("Set2", n_metrics)
    for i, col in enumerate(metric_cols):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, summary_df[col], width, label=col, color=cmap(i))

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df[model_col], rotation=30, ha="right")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig
