"""Create publication-ready tables and high-resolution figures from v3 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.catalog import model_catalog_records

MODEL_NAMES = {row["model_id"]: row["display_name"] for row in model_catalog_records()}


def _save(fig: plt.Figure, folder: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = folder / f"{stem}.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _load_metric_tables(results: Path, dataset: str) -> pd.DataFrame:
    paths = [
        path for path in sorted(results.glob(f"{dataset}_*_fold_metrics.csv"))
        if not path.name.startswith("nasa_to_")
    ]
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def generate_paper_outputs(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root)
    results = root / "artifacts" / "v3" / "results"
    figures = root / "artifacts" / "v3" / "figures"
    tables = root / "artifacts" / "v3" / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    manifest: list[dict[str, object]] = []

    feature_summaries = []
    for dataset in ("nasa", "calce", "oxford"):
        frame = pd.read_csv(root / "artifacts" / "v3" / "features" / dataset / "features.csv")
        feature_summaries.append({
            "dataset": dataset.upper() if dataset == "nasa" else dataset.title(),
            "batteries": frame["battery_id"].nunique(),
            "retained_cycles": len(frame),
            "soh_min": frame["SoH"].min(),
            "soh_median": frame["SoH"].median(),
            "soh_max": frame["SoH"].max(),
        })
    dataset_table = pd.DataFrame(feature_summaries)
    dataset_table.to_csv(tables / "table_1_dataset_summary.csv", index=False)

    metric_frames = {}
    for dataset in ("nasa", "calce", "oxford"):
        metrics = _load_metric_tables(results, dataset)
        metric_frames[dataset] = metrics
        summary = metrics.groupby("model", as_index=False).agg(
            mae=("mae", "mean"), mae_sd=("mae", "std"),
            rmse=("rmse", "mean"), adjusted_r2=("adjusted_r2", "mean"),
            mape=("mape", "mean"), within_5pp=("within_5pp", "mean"),
        ).sort_values("mae")
        summary.insert(1, "display_name", summary["model"].map(MODEL_NAMES))
        summary.to_csv(tables / f"table_{2 if dataset == 'nasa' else 3 if dataset == 'calce' else 4}_{dataset}_grouped_results.csv", index=False)

    protocol = pd.read_csv(results / "protocol_comparison.csv")
    protocol_summary = protocol.groupby(["protocol", "feature_contract"], as_index=False).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), r2=("r2", "mean"), within_5pp=("within_5pp", "mean")
    )
    protocol_summary.to_csv(tables / "table_5_protocol_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.1, 3.3))
    sns.barplot(data=protocol_summary, x="protocol", y="mae", hue="feature_contract", ax=ax)
    ax.set(xlabel="Evaluation protocol", ylabel="MAE (SOH percentage points)")
    ax.tick_params(axis="x", rotation=15)
    paths = _save(fig, figures, "figure_1_protocol_and_leakage_comparison")
    manifest.append({"artifact": "Figure 1", "purpose": "Quantitative V1/V2/V3 and feature-leakage comparison", "files": [str(p) for p in paths]})

    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    sns.barplot(data=dataset_table, x="dataset", y="retained_cycles", hue="dataset", legend=False, ax=ax)
    ax.set(xlabel="Dataset", ylabel="Retained discharge cycles")
    for i, row in dataset_table.iterrows():
        ax.text(i, row["retained_cycles"] + max(dataset_table["retained_cycles"]) * 0.02, f"{row['batteries']} cells", ha="center", fontsize=8)
    paths = _save(fig, figures, "figure_2_dataset_coverage")
    manifest.append({"artifact": "Figure 2", "purpose": "NASA, CALCE, and Oxford benchmark coverage", "files": [str(p) for p in paths]})

    nasa_summary = pd.read_csv(tables / "table_2_nasa_grouped_results.csv").head(20)
    fig, ax = plt.subplots(figsize=(7.1, 5.1))
    plot = nasa_summary.sort_values("mae", ascending=False)
    sns.barplot(data=plot, y="display_name", x="mae", hue="display_name", legend=False, ax=ax)
    ax.set(xlabel="Grouped test MAE (SOH percentage points)", ylabel="")
    paths = _save(fig, figures, "figure_3_nasa_all_models_mae")
    manifest.append({"artifact": "Figure 3", "purpose": "All 20 NASA grouped benchmark results", "files": [str(p) for p in paths]})

    averaged = pd.read_csv(results / "averaged_predictions.csv")
    best_nasa = nasa_summary.iloc[0]["model"]
    selected = averaged[(averaged["dataset"] == "NASA") & (averaged["model"] == best_nasa)]
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    sns.scatterplot(data=selected, x="y_true", y="y_pred", hue="battery_id", legend=False, s=14, alpha=0.65, ax=ax)
    low = min(selected["y_true"].min(), selected["y_pred"].min())
    high = max(selected["y_true"].max(), selected["y_pred"].max())
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Observed SOH (%)", ylabel="Predicted SOH (%)")
    paths = _save(fig, figures, "figure_4_nasa_predicted_vs_observed")
    manifest.append({"artifact": "Figure 4", "purpose": "Grouped held-out-battery prediction agreement", "files": [str(p) for p in paths]})

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    sns.scatterplot(data=selected, x="y_pred", y="residual", s=12, alpha=0.6, ax=axes[0])
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Predicted SOH (%)", ylabel="Residual (observed - predicted)")
    sns.histplot(selected["residual"], kde=True, ax=axes[1])
    axes[1].set(xlabel="Residual (SOH percentage points)", ylabel="Count")
    paths = _save(fig, figures, "figure_5_residual_diagnostics")
    manifest.append({"artifact": "Figure 5", "purpose": "Residual trend and distribution diagnostics", "files": [str(p) for p in paths]})

    importance = pd.read_csv(results / "extra_trees_permutation_importance.csv")
    importance_summary = importance.groupby("feature", as_index=False).agg(
        mae_increase=("mae_increase", "mean"), fold_std=("mae_increase", "std")
    ).sort_values("mae_increase", ascending=False)
    importance_summary.to_csv(tables / "table_6_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.1, 4.7))
    top = importance_summary.head(15).sort_values("mae_increase")
    ax.barh(top["feature"], top["mae_increase"], xerr=top["fold_std"], color="#2878B5", alpha=0.9)
    ax.set(xlabel="Permutation-induced MAE increase", ylabel="")
    paths = _save(fig, figures, "figure_6_extra_trees_permutation_importance")
    manifest.append({"artifact": "Figure 6", "purpose": "Held-out-battery Extra Trees permutation importance", "files": [str(p) for p in paths]})

    ablation = pd.read_csv(results / "feature_ablation.csv")
    ablation_summary = ablation.groupby("ablation", as_index=False).agg(mae=("mae", "mean"), std=("mae", "std"))
    ablation_summary.to_csv(tables / "table_7_feature_ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    sns.barplot(data=ablation_summary.sort_values("mae"), y="ablation", x="mae", hue="ablation", legend=False, ax=ax)
    ax.set(xlabel="MAE (SOH percentage points)", ylabel="")
    paths = _save(fig, figures, "figure_7_feature_ablation")
    manifest.append({"artifact": "Figure 7", "purpose": "Feature-family ablation", "files": [str(p) for p in paths]})

    external_rows = []
    for dataset in ("calce", "oxford"):
        summary_path = results / f"{dataset}_validation_summary.csv"
        summary = pd.read_csv(summary_path)
        summary["dataset"] = dataset.title()
        external_rows.append(summary)
    external = pd.concat(external_rows, ignore_index=True)
    external.to_csv(tables / "table_8_external_validation.csv", index=False)
    for number, validation, stem in [
        (8, "within_dataset_grouped", "external_grouped"),
        (9, "nasa_zero_shot", "nasa_zero_shot"),
    ]:
        plot = external[external["validation"] == validation].copy()
        plot["display_name"] = plot["model"].map(MODEL_NAMES)
        plot = plot.sort_values(["dataset", "mae"]).groupby("dataset").head(8)
        fig, ax = plt.subplots(figsize=(7.1, 4.0))
        sns.barplot(data=plot, y="display_name", x="mae", hue="dataset", ax=ax)
        ax.set(xlabel="MAE (SOH percentage points)", ylabel="")
        paths = _save(fig, figures, f"figure_{number}_{stem}")
        manifest.append({"artifact": f"Figure {number}", "purpose": validation.replace("_", " "), "files": [str(p) for p in paths]})

    stress = pd.read_csv(results / "sensor_stress.csv")
    stress_summary = stress.groupby("scenario", as_index=False).agg(mae=("mae", "mean"), std=("mae", "std"))
    stress_summary.to_csv(tables / "table_9_sensor_stress.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    sns.barplot(data=stress_summary, x="scenario", y="mae", hue="scenario", legend=False, ax=ax)
    ax.set(xlabel="Sensor scenario", ylabel="MAE (SOH percentage points)")
    ax.tick_params(axis="x", rotation=12)
    paths = _save(fig, figures, "figure_10_sensor_stress")
    manifest.append({"artifact": "Figure 10", "purpose": "Missing-temperature and measurement-noise stress tests", "files": [str(p) for p in paths]})

    ci = pd.read_csv(results / "best_model_cluster_bootstrap_ci.csv")
    ci.to_csv(tables / "table_10_cluster_bootstrap_intervals.csv", index=False)
    mae_ci = ci[ci["metric"] == "mae"].copy()
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(mae_ci))
    ax.errorbar(x, mae_ci["estimate"], yerr=[mae_ci["estimate"] - mae_ci["ci_low"], mae_ci["ci_high"] - mae_ci["estimate"]], fmt="o", capsize=4)
    ax.set_xticks(x, mae_ci["dataset"])
    ax.set(xlabel="Dataset", ylabel="MAE with 95% cluster-bootstrap CI")
    paths = _save(fig, figures, "figure_11_cluster_bootstrap_intervals")
    manifest.append({"artifact": "Figure 11", "purpose": "Battery-cluster bootstrap uncertainty", "files": [str(p) for p in paths]})

    fig, ax = plt.subplots(figsize=(7.1, 2.5))
    ax.axis("off")
    labels = ["Raw discharge records", "4.0–3.6 V observable segment", "Train-only preprocessing", "Battery-grouped validation", "External evaluation"]
    xs = np.linspace(0.07, 0.93, len(labels))
    for x, label in zip(xs, labels):
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=8, wrap=True,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#EAF3F8", "edgecolor": "#2878B5"})
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(right - 0.08, 0.5), xytext=(left + 0.08, 0.5), arrowprops={"arrowstyle": "->", "color": "#333333"})
    paths = _save(fig, figures, "figure_12_leakage_safe_workflow")
    manifest.append({"artifact": "Figure 12", "purpose": "Leakage-safe experimental workflow", "files": [str(p) for p in paths]})

    manifest_frame = pd.DataFrame(manifest)
    manifest_frame.to_csv(root / "artifacts" / "v3" / "paper_artifact_manifest.csv", index=False)
    (root / "artifacts" / "v3" / "paper_artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    print(generate_paper_outputs(args.project_root).to_string(index=False))


if __name__ == "__main__":
    main()
