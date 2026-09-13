"""Generate the ordered, leakage-aware v3 review notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"

SETUP = """from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "v3"
RESULTS = ARTIFACT_ROOT / "results"
FIGURES = ARTIFACT_ROOT / "figures"
FEATURES = ARTIFACT_ROOT / "features"
for path in (RESULTS, FIGURES, FEATURES):
    path.mkdir(parents=True, exist_ok=True)
print(f"Project root: {PROJECT_ROOT}")
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def py(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def make_notebook(title: str, purpose: str, cells: list[dict]) -> dict:
    intro = f"""# {title}

Purpose: {purpose}

Leakage rule: current-cycle full discharged capacity and SOH are labels only.
Preprocessing is fitted on training batteries, validation batteries control
selection and early stopping, and test batteries are scored exactly once.
"""
    notebook_cells = [md(intro), py(SETUP), *cells]
    for index, cell in enumerate(notebook_cells):
        cell["id"] = f"cell-{index:02d}"
    return {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (aiBatteryLifecycle)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "review_protocol": "v3-leakage-aware-2026-09",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SPECS = {
    "00_environment.ipynb": (
        "00 — Environment and Reproducibility",
        "Record software, hardware, seeds, source checksums, and the exact 20-model registry.",
        [
            md("## Fixed protocol\n\nSeeds: 17, 42, 2026. Primary criterion: macro per-battery MAE; tie-breakers: RMSE, serialized size, then CPU latency."),
            py("""import json, os, platform
import numpy as np, pandas as pd, sklearn, scipy, psutil, torch, tensorflow as tf
import xgboost, lightgbm
from src.models.catalog import model_catalog_records

environment = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "processor": platform.processor(),
    "logical_cpu_count": os.cpu_count(),
    "ram_gib": round(psutil.virtual_memory().total / 1024**3, 2),
    "cuda_available": torch.cuda.is_available(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "pytorch": torch.__version__,
    "tensorflow": tf.__version__,
    "xgboost": xgboost.__version__,
    "lightgbm": lightgbm.__version__,
    "seeds": [17, 42, 2026],
    "primary_metric": "macro_per_battery_mae",
}
(RESULTS / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
catalog = pd.DataFrame(model_catalog_records())
catalog.to_csv(RESULTS / "model_catalog.csv", index=False)
display(catalog)
environment"""),
            py("""checksum_file = PROJECT_ROOT / "datasets" / "checksums.sha256"
print(checksum_file.read_text(encoding="utf-8") if checksum_file.exists() else "Run the benchmark downloader first.")"""),
        ],
    ),
    "01_dataset_audit.ipynb": (
        "01 — Dataset Audit",
        "Inventory NASA, CALCE, and Oxford cells, cycles, conditions, missing channels, exclusions, and provenance.",
        [
            md("## Normalize raw cycles\n\nThis step engineers no predictors. Every exclusion is recorded with a reason."),
            py("""import pandas as pd
from scripts.data.build_benchmark_datasets import load_all_cycles
from src.data.adapters import cycle_inventory

cycles_by_dataset = load_all_cycles(PROJECT_ROOT / "datasets" / "raw")
inventories = {}
for name, cycles in cycles_by_dataset.items():
    inventory = cycle_inventory(cycles)
    inventory.to_csv(FEATURES / f"{name.lower()}_inventory.csv", index=False)
    inventories[name] = inventory
    display(inventory.groupby("battery_id").agg(
        cycles=("cycle_number", "count"),
        capacity_min_ah=("capacity_ah", "min"),
        capacity_max_ah=("capacity_ah", "max"),
    ))
audit = pd.concat(inventories.values(), ignore_index=True)
display(audit.groupby("dataset").agg(
    batteries=("battery_id", "nunique"),
    cycles=("cycle_number", "count"),
    measurements=("n_measurements", "sum"),
))"""),
        ],
    ),
    "02_feature_engineering.ipynb": (
        "02 — Leakage-Safe Partial-Cycle Features",
        "Build 18 scalar predictors and 64 x 5 sequences from the observable 4.0–3.6 V discharge segment.",
        [
            md("## Target and inputs\n\nSOH = 100 x Q_t / median(Q_1,Q_2,Q_3). Full-cycle Q_t is the target only; inputs use the partial voltage window and prior usage."),
            py("""import pandas as pd
from scripts.data.build_benchmark_datasets import build_all_benchmarks
summaries = build_all_benchmarks(PROJECT_ROOT / "datasets" / "raw", FEATURES)
pd.DataFrame(summaries)"""),
            py("""from src.utils.config import FEATURE_COLS_V3
forbidden = {"Capacity", "capacity_ah", "SoH", "delta_capacity", "soh_rolling_mean"}
assert not forbidden.intersection(FEATURE_COLS_V3)
print(f"Verified {len(FEATURE_COLS_V3)} predictors with no current-cycle target proxies.")"""),
        ],
    ),
    "03_protocol_comparison.ipynb": (
        "03 — Quantitative V1/V2/V3 Protocol Comparison",
        "Separate split-design effects from feature-leakage effects with a factorial comparison.",
        [
            md("## Design\n\nV1: legacy random-cycle holdout. V2: within-battery chronological 80/20. V3: five-fold battery-grouped validation. Legacy and safe feature sets are crossed where possible."),
            py("""from scripts.run_protocol_comparison import run_protocol_comparison
comparison = run_protocol_comparison(PROJECT_ROOT, seeds=(17, 42, 2026))
comparison.to_csv(RESULTS / "protocol_comparison.csv", index=False)
display(comparison)"""),
        ],
    ),
    "04_classical_ml.ipynb": (
        "04 — Classical Models",
        "Evaluate eight classical regressors under grouped nested validation.",
        [
            py("""import pandas as pd
from src.experiments.classical import run_grouped_tabular_benchmark
nasa = pd.read_csv(FEATURES / "nasa" / "features.csv")
metric_path = RESULTS / "nasa_classical_fold_metrics.csv"
prediction_path = RESULTS / "nasa_classical_predictions.csv"
if not metric_path.exists() or not prediction_path.exists():
    metrics, predictions = run_grouped_tabular_benchmark(nasa, dataset_name="NASA")
    metrics.to_csv(metric_path, index=False)
    predictions.to_csv(prediction_path, index=False)
else:
    metrics = pd.read_csv(metric_path)
display(metrics.groupby("model")[["mae", "rmse", "r2", "within_5pp"]].mean().sort_values("mae"))"""),
        ],
    ),
    "05_recurrent_models.ipynb": (
        "05 — Recurrent Models",
        "Evaluate Vanilla LSTM, Bidirectional LSTM, GRU, and Attention LSTM with validation-battery early stopping.",
        [
            py("""import numpy as np, pandas as pd
from src.experiments.deep import run_grouped_sequence_benchmark
data = np.load(FEATURES / "nasa" / "sequences.npz")
index = pd.read_csv(FEATURES / "nasa" / "sequence_index.csv")
models = ("vanilla_lstm", "bidirectional_lstm", "gru", "attention_lstm")
metric_path = RESULTS / "nasa_recurrent_fold_metrics.csv"
prediction_path = RESULTS / "nasa_recurrent_predictions.csv"
if not metric_path.exists() or not prediction_path.exists():
    metrics, predictions = run_grouped_sequence_benchmark(data["X"], index, dataset_name="NASA", model_ids=models)
    metrics.to_csv(metric_path, index=False)
    predictions.to_csv(prediction_path, index=False)
else:
    metrics = pd.read_csv(metric_path)
display(metrics.groupby("model")[["mae", "rmse", "r2", "within_5pp"]].mean().sort_values("mae"))"""),
        ],
    ),
    "06_transformer_models.ipynb": (
        "06 — Transformer Models",
        "Evaluate BatteryGPT, TFT, iTransformer, and physics-informed iTransformer.",
        [
            py("""import numpy as np, pandas as pd
from src.experiments.deep import run_grouped_sequence_benchmark
data = np.load(FEATURES / "nasa" / "sequences.npz")
index = pd.read_csv(FEATURES / "nasa" / "sequence_index.csv")
models = ("battery_gpt", "temporal_fusion_transformer", "itransformer", "physics_itransformer")
metric_path = RESULTS / "nasa_transformer_fold_metrics.csv"
prediction_path = RESULTS / "nasa_transformer_predictions.csv"
if not metric_path.exists() or not prediction_path.exists():
    metrics, predictions = run_grouped_sequence_benchmark(data["X"], index, dataset_name="NASA", model_ids=models)
    metrics.to_csv(metric_path, index=False)
    predictions.to_csv(prediction_path, index=False)
else:
    metrics = pd.read_csv(metric_path)
display(metrics.groupby("model")[["mae", "rmse", "r2", "within_5pp"]].mean().sort_values("mae"))"""),
        ],
    ),
    "07_graph_variational.ipynb": (
        "07 — Graph and Variational Models",
        "Evaluate Dynamic-Graph iTransformer and VAE-LSTM on the same untouched test batteries.",
        [
            py("""import numpy as np, pandas as pd
from src.experiments.deep import run_grouped_sequence_benchmark
data = np.load(FEATURES / "nasa" / "sequences.npz")
index = pd.read_csv(FEATURES / "nasa" / "sequence_index.csv")
models = ("dynamic_graph_itransformer", "vae_lstm")
metric_path = RESULTS / "nasa_graph_variational_fold_metrics.csv"
prediction_path = RESULTS / "nasa_graph_variational_predictions.csv"
if not metric_path.exists() or not prediction_path.exists():
    metrics, predictions = run_grouped_sequence_benchmark(data["X"], index, dataset_name="NASA", model_ids=models)
    metrics.to_csv(metric_path, index=False)
    predictions.to_csv(prediction_path, index=False)
else:
    metrics = pd.read_csv(metric_path)
display(metrics.groupby("model")[["mae", "rmse", "r2", "within_5pp"]].mean().sort_values("mae"))"""),
        ],
    ),
    "08_ensembles.ipynb": (
        "08 — Leakage-Safe Ensembles",
        "Review stacking and validation-error weighted ensembles fitted without outer-test labels.",
        [
            md("Ensemble predictions are produced in Notebook 04 so all base and meta models share the same outer test fold."),
            py("""import pandas as pd
metrics = pd.read_csv(RESULTS / "nasa_classical_fold_metrics.csv")
ensemble = metrics[metrics["model"].isin(["stacking_ensemble", "weighted_ensemble"])]
display(ensemble.groupby("model")[["mae", "rmse", "r2", "within_5pp"]].mean().sort_values("mae"))"""),
        ],
    ),
    "09_external_calce.ipynb": (
        "09 — CALCE External Validation",
        "Run all 20 models in grouped CALCE validation and NASA-trained models zero-shot on CALCE.",
        [
            py("""from scripts.run_external_validation import run_external_validation
summary = run_external_validation(PROJECT_ROOT, dataset="calce", seeds=(17, 42, 2026))
summary.to_csv(RESULTS / "calce_validation_summary.csv", index=False)
display(summary)"""),
        ],
    ),
    "10_external_oxford.ipynb": (
        "10 — Oxford External Validation",
        "Run all 20 models in grouped Oxford validation and NASA-trained models zero-shot on Oxford.",
        [
            py("""from scripts.run_external_validation import run_external_validation
summary = run_external_validation(PROJECT_ROOT, dataset="oxford", seeds=(17, 42, 2026))
summary.to_csv(RESULTS / "oxford_validation_summary.csv", index=False)
display(summary)"""),
        ],
    ),
    "11_stats_robustness.ipynb": (
        "11 — Statistics, Residuals, Robustness, and Ablations",
        "Compute cluster-bootstrap intervals, Wilcoxon-Holm tests, residual diagnostics, sensor stress tests, and feature ablations.",
        [
            py("""from scripts.run_statistical_analysis import run_statistical_analysis
outputs = run_statistical_analysis(PROJECT_ROOT, bootstrap_samples=10_000)
for name, table in outputs.items():
    display(name, table.head())"""),
        ],
    ),
    "12_paper_outputs.ipynb": (
        "12 — Publication Tables and Figures",
        "Generate every manuscript table and high-resolution figure from verified machine-readable outputs.",
        [
            py("""from scripts.generate_paper_outputs import generate_paper_outputs
manifest = generate_paper_outputs(PROJECT_ROOT)
display(manifest)"""),
            md("Raster figures are exported at 600 dpi and line plots also as PDF/SVG. The manuscript uses consistent Arabic numbering: Table 1, Table 2, and so on."),
        ],
    ),
}


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for filename, (title, purpose, cells) in SPECS.items():
        path = NOTEBOOKS / filename
        path.write_text(json.dumps(make_notebook(title, purpose, cells), indent=1), encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
