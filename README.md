---
title: AI Battery Lifecycle Predictor
emoji: 🔋
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
license: mit
---

# Leakage-Aware Lithium-Ion Battery SOH Estimation

This repository contains the reproducible code, notebooks, API, and web demo for
current-cycle state-of-health (SOH) estimation from the observable 4.0–3.6 V
portion of a discharge. It is the canonical source repository for the paper
revision; the Hugging Face Space is a deployment remote and the separate model
repository stores release artifacts.

The 2026 reviewer revision replaces the former v3 experiment because it used
current-cycle capacity-derived predictors. Those fields made the reported score
look much better than performance on a genuinely unseen battery. The new v3
contract excludes full-cycle capacity, current-cycle SOH, rolling SOH, and all
other target proxies from model inputs.

## Reproducible benchmark

- Target: current-cycle SOH, `100 × full-cycle capacity / robust initial capacity`.
- Inputs: 18 scalar summaries or a 64 × 5 sequence from the current discharge's
  4.0–3.6 V window; full-cycle capacity is retained only as the label.
- C-rate and prior equivalent-full-cycle inputs use public rated capacity
  (NASA 2.0 Ah, CALCE 1.1 Ah, Oxford 0.74 Ah), never a measured target capacity.
- Internal validation: disjoint train, validation, and test batteries; five
  grouped folds where the dataset has at least five cells; seeds 17, 42, 2026.
- External validation: NASA-trained frozen models evaluated zero-shot on CALCE
  and Oxford, plus grouped within-dataset cross-validation on each external set.
- Metrics: MAE, RMSE, MAPE, R², adjusted R², residual diagnostics, absolute
  ±5-SOH-percentage-point coverage, per-battery scores, 10,000-draw
  battery-cluster bootstrap intervals, and paired Wilcoxon–Holm tests.
- Model registry: exactly 20 evaluated entries—8 classical regressors, 10
  sequence architectures, and 2 leakage-safe ensembles.

After quality exclusions and partial-window eligibility checks, the normalized
benchmark contains 2,141 NASA cycles from 30 batteries, 3,789 CALCE cycles from
4 cells, and 519 Oxford records from 8 cells. Raw sources and SHA-256 digests are
documented in `datasets/SOURCES.md` and `datasets/checksums.sha256`.

## Notebook order

Run the notebooks in numerical order. Each notebook declares its purpose,
leakage rule, inputs, and saved outputs.

```text
00_environment.ipynb
01_dataset_audit.ipynb
02_feature_engineering.ipynb
03_protocol_comparison.ipynb
04_classical_ml.ipynb
05_recurrent_models.ipynb
06_transformer_models.ipynb
07_graph_variational.ipynb
08_ensembles.ipynb
09_external_calce.ipynb
10_external_oxford.ipynb
11_stats_robustness.ipynb
12_paper_outputs.ipynb
```

Machine-readable fold metrics and predictions are written under
`artifacts/v3/results`; publication tables and 600-dpi/PDF/SVG figures are
generated under `artifacts/v3/tables` and `artifacts/v3/figures`.

## Setup and rerun

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python scripts/data/download_benchmark_data.py
.venv/Scripts/python scripts/data/build_benchmark_datasets.py
jupyter lab notebooks/
```

The downloader uses only the official NASA PCoE, CALCE, and Oxford ORA sources
listed in the provenance file. Do not commit the raw archives.

## API contract

`POST /api/v3/predict` accepts the same 18 leakage-safe partial-discharge
features used by the tabular benchmark. It does not accept current-cycle
capacity, SOH, or a capacity delta. The endpoint returns SOH only; the former
heuristic RUL and fixed-width confidence values are intentionally not presented
as validated v3 outputs. The v1/v2 routes remain for backward compatibility and
are explicitly legacy.

```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

Swagger documentation is available at `/docs`. The recommendation and
multi-step simulation panels are exploratory prototypes; they are not validated
battery-management control policies and must not be used for safety-critical
operation.

## Repository layout

```text
api/                 FastAPI routes and model registry
datasets/            provenance and checksums (raw data ignored)
frontend/            React demonstration interface
notebooks/           ordered experimental record
scripts/             download, evaluation, statistics, and figure entry points
src/data/            dataset adapters and partial-cycle features
src/evaluation/      metrics, grouped splits, bootstrap, statistical tests
src/experiments/     tabular and sequence benchmark runners
src/models/          20-model implementation catalog
tests/               unit and integration checks
artifacts/v3/         generated, machine-readable revision outputs
```

## Scope

This is a research prototype, not a certified battery-management system. The
paper now frames charging recommendations as future decision-support research,
not as a demonstrated deployment claim.
