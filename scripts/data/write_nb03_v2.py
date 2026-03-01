"""
Script to write the v2 NB03 notebook (Classical ML) as a proper .ipynb JSON file.
"""
import json
import os

def make_cell(cell_type, source, execution_count=None):
    """Create a notebook cell dict."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source,
    }
    # Fix: each line except the last needs a trailing newline
    lines = cell["source"]
    fixed = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            fixed.append(line + "\n" if not line.endswith("\n") else line)
        else:
            fixed.append(line)
    cell["source"] = fixed
    
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = []
    return cell

cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 0: Title
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", """\
# 03 — Classical Machine Learning Models (v2)
## SOH & RUL Regression + Degradation State Classification

**v2 Changes from v1:**
- **Intra-battery chronological split** (80/20 per battery) instead of cross-battery group split
- Added **ExtraTrees** and **GradientBoosting** regressors (12 models total)
- Target metric: **≥ 95% within ±5% SOH accuracy**
- All artifacts saved to `artifacts/v2/` via versioning helpers
- Per-battery error analysis added

**Models trained (12 total):**
| # | Model | Type |
|---|-------|------|
| 1 | Ridge | Linear baseline |
| 2 | Lasso | L1-regularized |
| 3 | ElasticNet | L1+L2 blend |
| 4-6 | KNN (k=5, 10, 20) | Instance-based |
| 7 | SVR (RBF) | Kernel method |
| 8 | RandomForest | Bagging ensemble |
| 9 | ExtraTrees | Randomized trees *(new)* |
| 10 | GradientBoosting | Sequential boosting *(new)* |
| 11 | XGBoost | Optuna HPO, 100 trials |
| 12 | LightGBM | Optuna HPO, 100 trials |

**Evaluation:** MAE, RMSE, R², MAPE, ±5% tolerance accuracy, SHAP analysis, per-battery breakdown"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 1: Imports
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data.preprocessing import (
    FEATURE_COLS_SCALAR, TARGET_SOH, TARGET_RUL, TARGET_DEGRADATION,
)
from src.models.classical.classifiers import (
    train_rf_classifier, train_xgb_classifier, evaluate_classifier,
    DEGRADATION_LABELS,
)
from src.evaluation.metrics import (
    regression_metrics, tolerance_accuracy, build_summary_table,
    per_battery_evaluation,
)
from src.utils.plotting import (
    plot_actual_vs_predicted, plot_residuals, save_fig,
)
from src.utils.config import (
    ARTIFACTS_DIR, RANDOM_STATE, N_OPTUNA_TRIALS, CV_FOLDS,
    get_version_paths, ensure_version_dirs,
)

# Set up v2 artifact paths
V2 = ensure_version_dirs("v2")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.3)

print("v2 artifact paths:")
for k, v in V2.items():
    print(f"  {k}: {v}")
print("\\nSetup complete.")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 2: Load data header
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 1. Load Feature Dataset"))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 3: Load data
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
features_df = pd.read_csv(ARTIFACTS_DIR / "battery_features.csv")
print(f"Dataset shape: {features_df.shape}")
print(f"Columns: {list(features_df.columns)}")

# Select available features
available_cols = [c for c in FEATURE_COLS_SCALAR if c in features_df.columns]
print(f"\\nUsing {len(available_cols)} features: {available_cols}")

# Drop rows with NaN in key columns
features_df = features_df.dropna(subset=available_cols + [TARGET_SOH, TARGET_RUL]).copy()
print(f"After dropping NaN: {len(features_df)} samples")
print(f"Batteries: {sorted(features_df['battery_id'].unique())}")
print(f"SOH range: {features_df[TARGET_SOH].min():.1f}% — {features_df[TARGET_SOH].max():.1f}%")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 4: Split header
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", """\
## 2. Intra-Battery Chronological Split (v2 Key Change)

**v1 bug:** `group_battery_split()` held out entire batteries → model had to extrapolate to unseen batteries → poor real-world SOH predictions.

**v2 fix:** For each battery, sort by `cycle_number`, use **first 80%** of cycles for training and **last 20%** for testing. This ensures:
- The model sees early degradation patterns from ALL batteries
- Testing is on later (more degraded) cycles — realistic deployment scenario
- No temporal data leakage (test is always chronologically after train)"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 5: Split logic
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# v2: Intra-battery chronological split
# For each battery, sort by cycle_number, first 80% -> train, last 20% -> test

train_frames, test_frames = [], []

for bid in sorted(features_df["battery_id"].unique()):
    bat_df = features_df[features_df["battery_id"] == bid].sort_values("cycle_number")
    n = len(bat_df)
    split_idx = int(n * 0.8)
    train_frames.append(bat_df.iloc[:split_idx])
    test_frames.append(bat_df.iloc[split_idx:])

train_df = pd.concat(train_frames, ignore_index=True)
test_df = pd.concat(test_frames, ignore_index=True)

print(f"Train: {len(train_df)} samples from {train_df['battery_id'].nunique()} batteries")
print(f"Test:  {len(test_df)} samples from {test_df['battery_id'].nunique()} batteries")
print(f"\\nTrain SOH: {train_df[TARGET_SOH].min():.1f}% — {train_df[TARGET_SOH].max():.1f}% (mean {train_df[TARGET_SOH].mean():.1f}%)")
print(f"Test  SOH: {test_df[TARGET_SOH].min():.1f}% — {test_df[TARGET_SOH].max():.1f}% (mean {test_df[TARGET_SOH].mean():.1f}%)")

# Show per-battery split info
split_info = []
for bid in sorted(features_df["battery_id"].unique()):
    n_train = len(train_df[train_df["battery_id"] == bid])
    n_test = len(test_df[test_df["battery_id"] == bid])
    split_info.append({"battery": bid, "train": n_train, "test": n_test, "total": n_train + n_test})
pd.DataFrame(split_info).set_index("battery")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 6: Scaling header
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 3. Feature Scaling & Target Extraction"))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 7: Scale features
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# Fit StandardScaler on training data, transform both splits
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[available_cols].values)
X_test = scaler.transform(test_df[available_cols].values)

# Save scalers to v2 paths
joblib.dump(scaler, V2["scalers"] / "features_standard.joblib")
print(f"StandardScaler saved to v2 scalers")

# Also fit MinMaxScaler (needed by some API endpoints)
mm_scaler = MinMaxScaler()
mm_scaler.fit(train_df[available_cols].values)
joblib.dump(mm_scaler, V2["scalers"] / "features_minmax.joblib")
print(f"MinMaxScaler saved to v2 scalers")

# Also save a "linear_scaler" alias for API compatibility
joblib.dump(scaler, V2["scalers"] / "linear_scaler.joblib")
print(f"Linear scaler alias saved to v2 scalers")

# Extract targets
y_train_soh = train_df[TARGET_SOH].values
y_test_soh  = test_df[TARGET_SOH].values
y_train_rul = train_df[TARGET_RUL].values
y_test_rul  = test_df[TARGET_RUL].values

print(f"\\nX_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"y_train SOH: [{y_train_soh.min():.1f}, {y_train_soh.max():.1f}]  mean={y_train_soh.mean():.1f}")
print(f"y_test  SOH: [{y_test_soh.min():.1f}, {y_test_soh.max():.1f}]  mean={y_test_soh.mean():.1f}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 8: SOH Training header
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", """\
## 4. SOH Regression — Train All Models

Training 12 models directly (not via `regressors.py`) so we can save checkpoints to v2 paths."""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 9: Ridge, Lasso, ElasticNet
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# Storage for all results and trained models
results_soh = {}
trained_models = {}

# ── Ridge ────────────────────────────────────────────────────────────────────
print("Training Ridge...")
model_ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
model_ridge.fit(X_train, y_train_soh)
y_pred = model_ridge.predict(X_test)
results_soh["Ridge"] = regression_metrics(y_test_soh, y_pred)
results_soh["Ridge"]["within_5pct"] = tolerance_accuracy(y_test_soh, y_pred, 5.0)
trained_models["Ridge"] = model_ridge
print(f"  R2={results_soh['Ridge']['R2']:.4f}  MAE={results_soh['Ridge']['MAE']:.4f}  +/-5%={results_soh['Ridge']['within_5pct']:.3f}")

# ── Lasso ────────────────────────────────────────────────────────────────────
print("Training Lasso...")
model_lasso = Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000)
model_lasso.fit(X_train, y_train_soh)
y_pred = model_lasso.predict(X_test)
results_soh["Lasso"] = regression_metrics(y_test_soh, y_pred)
results_soh["Lasso"]["within_5pct"] = tolerance_accuracy(y_test_soh, y_pred, 5.0)
trained_models["Lasso"] = model_lasso
print(f"  R2={results_soh['Lasso']['R2']:.4f}  MAE={results_soh['Lasso']['MAE']:.4f}  +/-5%={results_soh['Lasso']['within_5pct']:.3f}")

# ── ElasticNet ───────────────────────────────────────────────────────────────
print("Training ElasticNet...")
model_en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000)
model_en.fit(X_train, y_train_soh)
y_pred = model_en.predict(X_test)
results_soh["ElasticNet"] = regression_metrics(y_test_soh, y_pred)
results_soh["ElasticNet"]["within_5pct"] = tolerance_accuracy(y_test_soh, y_pred, 5.0)
trained_models["ElasticNet"] = model_en
print(f"  R2={results_soh['ElasticNet']['R2']:.4f}  MAE={results_soh['ElasticNet']['MAE']:.4f}  +/-5%={results_soh['ElasticNet']['within_5pct']:.3f}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 10: KNN models
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# ── KNN (multiple k values) ──────────────────────────────────────────────────
for k in [5, 10, 20]:
    print(f"Training KNN (k={k})...")
    model_knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    model_knn.fit(X_train, y_train_soh)
    y_pred = model_knn.predict(X_test)
    name = f"KNN-{k}"
    results_soh[name] = regression_metrics(y_test_soh, y_pred)
    results_soh[name]["within_5pct"] = tolerance_accuracy(y_test_soh, y_pred, 5.0)
    trained_models[name] = model_knn
    print(f"  R2={results_soh[name]['R2']:.4f}  MAE={results_soh[name]['MAE']:.4f}  +/-5%={results_soh[name]['within_5pct']:.3f}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 11: SVR
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# ── SVR ──────────────────────────────────────────────────────────────────────
print("Training SVR (RBF kernel)...")
model_svr = SVR(kernel="rbf", C=10.0, gamma="scale")
model_svr.fit(X_train, y_train_soh)
y_pred = model_svr.predict(X_test)
results_soh["SVR"] = regression_metrics(y_test_soh, y_pred)
results_soh["SVR"]["within_5pct"] = tolerance_accuracy(y_test_soh, y_pred, 5.0)
trained_models["SVR"] = model_svr
print(f"  R2={results_soh['SVR']['R2']:.4f}  MAE={results_soh['SVR']['MAE']:.4f}  +/-5%={results_soh['SVR']['within_5pct']:.3f}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 12: Random Forest
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# ── Random Forest ────────────────────────────────────────────────────────────
print("Training Random Forest (500 trees)...")
model_rf = RandomForestRegressor(
    n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
)
model_rf.fit(X_train, y_train_soh)
y_pred = model_rf.predict(X_test)
results_soh["RandomForest"] = regression_metrics(y_test_soh, y_pred)
results_soh["RandomForest"]["within_5pct"] = tolerance_accuracy(y_test_soh, y_pred, 5.0)
trained_models["RandomForest"] = model_rf
print(f"  R2={results_soh['RandomForest']['R2']:.4f}  MAE={results_soh['RandomForest']['MAE']:.4f}  +/-5%={results_soh['RandomForest']['within_5pct']:.3f}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 13: ExtraTrees
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# ── ExtraTrees (new in v2) ───────────────────────────────────────────────────
print("Training ExtraTrees (500 trees)...")
model_et = ExtraTreesRegressor(
    n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
)
model_et.fit(X_train, y_train_soh)
y_pred = model_et.predict(X_test)
results_soh["ExtraTrees"] = regression_metrics(y_test_soh, y_pred)
results_soh["ExtraTrees"]["within_5pct"] = tolerance_accuracy(y_test_soh, y_pred, 5.0)
trained_models["ExtraTrees"] = model_et
print(f"  R2={results_soh['ExtraTrees']['R2']:.4f}  MAE={results_soh['ExtraTrees']['MAE']:.4f}  +/-5%={results_soh['ExtraTrees']['within_5pct']:.3f}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 14: GradientBoosting
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# ── GradientBoosting (new in v2) ─────────────────────────────────────────────
print("Training GradientBoosting...")
model_gb = GradientBoostingRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, random_state=RANDOM_STATE,
)
model_gb.fit(X_train, y_train_soh)
y_pred = model_gb.predict(X_test)
results_soh["GradientBoosting"] = regression_metrics(y_test_soh, y_pred)
results_soh["GradientBoosting"]["within_5pct"] = tolerance_accuracy(y_test_soh, y_pred, 5.0)
trained_models["GradientBoosting"] = model_gb
print(f"  R2={results_soh['GradientBoosting']['R2']:.4f}  MAE={results_soh['GradientBoosting']['MAE']:.4f}  +/-5%={results_soh['GradientBoosting']['within_5pct']:.3f}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 15: XGBoost with Optuna
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# ── XGBoost with Optuna HPO ─────────────────────────────────────────────────
import optuna
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"Training XGBoost (Optuna HPO, {N_OPTUNA_TRIALS} trials)...")

def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    model = XGBRegressor(**params, tree_method="hist", random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    scores = cross_val_score(model, X_train, y_train_soh, cv=CV_FOLDS, scoring="neg_mean_absolute_error")
    return -scores.mean()

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(xgb_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

model_xgb = XGBRegressor(**study_xgb.best_params, tree_method="hist", random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
model_xgb.fit(X_train, y_train_soh)
y_pred = model_xgb.predict(X_test)
results_soh["XGBoost"] = regression_metrics(y_test_soh, y_pred)
results_soh["XGBoost"]["within_5pct"] = tolerance_accuracy(y_test_soh, y_pred, 5.0)
trained_models["XGBoost"] = model_xgb
print(f"  Best params: {study_xgb.best_params}")
print(f"  R2={results_soh['XGBoost']['R2']:.4f}  MAE={results_soh['XGBoost']['MAE']:.4f}  +/-5%={results_soh['XGBoost']['within_5pct']:.3f}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 16: LightGBM with Optuna
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# ── LightGBM with Optuna HPO ────────────────────────────────────────────────
from lightgbm import LGBMRegressor

print(f"Training LightGBM (Optuna HPO, {N_OPTUNA_TRIALS} trials)...")

def lgbm_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }
    model = LGBMRegressor(**params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
    scores = cross_val_score(model, X_train, y_train_soh, cv=CV_FOLDS, scoring="neg_mean_absolute_error")
    return -scores.mean()

study_lgbm = optuna.create_study(direction="minimize")
study_lgbm.optimize(lgbm_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

model_lgbm = LGBMRegressor(**study_lgbm.best_params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
model_lgbm.fit(X_train, y_train_soh)
y_pred = model_lgbm.predict(X_test)
results_soh["LightGBM"] = regression_metrics(y_test_soh, y_pred)
results_soh["LightGBM"]["within_5pct"] = tolerance_accuracy(y_test_soh, y_pred, 5.0)
trained_models["LightGBM"] = model_lgbm
print(f"  Best params: {study_lgbm.best_params}")
print(f"  R2={results_soh['LightGBM']['R2']:.4f}  MAE={results_soh['LightGBM']['MAE']:.4f}  +/-5%={results_soh['LightGBM']['within_5pct']:.3f}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 17: Save all models
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# ── Save all SOH models to v2 paths ─────────────────────────────────────────
save_dir = V2["models_classical"]
model_filenames = {
    "Ridge": "ridge.joblib",
    "Lasso": "lasso.joblib",
    "ElasticNet": "elasticnet.joblib",
    "KNN-5": "knn_k5.joblib",
    "KNN-10": "knn_k10.joblib",
    "KNN-20": "knn_k20.joblib",
    "SVR": "svr.joblib",
    "RandomForest": "random_forest.joblib",
    "ExtraTrees": "extra_trees.joblib",
    "GradientBoosting": "gradient_boosting.joblib",
    "XGBoost": "xgboost.joblib",
    "LightGBM": "lightgbm.joblib",
}

for name, fname in model_filenames.items():
    path = save_dir / fname
    joblib.dump(trained_models[name], path)
    print(f"  Saved {name} -> {path}")

# Also save Optuna best params
joblib.dump(study_xgb.best_params, save_dir / "xgboost_best_params.joblib")
joblib.dump(study_lgbm.best_params, save_dir / "lightgbm_best_params.joblib")
print(f"\\nAll {len(model_filenames)} SOH models saved to {save_dir}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 18: Comparison header
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 5. SOH Model Comparison & Results"))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 19: Summary table + bar chart
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
summary = build_summary_table(results_soh)
summary = summary.sort_values("R2", ascending=False)
display(summary.round(4))

# Save to v2
results_path = V2["results"] / "classical_soh_results.csv"
summary.to_csv(results_path)
print(f"Saved to {results_path}")

# Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
models = summary.index.tolist()
x = np.arange(len(models))

axes[0].barh(x, summary["MAE"], color="steelblue")
axes[0].set_yticks(x); axes[0].set_yticklabels(models)
axes[0].set_xlabel("MAE (% SOH)"); axes[0].set_title("Mean Absolute Error")
axes[0].invert_yaxis()

axes[1].barh(x, summary["R2"], color="seagreen")
axes[1].set_yticks(x); axes[1].set_yticklabels(models)
axes[1].set_xlabel("R2"); axes[1].set_title("R2 Score")
axes[1].invert_yaxis()

axes[2].barh(x, summary["within_5pct"] * 100, color="coral")
axes[2].set_yticks(x); axes[2].set_yticklabels(models)
axes[2].set_xlabel("Accuracy (%)"); axes[2].set_title("Within +/-5% SOH Accuracy")
axes[2].axvline(x=95, color="red", linestyle="--", linewidth=1.5, label="95% target")
axes[2].legend()
axes[2].invert_yaxis()

plt.suptitle("Classical ML Comparison — SOH Prediction (v2)", fontsize=15, fontweight="bold")
plt.tight_layout()
save_fig(fig, "classical_soh_comparison_v2")
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 20: Best model analysis
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# Best model analysis
best_model_name = summary.index[0]
print(f"Best model: {best_model_name}")
print(f"  R2 = {summary.loc[best_model_name, 'R2']:.4f}")
print(f"  MAE = {summary.loc[best_model_name, 'MAE']:.4f}% SOH")
print(f"  +/-5% accuracy = {summary.loc[best_model_name, 'within_5pct']*100:.1f}%")

best = trained_models[best_model_name]
y_pred_best = best.predict(X_test)

fig = plot_actual_vs_predicted(
    y_test_soh, y_pred_best, label="SOH (%)",
    model_name=f"{best_model_name} (v2)", save_name="classical_best_actual_vs_pred_v2",
)
plt.show()

fig = plot_residuals(
    y_test_soh, y_pred_best, model_name=f"{best_model_name} (v2)",
    save_name="classical_best_residuals_v2",
)
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 21: Accuracy gate header
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", """\
## 6. Within +/-5% SOH Accuracy Gate

**Requirement:** At least 3 models must achieve >= 95% within-+/-5% accuracy.

This verifies the v2 intra-battery split yields production-quality predictions."""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 22: Accuracy gate check
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# Check which models pass the 95% within-+/-5% gate
GATE_THRESHOLD = 0.95  # 95%

passed = []
failed = []
for name in summary.index:
    acc = summary.loc[name, "within_5pct"]
    status = "PASS" if acc >= GATE_THRESHOLD else "FAIL"
    if acc >= GATE_THRESHOLD:
        passed.append(name)
    else:
        failed.append(name)
    print(f"  {name:20s}  within_5pct = {acc*100:5.1f}%  [{status}]")

print(f"\\n{'='*60}")
print(f"Models passing >=95% gate: {len(passed)} / {len(summary)}")
print(f"  Passed: {passed}")
if failed:
    print(f"  Failed: {failed}")

# Assert at least 3 models pass
assert len(passed) >= 3, f"GATE FAILED: Only {len(passed)} models achieved >=95% within-5% accuracy!"
print(f"\\nGATE PASSED: {len(passed)} models exceed 95% within +/-5% SOH accuracy")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 23: Per-battery header
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", """\
## 7. Per-Battery Error Analysis

How well does the best model perform on each individual battery's test cycles?"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 24: Per-battery metrics
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# Per-battery evaluation for the best model
best = trained_models[best_model_name]
y_pred_all = best.predict(X_test)

per_bat = per_battery_evaluation(
    y_test_soh, y_pred_all,
    battery_ids=test_df["battery_id"].values,
)
per_bat = per_bat.sort_values("MAE")
display(per_bat.round(4))

# Heatmap of per-battery MAE for top 5 models
top5 = summary.index[:5].tolist()
heatmap_data = {}
for mname in top5:
    m = trained_models[mname]
    yp = m.predict(X_test)
    pb = per_battery_evaluation(y_test_soh, yp, test_df["battery_id"].values)
    for _, row in pb.iterrows():
        if row["battery_id"] not in heatmap_data:
            heatmap_data[row["battery_id"]] = {}
        heatmap_data[row["battery_id"]][mname] = row["MAE"]

heatmap_df = pd.DataFrame(heatmap_data).T.sort_index()
fig, ax = plt.subplots(figsize=(14, max(6, len(heatmap_df) * 0.6)))
sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "MAE (% SOH)"})
ax.set_title("Per-Battery MAE — Top 5 Models (v2)", fontweight="bold")
ax.set_xlabel("Model")
ax.set_ylabel("Battery ID")
plt.tight_layout()
save_fig(fig, "per_battery_mae_heatmap_v2")
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 25: SHAP header
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 8. SHAP Feature Importance Analysis"))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 26: SHAP analysis
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
# SHAP for XGBoost (v2)
print("Computing SHAP values for XGBoost (v2 model)...")
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

plt.sca(axes[0])
shap.summary_plot(shap_values, X_test, feature_names=available_cols, show=False, max_display=12)
axes[0].set_title("SHAP Beeswarm — XGBoost (v2)", fontsize=13)

plt.sca(axes[1])
shap.summary_plot(shap_values, X_test, feature_names=available_cols, plot_type="bar", show=False, max_display=12)
axes[1].set_title("SHAP Feature Importance — XGBoost (v2)", fontsize=13)

plt.tight_layout()
save_fig(fig, "shap_xgboost_soh_v2")
plt.show()

# Also compute for Random Forest
print("\\nComputing SHAP values for Random Forest (v2 model)...")
explainer_rf = shap.TreeExplainer(model_rf)
shap_values_rf = explainer_rf.shap_values(X_test)

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values_rf, X_test, feature_names=available_cols, plot_type="bar", show=False, max_display=12)
ax.set_title("SHAP Feature Importance — Random Forest (v2)", fontsize=13)
plt.tight_layout()
save_fig(fig, "shap_rf_soh_v2")
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 27: RUL header
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", """\
## 9. RUL Regression

Train top models (RF, ExtraTrees, XGBoost, LightGBM) on remaining-useful-life target."""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 28: RUL training
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
results_rul = {}
rul_models = {}

rul_configs = [
    ("RandomForest", RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)),
    ("ExtraTrees",   ExtraTreesRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)),
    ("XGBoost",      XGBRegressor(**study_xgb.best_params, tree_method="hist", random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)),
    ("LightGBM",     LGBMRegressor(**study_lgbm.best_params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)),
]

for name, model in rul_configs:
    print(f"Training {name} for RUL...")
    model.fit(X_train, y_train_rul)
    y_pred = model.predict(X_test)
    results_rul[name] = regression_metrics(y_test_rul, y_pred)
    results_rul[name]["tolerance_acc_5cyc"] = tolerance_accuracy(y_test_rul, y_pred, 5.0)
    rul_models[name] = model
    print(f"  R2={results_rul[name]['R2']:.4f}  MAE={results_rul[name]['MAE']:.4f}")

rul_summary = build_summary_table(results_rul)
display(rul_summary.round(4))
rul_summary.to_csv(V2["results"] / "classical_rul_results.csv")
print(f"\\nSaved to {V2['results'] / 'classical_rul_results.csv'}")

# Save best RUL model
best_rul_name = rul_summary.sort_values("R2", ascending=False).index[0]
joblib.dump(rul_models[best_rul_name], V2["models_classical"] / "best_rul_model.joblib")
print(f"Best RUL model ({best_rul_name}) saved.")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 29: Classification header
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", "## 10. Degradation State Classification"))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 30: Classification
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("code", """\
y_train_cls = train_df[TARGET_DEGRADATION].values.astype(int)
y_test_cls = test_df[TARGET_DEGRADATION].values.astype(int)

print("Training Random Forest Classifier...")
clf_rf = train_rf_classifier(X_train, y_train_cls)
metrics_rf = evaluate_classifier(clf_rf, X_test, y_test_cls, "RF")
print(metrics_rf["classification_report"])

print("Training XGBoost Classifier...")
clf_xgb = train_xgb_classifier(X_train, y_train_cls)
metrics_xgb = evaluate_classifier(clf_xgb, X_test, y_test_cls, "XGB")
print(metrics_xgb["classification_report"])

# Save classifiers to v2
joblib.dump(clf_rf, V2["models_classical"] / "rf_classifier.joblib")
joblib.dump(clf_xgb, V2["models_classical"] / "xgb_classifier.joblib")
print(f"Classifiers saved to {V2['models_classical']}")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, metrics) in zip(axes, [("Random Forest", metrics_rf), ("XGBoost", metrics_xgb)]):
    cm = metrics["confusion_matrix"]
    disp = ConfusionMatrixDisplay(cm, display_labels=DEGRADATION_LABELS[:cm.shape[0]])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{name} — Confusion Matrix (v2)", fontweight="bold")

plt.tight_layout()
save_fig(fig, "classification_confusion_matrices_v2")
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 31: Summary
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(make_cell("markdown", """\
## 11. v2 Summary

### Key Results

| Metric | v1 (cross-battery split) | v2 (intra-battery chronological) |
|--------|--------------------------|----------------------------------|
| Split strategy | Hold out entire batteries | 80/20 per battery (temporal) |
| Best R2 | ~0.957 (RandomForest) | Expected >= 0.99 |
| +/-5% accuracy | Not measured | **>= 95% target** |
| Models | 10 | **12** (+ExtraTrees, +GradientBoosting) |
| Artifact path | `artifacts/models/classical/` | `artifacts/v2/models/classical/` |

### Why intra-battery split works better
- The model learns early degradation patterns from **all** batteries
- Testing on later cycles is a realistic "interpolation" scenario
- In production, we track a battery over time — the same battery's early history is available
- This matches how the API will actually be used

### What's saved to v2
- 12 SOH regression models + 2 Optuna param sets
- RUL models for top performers
- 2 classification models
- StandardScaler + MinMaxScaler + linear_scaler alias
- Results CSVs, figures, SHAP plots

All model checkpoints saved to `artifacts/v2/models/classical/`."""))

# ═══════════════════════════════════════════════════════════════════════════════
# Build the notebook
# ═══════════════════════════════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "03_classical_ml.ipynb")
out_path = os.path.abspath(out_path)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to {out_path}")
print(f"Total cells: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code:     {sum(1 for c in cells if c['cell_type'] == 'code')}")
