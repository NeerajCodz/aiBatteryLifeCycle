"""
scripts.models.retrain_classical
=================================
Retrain all classical ML models on RAW (unscaled) features using v2 split.

Tree-based models (RF, XGB, LGB) are scale-invariant, so no scaler
is needed.  Linear models (Ridge, Lasso, ElasticNet) and SVR DO need
scaling; we fit + save their scaler alongside the models.

All artifacts saved to artifacts/v2/ structure.

Run with:
    python scripts/models/retrain_classical.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

from src.data.preprocessing import FEATURE_COLS_SCALAR, TARGET_SOH
from src.utils.config import get_version_paths, ensure_version_dirs, RANDOM_STATE, METADATA_PATH, DATA_DIR

# ── Config ────────────────────────────────────────────────────────────────────
XGB_TRIALS = 20      # reduced from 100 for faster retraining
LGB_TRIALS = 20
CV_FOLDS   = 3

v2 = ensure_version_dirs("v2")
SAVE_DIR = v2["models_classical"]

TARGET = TARGET_SOH

# ── Data ─────────────────────────────────────────────────────────────────────
print("Loading data ...")
import pandas as pd
meta = pd.read_csv(METADATA_PATH)
frames = []
for _, row in meta.iterrows():
    fp = DATA_DIR / row['filename']
    df = pd.read_csv(fp)
    df['battery_id'] = row['battery_id']
    frames.append(df)

full = pd.concat(frames, ignore_index=True)
required = FEATURE_COLS_SCALAR + [TARGET, 'battery_id', 'cycle_number']
full = full.dropna(subset=[c for c in required if c in full.columns]).copy()

# Intra-battery chronological split (v2 split)
train_parts, test_parts = [], []
for _, grp in full.groupby('battery_id'):
    grp = grp.sort_values('cycle_number')
    cut = int(len(grp) * 0.8)
    train_parts.append(grp.iloc[:cut])
    test_parts.append(grp.iloc[cut:])

train_df = pd.concat(train_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

print(f"  Train: {len(train_df)} samples ({train_df.battery_id.nunique()} batteries)")
print(f"  Test:  {len(test_df)} samples ({test_df.battery_id.nunique()} batteries)")

X_train_raw = train_df[FEATURE_COLS_SCALAR].values
X_test_raw  = test_df[FEATURE_COLS_SCALAR].values
y_train = train_df[TARGET].values
y_test  = test_df[TARGET].values

# StandardScaler for linear + SVR models only
lin_scaler = StandardScaler()
X_train_sc = lin_scaler.fit_transform(X_train_raw)
X_test_sc  = lin_scaler.transform(X_test_raw)
lin_scaler_path = v2["scalers"] / "linear_scaler.joblib"
joblib.dump(lin_scaler, lin_scaler_path)
print(f"  Linear scaler saved → {lin_scaler_path.relative_to(Path.cwd())}")



def _save(model, name: str) -> None:
    p = SAVE_DIR / f"{name}.joblib"
    joblib.dump(model, p)


def _eval(model, X, y, name: str) -> None:
    p = model.predict(X)
    r2  = r2_score(y, p)
    mae = mean_absolute_error(y, p)
    print(f"  {name:25s}  R²={r2:.4f}  MAE={mae:.4f}")


# ── Tree-based models (raw features) ─────────────────────────────────────────
print("\n--- Random Forest ---")
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_raw, y_train)
_save(rf, "random_forest")
_eval(rf, X_test_raw, y_test, "Random Forest")

print("--- XGBoost ---")
try:
    import optuna
    from xgboost import XGBRegressor
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _xgb_obj(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        }
        from sklearn.model_selection import cross_val_score
        m = XGBRegressor(**params, tree_method="hist", random_state=RANDOM_STATE,
                         verbosity=0, n_jobs=-1)
        return -cross_val_score(m, X_train_raw, y_train, cv=CV_FOLDS,
                                scoring="neg_mean_absolute_error").mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(_xgb_obj, n_trials=XGB_TRIALS, show_progress_bar=False)
    best_xgb = XGBRegressor(**study.best_params, tree_method="hist",
                             random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    best_xgb.fit(X_train_raw, y_train)
    _save(best_xgb, "xgboost")
    _eval(best_xgb, X_test_raw, y_test, "XGBoost")
except Exception as e:
    print(f"  XGBoost failed: {e}")

print("--- LightGBM ---")
try:
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _lgb_obj(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        }
        from sklearn.model_selection import cross_val_score
        m = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
        return -cross_val_score(m, X_train_raw, y_train, cv=CV_FOLDS,
                                scoring="neg_mean_absolute_error").mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(_lgb_obj, n_trials=LGB_TRIALS, show_progress_bar=False)
    best_lgb = lgb.LGBMRegressor(**study.best_params, random_state=RANDOM_STATE,
                                  verbose=-1, n_jobs=-1)
    best_lgb.fit(X_train_raw, y_train)
    _save(best_lgb, "lightgbm")
    _eval(best_lgb, X_test_raw, y_test, "LightGBM")
except Exception as e:
    print(f"  LightGBM failed: {e}")

# ── Linear / distance models (scaled features) ───────────────────────────────
print("\n--- Ridge ---")
ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
ridge.fit(X_train_sc, y_train)
_save(ridge, "ridge")
_eval(ridge, X_test_sc, y_test, "Ridge")

print("--- Lasso ---")
lasso = Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000)
lasso.fit(X_train_sc, y_train)
_save(lasso, "lasso")
_eval(lasso, X_test_sc, y_test, "Lasso")

print("--- ElasticNet ---")
en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000)
en.fit(X_train_sc, y_train)
_save(en, "elasticnet")
_eval(en, X_test_sc, y_test, "ElasticNet")

print("--- SVR ---")
svr = SVR(kernel="rbf", C=10.0, gamma="scale")
svr.fit(X_train_sc, y_train)
_save(svr, "svr")
_eval(svr, X_test_sc, y_test, "SVR")

print("--- KNN k=5 ---")
knn5 = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)
knn5.fit(X_train_sc, y_train)
_save(knn5, "knn_k5")
_eval(knn5, X_test_sc, y_test, "KNN k=5")

print("--- KNN k=10 ---")
knn10 = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn10.fit(X_train_sc, y_train)
_save(knn10, "knn_k10")
_eval(knn10, X_test_sc, y_test, "KNN k=10")

print("--- KNN k=20 ---")
knn20 = KNeighborsRegressor(n_neighbors=20, weights="distance", n_jobs=-1)
knn20.fit(X_train_sc, y_train)
_save(knn20, "knn_k20")
_eval(knn20, X_test_sc, y_test, "KNN k=20")

# ── Save the sequence scaler for deep models ─────────────────────────────────
print("\n--- Deep sequence scaler ---")
data = np.load(str(ARTIFACTS_DIR / "battery_sequences.npz"), allow_pickle=True)
X_seq = data["X_multi"]     # (N, 32, n_features)
bids  = data["bids_multi"]

unique_bids = np.unique(bids)           # sorted numpy array — deterministic
rng = np.random.RandomState(42)
rng.shuffle(unique_bids)
n_train_seq = int(0.8 * len(unique_bids))
train_bats_seq = set(unique_bids[:n_train_seq])
train_mask = np.isin(bids, list(train_bats_seq))
X_seq_train = X_seq[train_mask]

n_s, seq_l, n_f = X_seq_train.shape
seq_scaler = StandardScaler().fit(X_seq_train.reshape(-1, n_f))
joblib.dump(seq_scaler, ARTIFACTS_DIR / "scalers" / "sequence_scaler.joblib")
print(f"  Sequence scaler saved → artifacts/scalers/sequence_scaler.joblib")
print(f"  Fit on {n_s} windows × {seq_l} timesteps, {n_f} features")
print(f"  Mean range: [{seq_scaler.mean_.min():.3f}, {seq_scaler.mean_.max():.3f}]")

print("\n✅ All classical models retrained on raw features.")
print("   Tree models: no scaler needed at inference.")
print("   Linear/KNN/SVR: use artifacts/scalers/linear_scaler.joblib")
print("   Deep models: use artifacts/scalers/sequence_scaler.joblib")
