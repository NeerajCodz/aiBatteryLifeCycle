# Models Documentation

## Overview

The system trains 24+ models across three generations, then selects the best via unified evaluation.
All model metadata, metrics, and configuration are stored in `artifacts/{version}/models.json` and
loaded dynamically by the registry — there is no hardcoded model catalog.

**Champion v1:** Random Forest (R² = 0.957, MAE = 4.78) — cross-battery group split, 12 features.
**Champion v2:** ExtraTrees (R² = 0.967, MAE = 1.17) — intra-battery chronological split, 12 features.
**Champion v3:** XGBoost (R² = 0.987, MAE = 0.92, 99.5% within ±5%) — cross-battery grouped split, 18 features.

---

## Model Versioning

Models are organized into three generations. Each version has its own `models.json` that defines
the available models, their scores, feature set, scalers, and ensemble configuration.

| Generation | Version | Family | Features | Split | Champion |
|:---:|:---:|---|:---:|---|---|
| **v1** | 1.0 | Classical ML | 12 | Cross-battery group | Random Forest |
| **v2** | 2.0 | Classical + Deep | 12 | Intra-battery chrono | ExtraTrees |
| **v3** | 3.0 | Classical + Deep + Ensemble | 18 | Cross-battery grouped | XGBoost |

### BestEnsemble (v3.0)

The weighted-average ensemble combines the top classical models (R²-proportional weights):

$$\hat{y} = \frac{\sum_{i} w_i \cdot \hat{y}_i}{\sum_{i} w_i}$$

Components and weights are defined in `artifacts/v3/models.json` and loaded dynamically.
v3 ensemble components: XGBoost, RandomForest, ExtraTrees, VanillaLSTM, TFT.

---

## v3 Results Summary (Production)

| Rank | Model | R² | MAE | Within ±5% | Family |
|------|-------|----|-----|------------|--------|
| 1 | XGBoost | 0.9866 | 0.92 | 99.5% | Classical |
| 2 | GradientBoosting | 0.9860 | 0.94 | 99.4% | Classical |
| 3 | LightGBM | 0.9826 | 1.05 | 99.0% | Classical |
| 4 | Random Forest | 0.9814 | 1.10 | 98.8% | Classical |
| 5 | Best Ensemble | 0.9810 | 1.02 | 99.2% | Ensemble |
| 6 | ExtraTrees | 0.9701 | 1.38 | 97.8% | Classical |

### v3 Classification Quality (Degradation Classes)

The v3 notebooks now also report degradation-state quality by binning SOH into
4 classes (`<70`, `70-80`, `80-90`, `>=90`) and computing macro/weighted F1.

| Model | F1 Macro | F1 Weighted | Notes |
|------|----------|-------------|-------|
| GradientBoosting | ~0.89 | ~0.94 | Best classical class balance |
| XGBoost | ~0.92 | ~0.95 | Strong boundary discrimination |
| Best Ensemble | tracked in NB08/NB09 | tracked in NB08/NB09 | Mixed classical + deep |

## v2 Results Summary

| Rank | Model | R² | MAE | Within ±5% | Family |
|------|-------|----|-----|------------|--------|
| 1 | ExtraTrees | 0.9673 | 1.17 | 99.1% | Classical |
| 2 | LightGBM | 0.9582 | 1.38 | 98.4% | Classical |
| 3 | SVR | 0.9474 | 1.67 | 95.1% | Classical |
| 4 | TFT | 0.881 | 3.93 | — | Transformer |
| 5 | BatteryGPT | 0.881 | 10.71 | — | Transformer |

## v1 Results Summary (Legacy)

| Rank | Model | R² | MAE | Family |
|------|-------|----|-----|--------|
| 1 | Random Forest | 0.957 | 4.78 | Classical |
| 2 | LightGBM | 0.928 | 5.53 | Classical |
| 3 | XGBoost | 0.847 | 8.06 | Classical |
| 4 | SVR | 0.805 | 7.56 | Classical |

---

## 1. Classical Machine Learning

### 1.1 Linear Models

| Model | Regularization | Key Hyperparameters |
|-------|---------------|---------------------|
| Ridge | L2 | α (cross-validated) |
| Lasso | L1 | α (cross-validated) |
| ElasticNet | L1 + L2 | α, l1_ratio |

### 1.2 Instance-Based
- **KNN** (k=3, 5, 7): Distance-weighted, Minkowski metric

### 1.3 Kernel
- **SVR** (RBF): C, γ, ε via grid search

### 1.4 Tree Ensembles
- **Random Forest:** 500 trees, max_depth=None
- **XGBoost:** 100 Optuna trials, objective=reg:squarederror
- **LightGBM:** 100 Optuna trials, metric=MAE

All classical models use **5-fold battery-grouped CV** for validation.

---

## 2. Deep Learning — LSTM/GRU Family

Built with PyTorch. Input: sliding windows of 32 cycles × 12 features.

### 2.1 Vanilla LSTM
- 2 layers, hidden_dim=128, dropout=0.2
- MAE loss, Adam optimizer

### 2.2 Bidirectional LSTM
- Same as Vanilla but processes sequences in both directions
- Doubles hidden representation

### 2.3 GRU
- 2-layer GRU (fewer parameters than LSTM)
- Simpler gating mechanism (reset + update gates)

### 2.4 Attention LSTM
- 3-layer LSTM + Additive Attention mechanism
- Learns to weight important time steps
- Attention weights are interpretable

### Training Protocol
- **Optimizer:** Adam (lr=1e-3)
- **Scheduler:** CosineAnnealingLR
- **Early stopping:** patience=20
- **Gradient clipping:** max_norm=1.0
- **Uncertainty:** MC Dropout (50 forward passes, p=0.2)

---

## 3. Transformer Architectures

### 3.1 BatteryGPT
- Nano GPT-style decoder-only Transformer
- d_model=64, nhead=4, 2 layers
- Positional encoding + causal mask
- Lightweight (~50K parameters)

### 3.2 Temporal Fusion Transformer (TFT)
- Variable Selection Network for feature importance
- Gated Residual Networks for non-linear processing
- Multi-head attention with interpretable weights
- Originally designed for multi-horizon forecasting

### 3.3 iTransformer (Inverted)
- Inverts the attention axis: attends across features, not time
- Feature-wise multi-head attention + temporal convolution
- Built with TensorFlow/Keras

### 3.4 Physics-Informed iTransformer
- Dual-head: primary SOH head + auxiliary physics head (ΔQ prediction)
- Joint loss: L = L_soh + λ × L_physics (λ=0.3)
- Physics constraint regularizes learning

### 3.5 Dynamic-Graph iTransformer
- Adds Dynamic Graph Convolution layer
- Learns inter-feature adjacency matrix dynamically
- Fuses local (graph) and global (attention) representations

---

## 4. VAE-LSTM

- **Encoder:** 2-layer Bi-LSTM → μ, log σ² (latent_dim=16)
- **Reparameterization:** z = μ + σ · ε
- **Decoder:** 2-layer LSTM → reconstructed sequences
- **Health Head:** MLP(z) → SOH
- **Loss:** L_recon + β · KL + L_soh (β annealing over 30 epochs)
- **Anomaly Detection:** 3σ threshold on reconstruction error

---

## 5. Ensemble Methods

### 5.1 Stacking Ensemble
- Base models generate out-of-fold predictions
- Ridge regression as meta-learner
- Combines diverse model predictions

### 5.2 Weighted Average Ensemble (v2.6.0)
- Optimizes weights via L-BFGS-B (minimize MAE)
- Constraint: weights sum to 1, all ≥ 0
- Usually achieves best overall performance
- Registered as a v2 patch — no separate generation needed

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | mean(\|y - ŷ\|) | Average absolute error |
| MSE | mean((y - ŷ)²) | Penalizes large errors |
| RMSE | √MSE | Same units as target |
| R² | 1 - SS_res/SS_tot | Explained variance (1.0 = perfect) |
| MAPE | mean(\|y - ŷ\|/y) × 100 | Percentage error |
| Tolerance Accuracy | fraction within ±2% | Practical precision |

---

## 6. Vectorized Simulation (`predict_array`)

### Overview

The `ModelRegistry.predict_array(X: np.ndarray, model_name: str) -> np.ndarray` method enables
batch prediction for the simulation pipeline without Python-level loops.

- **Input:** `X` — shape `(N, n_features)` where N is the number of simulation steps
- **Output:** flat `np.ndarray` of shape `(N,)` — SOH predictions for each step
- Automatically loads and applies the correct scaler via `_load_scaler(model_name)`
- Dispatches to the correct backend (sklearn `.predict()`, XGBoost/LightGBM `.predict()`, PyTorch `.forward()` batch, Keras `.predict()`)

### Simulation Pipeline (`api/routers/simulate.py`)

Each simulated battery follows this vectorized path:

1. **Vectorized feature matrix** assembled all at once using `np.arange` for cycle indices, scalar broadcasting for temperature/current/cutoff
2. **All engineered features** (SOC, cycle_norm, temp_norm, Δfeatures) computed column-by-column using numpy — no step loop
3. **`predict_array(X, model_name)`** called once per battery \u2192 entire SOH trajectory in one forward pass
4. **RUL** computed via `np.searchsorted` on the reversed-SOH array with the EOL threshold \u2192 O(log N) rather than O(N)
5. **Degradation state** classified by SOH thresholds using `np.select([soh > 0.9, soh > 0.8, soh > 0.7], [...])`

### Physics Fallback (Arrhenius)

When no ML model is selected, pure physics degradation uses Arrhenius kinetics:

$$Q_{\text{loss}} = A \cdot \exp\!\left(-\frac{E_a}{R \cdot T}\right) \cdot N^z$$

where $A = 31630$, $E_a = 17126\ \text{J/mol}$, $R = 8.314\ \text{J/(mol·K)}$, $z = 0.55$, and $T$ is temperature in Kelvin.

### Performance

Vectorization replaces an O(N·k) Python loop (N steps × k overhead) with:

- Feature assembly: one `np.column_stack` call
- Prediction: single framework forward pass
- RUL: `np.searchsorted` O(log N)

For a 1 000-cycle simulation of 10 batteries this is **10–50× faster** than the loop-based equivalent.
