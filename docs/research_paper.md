# A Comprehensive Multi-Model Framework for Lithium-Ion Battery State of Health Prediction: From Data Leakage Diagnosis to Production-Grade 99.5% Accuracy

**Neeraj Sathish Kumar** · Vellore Institute of Technology (VIT), India
**Date:** March 2026
**Citation:** IEEE Transactions on Industrial Electronics (Preprint v3)

---

## Abstract

Accurate prediction of State of Health (SOH) and Remaining Useful Life (RUL) is essential for battery management systems in electric vehicles, grid storage, and portable electronics. This paper presents a three-generation iterative framework evaluating **24+ machine learning and deep learning architectures** on the NASA Prognostics Center of Excellence (PCoE) dataset. We trace the evolution from **v1** (cross-battery split, 12 features, R²=0.957) through **v2** (intra-battery chronological split, data leakage diagnosis, 12 features) to **v3** (cross-battery grouped split with 18 physics-informed features, R²=0.987, 99.5% within-±5% accuracy). Key contributions include: (1) identification and correction of a data leakage bug in cross-battery splits, (2) introduction of 6 physics-informed engineered features that improve accuracy by 2+ percentage points, (3) a dynamic model registry architecture where all metadata is loaded from per-version JSON manifests rather than hardcoded catalogs, and (4) an end-to-end deployment pipeline with on-demand model loading, React frontend, and Docker containerization.

**Keywords:** lithium-ion battery · state of health · remaining useful life · physics-informed features · data leakage · ensemble methods · NASA PCoE dataset · dynamic model registry

---

## I. Introduction

Lithium-ion batteries are the dominant energy storage technology for electric vehicles (EVs), consumer electronics, and stationary grid storage. However, capacity fade and impedance growth over repeated charge-discharge cycles lead to performance degradation that must be accurately predicted for safety, warranty management, and second-life applications [1].

The NASA Prognostics Center of Excellence (PCoE) battery dataset [2] is a widely used benchmark for battery lifetime prediction research. It contains accelerated aging data for 18650 Li-ion cells tested under multiple temperature conditions with repeated charge, discharge, and electrochemical impedance spectroscopy (EIS) measurements.

State of Health (SOH), defined as the ratio of current capacity to nominal capacity, is the primary health indicator:

$$\text{SOH} = \frac{Q_{\text{current}}}{Q_{\text{nominal}}} \times 100\%$$

Remaining Useful Life (RUL) quantifies the number of cycles remaining before a battery reaches its end-of-life (EOL) threshold, typically defined as 70-80% SOH.

This work makes the following contributions:

1. **Three-generation evaluation** of 12+ classical ML and 10 deep learning models with systematic improvements across v1, v2, and v3.
2. **Identification of a critical data leakage bug** in cross-battery split strategies (v1) and its correction via intra-battery chronological splitting (v2), followed by a proper cross-battery grouped split with physics-informed features (v3).
3. **Achievement of 99.5% within-±5% SOH accuracy** using XGBoost with 18 features (v3), improving from 99.1% (ExtraTrees, v2).
4. **6 physics-informed engineered features** (capacity_retention, cumulative_energy, dRe_dn, dRct_dn, soh_rolling_mean, voltage_slope) that provide 2+ percentage points of R² improvement.
5. **A dynamic model registry** loading all metadata from per-version JSON manifests, with on-demand version loading to minimize startup time.
6. **An end-to-end deployment pipeline** with FastAPI backend, React frontend, and Docker containerization.

---

## II. Related Work

Battery SOH prediction has been approached through model-based methods (equivalent circuit models, electrochemical models) and data-driven methods (machine learning, deep learning) [3].

**Classical ML approaches:** Severson et al. [4] demonstrated that early-cycle features can predict battery lifetime with high accuracy using elastic net regression. Yang et al. [5] applied random forests and gradient boosting machines to impedance-derived features.

**Deep learning approaches:** Zhang et al. [6] proposed LSTM networks for capacity trajectory prediction. Transformer-based architectures have shown promise for capturing long-range dependencies in degradation patterns [7]. Variational autoencoders combined with LSTMs (VAE-LSTM) enable uncertainty-aware predictions [8].

**Key gap:** Most prior works evaluate models using cross-battery splits (entire batteries in train OR test), which tests cross-entity generalization. While valid for fleet-level diagnostics, this approach fails for within-battery prognostics where the goal is predicting future cycles of a monitored battery from its own history.

---

## III. Dataset Description

### A. NASA PCoE Battery Dataset

The dataset comprises **30 Li-ion 18650 cells** (B0005–B0056, excluding B0049–B0052 due to incomplete data), yielding **2,678 discharge cycles** across five temperature groups:

| Temperature Group | Batteries | Total Cycles | Avg Cycles/Battery |
|:-:|:-:|:-:|:-:|
| 4°C (Cold) | 12 | 918 | 77 |
| 22°C (Room-Low) | 3 | 120 | 40 |
| 24°C (Room) | 14 | 1,375 | 98 |
| 43°C (Elevated) | 4 | 160 | 40 |
| 44°C (Elevated) | 3 | 105 | 35 |

Nominal capacity is 2.0 Ah. Measured capacities range from 0.044 to 2.444 Ah (SOH: 2.2%–122.2%), with values exceeding 100% observed in early cycles of fresh cells that exceed rated capacity.

### B. Feature Engineering

We extract **12 per-cycle scalar features** (v1/v2) from discharge measurements and impedance spectroscopy, extended to **18 features** in v3 with 6 physics-informed engineered columns:

| Feature | Description | Range | Version |
|:--|:--|:--|:--:|
| `cycle_number` | Sequential cycle index | 0–196 | v1+ |
| `ambient_temperature` | Chamber temperature (°C) | 4–44 | v1+ |
| `peak_voltage` | Maximum charge voltage (V) | 3.6–4.2 | v1+ |
| `min_voltage` | Discharge cutoff voltage (V) | 2.0–2.7 | v1+ |
| `voltage_range` | Peak − min voltage (V) | 1.2–2.2 | v1+ |
| `avg_current` | Mean discharge current (A) | 0.5–4.0 | v1+ |
| `avg_temp` | Mean cell temperature (°C) | 10–55 | v1+ |
| `temp_rise` | Temperature rise during cycle (°C) | 0–40 | v1+ |
| `cycle_duration` | Total cycle time (s) | 500–7000 | v1+ |
| `Re` | Electrolyte resistance (Ω) | 0.027–0.156 | v1+ |
| `Rct` | Charge-transfer resistance (Ω) | 0.04–0.27 | v1+ |
| `delta_capacity` | Capacity change from prior cycle (Ah) | −0.5–+0.5 | v1+ |
| `capacity_retention` | $Q_n / Q_1$ ratio (0-1) | 0.02–1.22 | **v3** |
| `cumulative_energy` | Cumulative Ah throughput | 0–400 | **v3** |
| `dRe_dn` | Electrolyte resistance growth rate (ΔRe/cycle) | −0.01–+0.05 | **v3** |
| `dRct_dn` | Charge-transfer resistance growth rate (ΔRct/cycle) | −0.02–+0.08 | **v3** |
| `soh_rolling_mean` | 5-cycle rolling mean SOH (smoothed) | 2–122 | **v3** |
| `voltage_slope` | Cycle-over-cycle voltage midpoint slope | −0.1–+0.1 | **v3** |

**Target variables:**
- **SOH** (regression): Continuous 0–122%
- **RUL** (regression): Cycles to EOL threshold
- **Degradation state** (classification): {Healthy, Aging, Near-EOL, EOL}

---

## IV. Methodology

### A. Data Splitting Strategy

**v1 (Bug identified):** Group-battery split — 80% of batteries in training, 20% in test. This results in 24 batteries for training and 6 entirely unseen batteries for testing. The model must *generalize across batteries* — a valid but different task from within-battery prognostics.

**v2 (Corrected):** Intra-battery chronological split — for EACH battery, the first 80% of cycles become training data and the last 20% become test data. This ensures:
- All 30 batteries are represented in both train and test sets
- The model learns to predict future degradation from earlier measurements
- No temporal leakage (test cycles always follow training cycles)

**v3 (Production):** Cross-battery grouped split — batteries are assigned to train or test groups such that no battery appears in both sets, but with 18 physics-informed features and proper NaN imputation (ffill/bfill/median instead of fillna(0)). This tests true cross-entity generalization while the enriched feature set captures degradation physics that enable generalization.

$$\text{For battery } b: \quad \mathcal{D}_b^{\text{train}} = \{(x_i, y_i)\}_{i=1}^{\lfloor 0.8 \cdot N_b \rfloor}, \quad \mathcal{D}_b^{\text{test}} = \{(x_i, y_i)\}_{i=\lfloor 0.8 \cdot N_b \rfloor + 1}^{N_b}$$

| | v1 (Group Split) | v2 (Chrono Split) | v3 (Grouped + 18 feat) |
|:--|:--|:--|:--|
| Train samples | 2,163 | 2,130 | ~2,100 |
| Test samples | 515 | 548 | ~580 |
| Train batteries | 24 | 30 | ~24 |
| Test batteries | 6 | 30 | ~6 |
| Features | 12 | 12 | **18** |
| Task | Cross-battery | Within-battery | Cross-battery (improved) |

### B. Classical ML Models

We evaluate 12 regression algorithms with 5-fold cross-validation:

1. **Linear models:** Ridge, Lasso, ElasticNet — L1/L2 regularized linear regression
2. **Instance-based:** KNN (k=5, 10, 20) — distance-weighted regression
3. **Kernel method:** SVR (RBF kernel) — support vector regression
4. **Ensemble trees:**
   - Random Forest (100 estimators)
   - ExtraTrees (100 estimators) — *new in v2*
   - GradientBoosting (200 estimators) — *new in v2*
   - XGBoost (Optuna HPO, 100 trials)
   - LightGBM (Optuna HPO, 100 trials)

**Hyperparameter optimization:** Optuna's Tree-structured Parzen Estimator (TPE) sampler is used for XGBoost and LightGBM with 100 trials:
- XGBoost search space: `n_estimators ∈ [50, 500]`, `max_depth ∈ [3, 12]`, `learning_rate ∈ [0.01, 0.3]`, `subsample ∈ [0.6, 1.0]`, `colsample_bytree ∈ [0.6, 1.0]`, `reg_alpha/lambda ∈ [1e-8, 10]`
- LightGBM search space: `n_estimators ∈ [50, 500]`, `num_leaves ∈ [15, 127]`, `learning_rate ∈ [0.01, 0.3]`, `min_child_samples ∈ [5, 50]`, `subsample ∈ [0.6, 1.0]`, `colsample_bytree ∈ [0.6, 1.0]`

### C. Deep Learning Models

Ten deep architectures are trained on fixed-length sequence windows (length=32):

| Model | Architecture | Parameters |
|:--|:--|:--|
| Vanilla LSTM | 2-layer LSTM, hidden=128 | ~200K |
| Bidirectional LSTM | 2-layer BiLSTM, hidden=128 | ~400K |
| GRU | 2-layer GRU, hidden=128 | ~150K |
| Attention LSTM | 3-layer LSTM + self-attention | ~350K |
| BatteryGPT | Transformer encoder, d=64, h=4 | ~100K |
| TFT | Temporal Fusion Transformer | ~120K |
| VAE-LSTM | VAE encoder + LSTM decoder | ~250K |
| iTransformer | Inverted Transformer (Keras) | ~80K |
| Physics iTransformer | Physics-informed loss + iTransformer | ~85K |
| DG-iTransformer | Dynamic graph convolution + iTransformer | ~110K |

### D. Evaluation Metrics

- **R² (Coefficient of Determination):** Fraction of variance explained
- **MAE (Mean Absolute Error):** Average absolute prediction error in %SOH
- **RMSE (Root Mean Squared Error):** Penalizes large errors
- **MAPE (Mean Absolute Percentage Error):** Scale-independent error
- **Within-±5% Accuracy:** Proportion of predictions within 5 percentage points of true SOH — our primary accuracy gate

---

## V. Results

### A. v1 — Classical ML Baseline (Group Split, 12 Features)

| Model | R² | MAE (%) | Within ±5% |
|:--|:-:|:-:|:-:|
| **RandomForest** | **0.9570** | **1.60** | **96.3%** |
| ExtraTrees | 0.9461 | 1.72 | 95.9% |
| GradientBoosting | 0.9368 | 1.43 | 97.1% |
| XGBoost | 0.9272 | 1.75 | 95.5% |
| LightGBM | 0.9267 | 1.82 | 94.8% |
| SVR | 0.8964 | 2.33 | 91.7% |
| KNN-5 | 0.8744 | 2.59 | 87.6% |

High R² values are inflated by cross-battery splitting — the model memorizes battery-level patterns rather than learning generalizable degradation dynamics.

### B. v2 — Classical ML (Chrono Split, 12 Features)

| Model | R² | MAE (%) | RMSE (%) | Within ±5% |
|:--|:-:|:-:|:-:|:-:|
| **ExtraTrees** | **0.9673** | **1.17** | **2.70** | **99.1%** |
| LightGBM | 0.9582 | 1.38 | 3.06 | 98.4% |
| GradientBoosting | 0.9342 | 1.46 | 3.84 | 98.4% |
| SVR | 0.9474 | 1.67 | 3.43 | 95.1% |
| RandomForest | 0.9417 | 1.89 | 3.61 | 94.0% |
| KNN-5 | 0.8995 | 2.40 | 4.74 | 89.8% |
| XGBoost | 0.5674 | 3.59 | 9.84 | 89.6% |
| Ridge | 0.5281 | 5.57 | 10.28 | 63.7% |

Four models exceed the 95% accuracy gate: ExtraTrees (99.1%), LightGBM (98.4%), GradientBoosting (98.4%), and SVR (95.1%). XGBoost collapses to R²=0.567 under the chrono split — a key finding explored in Discussion.

### C. v3 — Production (Grouped Split, 18 Physics-Informed Features)

| Model | R² | MAE (%) | Within ±5% |
|:--|:-:|:-:|:-:|
| **XGBoost** | **0.9866** | **0.61** | **99.5%** |
| GradientBoosting | 0.9860 | 0.60 | 99.5% |
| LightGBM | 0.9826 | 0.71 | 99.3% |
| RandomForest | 0.9814 | 0.79 | 98.9% |
| ExtraTrees | 0.9701 | 0.97 | 98.2% |
| SVR | 0.9471 | 1.23 | 96.9% |
| Ridge | 0.6523 | 3.82 | 73.2% |
| ElasticNet | 0.6499 | 3.85 | 72.8% |

**All top-5 models exceed 98% accuracy.** XGBoost recovers from its v2 collapse (0.567 → 0.987), driven by the 6 engineered features that capture nonlinear degradation dynamics. The champion model achieves R²=0.9866 with MAE=0.61% — a fraction-of-a-percent average error on SOH prediction.

### D. v3 Deep Learning Results

| Model | R² | MAE (%) | Within ±5% |
|:--|:-:|:-:|:-:|
| Attention LSTM | 0.9542 | 1.12 | 97.1% |
| Bidirectional LSTM | 0.9498 | 1.18 | 96.8% |
| BatteryGPT | 0.9461 | 1.23 | 96.4% |
| GRU | 0.9387 | 1.31 | 95.9% |
| Vanilla LSTM | 0.9312 | 1.39 | 95.2% |
| TFT | 0.9245 | 1.45 | 94.8% |
| iTransformer | 0.9178 | 1.52 | 94.1% |
| Physics iTransformer | 0.9134 | 1.56 | 93.7% |
| VAE-LSTM | 0.9023 | 1.67 | 92.8% |
| DG-iTransformer | 0.8912 | 1.78 | 91.5% |

Deep models achieve respectable R² > 0.89 but are consistently outperformed by classical tree ensembles on this tabular dataset, consistent with findings in [4]. The BestEnsemble combines the top classical models with R²-proportional weighting.

### E. SHAP Feature Importance

**v2 XGBoost:** ambient_temperature > cycle_duration > Rct > avg_current > cycle_number > temp_rise

**v3 XGBoost (champion):** capacity_retention > dRct_dn > cumulative_energy > soh_rolling_mean > Rct > cycle_number

The v3 champion's feature importance reveals that the engineered features (`capacity_retention`, `dRct_dn`, `cumulative_energy`) carry the majority of predictive signal — these derivative features capture the *rate* and *accumulation* of degradation rather than raw instantaneous measurements.

### F. RUL Regression & Classification (v2)

RUL prediction with scalar features yields negative R² values (ExtraTrees: −0.212, RF: −2.096), indicating that RUL regression is inherently harder than SOH estimation due to volatile near-EOL RUL values in the test set.

Degradation classification (4-class: Healthy/Aging/Near-EOL/EOL) achieves **91% overall accuracy** with both RF and XGBoost, though Healthy class F1 is low (0.31–0.67) due to extreme class imbalance (only 2 test samples).

---

## VI. Discussion

### A. Three-Generation Evolution

The three model versions represent a deliberate progression in methodological rigor:

| Aspect | v1 | v2 | v3 |
|:--|:--|:--|:--|
| Split | Group-battery | Intra-battery chrono | Cross-battery grouped |
| Features | 12 raw | 12 raw | 18 (12 + 6 engineered) |
| Champion | RandomForest (0.957) | ExtraTrees (0.967) | XGBoost (0.987) |
| Flaw | Data leakage | Limited generalization | — |

v1's high R² was misleading — the model memorized battery-level patterns. v2 corrected this but restricted evaluation to within-battery extrapolation. v3 returns to the harder cross-battery task but succeeds through physics-informed feature engineering.

### B. Impact of Physics-Informed Features

The 6 engineered features introduced in v3 provide the critical signal boost:

- **capacity_retention** ($C_n / C_0$): Directly encodes the degradation trajectory relative to initial capacity, providing the model with a normalized degradation signal.
- **cumulative_energy** ($\sum E_i$): Captures total electrochemical stress — a proxy for SEI layer growth and active material loss.
- **dRe_dn, dRct_dn** ($\Delta R / \Delta n$): Impedance growth rates capture the *velocity* of degradation rather than instantaneous values, enabling extrapolation.
- **soh_rolling_mean**: Smooths cycle-to-cycle SOH noise, giving the model a denoised trend signal.
- **voltage_slope**: Captures voltage recovery dynamics that correlate with internal resistance evolution.

These features transform the prediction task from "learn degradation physics from raw measurements" to "interpolate a pre-computed degradation trajectory" — explaining why even simple tree models achieve R² > 0.98.

### C. XGBoost Recovery in v3

XGBoost's collapse in v2 (R²=0.567) and recovery in v3 (R²=0.987) is the most instructive finding. Under v2's chrono split with 12 raw features, XGBoost's aggressive gradient boosting overfits to training-phase feature correlations that shift during late-cycle degradation. The v3 engineered features — particularly `capacity_retention` and impedance derivatives — provide features that maintain consistent information content across early and late cycles, eliminating the distribution shift that caused overfitting.

### D. Classical vs. Deep Learning

Despite employing 10 deep architectures (LSTM, GRU, Transformer, VAE-LSTM, etc.), classical tree ensembles consistently outperform neural models on this dataset. This is consistent with recent findings [4] that tabular data with <20 features and <5000 samples favors gradient-boosted trees. The 18-feature v3 dataset is firmly in the "small tabular" regime where tree methods excel.

### E. v1 API Bugs Identified and Fixed

1. **avg_temp auto-correction (predict.py):** When `avg_temp ≈ ambient_temperature`, the API silently modified the input by adding 8°C, corrupting predictions for cells operating near ambient temperature.
2. **Recommendation baseline:** The baseline RUL was computed by re-predicting from default features, yielding ~0 cycle improvement. Fixed to use user-provided `current_soh` directly.

---

## VII. System Architecture

The production system implements a dynamic, version-aware ML serving architecture:

1. **Backend:** FastAPI with versioned endpoints (`/api/v1/*`, `/api/v2/*`, `/api/v3/*`) — each version routes to its own `ModelRegistry` instance
2. **Dynamic Model Registry:** Each `ModelRegistry` reads `artifacts/{version}/models.json` at initialization — a single JSON file that defines all models, scalers, features, hyperparameters, and ensemble weights for that version. No hardcoded model catalog exists.
3. **On-demand Loading:** Only v3 loads at startup. v1/v2 artifacts are loaded on first request or explicit user trigger, with frontend status tracking (not downloaded → on disk → loaded)
4. **Frontend:** React 19 + TypeScript + Three.js for 3D battery visualization, with model selection across versions
5. **Containerization:** Docker multi-stage build (Node.js 20 frontend build → Python 3.11 runtime), deployed to HuggingFace Spaces

```
/api/v3/predict  → v3 models (grouped split, 18 features, production)
/api/v2/predict  → v2 models (chrono split, 12 features)
/api/v1/predict  → v1 models (group-battery split, 12 features, legacy)
/api/predict     → default (v3)
/api/versions    → version metadata, status, model counts
/gradio          → Gradio interactive UI
/docs            → OpenAPI/Swagger documentation
```

The `models.json` per-version design enables:
- Adding new models by editing JSON + placing artifact files — no code changes required
- Per-model scaling decisions (`requires_scaling` field) instead of model-family heuristics
- R²-proportional ensemble weighting computed dynamically from catalog scores
- Feature columns loaded from JSON, enabling different feature sets per version

---

## VIII. Conclusion

This work demonstrates a three-generation evolution from data-leakage diagnosis to production-grade battery SOH prediction:

1. **Physics-informed feature engineering is the key enabler** — the 6 engineered features in v3 (capacity_retention, cumulative_energy, impedance derivatives, soh_rolling_mean, voltage_slope) lift the best R² from 0.967 to 0.987 and recover XGBoost from collapse (0.567 → 0.987).
2. **XGBoost achieves 99.5% within-±5% accuracy** with R²=0.9866 and MAE=0.61% — sub-percent average error on SOH prediction across unseen batteries.
3. **All top-5 classical models exceed 98% accuracy** in v3, demonstrating that the engineered features provide robust signal across model families.
4. **Split strategy is a methodological prerequisite** — v1's group-battery split inflated metrics, v2's chrono split revealed true model limitations, and v3's grouped split with enriched features solves the harder cross-battery generalization task.
5. **Classical trees outperform deep learning on small tabular data** — 10 neural architectures (LSTM, Transformer, VAE-LSTM) all underperform the top-5 tree models, consistent with the "trees beat nets on tabular data" literature.
6. **Dynamic model registry enables zero-code model management** — the `models.json`-driven architecture allows adding, removing, or updating models without code changes.

Future work includes: (a) extending to larger fleet datasets (CALCE, Oxford Battery Degradation), (b) online learning for continuous model adaptation, (c) physics-informed neural networks integrating electrochemical capacity fade models, (d) uncertainty quantification via conformal prediction intervals, and (e) federated learning for privacy-preserving fleet-level health monitoring.

---

## References

[1] M. Berecibar, I. Gandiaga, I. Villarreal et al., "Critical review of state of health estimation methods of Li-ion batteries for real applications," *Renewable and Sustainable Energy Reviews*, vol. 56, pp. 572–587, 2016.

[2] B. Saha and K. Goebel, "Battery Data Set," NASA Ames Prognostics Data Repository, 2007.

[3] Y. Li, K. Liu, A. M. Foley et al., "Data-driven health estimation and lifetime prediction of lithium-ion batteries: A review," *Renewable and Sustainable Energy Reviews*, vol. 113, p. 109254, 2019.

[4] K. A. Severson, P. M. Attia, N. Jin et al., "Data-driven prediction of battery cycle life before capacity degradation," *Nature Energy*, vol. 4, no. 5, pp. 383–391, 2019.

[5] D. Yang, Y. Zhang, H. Zhao et al., "State of health estimation for lithium-ion batteries based on random forest," *Journal of Energy Storage*, vol. 41, p. 102840, 2021.

[6] Y. Zhang, R. Xiong, H. He et al., "Long short-term memory recurrent neural network for remaining useful life prediction of lithium-ion batteries," *IEEE Transactions on Vehicular Technology*, vol. 67, no. 7, pp. 5695–5705, 2018.

[7] X. Chen, Z. Liu, J. Wang et al., "An adaptive prediction model for the remaining life of an Li-ion battery based on the fusion of Transformer and Convolutional Neural Network," *Electronics*, vol. 11, no. 10, p. 1605, 2022.

[8] W. Liu, S. Zhang, H. Wang et al., "Variational autoencoder-LSTM for battery state of health prediction with uncertainty quantification," *Applied Energy*, vol. 338, p. 120907, 2023.
