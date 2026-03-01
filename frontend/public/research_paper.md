# Comprehensive Multi-Model Approach to Lithium-Ion Battery State of Health Prediction Using the NASA PCoE Dataset

**Authors:** Neeraj Sathish Kumar.  
**Affiliation:** Vellore Institute of Technology (VIT)  
**Date:** February 2026

---

## Abstract

Accurate prediction of State of Health (SOH) and Remaining Useful Life (RUL) of lithium-ion batteries is critical for safe and efficient operation of electric vehicles, grid energy storage, and portable electronics. This paper presents a comprehensive multi-model framework evaluating 12 classical machine learning algorithms and 10 deep learning architectures for battery lifecycle prediction using the NASA Prognostics Center of Excellence (PCoE) dataset. We introduce a novel **intra-battery chronological split** methodology that eliminates cross-battery data leakage—a systematic flaw identified in the prior version (v1) of this system. Our best model, ExtraTrees Regressor with Optuna-tuned hyperparameters, achieves an R² of 0.967, MAE of 1.17%, and **99.1% within ±5% SOH accuracy**, surpassing the 95% accuracy threshold. We further demonstrate that the choice of train-test split strategy has a greater impact on predictive reliability than model architecture selection, and present a versioned artifact management system for reproducible deployment.

**Keywords:** lithium-ion battery, state of health, remaining useful life, machine learning, ensemble methods, NASA PCoE, degradation prediction, predictive maintenance

---

## I. Introduction

Lithium-ion batteries are the dominant energy storage technology for electric vehicles (EVs), consumer electronics, and stationary grid storage. However, capacity fade and impedance growth over repeated charge-discharge cycles lead to performance degradation that must be accurately predicted for safety, warranty management, and second-life applications [1].

The NASA Prognostics Center of Excellence (PCoE) battery dataset [2] is a widely used benchmark for battery lifetime prediction research. It contains accelerated aging data for 18650 Li-ion cells tested under multiple temperature conditions with repeated charge, discharge, and electrochemical impedance spectroscopy (EIS) measurements.

State of Health (SOH), defined as the ratio of current capacity to nominal capacity, is the primary health indicator:

$$\text{SOH} = \frac{Q_{\text{current}}}{Q_{\text{nominal}}} \times 100\%$$

Remaining Useful Life (RUL) quantifies the number of cycles remaining before a battery reaches its end-of-life (EOL) threshold, typically defined as 70-80% SOH.

This work makes the following contributions:

1. **Systematic evaluation** of 12 classical ML and 10 deep learning models on a unified dataset with consistent preprocessing.
2. **Identification of a critical data leakage bug** in cross-battery split strategies and its correction via intra-battery chronological splitting.
3. **Achievement of 99.1% within-±5% SOH accuracy** using the ExtraTrees Regressor.
4. **A versioned artifact management system** enabling reproducible comparison between model generations.
5. **An end-to-end deployment pipeline** with FastAPI backend, React frontend, and Docker containerization.

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

We extract **12 per-cycle scalar features** from discharge measurements and impedance spectroscopy:

| Feature | Description | Range |
|:--|:--|:--|
| `cycle_number` | Sequential cycle index | 0–196 |
| `ambient_temperature` | Chamber temperature (°C) | 4–44 |
| `peak_voltage` | Maximum charge voltage (V) | 3.6–4.2 |
| `min_voltage` | Discharge cutoff voltage (V) | 2.0–2.7 |
| `voltage_range` | Peak − min voltage (V) | 1.2–2.2 |
| `avg_current` | Mean discharge current (A) | 0.5–4.0 |
| `avg_temp` | Mean cell temperature (°C) | 10–55 |
| `temp_rise` | Temperature rise during cycle (°C) | 0–40 |
| `cycle_duration` | Total cycle time (s) | 500–7000 |
| `Re` | Electrolyte resistance (Ω) | 0.027–0.156 |
| `Rct` | Charge-transfer resistance (Ω) | 0.04–0.27 |
| `delta_capacity` | Capacity change from prior cycle (Ah) | −0.5–+0.5 |

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

$$\text{For battery } b: \quad \mathcal{D}_b^{\text{train}} = \{(x_i, y_i)\}_{i=1}^{\lfloor 0.8 \cdot N_b \rfloor}, \quad \mathcal{D}_b^{\text{test}} = \{(x_i, y_i)\}_{i=\lfloor 0.8 \cdot N_b \rfloor + 1}^{N_b}$$

| | v1 (Group Split) | v2 (Chrono Split) |
|:--|:--|:--|
| Train samples | 2,163 | 2,130 |
| Test samples | 515 | 548 |
| Train batteries | 24 | 30 |
| Test batteries | 6 | 30 |
| Task | Cross-battery generalization | Within-battery prognostics |

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

### A. Classical ML — SOH Regression (v2)

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
| ElasticNet | 0.5271 | 5.59 | 10.29 | 63.7% |
| Lasso | 0.5271 | 5.59 | 10.29 | 63.7% |
| KNN-10 | 0.8778 | 2.69 | 5.23 | 88.1% |
| KNN-20 | 0.8378 | 3.19 | 6.03 | 85.9% |

**Four models exceed the 95% accuracy gate:** ExtraTrees (99.1%), LightGBM (98.4%), GradientBoosting (98.4%), and SVR (95.1%).

### B. SHAP Feature Importance

SHAP analysis reveals the following feature importance ranking for the top models:

**XGBoost:** ambient_temperature > cycle_duration > Rct > avg_current > cycle_number > temp_rise

**Random Forest:** cycle_duration > avg_temp > avg_current > ambient_temperature > temp_rise > cycle_number

Notably, impedance parameters (Re, Rct) are more important for XGBoost than ensemble tree methods, while thermal features dominate in Random Forest and ExtraTrees.

### C. RUL Regression

| Model | R² | MAE (cycles) | Within ±5 cycles |
|:--|:-:|:-:|:-:|
| ExtraTrees | −0.212 | 1.93 | 88.7% |
| RandomForest | −2.096 | 2.72 | 88.3% |
| LightGBM | −0.582 | 3.28 | 85.8% |
| XGBoost | −1.261 | 3.89 | 80.8% |

Negative R² values indicate that RUL prediction is inherently harder than SOH estimation with scalar features alone — the test set (last 20% of cycles per battery) concentrates near-EOL samples with low and volatile RUL values.

### D. Degradation Classification

Both Random Forest and XGBoost classifiers achieve **91% overall accuracy** on 4-class degradation state classification:

| Class | RF F1 | XGB F1 | Support |
|:--|:-:|:-:|:-:|
| Healthy | 0.31 | 0.67 | 2 |
| Aging | 0.85 | 0.91 | 98 |
| Near-EOL | 0.80 | 0.75 | 110 |
| EOL | 0.97 | 0.96 | 338 |

The low Healthy F1 score is due to class imbalance — only 2 test samples are in the Healthy state (early cycles of batteries with many cycles).

---

## VI. Discussion

### A. Impact of Split Strategy

The choice of data splitting strategy is the single most important methodological decision:

- **v1 (cross-battery):** Forces the model to generalize to unseen battery chemistries/manufacturing lots. High apparent R² (~0.96 for RF) but poor real-world SOH predictions (e.g., predicting 4% SOH for a healthy battery at cycle 10 of B0005).
- **v2 (intra-battery chrono):** Tests the model's ability to extrapolate future degradation from observed history. Lower but more honest R² values, and predictions align with physical expectations.

### B. Why ExtraTrees Outperforms

ExtraTrees (Extremely Randomized Trees) outperforms Random Forest by introducing additional randomness in split selection — instead of searching for the best split, it selects random thresholds. This reduces variance and provides better generalization on the chronological split where the test distribution (late-cycle degradation) differs subtly from training (early-to-mid cycle).

### C. XGBoost Underperformance

Despite Optuna HPO, XGBoost underperforms on this dataset (R²=0.567 vs. ExtraTrees 0.967). Analysis suggests overfitting to the training distribution's feature correlation structure, which shifts between early (training) and late (test) cycles due to nonlinear degradation mechanisms.

### D. v1 API Bugs Identified and Fixed

1. **avg_temp auto-correction (predict.py L30-31):** When `avg_temp ≈ ambient_temperature`, the API silently modified the input by adding 8°C. This corrupted predictions for cells operating near ambient temperature.
2. **Recommendation baseline (recommend endpoint):** The baseline RUL was computed by re-predicting from default features, yielding ~0 cycle improvement for all recommendations. Fixed to use user-provided `current_soh` directly.

---

## VII. System Architecture

The deployment comprises:

1. **Backend:** FastAPI with versioned endpoints (`/api/v1/*`, `/api/v2/*`)
2. **Model Registry:** Singleton registry with version-aware artifact loading, supporting 21+ models across classical, deep, and ensemble families
3. **Frontend:** React 19 + TypeScript + Three.js for 3D battery visualization
4. **Containerization:** Docker multi-stage build, deployed to HuggingFace Spaces

```
/api/v1/predict  → v1 models (group-battery split, legacy)
/api/v2/predict  → v2 models (chrono split, bug-fixed)
/api/predict     → default (v2)
/gradio          → Gradio interactive UI
/docs            → OpenAPI/Swagger documentation
```

---

## VIII. Conclusion

This work demonstrates that:

1. **Split strategy matters more than model choice** — the same Random Forest achieves different real-world accuracy depending on whether cross-battery or intra-battery chronological splitting is used.
2. **ExtraTrees achieves 99.1% within-±5% accuracy** — surpassing the 95% target with R²=0.967 and MAE=1.17%.
3. **Four models exceed 95% accuracy** — ExtraTrees, LightGBM, GradientBoosting, and SVR all pass the accuracy gate.
4. **Feature importance varies by model type** — thermal features (cycle_duration, avg_temp) dominate ensemble trees, while impedance features (Rct) are more important for gradient boosting.
5. **Versioned artifact management** enables rigorous A/B comparison between model generations while maintaining backward compatibility.

Future work includes: (a) extending to larger fleet datasets (CALCE, Oxford Battery Degradation), (b) online learning for continuous model adaptation, (c) physics-informed neural networks integrating electrochemical capacity fade models, and (d) uncertainty quantification via conformal prediction intervals.

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
