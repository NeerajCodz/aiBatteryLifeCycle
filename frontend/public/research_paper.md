# State of Health Estimation and Target Leakage Auditing for Lithium Ion Batteries

S Neeraj
School of Computer Science
VIT Vellore
neeraj.s2025@vitstudent.ac.in

Anuradha Pradeep
School of Computer Science
VIT Vellore
anuradha.pradeep2025@vitstudent.ac.in

Shourya Kumar
School of Mechanical Engineering
VIT Vellore
shourya.kumar2025a@vitstudent.ac.in

Abstract—Accurate state-of-health (SOH) estimation requires inputs available when an estimate is made and testing on batteries excluded from fitting. We compare 20 regressors using 18 scalar features or 64 × 5 sequences from the observable 4.0–3.6 V discharge segment. NASA uses disjoint training, validation, and test batteries over five folds and three seeds; CALCE and Oxford provide separate grouped evaluations and untouched zero-shot targets for NASA-trained models. The original audit reports grouped ExtraTrees MAE 4.145 with observable inputs and 0.436 SOH percentage points (pp) after adding current-target proxies; matched-count and temperature-quota controls support this within-split leakage diagnostic without isolating a causal split effect. Under the corrected protocol, the Weighted Average Ensemble has the lowest NASA battery-macro MAE (3.536 pp); its averaged held-out predictions yield sample MAE 3.682 pp, RMSE 6.011 pp, MAPE 4.37%, R² 0.610, adjusted R² 0.607, and 79.2% within ±5 absolute SOH pp. The best frozen NASA models obtain MAE 6.955 pp on CALCE and 4.615 pp on Oxford, much worse than the respective within-dataset fits. Secondary NASA health-band diagnostics use validation-residual scores; capacity/RUL trajectories are retrospective descriptions with censoring. These laboratory results support leakage-aware benchmarking, not a validated charging policy or field deployment.

Keywords—battery state of health, battery-grouped validation, data leakage, domain shift, external validation, partial discharge, regression

## I. INTRODUCTION

Lithium-ion battery degradation changes usable energy, power capability, and safety margins, making SOH estimation relevant to battery management, warranty assessment, and second-life screening [1], [2]. SOH is not directly sensed; it is inferred from measurements such as voltage, current, temperature, or impedance. Data-driven estimators can represent nonlinear relationships without identifying every electrochemical parameter, but their reported error depends as much on the target and split design as on the model architecture [1], [13].

A deployment-oriented estimate should meet two constraints. First, every predictor must be available at the intended prediction time. A feature computed from the same full-cycle capacity used to define SOH is a target proxy, even if it has a different name. Second, observations from one physical cell should not appear in both model fitting and final testing when the claim concerns a new battery. Random cycle splits and within-cell chronological splits answer different questions from cross-battery validation and can yield much smaller errors.

The previously circulated 99.1% within-five-point figure is withdrawn as a performance contribution. It motivates the leakage audit only: no abstract, introduction or conclusion claim treats it as unseen-battery accuracy, and no charging-control policy is evaluated in this paper.

Recent studies examine partial curves and transformers [12], [16], [18], [24], weak-label learning [15], and transfer across domains [17], [19]. A recent review also distinguishes laboratory evaluation from practical use [25]. Scores are not directly comparable when voltage windows, label availability, physical split units, and destination-data access differ. A high within-dataset R² alone cannot establish low absolute error, calibration, sensor robustness, or transfer to another chemistry and protocol.

Tree/deep fusion is also used for energy-demand time-series forecasting [26]. That methodological parallel does not transfer forecasting accuracy, charging benefit, or validation guarantees to battery SOH; the prediction time, physical unit and leakage pathways must be established for the battery task itself.

This study makes five contributions. It defines a leakage-safe current-cycle SOH task using only a fixed partial-discharge window; evaluates eight classical, ten sequence, and two ensemble models using disjoint train, validation, and test batteries; audits split sensitivity and target-proxy leakage using V1/V2/V3 comparisons plus paired sample-count, predictor-count and temperature-quota controls; and tests both grouped within-dataset performance and zero-shot NASA-to-CALCE/Oxford transfer. A secondary NASA analysis evaluates SOH-derived health bands and reports censoring-aware descriptive capacity/RUL trajectories. The software is supplied as a reproducible research prototype. Charging recommendations are outside the validated endpoint and are not claimed as an operational control policy.

## II. RELATED WORK

### A. Model based and data driven estimation

Equivalent-circuit and electrochemical models offer physical structure but require identification and may be sensitive to operating-condition mismatch [2]. Earlier Bayesian, ensemble, and health-indicator studies address prognostics or feature choice [6]–[8], while long-horizon Transformer and uncertainty-aware work target future capacity or remaining life [10], [11]. Classical regression can exploit compact indicators with modest compute; recurrent and attention models can learn from sampled curves [9], [20]–[23]. These endpoint and input differences prevent a model-family ranking from substituting for a common observation window and split protocol.

### B. Partial curves and representation learning

Partial charging curves have been used for Gaussian-process and deep capacity estimation [12], [18], [24]. Shu et al. apply a transformer to a charging-voltage segment [18], and Lyu et al. combine relaxation-voltage features with recurrent and random-forest components under multiple operating conditions [16]. Wang et al. reduce reliance on annotated data through weak-label pretraining followed by target-domain fine-tuning [15]. These studies support informative partial observations, but their observation windows and adaptation procedures differ from a fixed partial-discharge, zero-shot test.

### C. Generalization and the unresolved evaluation gap

Domain generalization [17] and cross-domain transfer [19] directly address distribution change. They also reveal a distinction that matters for comparison: using target labels for adaptation or selection is different from evaluating a frozen source model. Our design therefore joins observable input timing, physical-cell separation, uncertainty analysis, sensor perturbations, and unadapted NASA-to-CALCE/Oxford testing under one recorded protocol. This is an evaluation contribution, not a claim of a new neural architecture.

## III. DATA AND PREDICTION CONTRACT

### A. Data sources and battery specifications

NASA PCoE records accelerated aging of 2 Ah 18650 lithium-ion cells under multiple temperatures and discharge protocols [3]. We parse 34 raw identifiers and apply the documented quality exclusion to B0049–B0052, retaining 30 cells. CALCE CS2_35–CS2_38 are 1.1 Ah LiCoO₂ prismatic cells distributed by CALCE [5]. Oxford Battery Degradation Dataset 1 contains eight 0.74 Ah Kokam pouch cells aged at 40 °C; periodic characterization discharges supply the records used here [4]. Table 1 summarizes the retained data.

Table 1. Eligible physical cells, discharges, and robust-reference SOH range by dataset

| Dataset | Cell specification | Cells | Cycles | SOH range (%) |
| --- | --- | --- | --- | --- |
| NASA PCoE | 18650; 2 Ah | 30 | 2141 | 56.97–116.60 |
| CALCE CS2 | LCO prismatic; 1.1 Ah | 4 | 3789 | 26.38–100.12 |
| Oxford | Kokam pouch; 0.74 Ah | 8 | 519 | 62.80–101.24 |

[Download Table 1 as CSV](/research/reviewer4/tables/table_01.csv)

All archives were obtained from their official repositories, and SHA-256 digests are stored with the code. The adapter records every excluded cycle and reason. Across datasets, incomplete segments with fewer than three samples, nonpositive capacity labels, pre-characterization partial discharges, and values above 120% of the robust initial reference are excluded before splitting. One Oxford temperature channel whose median differed from its 40 °C chamber setting by more than 20 °C is marked missing, replaced by ambient temperature, and identified by temperature_observed = 0 rather than discarded.

### B. Target definition

For cell b and discharge t, the label is SOHb,t = 100 Qb,t / Qb,ref, where Qb,t is the full discharged capacity. To reject intentionally partial diagnostics, full-capacity candidates must be at least 85% of the cell's 95th-percentile valid capacity; Qb,ref is the median of the earliest three candidates. This whole-cell statistic is used only to calibrate retrospective evaluation labels, never as an input. The label can exceed 100% because early measured capacity may exceed the robust reference. Qb,t and every function of the current label are prohibited inputs.

### C. Partial discharge features

Each eligible record is cropped to the observable 4.0–3.6 V interval. The ordered 18 scalar predictors are cycle_index, prior_equivalent_full_cycles, ambient_temperature, segment_duration_s, segment_energy_wh, voltage_start_v, voltage_end_v, voltage_mean_v, voltage_std_v, voltage_linear_slope_v_per_s, voltage_curvature, current_mean_c_rate, current_std_c_rate, temperature_mean_c, temperature_std_c, temperature_rise_c, temperature_slope_c_per_s, and temperature_observed. Prior equivalent full cycles use only preceding discharges; C-rate and usage normalization use public rated capacity (2.0, 1.1, or 0.74 Ah), never the current-cycle label. We prespecified and retained all 18 predictors; no correlation-based feature elimination or test-set-driven selection was applied. Correlated signals can share predictive information, so permutation importance and family ablations assess their contribution without claiming an independent causal effect for each variable. Sequence models receive 64 resampled bins of normalized time, voltage, C-rate, temperature, and observation status. Imputation and scaling are fitted on training cells only.

The feature set was selected by availability at completion of the partial discharge and by coverage of usage history, voltage shape, current loading, segment energy, and thermal response, rather than by test-set ranking. Table 7 records the families, dimensions and interpretations. Duration/energy and voltage slope may respond to loading and polarization; thermal summaries describe heat/exposure without identifying a unique degradation mechanism. All 18 were retained even when correlated. A 64-bin sequence standardizes input length for ten neural models; it is an engineering resolution, not a demonstrated optimal sampling rate.

The fixed 4.0–3.6 V discharge window restricts inputs to an earlier observable segment rather than full-cycle capacity and offers a common voltage interval across these archives. Its endpoints and 64 bins were fixed for harmonization, not optimized on held-out scores. Temperature and C-rate change polarization, time spent in the window and eligibility; the window is not an invariant state-of-charge interval. It is not adjusted by temperature, C-rate or chemistry in these experiments. The supplement reports exposure-specific support and errors; neither alternative-window superiority nor rate/temperature invariance has been tested. Prospective acquisition and window-sensitivity studies remain necessary.

### D. Descriptive health and system context

The class counts quantify imbalance before secondary classification; SOH bands are fixed before evaluation.

![Figure 1](/research/reviewer4/figures/figure_01_health_state_distribution.svg)

Fig. 1. NASA health-state distribution using the revised cell-reference SOH label. Healthy ≥90%, Aging 80–90%, Near-EOL 70–80%, and EOL <70%; these are descriptive bands rather than operational safety limits.

Temperature groups have different cycle and cell support; their distribution does not establish a temperature effect on aging.

![Figure 2](/research/reviewer4/figures/figure_02_health_state_by_temperature.svg)

Fig. 2. NASA health-state counts by recorded ambient temperature. Unequal exposure and cell coverage preclude a causal interpretation of temperature differences.

Correlated voltage and thermal summaries share information. All 18 prespecified predictors remain in the benchmark; this plot does not establish causal importance.

![Figure 3](/research/reviewer4/figures/figure_03_observable_feature_correlation.svg)

Fig. 3. Pairwise Pearson correlation of the 18 observable partial-discharge predictors in eligible NASA records. Full-cycle capacity and current SOH are excluded; undefined correlations are shown as blank cells. This descriptive plot is not a test-set feature-selection procedure.

The cell panels expose heterogeneous degradation rates and non-monotonic measurements that a single overlaid plot obscures. Dashed trend lines summarize observations and do not predict future life.

![Figure 4](/research/reviewer4/figures/figure_04_battery_soh_trajectories.svg)

Fig. 4. All 30 eligible NASA battery SOH trajectories, with descriptive linear trend lines and a 70% reference line. Panels use a shared SOH axis; cycle extent differs by cell.

Exact retrospective RUL is available only for cells with an observed threshold crossing. Finite recording length must not be treated as an EOL event; no predictive RUL performance is reported.

![Figure 5](/research/reviewer4/figures/figure_05_capacity_and_observed_rul.svg)

Fig. 5. Illustrative measured capacity and retrospective cycles to the first observed SOH <70% crossing. Of 30 NASA cells, 3 cross the threshold and 27 are right-censored. Censored cells have no exact RUL curve; these are observed labels, not RUL predictions.

Figure 6 separates the reproducible research architecture from the unvalidated recommendation and simulation components. The released API default is SVR; the Weighted Average Ensemble is the paper benchmark winner.

![Figure 6](/research/reviewer4/figures/figure_06_full_system_architecture.svg)

Fig. 6. Research system architecture. The grouped SOH evaluation supplies model and result artifacts; the API serves the packaged classical subset. Recommendation and simulation components remain exploratory and do not establish charging benefit.

Figure 7 summarizes retained records across the three independent laboratory sources. Cycle counts and physical-cell counts represent different units of evidence.

![Figure 7](/research/reviewer4/figures/figure_07_dataset_coverage.svg)

Fig. 7. Eligible discharges and physical cells after quality checks. NASA contributes 30 cells, CALCE four, and Oxford eight; battery identity determines grouped partitions.

## IV. METHODS

### A. Quantitative V1 V2 V3 audit

We reconstruct three split designs using the same ExtraTrees hyperparameters. Each compares the 18 observable features with those same 18 plus four deliberately invalid current-target proxies. V1 randomly holds out cycles; V2 holds out the final 20% of retained rows within each battery; V3 holds out whole batteries. This is a split-by-input diagnostic, not a causal decomposition: training counts, exposure composition, and test populations differ between protocols. Table 2 and Fig. 8 retain the original audit; the added controls in Section VII-C address sample and predictor count and training-temperature composition. They do not make the cross-protocol test populations exchangeable.

For retained cell rows ordered by cycle number, let k index the retained discharge, Qk its full capacity, Qref its retrospective reference, and yk = 100 Qk/Qref. The deliberately leaked features are Qk/Qref, the cumulative sum of Qj for j ≤ k, Qk − Qk−1 (zero at k = 1), and the arithmetic mean of yj over j = max(1, k−4), …, k. Windows reset per battery, use the current row, and follow retained-row order even when cycle numbers skip. In particular, retention equals yk/100 exactly. Table 6 and Supplement S1 provide units and generation commands; none of these four features enters the observable benchmark.

Table 2. ExtraTrees factorial audit; MAE and RMSE are SOH pp, coverage is within ±5 pp

| Protocol | Features | MAE | RMSE | R² | Within 5 pp (%) |
| --- | --- | --- | --- | --- | --- |
| v1 random cycle | legacy target proxies | 0.068 | 0.317 | 0.999 | 99.8 |
| v1 random cycle | safe partial cycle | 0.732 | 1.632 | 0.972 | 98.5 |
| v2 chronological | legacy target proxies | 0.434 | 1.375 | 0.983 | 97.3 |
| v2 chronological | safe partial cycle | 2.576 | 4.129 | 0.843 | 88.5 |
| v3 grouped battery | legacy target proxies | 0.436 | 1.056 | 0.979 | 98.1 |
| v3 grouped battery | safe partial cycle | 4.145 | 5.958 | 0.261 | 76.5 |

[Download Table 2 as CSV](/research/reviewer4/tables/table_02.csv)

![Figure 8](/research/reviewer4/figures/figure_08_protocol_audit.svg)

Fig. 8. ExtraTrees factorial audit of split unit and feature leakage. V1 random cycles, V2 within-cell chronological, and V3 held-out batteries are audit protocols; MAE is in absolute SOH percentage points.

### B. Models and training constraints

The fixed catalog comprises Extra Trees, gradient boosting, random forest, XGBoost, LightGBM, support-vector regression, ridge regression, k-nearest neighbours, vanilla and bidirectional LSTM, GRU, attention LSTM, BatteryGPT, temporal fusion transformer, iTransformer, physics-informed iTransformer, dynamic-graph iTransformer, VAE-LSTM, validation-error weighted averaging, and ridge stacking. Ensembles use validation predictions only. Neural models stop on validation-battery loss with a maximum of 200 epochs and patience of 20; the outer test fold is never used for stopping or selection.

Tree ensembles use 250–400 estimators with fixed seed, and scaled SVR, ridge, and seven-neighbour distance-weighted KNN serve as non-tree baselines. Recurrent models use two 64-unit layers. PyTorch transformer models use dimension 64, four heads, and two blocks; TensorFlow iTransformer variants use dimension 32, four heads, and two blocks. VAE-LSTM uses a 16-dimensional latent state with up to 30 warm-up epochs. Neural optimization uses Adam at 10^-3. Ordinary PyTorch models stop on mean validation-batch MAE after training-target normalization. VAE-LSTM stops on reconstruction plus annealed KL and health-prediction loss; the physics-informed TensorFlow model stops on the weighted validation loss of both heads (auxiliary weight 0.3). These monitored objectives are not identical to the raw SOH MAE reported for evaluation. Batch size is 64. These fixed settings are shared across datasets rather than retuned on external labels.

### C. Validation and statistical analysis

NASA and Oxford use five outer battery-grouped folds; CALCE uses four because it has only four cells. In each outer fold, a grouped 20% of the remaining cells forms validation data. Seeds 17, 42, and 2026 repeat partitioning and initialization. Battery-macro MAE identifies the lowest-error model descriptively after benchmark evaluation; it is not a nested, unbiased deployment-model selection estimate. Hyperparameters are fixed; validation batteries alone determine early stopping and ensemble fitting. For headline sample metrics, repeated held-out predictions for each record are averaged first and metrics are recomputed on those records. We distinguish that estimand from the mean of separate fold metrics. MAE = mean(|y − ŷ|), RMSE = √mean((y − ŷ)²), R² = 1 − SSE/SST, and within-5-pp coverage = mean(1[|y − ŷ| ≤ 5]). Adjusted R² uses p = 18 and is descriptive rather than a valid neural-network complexity correction. We also report MAPE, median and 95th-percentile absolute error, signed bias, residuals, and fold-level training versus held-out-battery gaps. Training MAE is in-sample and is not an unbiased validation score. All absolute errors are SOH percentage points; within-5 coverage is not R².

The five-fold design is not a 50% holdout. Table 8 gives test-fold SOH counts for all three seeds; Supplement S2 lists every training/validation/test battery and role-specific SOH quartiles, ranges and exposure support. GroupKFold shuffles cell identities without SOH or temperature stratification. Fold balance is therefore not claimed: several folds have no EOL observations, and only three cells contribute any EOL cycles. Missing-class ROC values remain undefined where applicable.

Uncertainty is estimated by 10,000 battery-cluster bootstrap resamples. Paired per-battery MAE differences are tested with two-sided Wilcoxon signed-rank tests and Holm family-wise correction. Held-out-battery permutation importance uses ten repeats. Ablations remove thermal, voltage, or current feature families. Stress tests remove temperature inputs or add Gaussian noise with standard deviation equal to 5% of the training dispersion to voltage/current features.

The signed-rank unit is one battery-specific MAE after averaging repeated held-out predictions, giving 30 paired NASA cells, not 15 folds or 90 seed-cell replicates. The Holm family contains exactly 19 comparisons of Weighted Average Ensemble against each other catalog model, with two-sided tests and zero_method = zsplit. The reference was identified from the same benchmark, so these adjusted results are exploratory and do not correct that selection. Bootstrap intervals condition on fitted models and observed cells; shared training data also induce dependence between held-out-cell errors. With only four CALCE cells, percentile endpoints have limited stability and 10,000 draws do not create more independent cells. No CALCE population-level superiority claim is made.

### D. External validation

Within-dataset results retrain every model using grouped folds on CALCE or Oxford. Zero-shot results train and select models only on NASA; the fitted preprocessing and model are then frozen and applied to all target records. CALCE and Oxford labels are accessed only for final scoring. No target-domain fine-tuning, calibration, threshold selection, or feature normalization is permitted. Freezing includes the source-fitted median imputer, any scalar/channel standardization, training-target normalization, ensemble weights and stacking coefficients. Interpolation inside an observed partial segment is a fixed per-record operation with no pooled target-domain fit. Ambient temperature is numeric in degrees Celsius and temperature_observed is binary; neither uses a target-fitted encoder. Section IV-G gives the complete preprocessing contract.

### E. Reproducibility and software

Experiments ran on Windows-11-10.0.26200-SP0 with Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 20 logical CPU cores, 15.71 GiB RAM, and no CUDA device. The locked environment used Python 3.12.10, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.9.1, SciPy 1.18.1, PyTorch 2.14.0+cpu, TensorFlow 2.20.0, XGBoost 2.1.4, and LightGBM 4.7.0. The repository stores ordered notebooks, source checksums, row-level predictions, fold metrics, exclusions, and figure source tables.

### F. Secondary health band evaluation

The secondary analysis derives four descriptive health bands from SOH: Healthy ≥90%, Aging 80–90%, Near-EOL 70–80%, and EOL <70%. The same fixed boundaries apply to observed labels and averaged held-out SOH estimates. These are research labels rather than validated safety or serviceability decisions. All 20 NASA regressors are evaluated; no separately trained classifier is claimed.

For ROC scoring, each model-fold estimate is combined with its empirical validation residuals r = observed SOH − predicted SOH. The probability of a band is the proportion of validation residuals for which predicted SOH + r falls in that band. Calibration uses validation batteries only. For ensembles, leave-one-validation-battery-out meta fitting generates calibration predictions; the final test ensemble is fitted on all validation batteries. Test labels never influence calibration, ensemble weights, or stopping. Validation data also support early stopping; the empirical scores do not provide a distribution-free calibration guarantee.

Necessary NASA folds were rerun with the original grouped splits, seeds 17, 42, and 2026, maximum 200 epochs, patience 20, and batch size 64; validation and test predictions are retained in a separate restoration package. Reruns use one CPU thread per worker and explicitly seed TensorFlow initialization; neural estimates may differ from archived benchmark estimates. Three held-out SOH estimates and four-band probabilities are averaged per model and cycle before pooled evaluation. Confusion matrices use bands of averaged SOH, whereas ROC curves use averaged calibrated probabilities. Balanced accuracy is the mean recall over supported classes; macro F1 uses all four fixed bands. One-versus-rest AUC is undefined when a class lacks positive or negative examples. Secondary rerun results are distinct from the preserved regression benchmark.

The capacity/RUL plot is descriptive. Retrospective RUL is the difference between a recorded cycle number and the first later observed SOH <70% crossing, clipped at zero after that event. Cells with no observed crossing are right-censored and receive no exact RUL curve. This manuscript does not report predictive RUL accuracy.

### G. Frozen preprocessing and partition contract

Scalar pipelines fit a median imputer on training rows only; SVR, Ridge and KNN additionally fit StandardScaler on those rows, whereas tree pipelines do not standardize. SequenceStandardizer fits channel medians, means and population standard deviations on training sequences flattened across time; zero scales are set to one. Neural targets use the training SOH mean and standard deviation. These fitted quantities are reused unchanged for validation, test and frozen transfer. No interpolation, imputation statistic or temperature encoder is estimated from a target-domain pool.

Inside each observed partial segment, missing sequence voltage/current/temperature values are linearly interpolated with endpoint filling before resampling on 64 normalized-time bins. Scalar thermal traces use the same within-segment temperature filling. When temperature is entirely absent, the predeclared ambient value replaces it and the observation indicator is zero; a wholly missing model input is handled by the source-trained imputer. This interpolation may use later samples inside the already completed partial segment, so the supported prediction time is segment completion, not every instant within it. Oxford current reconstruction and its fixed channel-quality rule are dataset-adapter operations, not target-trained normalization.

For each NASA seed-fold fit, 19 training, five validation and six test cells are mutually disjoint. Base learners fit training cells; inverse validation-MAE weights and the Ridge stacking meta-model fit validation predictions only and are then frozen for that fold's test cells. Weights are re-estimated independently for each outer fold/seed, never from its test labels. Separate full-source transfer ensembles learn NASA out-of-fold weights/meta coefficients and refit base models on NASA before scoring CALCE/Oxford. The secondary health-band calibration further cross-fits meta predictions by validation battery, as specified in Section IV-F.

Let ŷ̄bt be the arithmetic mean of the three held-out predictions for cycle t of battery b. Sample MAE = Σb Σt |ybt − ŷ̄bt| / Σb nb. Battery-macro MAE = (1/B) Σb [(1/nb) Σt |ybt − ŷ̄bt|]. Fold-mean MAE averages the 15 separate NASA seed-fold MAEs before prediction pooling and is a third estimand. Table 3 reports sample MAE; Table 4 labels battery-macro and sample columns separately. Figure 9 is a fold-diagnostic display, not a competing estimate of Table 3 or evidence that one aggregation is superior. Supplement S5 exports all three for every NASA model.

## V. RESULTS

### A. The former headline score was not valid

Table 2 separates two sources of optimism. With the safe feature set, Extra Trees MAE rises from 0.732 under a random-cycle split to 2.576 under within-battery chronological testing and 4.145 under held-out-battery testing. Adding current-cycle target proxies reduces grouped MAE to 0.436. The earlier 99.1% within-five-point statement therefore cannot be interpreted as cross-battery performance and is withdrawn.

### B. NASA grouped comparison

Across all 20 candidates, Weighted Average Ensemble has the lowest NASA mean per-battery MAE (3.536 pp). On averaged held-out predictions its sample MAE is 3.682 pp, RMSE 6.011 pp, MAPE 4.37%, R² 0.610, adjusted R² 0.607, and within-5-pp coverage 79.2%. The median absolute error is 1.815 pp, the 95th percentile is 16.759 pp, and signed bias is −0.817 pp. Thus the moderate R² and long error tail qualify the apparently favorable threshold coverage (Figs. 10–11). Mean-of-fold metrics give MAE 3.779 pp and RMSE 5.347 pp, as plotted in Fig. 9; these are not the same aggregation as the sample statistics in Tables 3–4.

Table 3. Sample MAE in SOH pp from averaged held-out predictions. Grouped columns use separate in-dataset fits; transfer columns use frozen NASA fits

| Model | NASA grouped | CALCE grouped | Oxford grouped | NASA to CALCE | NASA to Oxford |
| --- | --- | --- | --- | --- | --- |
| Extra Trees | 4.15 | 1.26 | 0.66 | 9.33 | 4.69 |
| Gradient Boosting | 3.96 | 1.29 | 0.80 | 8.25 | 5.91 |
| Random Forest | 3.83 | 1.25 | 0.63 | 8.44 | 5.09 |
| XGBoost | 3.96 | 1.21 | 0.84 | 7.86 | 4.62 |
| LightGBM | 3.86 | 1.22 | 0.66 | 7.82 | 5.27 |
| Support Vector Regression | 3.75 | 2.83 | 2.09 | 17.37 | 7.34 |
| Ridge Regression | 5.84 | 1.55 | 0.74 | 29.31 | 478.83 |
| K-Nearest Neighbours | 4.77 | 1.74 | 1.54 | 13.17 | 5.85 |
| Vanilla LSTM | 7.55 | 6.25 | 4.42 | 13.91 | 8.95 |
| Bidirectional LSTM | 6.74 | 5.76 | 3.83 | 14.13 | 9.87 |
| GRU | 7.60 | 3.24 | 3.14 | 14.34 | 8.48 |
| Attention LSTM | 7.94 | 3.88 | 5.62 | 14.17 | 9.86 |
| BatteryGPT | 8.40 | 5.19 | 4.03 | 12.98 | 9.81 |
| Temporal Fusion Transformer | 7.35 | 4.22 | 4.55 | 16.64 | 8.25 |
| iTransformer | 6.87 | 4.69 | 1.49 | 19.97 | 7.45 |
| Physics-Informed iTransformer | 6.99 | 5.13 | 1.29 | 20.41 | 8.05 |
| Dynamic-Graph iTransformer | 6.39 | 4.54 | 3.23 | 16.68 | 8.44 |
| VAE-LSTM | 7.49 | 6.46 | 4.70 | 14.46 | 11.02 |
| Stacking Ensemble | 4.00 | 1.15 | 0.88 | 13.17 | 30.26 |
| Weighted Average Ensemble | 3.68 | 1.29 | 0.69 | 6.95 | 51.87 |

[Download Table 3 as CSV](/research/reviewer4/tables/table_03.csv)

Table 4. Grouped-dataset winners on averaged held-out predictions; all errors are SOH pp

| Dataset | Winner | Battery MAE | Sample MAE | RMSE | R² | Adj. R² | Within 5 pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NASA | Weighted avg. | 3.536 | 3.682 | 6.011 | 0.610 | 0.607 | 79.2% |
| CALCE | Stacking | 1.153 | 1.150 | 1.855 | 0.989 | 0.989 | 97.5% |
| Oxford | Random forest | 0.622 | 0.625 | 1.056 | 0.976 | 0.975 | 99.6% |

[Download Table 4 as CSV](/research/reviewer4/tables/table_04.csv)

![Figure 9](/research/reviewer4/figures/figure_09_nasa_twenty_model_comparison.svg)

Fig. 9. All 20 NASA models under the common grouped protocol. Bars report mean fold MAE, which differs from the battery-macro selection statistic and sample MAE.

![Figure 10](/research/reviewer4/figures/figure_10_nasa_observed_vs_predicted.svg)

Fig. 10. NASA Weighted Average Ensemble averaged held-out predictions. The diagonal represents perfect agreement and is not a fitted calibration line.

![Figure 11](/research/reviewer4/figures/figure_11_residual_diagnostics.svg)

Fig. 11. NASA ensemble residual trend and distribution. Residuals are observed minus predicted SOH; asymmetry and tails motivate bias and error-quantile reporting.

### C. Secondary health band results

The NASA secondary evaluation contains Healthy 1089, Aging 573, Near-EOL 394, EOL 85 cycles. The largest pooled balanced accuracy is 0.670 for Stacking Ensemble; the largest macro F1 is 0.639 for Stacking Ensemble. These are descriptive rankings after evaluation, not test-driven model selection or evidence of operational classification reliability. Table 5 reports all 20 models, and Figures 12–17 retain separate family-level confusion and ROC views.

For the model with the largest balanced accuracy, pooled per-class recalls are Healthy 0.781, Aging 0.733, Near-EOL 0.670, EOL 0.494, and macro one-versus-rest AUC is 0.888. These recall values measure fixed-band decisions, whereas AUC measures ranking from the validation-residual scores. The EOL band has only three contributing batteries, so its cycle count overstates the number of independent units supporting that result.

Table 5. NASA secondary health band metrics from averaged held-out rerun predictions

| SOH model | Balanced accuracy | Macro F1 | Macro OVR AUC |
| --- | --- | --- | --- |
| Attention LSTM | 0.332 | 0.285 | 0.587 |
| BatteryGPT | 0.325 | 0.308 | 0.616 |
| Bidirectional LSTM | 0.332 | 0.303 | 0.601 |
| Dynamic-Graph iTransformer | 0.454 | 0.430 | 0.701 |
| Extra Trees | 0.517 | 0.500 | 0.818 |
| Gradient Boosting | 0.537 | 0.513 | 0.835 |
| GRU | 0.320 | 0.269 | 0.616 |
| iTransformer | 0.480 | 0.449 | 0.689 |
| K-Nearest Neighbours | 0.548 | 0.537 | 0.853 |
| LightGBM | 0.536 | 0.514 | 0.855 |
| Physics-Informed iTransformer | 0.480 | 0.449 | 0.689 |
| Random Forest | 0.531 | 0.510 | 0.853 |
| Ridge Regression | 0.526 | 0.514 | 0.855 |
| Stacking Ensemble | 0.670 | 0.639 | 0.888 |
| Support Vector Regression | 0.587 | 0.554 | 0.886 |
| Temporal Fusion Transformer | 0.320 | 0.304 | 0.600 |
| VAE-LSTM | 0.324 | 0.302 | 0.588 |
| Vanilla LSTM | 0.308 | 0.277 | 0.592 |
| Weighted Average Ensemble | 0.571 | 0.546 | 0.847 |
| XGBoost | 0.526 | 0.505 | 0.852 |

[Download Table 5 as CSV](/research/reviewer4/tables/table_05.csv)

OVR denotes one-versus-rest. These metrics summarize descriptive SOH-derived bands rather than a separately fitted classification endpoint. Class support and per-class recall are supplied in the reproducibility package.

Figures 12 and 13 show where the eight classical models confuse adjacent SOH bands and how their validation-residual scores rank each band. Counts and row percentages expose the effect of class imbalance.

![Figure 12](/research/reviewer4/figures/figure_12_classical_confusion_matrices.svg)

Fig. 12. NASA classical model health-band confusion matrices derived from averaged held-out SOH estimates in the secondary reruns. Cells show counts and observed-class percentages. H Healthy, A Aging, N Near-EOL, E EOL; color intensity represents row-normalized frequency.

![Figure 13](/research/reviewer4/figures/figure_13_classical_roc_curves.svg)

Fig. 13. NASA classical one-versus-rest ROC curves from the secondary reruns. Scores integrate an empirical validation-residual distribution around each SOH estimate. Residuals come from disjoint validation batteries. Legends give pooled AUC; unsupported classes are undefined.

Figures 14 and 15 provide the same secondary analysis for all ten sequence models. AUC measures ranking across thresholds; it does not establish reliable decisions at a fixed health-band boundary.

![Figure 14](/research/reviewer4/figures/figure_14_deep_confusion_matrices.svg)

Fig. 14. NASA deep model health-band confusion matrices derived from averaged held-out SOH estimates in the secondary reruns. Cells show counts and observed-class percentages. H Healthy, A Aging, N Near-EOL, E EOL; color intensity represents row-normalized frequency.

![Figure 15](/research/reviewer4/figures/figure_15_deep_roc_curves.svg)

Fig. 15. NASA deep one-versus-rest ROC curves from the secondary reruns. Scores integrate an empirical validation-residual distribution around each SOH estimate. Residuals come from disjoint validation batteries. Legends give pooled AUC; unsupported classes are undefined.

Figures 16 and 17 show the two ensembles. Their calibration residuals come from validation-battery cross-fitting of ensemble weights or the stacking model, rather than in-sample meta predictions.

![Figure 16](/research/reviewer4/figures/figure_16_ensemble_confusion_matrices.svg)

Fig. 16. NASA ensemble model health-band confusion matrices derived from averaged held-out SOH estimates in the secondary reruns. Cells show counts and observed-class percentages. H Healthy, A Aging, N Near-EOL, E EOL; color intensity represents row-normalized frequency.

![Figure 17](/research/reviewer4/figures/figure_17_ensemble_roc_curves.svg)

Fig. 17. NASA ensemble one-versus-rest ROC curves from the secondary reruns. Scores integrate an empirical validation-residual distribution around each SOH estimate. Ensemble residuals use battery-cross-fitted meta predictions. Legends give pooled AUC; unsupported classes are undefined.

### D. External validation

Grouped fits trained separately within CALCE yield the lowest battery-macro MAE of 1.153 pp for Stacking Ensemble (sample MAE 1.150 pp, RMSE 1.855 pp, R² 0.989, within-5 coverage 97.5%). Oxford's grouped winner is Random Forest at macro MAE 0.622 pp (sample MAE 0.625 pp, RMSE 1.056 pp, R² 0.976, coverage 99.6%). These are in-dataset retraining results, not NASA-model transfer scores; Fig. 18 shows the distinction.

In frozen zero-shot testing, the lowest NASA-to-CALCE sample MAE is 6.955 pp from Weighted Average Ensemble, and the lowest NASA-to-Oxford MAE is 4.615 pp from XGBoost (Fig. 19). Target records do not refit scaling, select models, or calibrate predictions. The gap from the grouped CALCE and Oxford results is evidence of substantial dataset shift, although these laboratory datasets cannot quantify field performance.

![Figure 18](/research/reviewer4/figures/figure_18_external_grouped.svg)

Fig. 18. Five lowest sample-MAE models in grouped CALCE and Oxford fits, trained separately within each target dataset.

![Figure 19](/research/reviewer4/figures/figure_19_external_zero_shot.svg)

Fig. 19. Five lowest sample-MAE frozen NASA models on CALCE and Oxford, without target-domain refitting or calibration. These are post-evaluation rankings, not a target-domain selection policy.

### E. Importance ablations and sensor robustness

In the held-out-battery ExtraTrees analysis, permuting cycle index increases MAE by 1.792 pp, prior equivalent full cycles by 0.721, mean temperature by 0.511, ambient temperature by 0.461, and segment energy by 0.323 (Fig. 20). Correlated features share information, so these are predictive diagnostics, not causal effects. Retraining with all safe features yields fold-mean MAE 4.384 pp; time/usage only gives 5.568, without voltage 4.759, without temperature 4.297, and without current 4.320 pp (Fig. 21). The small temperature-ablation improvement must not be read as proof that temperature is useless.

The added paired temperature-ablation analysis retains all held-out cycle predictions and resamples whole cells. The sample-MAE change after retraining without temperature is -0.130 pp (95% interval -0.352 to 0.133); this interval includes zero. It concerns sample-weighted pooled errors, whereas the 4.297 versus 4.384 comparison is a mean of five fold MAEs. Figure 26 and Section VII-D add feature/block permutation intervals and an explicit predictive interaction contrast.

In a separate sensor-stress experiment, corrupting the trained full model's temperature inputs raises fold-mean MAE from 4.384 to 5.373 pp; adding voltage/current noise at 5% of training standard deviation yields 4.421 pp (Fig. 22). Unlike ablation, this experiment does not retrain a reduced-input model. The synthetic perturbations are not hardware fault-injection evidence.

![Figure 20](/research/reviewer4/figures/figure_20_permutation_importance.svg)

Fig. 20. Eight largest held-out-battery ExtraTrees permutation effects. Correlated predictors share information; these effects are predictive diagnostics rather than causal mechanisms.

![Figure 21](/research/reviewer4/figures/figure_21_feature_ablation.svg)

Fig. 21. Feature-family ablations after retraining ExtraTrees. Removing a family differs from corrupting a trained model’s inputs.

![Figure 22](/research/reviewer4/figures/figure_22_sensor_stress.svg)

Fig. 22. Synthetic sensor stress for the trained full-feature ExtraTrees model: unmodified inputs, missing temperature, and voltage/current noise at 5% of training standard deviation.

### F. Uncertainty and paired comparisons

For the selected NASA model, 10,000 battery-cluster bootstrap resamples give sample MAE 3.682 pp with a 95% interval of 1.879–5.966 pp and within-5 coverage 79.2% with a 61.4–94.4% interval (Fig. 23). Exploratory paired battery-level Wilcoxon tests with Holm adjustment reject a zero signed-rank location difference in 13 of 19 comparisons at adjusted p < 0.05; they do not test equality of means. The comparison with the released SVR model is not significant (adjusted p = 0.824). Across 15 folds, the Weighted Average Ensemble has mean in-sample training MAE 0.562 pp versus held-out-battery MAE 3.779 pp, a 3.217 pp generalization gap. This substantial gap cautions against interpreting a fitted training score as deployment performance. Fold-level gaps, residuals, and predictions are supplied with the reproducibility artifacts; no single R² score substitutes for these checks.

For CALCE's grouped Stacking Ensemble, the conditional sample-MAE interval is 0.745–1.543 pp around 1.150 pp. Omitting one of its four observed cells at a time changes the already-fitted prediction-table sample MAE to 0.968–1.345 pp (Supplement S6). This is a cell-influence sensitivity, not retraining or independent external replication. CALCE and Oxford grouped model rankings are post-evaluation summaries; frozen NASA transfer remains a separate experiment.

![Figure 23](/research/reviewer4/figures/figure_23_battery_cluster_intervals.svg)

Fig. 23. Sample MAE with 95% battery-cluster bootstrap intervals for each dataset’s grouped winner, using 10,000 cluster resamples. CALCE has only four independent cells.

## VI. RESEARCH PROTOTYPE IMPLEMENTATION

The GitHub source repository, Hugging Face model repository, and Hugging Face Space are separate release surfaces. The 20-model paper comparison selects Weighted Average Ensemble by NASA battery-macro MAE, whereas the eight packaged classical models use SVR as the API default (macro MAE 3.628 pp among deployable files). The ensemble is not one of those packaged model files. The FastAPI v3 endpoint accepts the ordered 18-feature partial-discharge contract and returns current-cycle SOH; it exposes neither a validated v3 RUL estimate nor a calibrated prediction interval. Fig. 24 separates observable inputs from the full-capacity label. The React/Docker interface is a research prototype. Ordered notebooks, dataset checksums, split records, prediction CSVs, metric tables, and figure sources are published at https://github.com/neerajcodz/aiBatterylifecycle and https://huggingface.co/NeerajCodz/aiBatteryLifeCycle.

Recommendation and multi-step simulation screens are retained only as exploratory software prototypes. They have not been evaluated as closed-loop charging policies, do not represent electrochemical safety constraints, and must not be used for operational decisions. A credible charging-optimization claim would require controlled prospective experiments, cell and pack constraints, uncertainty-aware objectives, and validation across fast-charging and field duty cycles [14].

The v3 CSV supplement supplies full fold, transfer, importance, ablation, stress, bootstrap, and train/test results (selected-ensemble folds: results/nasa_classical_fold_metrics.csv). Its aggregations are distinct from manuscript Tables 3–4.

![Figure 24](/research/reviewer4/figures/figure_24_leakage_safe_workflow.svg)

Fig. 24. Leakage-safe signal and label paths. Full-cycle capacity constructs the retrospective SOH label and never enters predictors; calibration uses validation batteries and evaluation uses held-out batteries.

## VII. ADDITIONAL AUDIT AND REPRODUCIBILITY RESULTS

This section makes the audit reproducible while preserving the original 24 figures and benchmark results. Tables 6–9 specify the invalid proxy definitions, observable feature rationale, fold support and actual stopping counts. Figures 25–27 add controlled audit sensitivities, conditional importance intervals and validation/test diagnostics. Complete per-fit records are supplied separately; no original result is silently replaced.

### A. Proxy and feature definitions

Table 6. Deliberately invalid target-proxy definitions used only in the audit

| Proxy | Exact calculation | Unit | Window and boundary |
| --- | --- | --- | --- |
| Current retention | Qk / Qref | Fraction | Current retained row; exactly yk / 100 |
| Current cumulative capacity | Σj=1..k Qj | Ah | Expanding sum including current retained row |
| Current capacity difference | Qk − Qk−1 | Ah | Previous retained row; first row is zero |
| Current SOH rolling mean | mean(yj), j = max(1,k−4)..k | SOH % | Trailing five retained rows, current included; minimum one |

[Download Table 6 as CSV](/research/reviewer4/tables/table_06.csv)

Rows are sorted by battery_id and cycle_number before proxy construction; group boundaries reset all accumulations and windows. Qref is the same retrospective reference used for the target. The audit is reproduced by scripts/paper/r4_analysis.py controls; the source definitions are application/scripts/run_protocol_comparison.py::_add_legacy_target_proxies. The previous-cycle throughput feature in the observable contract is different: it excludes Qk and uses rated capacity as its denominator.

Table 7. Observable feature families and representation rationale

| Family | Count | Included information | Rationale |
| --- | --- | --- | --- |
| Usage and duration | 3 | Cycle index; prior equivalent full cycles; segment duration | Exposure and elapsed usage available by segment completion |
| Voltage shape | 6 | Start, end, mean, standard deviation, linear slope, quadratic curvature | Partial-discharge shape and polarization-sensitive summaries |
| Current loading | 2 | Mean and standard deviation of absolute C-rate | Loading normalized by published rated Ah, not measured Qk |
| Partial energy | 1 | Σ \|Ii\| Vi max(Δti,0) / 3600 | Energy inside the partial segment only; not full capacity |
| Thermal context | 6 | Ambient, mean, standard deviation, range, slope, observation indicator | Exposure, within-segment heating and missingness; not causal degradation |
| Sequence representation | 64 × 5 | Normalized time, voltage, absolute C-rate, temperature, observation indicator | Linear resampling of the same completed partial segment |

[Download Table 7 as CSV](/research/reviewer4/tables/table_07.csv)

Scalar standard deviations use population normalization (ddof = 0); slopes are degree-one least-squares coefficients versus elapsed seconds; voltage curvature is the quadratic coefficient versus normalized segment time. The thermal rise feature is max(T) − min(T), not endpoint difference. Segment extraction starts at the first sample ≤4.0 V and ends at the first subsequent sample ≤3.6 V; it does not interpolate exact voltage-boundary crossings. At least three samples and positive sequence duration are required. Full implementations are in src/data/partial_cycle.py and the supplement.

### B. Split support and stopping records

Table 8. NASA test-fold support; every fold has 19 training, five validation and six test cells

| Seed | Fold | Cycles | Healthy | Aging | Near-EOL | EOL | SOH range (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | 1 | 423 | 202 | 105 | 114 | 2 | 69.7–114.0 |
| 17 | 2 | 590 | 224 | 204 | 162 | 0 | 72.6–109.8 |
| 17 | 3 | 584 | 328 | 125 | 68 | 63 | 57.0–110.0 |
| 17 | 4 | 270 | 187 | 67 | 16 | 0 | 75.3–113.8 |
| 17 | 5 | 274 | 148 | 72 | 34 | 20 | 65.4–116.6 |
| 42 | 1 | 316 | 184 | 86 | 46 | 0 | 74.1–114.0 |
| 42 | 2 | 451 | 201 | 124 | 124 | 2 | 69.7–113.8 |
| 42 | 3 | 504 | 195 | 103 | 123 | 83 | 57.0–116.6 |
| 42 | 4 | 447 | 167 | 192 | 88 | 0 | 72.8–110.0 |
| 42 | 5 | 423 | 342 | 68 | 13 | 0 | 75.3–110.0 |
| 2026 | 1 | 479 | 298 | 93 | 68 | 20 | 65.4–116.6 |
| 2026 | 2 | 371 | 158 | 120 | 93 | 0 | 72.8–105.9 |
| 2026 | 3 | 449 | 284 | 99 | 64 | 2 | 69.7–102.0 |
| 2026 | 4 | 477 | 244 | 155 | 78 | 0 | 74.5–113.8 |
| 2026 | 5 | 365 | 105 | 106 | 91 | 63 | 57.0–110.7 |

[Download Table 8 as CSV](/research/reviewer4/tables/table_08.csv)

Class boundaries are the fixed secondary bands: Healthy ≥90%, Aging 80–90%, Near-EOL 70–80%, EOL <70%. Role-specific memberships, class support and quartiles are in Supplement S2 and nasa_split_inventory.csv; no redistribution was made to improve the balance after observing scores.

Table 9. Original NASA deep-model stopping and generalization diagnostics; 15 fits per model

| Model | Executed epochs min / median / max | Train MAE | Test MAE | Test − train |
| --- | --- | --- | --- | --- |
| Attention LSTM | 21 / 45 / 94 | 4.922 | 8.171 | 3.249 |
| BatteryGPT | 22 / 39 / 99 | 4.275 | 8.475 | 4.200 |
| Bidirectional LSTM | 21 / 52 / 89 | 4.468 | 6.930 | 2.463 |
| Dynamic-Graph iTransformer | 21 / 44 / 91 | 3.161 | 6.800 | 3.639 |
| GRU | 21 / 32 / 85 | 5.963 | 7.619 | 1.656 |
| iTransformer | 21 / 39 / 56 | 3.231 | 7.295 | 4.065 |
| Physics-Informed iTransformer | 21 / 33 / 78 | 3.497 | 7.533 | 4.036 |
| Temporal Fusion Transformer | 21 / 53 / 120 | 4.547 | 7.658 | 3.110 |
| VAE-LSTM | 25 / 56 / 116 | 4.134 | 7.790 | 3.656 |
| Vanilla LSTM | 22 / 28 / 52 | 5.854 | 7.494 | 1.640 |

[Download Table 9 as CSV](/research/reviewer4/tables/table_09.csv)

All errors are fold-mean SOH pp. Executed epochs include patience epochs; they are not the selected best epoch. The complete 420 grouped-fit records cover 150 NASA, 120 CALCE and 150 Oxford fits; two vanilla-LSTM fits reached 200 epochs (one CALCE, one Oxford), so hitting the cap is distinguished from patience stopping. The 150 cached NASA secondary reruns provide validation/test MAE and their difference, shown in Fig. 27 and Supplement S4. Original validation histories and best-epoch identities are not recoverable from the preserved release.

### C. Matched audit sensitivity

The added audit runs 63 fixed-hyperparameter fits per design (21 seed/split instances × three input contracts). In the first design all fits train on 1,152 rows, the smallest original training partition. Within a split, observable, observable-plus-real-proxy and observable-plus-shuffled-proxy fits use exactly the same sampled training and test rows. The shuffled four columns preserve each partition's marginal proxy values but remove their row correspondence; no held-out rows are used as training donors. This controls input dimensionality as well as count, although a shuffled column is not a universal negative control for every alternative feature family.

The second design trains on the common per-temperature minima across all 21 candidate training partitions: 416 rows at 4 °C, 380 at 24 °C and 40 at 43 °C (836 total). Temperatures 22 °C and 44 °C have zero common training support and are omitted only from this sensitivity's training rows; their held-out records remain scored. Per-temperature and C-rate error/support tables are exported. The rare exposure cells, differing test populations and residual battery/history confounding prevent causal cross-protocol attribution even after these controls.

![Figure 25](/research/reviewer4/figures/figure_25_matched_proxy_controls.svg)

Fig. 25. Additional ExtraTrees controls. Within each split, all contracts use identical training and test rows, fixed hyperparameters, and either 18 observable features or 18 plus four real/shuffled proxy features. Panel (a) fixes training count at 1,152; panel (b) uses 836 rows with common temperature quotas. Test populations differ across V1/V2/V3, so between-protocol differences are descriptive, not isolated causal split effects.

For grouped V3, equal-count mean fold MAEs are 4.171 pp (observable), 4.181 pp (shuffled proxies), and 0.439 pp (real proxies). Temperature-quota counterparts are 4.190, 4.189, and 0.429 pp. These repeated within-split contrasts support the leakage interpretation; the original V1/V2/V3 means remain descriptive comparisons.

### D. Importance uncertainty and interaction diagnostic

For seed 42, five grouped ExtraTrees fits reproduce the full-feature and temperature-ablation experiment. We permute individual features and joint thermal/voltage blocks on held-out records ten times, with paired battery-cluster resampling of the resulting absolute-error differences. Joint shuffling uses one donor permutation for all columns in a block. Pooled shuffling can break between-cell exposure structure; within-battery shuffling preserves each cell's ambient context while perturbing within-cell variation. Neither estimates a causal feature effect or samples model-retraining uncertainty. Intervals are pointwise, with no family-wise claim across the inspected features.

![Figure 26](/research/reviewer4/figures/figure_26_grouped_permutation_intervals.svg)

Fig. 26. Joint feature permutation and an additive interaction contrast for five held-out NASA ExtraTrees fits (seed 42). Each point averages ten permutations; bars are pointwise 95% intervals from 10,000 paired battery-cluster resamples, conditional on fitted models. Temperature has six features and voltage six. Within-battery shuffling preserves each cell's exposure context. The interaction contrast is Δ(T,V) − Δ(T) − Δ(V); it is a predictive perturbation diagnostic, not a causal interaction or multiplicity-adjusted test.

Pooled thermal-block permutation increases sample MAE by 1.235 pp (0.672 to 1.940), whereas within-battery thermal permutation yields 0.132 pp (-0.007 to 0.266). Thus the apparent temperature importance depends strongly on which exposure structure the perturbation preserves. The joint-minus-additive temperature/voltage contrast is -0.126 pp (-0.379 to 0.069) under pooled permutation and 0.057 pp (0.009 to 0.104) within cells. These conditional interaction diagnostics can reflect redundancy and extrapolation under perturbation; they do not establish an electrochemical mechanism. The paired retraining interval crossing zero remains the basis for treating the small ablation improvement as inconclusive.

### E. Validation and test diagnostics

![Figure 27](/research/reviewer4/figures/figure_27_secondary_validation_test_gap.svg)

Fig. 27. Validation and held-out-test MAE for all ten deep models in the saved NASA secondary reruns, averaged over 15 seed-fold fits per model. Validation batteries support stopping and therefore yield optimistic diagnostics; test batteries are disjoint. These rerun outputs are kept separate from the original regression results in Tables 3–4 and 9.

Across the secondary NASA reruns, model-level mean test-minus-validation MAE ranges from 1.200 to 3.529 pp. Validation batteries participate in stopping and are not an independent final performance set. Original and secondary stopping/score records are separately labeled in the supplement; the reruns do not replace the primary benchmark or reconstruct missing original training histories. No model was stopped, tuned or assigned ensemble weights using its outer test batteries.

## VIII. DISCUSSION

The principal result is methodological. The original diagnostic reports safe-feature ExtraTrees MAE below one point for random-cycle testing and above four points for held-out-battery testing, but these tests differ in populations, histories and exposure composition. The matched controls in Fig. 25 narrow the within-split conclusion: access to current-target proxies lowers error far more than adding equally many shuffled proxies on the same sampled training and test rows. This demonstrates an invalid information pathway; it does not assign a causal MAE penalty to grouping alone. The corrected benchmark estimates whether an observable partial discharge predicts retrospective current SOH on a physically unseen battery.

No architecture dominates every domain. On NASA, the Weighted Ensemble's mean-of-fold MAE is 3.779 pp, versus 3.931 for SVR, 3.963 for Random Forest, and 6.800 for the best sequence model, Dynamic-Graph iTransformer. The common feature contract is learnable within CALCE and Oxford, but their frozen transfer errors are materially higher. Prior work with labeled target adaptation [15], [19] answers a different question; adaptation should be evaluated explicitly rather than counted as zero-shot validation.

Feature importance and ablations connect the selected predictors to performance, but they do not identify degradation  mechanisms. Temperature absence is partly mitigated by the observation indicator and imputation, whereas real sensors may exhibit drift, correlated failure, or timing error beyond the synthetic stresses used here. The cluster bootstrap accounts for within-cell dependence at the battery level, although uncertainty remains wide with only four CALCE cells.

## IX. LIMITATIONS

All three sources are laboratory aging datasets. They do not reproduce the full chemistry, manufacturer, pack-interaction, fast-charge, calendar-aging, and driving diversity of operational EV fleets. Dataset adapters harmonize signals but cannot remove protocol mismatch. The robust reference uses whole-cell capacity history to define retrospective labels and would need a prospective initialization procedure in deployment. NASA quality exclusions are fixed before fitting; Oxford current is reconstructed from capacity change and time; CALCE contributes only four cells to grouped testing. The primary endpoint is current-cycle SOH; secondary health bands are derived retrospectively from SOH estimates. RUL curves are descriptive, and charging benefit and safety risk remain unvalidated. Prospective fleet and pack-level validation is required before deployment.

The secondary four-band results use NASA laboratory records only. The 85 EOL-band cycles come from just three independent batteries; their recall and ranking scores are sensitive to this limited cell support. Band thresholds are conventional analysis choices, not independently validated battery safety limits. Empirical residual scores assume validation errors are relevant to held-out cells and may be poorly calibrated under cell or domain shift; ROC AUC does not establish probability calibration. Rerun outputs and original regression outputs are kept separate. Censoring limits retrospective RUL descriptions, and no future-life predictor is evaluated.

The 4.0–3.6 V window and 64-point resolution have not been validated as optimal or invariant across chemistry, temperature or C-rate. Equal-count and temperature-quota controls still leave different V1/V2/V3 test populations; they cannot identify a pure causal grouping effect. Original deep training artifacts contain total executed epochs and training/test MAE but not recoverable per-epoch validation histories or restored-best-epoch identifiers. We therefore disclose original stopping counts separately from cached secondary validation/test diagnostics rather than fabricating missing histories. All added intervals are conditional, pointwise and exploratory.

## X. CONCLUSION

The original 99.1% within-five-point statement is withdrawn as evidence of held-out-battery performance. With the observable partial-discharge contract and physically disjoint test cells, the Weighted Average Ensemble reaches NASA battery-macro MAE 3.536 pp and sample MAE 3.682 pp, with a wide battery-bootstrap interval and a substantial residual tail. Separate grouped CALCE/Oxford fits perform well, but zero-shot NASA transfer is weaker. The public notebooks, predictions, exclusions, tables, and high-resolution plots make the evaluation auditable. Prospective pack/fleet testing, domain adaptation, and controlled charging-policy studies are required before operational claims.

The restored secondary analysis achieves a largest balanced accuracy of 0.670 and a largest macro F1 of 0.639, with fixed SOH bands and battery-grouped evaluation. These pooled diagnostics supplement current-cycle SOH regression; they do not establish calibrated safety decisions. Only three of the thirty NASA cells have an observed SOH <70% crossing, so retrospective RUL remains descriptive and censored for the other cells.

## XI. REFERENCES

[1] Y. Li et al., “Data-driven health estimation and lifetime prediction of lithium-ion batteries: A review,” Renew. Sustain. Energy Rev., vol. 113, 109254, 2019.

[2] M. Berecibar et al., “Critical review of state of health estimation methods of Li-ion batteries for real applications,” Renew. Sustain. Energy Rev., vol. 56, pp. 572–587, 2016.

[3] B. Saha and K. Goebel, “Battery Data Set,” NASA Ames Prognostics Data Repository, 2007.

[4] D. Howey and C. Birkl, “Oxford Battery Degradation Dataset 1,” University of Oxford, 2017, doi: 10.5287/bodleian:KO2kdmYGg.

[5] Center for Advanced Life Cycle Engineering, “CALCE Battery Data,” University of Maryland, accessed Sep. 13, 2026. [Online]. Available: https://calce.umd.edu/data

[6] W. He, N. Williard, M. Osterman, and M. Pecht, “Prognostics of lithium-ion batteries based on Dempster–Shafer theory and the Bayesian Monte Carlo method,” J. Power Sources, vol. 196, no. 23, pp. 10314–10321, 2011.

[7] Y. Xing, E. W. M. Ma, K. L. Tsui, and M. Pecht, “An ensemble model for predicting the remaining useful performance of lithium-ion batteries,” Microelectron. Reliab., vol. 53, no. 6, pp. 811–820, 2013.

[8] N. Williard, W. He, M. Osterman, and M. Pecht, “Comparative analysis of features for determining state of health in lithium-ion batteries,” Int. J. Prognostics Health Manage., vol. 4, pp. 1–7, 2013.

[9] Y. Zhang, R. Xiong, H. He, and M. G. Pecht, “Long short-term memory recurrent neural network for remaining useful life prediction of lithium-ion batteries,” IEEE Trans. Veh. Technol., vol. 67, no. 7, pp. 5695–5705, 2018.

[10] X. Chen et al., “An adaptive prediction model for the remaining life of a Li-ion battery based on the fusion of Transformer and convolutional neural network,” Electronics, vol. 11, no. 10, 1605, 2022.

[11] K. Liu, Y. Shang, Q. Ouyang, and W. D. Widanage, “A data-driven approach with uncertainty quantification for predicting future capacities and remaining useful life of lithium-ion battery,” IEEE Trans. Ind. Electron., vol. 68, no. 4, pp. 3170–3180, 2021.

[12] J. Tian et al., “Deep neural network battery charging curve prediction using 30 points collected in 10 min,” Joule, vol. 5, no. 6, pp. 1521–1534, 2021.

[13] K. A. Severson et al., “Data-driven prediction of battery cycle life before capacity degradation,” Nat. Energy, vol. 4, pp. 383–391, 2019.

[14] P. M. Attia et al., “Closed-loop optimization of fast-charging protocols for batteries with machine learning,” Nature, vol. 578, pp. 397–402, 2020.

[15] T. Wang, Z. Ma, S. Zou, Z. Chen, and P. Wang, “Lithium-ion battery state-of-health estimation: A self-supervised framework incorporating weak labels,” Appl. Energy, vol. 355, 122332, 2024, doi: 10.1016/j.apenergy.2023.122332.

[16] G. Lyu, H. Zhang, and Q. Miao, “An adaptive and interpretable SOH estimation method for lithium-ion batteries based on relaxation voltage cross-scale features and multi-LSTM-RFR2,” Energy, vol. 304, 132167, 2024, doi: 10.1016/j.energy.2024.132167.

[17] L. Chen et al., “Domain generalization-based state-of-health estimation of lithium-ion batteries,” J. Power Sources, vol. 610, 234696, 2024, doi: 10.1016/j.jpowsour.2024.234696.

[18] X. Shu et al., “State of health estimation for lithium-ion batteries based on voltage segment and transformer,” J. Energy Storage, vol. 108, 115200, 2025, doi: 10.1016/j.est.2024.115200.

[19] Z. Cao, W. Gao, Y. Fu, N. V. Kurdkandi, and C. Mi, “A general framework for lithium-ion battery state of health estimation: From laboratory tests to machine learning with transferability across domains,” Appl. Energy, vol. 381, 125086, 2025, doi: 10.1016/j.apenergy.2024.125086.

[20] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. ACM SIGKDD, 2016, pp. 785–794.

[21] G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” in Proc. NeurIPS, 2017, pp. 3149–3157.

[22] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, pp. 1735–1780, 1997.

[23] A. Vaswani et al., “Attention is all you need,” in Proc. NeurIPS, 2017, pp. 5998–6008.

[24] R. R. Richardson, M. A. Osborne, and D. A. Howey, “Gaussian process regression for in-situ capacity estimation of lithium-ion batteries,” IEEE Trans. Ind. Informat., vol. 15, no. 1, pp. 127–138, 2019.

[25] S. Fu et al., “Recent progress in state of health estimation for lithium-ion batteries: From laboratory to practical application,” Renew. Sustain. Energy Rev., vol. 226, pt. B, 116323, 2026, doi: 10.1016/j.rser.2025.116323.

[26] G. Chen, “Research on an energy economic demand time series prediction model based on the integration of XGBoost and deep learning,” Journal of Discovery Core, vol. 1, no. 1, pp. 1–18, 2026, doi: 10.67541/jdc2601.
