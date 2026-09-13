# Models and evaluation

The authoritative catalog in `src/models/catalog.py` contains exactly 20
entries: eight tabular regressors, four recurrent networks, four
transformer-family models, one dynamic-graph iTransformer, one VAE-LSTM, and
two validation-only tabular ensembles.

All datasets use disjoint battery identities for outer testing and validation
batteries for stopping or ensemble fitting. Seeds are 17, 42, and 2026.
Neural runs use a 200-epoch cap, patience 20, and batch size 64. The
physics-informed iTransformer integrates the current C-rate channel and applies
an auxiliary-head loss weight of 0.3.

Selection uses macro per-battery MAE, followed by RMSE, serialized size, and CPU
latency. Reported diagnostics also include sample MAE/RMSE/MAPE, R², adjusted
R² for tabular fits, bias, median/p95 error, absolute five-point coverage,
residuals, train-test gaps, battery-cluster bootstrap intervals, Wilcoxon–Holm
tests, ablations, and sensor stress.

The former 99.1% and pre-review v3 results are withdrawn. Final corrected
metrics are generated from `artifacts/v3/results`; no number in this document
is hard-coded as a deployment claim.
