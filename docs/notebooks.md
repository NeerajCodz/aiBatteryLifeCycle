# Notebook Guide

## Execution Order

Run the notebooks sequentially — each builds on artifacts from previous ones.
All 9 notebooks have been fully executed and validated.

| # | Notebook | Purpose | Key Results |
|---|----------|---------|-------------|
| 01 | `01_eda.ipynb` | Exploratory Data Analysis | 30 batteries, 2678 cycles, 5 temp groups |
| 02 | `02_feature_engineering.ipynb` | Feature extraction + sliding windows | 2678×19 features, 1734 sequences (32×12) |
| 03 | `03_classical_ml.ipynb` | Classical ML (Optuna HPO) | **RF R²=0.957** (best overall), XGB RUL R²=0.536 |
| 04 | `04_lstm_rnn.ipynb` | LSTM/GRU family (CUDA) | Vanilla LSTM R²=0.507 best of family |
| 05 | `05_transformer.ipynb` | BatteryGPT, TFT, iTransformer | **TFT R²=0.881** (best DL) |
| 06 | `06_dynamic_graph.ipynb` | Dynamic-Graph iTransformer | DG-iTransformer R²=0.595 |
| 07 | `07_vae_lstm.ipynb` | VAE-LSTM + anomaly detection | R²=0.730, UMAP latent space |
| 08 | `08_ensemble.ipynb` | Stacking & Weighted Average | Weighted Avg R²=0.886 (TFT weight=93.5%) |
| 09 | `09_evaluation.ipynb` | Unified comparison & recommendations | 22 models ranked, RF champion |

## Artifacts Produced

| Notebook | Files |
|----------|-------|
| 02 | `battery_features.csv`, `battery_sequences.npz`, scalers |
| 03 | `.joblib` models, `classical_soh_results.csv`, `classical_rul_results.csv` |
| 04 | `.pt` checkpoints (vanilla/bi/gru/attention LSTM), `lstm_soh_results.csv` |
| 05 | `.pt` (BatteryGPT, TFT), `.keras` (iTransformer), `transformer_soh_results.csv` |
| 06 | `.keras` checkpoint, `dg_itransformer_results.json` |
| 07 | `vae_lstm.pt`, `vae_lstm_results.json`, UMAP plots |
| 08 | `ensemble_results.csv`, weight/comparison plots |
| 09 | `unified_results.csv`, `final_rankings.csv`, radar/CED/comparison plots |

## Key Dependencies

- **Notebook 02** produces `battery_sequences.npz` (sliding windows) used by all deep learning notebooks
- **Notebook 02** produces `battery_features.csv` used by notebook 03
- **Notebooks 04-07** produce model checkpoints used by notebook 08 (ensemble)
- **All result CSVs** are consumed by notebook 09 for unified evaluation

## GPU Notes

- PyTorch notebooks (04, 07, 08) use CUDA when available (`torch.cuda.is_available()`)
- TensorFlow/Keras notebooks (05, 06) run CPU-only on Windows (no native TF GPU support)
- Classical ML (notebook 03) is CPU-only, completes in ~2-5 minutes
- Deep learning notebooks: ~2-10 min on GPU, ~15-60 min on CPU
