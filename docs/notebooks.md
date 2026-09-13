# Notebook order

| No. | Notebook | Output |
| ---: | --- | --- |
| 00 | Environment and reproducibility | versions, hardware, checksums, catalog |
| 01 | Dataset audit | per-cell/cycle inventory |
| 02 | Feature engineering | aligned scalar and 64×5 sequence bundles |
| 03 | Protocol comparison | random, chronological, grouped × safe/leaked |
| 04 | Classical models | eight regressors plus two ensembles |
| 05 | Recurrent models | LSTM, BiLSTM, GRU, attention LSTM |
| 06 | Transformer models | BatteryGPT, TFT, iTransformer variants |
| 07 | Graph and variational models | dynamic graph and VAE-LSTM |
| 08 | Ensembles | validation-only ensemble audit |
| 09 | CALCE validation | grouped and NASA-frozen zero-shot |
| 10 | Oxford validation | grouped and NASA-frozen zero-shot |
| 11 | Statistics and robustness | CI, paired tests, residuals, ablations, stress |
| 12 | Paper outputs | publication tables and 600-dpi/vector figures |

Run them numerically. Long model notebooks reuse complete machine-readable
results when present so an interrupted review workflow can resume safely.
