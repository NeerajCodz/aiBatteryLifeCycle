# Reviewer 4 scientific and application contract

Revision: 1 November 2026, addressing the review dated 28 October. The paper is
**State of Health Estimation and Target Leakage Auditing for Lithium Ion
Batteries**. The former 99.1% contribution is withdrawn. Charging optimization
and predictive RUL accuracy are not claimed. Existing SOH prediction schemas
and model defaults remain valid.

## Inputs, labels and audit-only proxies

Order retained records by battery and cycle; windows reset at every battery.
For retained discharge k, let Q_k be measured full capacity and Q_ref the
retrospective robust reference. The label is y_k = 100 Q_k / Q_ref. Invalid
audit proxies are Q_k/Q_ref; sum(Q_j, j=1..k); Q_k−Q_(k−1), first row zero;
and mean(y_j, j=max(1,k−4)..k), minimum one retained row. All include the current
row and are excluded from the observable input contract. Generation:
`scripts/run_protocol_comparison.py::_add_legacy_target_proxies`.

The 18 observable features cover usage/duration (3), voltage shape (6), current
loading (2), partial energy (1) and thermal context (6). The 64 × 5 sequence
contains normalized time, voltage, absolute rated-capacity C-rate, temperature
and thermal observation flag. Rated capacities are NASA 2.0 Ah, CALCE 1.1 Ah
and Oxford 0.74 Ah. The fixed measured 4.0–3.6 V segment is a harmonization
choice, not a demonstrated optimal or invariant SOC window. Temperature, rate
and chemistry affect it; alternative windows/resolutions were not evaluated.

Scalar medians and applicable SVR/Ridge/KNN scalers fit training rows only.
Sequence medians, channel means/population scales and neural target
normalization fit training records only. Interpolation with endpoint filling
occurs inside each completed observed segment, before fixed resampling. It is
not fitted on an external record pool. Ambient temperature is numeric Celsius,
with a binary observation flag. Frozen transfer reuses source preprocessing
and source-only ensemble parameters. Oxford current reconstruction is a
dataset-adapter rule, not target-fitted normalization.

## Splits, aggregation and classification

NASA: 30 cells, 2,141 eligible cycles; seeds 17, 42, 2026, five folds each,
with 19 train / five validation / six test cells in every fold. All three
roles are battery-disjoint. GroupKFold does not stratify SOH or temperature;
some folds have no EOL cycles. Full inventories and support are published.

Average held-out SOH predictions per cycle before pooled evaluation. Sample
MAE weights cycles equally; battery-macro MAE weights cells equally; fold-mean
MAE averages separate fit scores and is not interchangeable with either.
Preserved NASA weighted-ensemble scores are 3.536 pp battery-macro, 3.682 pp
sample and 3.779 pp fold-mean MAE.

Secondary classes use identical observed/predicted boundaries: Healthy ≥90%,
Aging 80≤SOH<90%, Near-EOL 70≤SOH<80%, EOL <70%. Empirical residual calibration
uses validation records only, with ensemble validation meta-predictions
cross-fitted by battery. Repeated probabilities are averaged per cycle. AUC
is undefined if either positives or negatives are absent. This secondary
analysis does not change the deployed model or add a probability API claim.

## Controlled audit and inference

The new analysis adds 126 ExtraTrees fits. Equal-count fits use 1,152 training
rows. Common-temperature-quota fits use 836 rows: 416 at 4 °C, 380 at 24 °C,
40 at 43 °C. Each contrast uses exactly the same train/test rows for safe 18,
safe 18 plus four partition-wise shuffled proxies, and safe 18 plus four real
proxies. Zero common train quotas at 22/44 °C do not remove those test records.
Cross-protocol test populations still differ, so differences are descriptive.

Inference resamples batteries, conditional on fitted predictions; it does not
include retraining uncertainty. Wilcoxon uses one paired score per NASA cell,
two-sided zsplit, with Holm adjustment across 19 comparisons against the
weighted ensemble. Selecting that reference from the same benchmark makes
inference exploratory. Four CALCE cells cannot establish stable population
intervals; the supplement gives leave-one-observed-cell-out influence without
refitting. Temperature permutation intervals are pointwise and conditional.
Within-cell and pooled shuffles preserve different exposure structures. The
paired no-temperature retraining sample-MAE effect −0.130 pp has interval
−0.352 to 0.133; the small improvement is inconclusive.

## Original versus prospective training records

420 original grouped-fit epoch totals and train/test scores were recovered.
Executed epochs include patience and are not the restored-best epoch. Original
validation histories and best-epoch identifiers are unavailable. The 150
secondary NASA validation/test diagnostics are explicitly separate.

New `run_sequence_benchmark.py` runs save full train/validation monitor losses,
selected best epoch, executed epochs/cap status, all three role predictions,
split IDs, train-only preprocessing and target scale in `training_audit/`.
PyTorch and TensorFlow helper return contracts are unchanged. Use a new output
directory to avoid replacing archived results:

```bash
python scripts/run_sequence_benchmark.py nasa recurrent --result-dir artifacts/v3/reviewer4_reruns
```

Completed old checkpoints are skipped; the new code cannot recover their
missing histories. New audit files are evidence for the new fit only.

## Paper assets and regeneration

`frontend/public/research/reviewer4/manifest.json` connects every figure and
table to downloads and hashes. The complete manuscript text is generated into
`docs/research_paper.md` and its frontend copy from the verified Word package.
The static release includes all 27 vector SVG/PDF figures, nine CSV manuscript
tables, original and secondary audit tables, and three PDFs. Source PNGs at
600 dpi and editable Word documents remain in the full review archive.

From the enclosing review workspace, with original source artifacts available:

```bash
python application/scripts/run_reviewer4_analysis.py all
python application/scripts/publish_reviewer4.py --review-root paper/revision-2026-11-r4
```

The analysis entry point is the versioned copy of the manuscript's exact
control/importance analysis. It expects the accompanying review archive's
workspace-relative feature, prediction and calibration files; metadata analysis
also verifies the prior restoration split records. It writes only the new
revision's reproducibility directory and preserves primary results.

All five suggested papers were assessed against publisher records. Chen's
energy-demand fusion paper is included only as methodological context; no
results are transferred to battery SOH. The response documents decisions for
the other endpoints and corrects the export paper's third-author metadata.
