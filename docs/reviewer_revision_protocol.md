# 2026 Reviewer Revision Protocol

The pre-review v3 state is preserved by the annotated tag
`v3-pre-review-2026-09`. Development occurs on `reviewer-revision-2026-09`.

## Locked analysis decisions

1. Predict current-cycle SOH from the 4.0–3.6 V observable discharge segment.
2. Treat full-cycle discharged capacity and SOH as labels only.
   Use public rated capacity, not measured target capacity, for C-rate and prior
   equivalent-full-cycle normalization.
3. Use disjoint battery groups for training, validation, and test partitions.
4. Use validation batteries for early stopping and ensemble selection; score
   test batteries once.
5. Run seeds 17, 42, and 2026 and report per-fold values and aggregate results.
6. Evaluate the same 20 named models on NASA, CALCE, and Oxford.
7. Report zero-shot NASA-to-CALCE/Oxford transfer separately from grouped
   within-dataset validation.
8. Select a release model by macro per-battery MAE, with RMSE, artifact size,
   and CPU latency as stated tie-breakers.
9. Treat ±5 coverage as the fraction within five absolute SOH percentage
   points; never call it R² or generic accuracy.
10. Label recommendations and simulations as exploratory, unvalidated
    prototypes.

## Quantitative v1/v2/v3 audit

`scripts/run_protocol_comparison.py` crosses three split designs with a safe
feature contract and a reconstructed legacy target-proxy contract. These are
descriptive contrasts, not a causal decomposition: training counts, exposure
composition and test populations differ. Reviewer 4 adds 126 paired fits with
common training counts and temperature quotas. Even those controls do not make
the cross-protocol test populations exchangeable. Exact definitions, records
and limitations are in [the Reviewer 4 contract](reviewer4_revision.md).

## Reproducibility outputs

Every row-level prediction, fold metric, exclusion reason, bootstrap interval,
paired test, feature ablation, sensor stress test, and figure source table is
saved in a portable CSV/JSON/NPZ artifact. The final manuscript is generated
only after these files pass the test suite and consistency checks.
