# Leakage-aware battery SOH reviewer revision

The former 99.1% headline result has been withdrawn because target-derived
capacity features and non-independent cycle splits made it unsuitable evidence
for unseen-battery performance.

The corrected study predicts current-cycle SOH from the observable 4.0–3.6 V
partial-discharge segment. It evaluates 20 models with battery-grouped
train/validation/test partitions, three fixed seeds, external CALCE and Oxford
benchmarks, NASA-frozen zero-shot transfer, standard regression metrics,
residuals, 10,000-draw battery-cluster bootstrap intervals, paired
Wilcoxon–Holm tests, feature ablations, and sensor stress tests.

Recommendation and simulation screens are exploratory prototypes. They are not
validated RUL estimators, charging policies, or safety controls.

The complete clean manuscript, Word redline, and response to reviewers are
distributed with the reviewer-revision artifact package.
