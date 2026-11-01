# Version policy

- v1 and v2 are retained as historical API/model contracts.
- The pre-review v3 is preserved by the annotated tag
  `v3-pre-review-2026-09`.
- Reviewer-revision v3.0 was developed on `reviewer-revision-2026-09` and merged
  into `main`. The Reviewer 4 application revision updates documentation,
  research assets and prospective audit capture, retaining the v3 prediction
  contract and deployed model defaults.

The revised v3 predicts current-cycle SOH from a 4.0–3.6 V partial-discharge
window. A release is valid only when its `models.json`, `datamap.json`,
`dataset.json`, checksums, code commit, and paper tables refer to the same
artifact set.
