# Repository structure

```text
api/                 versioned FastAPI routes and model registry
datasets/            official-source provenance and checksums
docs/                active reviewer-revision documentation
frontend/            React research interface
notebooks/           ordered 00–12 experimental record
scripts/             active benchmark, statistics, release, and download tools
scripts/legacy/      preserved pre-review generators and upload helpers
src/data/            adapters and leakage-safe partial-cycle features
src/evaluation/      metrics, grouped protocol, bootstrap, paired tests
src/experiments/     tabular and sequence benchmark implementations
src/models/          model implementations and authoritative 20-model catalog
tests/               unit and integration checks
artifacts/v3/         generated results; ignored here and released separately
```

GitHub is canonical. Model artifacts live in the separate Hugging Face model
repository, and the Space remains an unmodified deployment remote until review
approval.
