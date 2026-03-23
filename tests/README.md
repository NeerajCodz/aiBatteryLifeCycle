# Tests Directory

Pytest-based test suite for API routers, schemas, utility modules, feature engineering,
model-registry helpers, and lightweight model wrappers.

## Running tests

Run full suite:

```bash
python -m pytest -q
```

Run a specific module:

```bash
python -m pytest -q tests/routers/test_visualize_router.py
```

Run by feature folder:

```bash
python -m pytest -q tests/data
python -m pytest -q tests/routers
python -m pytest -q tests/models
```

## Test layout

- `tests/api`:
  - schema validation (`api.schemas`)
  - `api.main` version/model lifecycle endpoints
- `tests/routers`:
  - `predict`, `predict_v2`, `predict_v3`
  - `simulate`
  - `visualize`
- `tests/data`:
  - `src.data.loader`
  - `src.data.preprocessing`
  - `src.data.features`
- `tests/evaluation`:
  - metrics and recommendations
- `tests/models`:
  - model registry helpers
  - ensemble/classical model wrappers
- `tests/utils`:
  - config paths
  - logger
  - plotting helpers

## Current coverage scope

- API routers:
  - `api.routers.predict`
  - `api.routers.predict_v2`
  - `api.routers.predict_v3`
  - `api.routers.simulate`
  - `api.routers.visualize`
- API core helpers in `api.main`
- Data/feature utilities:
  - `src.data.loader`
  - `src.data.preprocessing`
  - `src.data.features`
- Evaluation utilities:
  - `src.evaluation.metrics`
  - `src.evaluation.recommendations`
- Core config/logger/plotting helpers
- Model registry pure/helper behavior
- Lightweight smoke tests for classical models and ensemble wrappers

## Notes

- Tests are designed to be deterministic and fast.
- Heavyweight model loading is monkeypatched/stubbed where possible.
- Matplotlib is forced to a non-GUI backend (`Agg`) in `tests/conftest.py` for CI/headless safety.
- Some tests write temporary or small artifacts under existing project artifact paths.
