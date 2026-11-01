# Application deployment

Canonical source: https://github.com/neerajcodz/aiBatterylifecycle.
Existing Docker Space: https://huggingface.co/spaces/NeerajCodz/aiBatteryLifeCycle.
App: https://neerajcodz-aibatterylifecycle.hf.space.

## Local verification

From the repository root:

```bash
python -m pytest -q
cd frontend
npm ci
npm run build
cd ..
uvicorn api.main:app --host 127.0.0.1 --port 7860
```

FastAPI serves `frontend/dist` with the API. `/health` reports process health;
`/api/versions` reports asynchronous model readiness; `/docs` describes the API.
A healthy process may still be downloading models. `/api/v3/predict` estimates
current-cycle SOH; `/api/predict` is its alias. v1/v2 remain legacy.

## Docker and models

```bash
docker build -t battery-soh .
docker run --rm -p 7860:7860 battery-soh
```

The build uses Node 20 and Python 3.12, CPU Torch/TensorFlow, and port 7860.
Startup downloads metadata and selected v3 models from the existing
`NeerajCodz/aiBatteryLifeCycle` model repository. Do not replace that release
with secondary audit fits. Existing manifest model defaults are retained.

## Reviewer 4 research release

`frontend/public/research/reviewer4/` is included in the Vite build: exact
verified manuscript, response and supplement PDFs, all 27 vector SVG/PDF
figures, nine CSV manuscript tables, supporting audit tables and hashed
manifest. Editable Word and 600-dpi PNG files remain in the enclosing review
archive. Research displays the full paper; v3 Metrics → Paper Assets displays
the same figure set. These assets do not depend on model downloads.

```bash
python scripts/publish_reviewer4.py --review-root ../paper/revision-2026-11-r4
```

## Release to the existing Space

Push the canonical source and deploy its committed changes through the Hub API:

```bash
git push origin HEAD:main
python scripts/deploy_space.py --source-base <last-deployed-source-commit> --resume
```

The API handles binary PDF storage, which plain Git pushes reject. It uploads
only committed blobs changed since the supplied source base, guards against
concurrent Space edits with a parent commit, and records the source commit in
the Space commit description. Hub and GitHub commit IDs differ. Never use a
force push to reconcile them. Credentials come from the Hub token store or
existing Git credential helper; they are not written to command arguments.
Observe the Docker build/runtime until `RUNNING`. `--resume` resumes a paused
Space on its existing hardware; it does not upgrade hardware.
Verify the deployed commit, `/health`, `/api/versions`, research manifest and
PDF, a figure and a valid 18-feature SOH prediction; inspect desktop/mobile
research navigation. A successful push alone does not verify deployment.

Runtime: `LOG_LEVEL=INFO`, one Uvicorn worker, port 7860. Logs rotate under
`artifacts/logs/battery_lifecycle.log`. Predictions still exclude current
capacity/SOH and rolling target proxies. Recommendations and simulations
remain exploratory prototypes.
