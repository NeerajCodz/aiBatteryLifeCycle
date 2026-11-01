# Frontend

The React interface defaults to v3.

- Prediction collects all 18 observable partial-discharge values and displays
  current-cycle SOH.
- Recommendations show a prominent unvalidated-prototype disclaimer.
- Simulation is an illustrative physics trajectory; the model selector is
  visibly reference-only and fitted v3 models are not invoked for forecasting.
- Metrics and figures read versioned manifests and generated artifacts.
- The Research tab contains the full Reviewer 4 manuscript with all 27 figures,
  its nine tables and downloadable manuscript/response/supplement PDFs.
- Metrics → Paper Assets uses the static Reviewer 4 manifest for v3, so paper
  figures remain separate from older model-release diagnostic displays. It
  exposes genuine SVG/PDF exports and CSV tables. Metric weighting definitions
  and the original-versus-secondary provenance limitation appear beside them.

Run `npm ci` and `npm run build` inside `frontend/`. The current build is
successful; Vite reports only a bundle-size optimization warning.
