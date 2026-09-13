# Frontend

The React interface defaults to v3.

- Prediction collects all 18 observable partial-discharge values and displays
  current-cycle SOH.
- Recommendations show a prominent unvalidated-prototype disclaimer.
- Simulation is an illustrative physics trajectory; the model selector is
  visibly reference-only and fitted v3 models are not invoked for forecasting.
- Metrics and figures read versioned manifests and generated artifacts.
- The paper page summarizes the corrected protocol and links users to the
  reviewer-revision artifact package.

Run `npm ci` and `npm run build` inside `frontend/`. The current build is
successful; Vite reports only a bundle-size optimization warning.
