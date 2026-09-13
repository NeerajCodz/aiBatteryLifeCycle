# Architecture

```text
official NASA / CALCE / Oxford archives
                 |
         dataset adapters
                 |
  4.0–3.6 V feature/sequence builder
                 |
 battery-grouped train / validation / test
        |                     |
 20-model benchmark      statistics + figures
        |                     |
 deployable tabular      manuscript artifacts
 pipelines + manifest
        |
 FastAPI v3 -> React and Gradio research interfaces
```

GitHub is the canonical source repository. The Hugging Face model repository
stores versioned release artifacts, and the Hugging Face Space is a deployment
remote. The pre-review state is preserved by tag; reviewer work stays on a
review branch until approved.

The v3 fitted endpoint estimates current-cycle SOH only. Multi-step simulation
and operating-condition scenarios are isolated, explicitly exploratory physics
prototypes and never call the SOH model as a causal intervention model.
