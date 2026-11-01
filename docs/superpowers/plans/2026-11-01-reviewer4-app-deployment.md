# Reviewer 4 application and deployment plan

Goal: synchronize the existing SOH application and repository documentation with the completed Reviewer 4 manuscript, then update the existing Hugging Face Space.

The user authorized code, documentation and deployment. Use the existing React/FastAPI/Docker stack and current deployment remote. Work in the clean application checkout; preserve published model artifacts, prediction schemas and earlier manuscripts.

## Implementation

- [x] Add tested, opt-in sequence audit capture: validation predictions, train/validation/test battery identities, train-only preprocessing parameters, actual executed epochs, selected best epoch and complete monitored-loss history. Keep old helper return contracts and archived regression records unchanged. Allow a separate result directory for new audited runs.
- [x] Publish the verified Reviewer 4 PDF, response, supplement, all 27 vector figures and machine-readable audit tables as a versioned static research bundle with source hashes. Add the corresponding research view and downloadable assets; label the three metric aggregations and original versus secondary diagnostics explicitly.
- [x] Update README, research/protocol/model/deployment documentation and changelog, including exact proxy definitions, frozen preprocessing, inference units and limitations. Include a portable copy of the revision analysis entry point.
- [x] Run focused and full regression tests, frontend production build and a fresh code review. Check the actual research page in the browser.
- [ ] Commit the tested changes, push the canonical repository and existing Space, wait for its build, and verify the deployed commit, health, research assets and SOH endpoint.

Review focus: no target-derived API inputs; no pooled metric presented as battery-macro MAE; no archived best epochs invented; no missing-class or four-cell uncertainty concealed; no lost previous figure topics.
