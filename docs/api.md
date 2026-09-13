# API contract

The deployment starts with reviewer-revision v3. `POST /api/predict` and
`POST /api/v3/predict` accept the same required 18-field partial-discharge
request and return current-cycle SOH. Human-facing cycle numbers are 1-based;
the stored `cycle_index` predictor is 0-based.

The fields summarize the observable 4.0–3.6 V segment: prior usage, ambient
temperature, duration and energy, voltage distribution/shape, mean and
variation of absolute current in C-rate, and temperature distribution/shape
plus a sensor-observed flag. C-rate and prior equivalent full cycles use public
rated capacity, never current measured full-cycle capacity.
Unknown fields are rejected, so target-derived values such as `capacity_ah`,
`SoH`, or `delta_capacity` cannot be silently accepted by the v3 request path.

The v3 response deliberately sets `rul_cycles`, `confidence_lower`, and
`confidence_upper` to null. Recommendation and simulation responses carry
`validation_status: exploratory_prototype` and a safety disclaimer. Explicit
`/api/v1` and `/api/v2` routes retain historical contracts and are loaded
only on demand.

OpenAPI details are available at `/docs` when the server is running.
