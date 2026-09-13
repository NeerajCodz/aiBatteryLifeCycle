# Dataset and feature contract

| Dataset | Rated capacity | Retained cells | Retained records |
| --- | ---: | ---: | ---: |
| NASA PCoE | 2.00 Ah | 30 | 2,141 |
| CALCE CS2 | 1.10 Ah | 4 | 3,789 |
| Oxford Dataset 1 | 0.74 Ah | 8 | 519 |

SOH is the full discharged capacity divided by a robust initial reference and
multiplied by 100. Capacity is used only to construct retrospective labels.
Inputs are computed before the complete current-cycle capacity is known.

The scalar contract has 18 fields listed in `src/utils/config.py`. Sequence
models receive 64 bins with normalized time, voltage, absolute current C-rate,
temperature, and a temperature-observed indicator. Rated capacities above—not
measured target capacities—normalize current and prior throughput.

NASA identifiers B0049–B0052 are excluded by the fixed quality list. Every
record-level exclusion is written to `artifacts/v3/features/*/exclusions.csv`.
The Oxford MATLAB serial-day clock is converted to seconds. A grossly corrupt
Oxford temperature channel is treated as missing, replaced by ambient
temperature, and flagged rather than silently used.

Official URLs, requested citations, terms, and SHA-256 values are in
`datasets/SOURCES.md` and `datasets/checksums.sha256`.
