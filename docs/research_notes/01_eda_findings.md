# EDA Research Notes — Notebook 01

**Date:** 2025-07-15  
**Researcher:** AI Battery Lifecycle Research Pipeline  
**Dataset:** NASA PCoE Li-ion Battery Dataset (cleaned)

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total batteries (after exclusions) | 30 |
| Excluded batteries | B0049–B0052 (corrupt/incomplete) |
| Total cycle records | 7,317 |
| Discharge cycles | 2,678 |
| Charge cycles | 2,715 |
| Impedance measurements | 1,908 |
| Nominal capacity | 2.0 Ah (18650 Li-ion) |
| Temperature groups | 5: {4, 22, 24, 43, 44} °C |
| Mean cycles per battery | 89 |
| Max cycles (any battery) | 196 |
| Batteries reaching EOL | 22 / 30 (73.3%) |

## 2. Critical Discovery: Temperature Groups

**Original documentation claimed 3 temperature groups** (4°C, 24°C, 43°C).  
**Actual data contains 5 distinct groups:**

| Temperature (°C) | # Batteries | Category |
|---|---|---|
| 4 | 12 | Cold |
| 22 | 3 | Near-ambient |
| 24 | 14 | Room temperature |
| 43 | 4 | Elevated |
| 44 | 3 | Elevated |

> **Note:** Some batteries appear in multiple temperature groups (tested at different conditions).  
> The 22°C and 44°C groups have never been separately analyzed in published literature using this dataset — this represents a potential novel contribution.

## 3. Capacity Fade Analysis

### 3.1 Overall
- Capacity range: **[0.0441, 2.4441] Ah**
- SOH range: **[2.2%, 122.2%]**
- Values >100% SOH indicate initial measured capacity above the 2.0 Ah rated nominal — common in fresh Li-ion cells
- Clear exponential-like decay visible across all batteries, with some showing abrupt drops (regeneration artifacts)

### 3.2 Temperature Effects (Key Finding)
| Temperature | Mean Capacity (Ah) | Std Dev | Observations |
|---|---|---|---|
| 4°C (Cold) | 0.91 | 0.46 | **54.5% capacity reduction vs nominal** — severe cold degradation |
| 22°C | ~1.50 | ~0.25 | Near-room performance |
| 24°C (Room) | 1.54 | 0.26 | Widest distribution (most batteries) |
| 43°C (Elevated) | 1.72 | 0.07 | Narrowest distribution — rapid degradation with fewer cycles |
| 44°C | ~1.65 | ~0.15 | Similar to 43°C |

**Research Implication:** Cold-temperature operation (4°C) is more damaging to Li-ion cycle life than elevated temperature (43°C) in terms of absolute capacity. However, elevated-temperature batteries have fewer cycles before EOL.

### 3.3 SOH Distribution by Temperature
- **4°C:** Extremely wide, flat KDE centered at ~55% — batteries spend most of their life in degraded state
- **22°C:** Tight peak at ~82%
- **24°C:** Bimodal — some batteries healthy (~90%), others significantly degraded (~65%)
- **43°C:** Very tight KDE peak at ~85% — consistent but short-lived
- **44°C:** Peak at ~90%, narrow

## 4. Impedance Analysis

- **1,908 impedance measurement records** available
- **Electrolyte resistance (Re):** Clear upward trend with cycle number — SEI layer growth
- **Charge transfer resistance (Rct):** More dramatic increase, with outlier spikes (notably B0034)
- **Phase Space (Re vs Rct):** Distinct battery-group clusters visible, with drift from lower-left (healthy) to upper-right (degraded) confirming dual-resistance aging signature

## 5. Voltage Surface Analysis

- 3D discharge voltage surface for B0005 shows:
  - Progressive "sinking" of voltage plateau with aging
  - Voltage knee region shifts earlier (less capacity delivered)
  - End-of-discharge voltage drops become sharper in later cycles
  - Surface is smooth except for noise near cutoff voltage region

## 6. Anomalies Detected

1. **B0034 impedance spikes** — Rct shows anomalous jumps, possibly measurement artifact or internal micro-short
2. **SOH > 100%** — 122.2% max indicates initial capacity above nominal; may need capping at 100% for model inputs
3. **Very low capacity (0.044 Ah)** — Some cycles with near-zero capacity, likely incomplete discharge or measurement error; should be filtered in preprocessing
4. **Battery overlap across temperature groups** — Same batteries tested at multiple temperatures complicates group-independent analysis

## 7. Implications for Modeling

1. **Feature engineering should include temperature as a critical feature** — capacity degradation mechanisms differ fundamentally between cold and hot operation
2. **SOH capping** at 100% should be considered for target labels
3. **Outlier filtering** for capacity < 0.1 Ah recommended (likely measurement artifacts)
4. **Impedance features (Re, Rct)** are strong predictors — clear monotonic relationship with SOH
5. **Cross-temperature generalization** is the most challenging task — models should be evaluated for transfer across temperature domains
6. **Sequence length** varies significantly (89 mean, 196 max) — variable-length handling or padding strategies needed

---

*Generated figures saved to `artifacts/figures/`:*
- `capacity_fade_all.png`
- `capacity_fade_by_temp.png`
- `capacity_violin_box.png`
- `impedance_evolution.png`
- `re_vs_rct_phase.png`
- `soh_distribution.png`
- `voltage_surface_3d.png`
- `voltage_surface_3d_interactive.html`
- `capacity_fade_interactive.html`
