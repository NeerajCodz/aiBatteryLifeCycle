# Dataset Documentation

## NASA PCoE Li-ion Battery Dataset

### Source
- **Repository:** NASA Prognostics Center of Excellence (PCoE)
- **Reference:** B. Saha and K. Goebel (2007). *Battery Data Set*, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA
- **URL:** https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

### Cells
- **Type:** Li-ion 18650
- **Nominal capacity:** 2.0 Ah
- **Count:** 30 batteries after cleaning (from original 36)
- **Total discharge cycles:** 2,678
- **Sliding windows generated:** 1,734 (window size = 32 cycles)

### Temperature Groups (discovered in EDA)

| Group | Temperature | # Batteries | # Cycles |
|-------|-------------|-------------|----------|
| 1 | 4°C | 3 | ~200 |
| 2 | 22°C | 4 | ~280 |
| 3 | 24°C | 16 | ~1700 |
| 4 | 43°C | 4 | ~320 |
| 5 | 44°C | 3 | ~180 |

Note: 5 temperature groups were discovered (not 3 as originally assumed).

### End-of-Life Definitions
- **30% capacity fade:** 1.4 Ah (default threshold)
- **20% capacity fade:** 1.6 Ah (alternative)

### Cycle Types

#### Discharge
Columns: `Voltage_measured`, `Current_measured`, `Temperature_measured`, `Current_load`, `Voltage_load`, `Time`

#### Charge
Columns: `Voltage_measured`, `Current_measured`, `Temperature_measured`, `Current_charge`, `Voltage_charge`, `Time`

#### Impedance
Columns: `Sense_current` (Re + Im), `Battery_current` (Re + Im), `Current_ratio` (Re + Im), `Battery_impedance` (Re + Im)

### Metadata Schema (metadata.csv)
- `type`: cycle type (charge, discharge, impedance)
- `start_time`: MATLAB datenum
- `ambient_temperature`: °C
- `battery_id`: identifier
- `test_id`: test sequence number
- `uid`: unique identifier
- `filename`: path to cycle CSV
- `Capacity`: measured capacity (Ah)
- `Re`: electrolyte resistance (Ω)
- `Rct`: charge transfer resistance (Ω)

### Feature Engineering

#### Per-Cycle Scalar Features (12 dimensions)
1. `cycle_number` — sequential cycle index
2. `ambient_temperature` — environment temperature
3. `peak_voltage` — max voltage in cycle
4. `min_voltage` — min voltage in cycle
5. `voltage_range` — peak - min
6. `avg_current` — mean current magnitude
7. `avg_temp` — mean cell temperature
8. `temp_rise` — max - min temperature
9. `cycle_duration` — total time (s)
10. `Re` — electrolyte resistance
11. `Rct` — charge transfer resistance
12. `delta_capacity` — capacity change from previous cycle

#### Derived Targets
- **SOC:** Coulomb counting (integrated current)
- **SOH:** (Current capacity / Nominal capacity) × 100%
- **RUL:** Cycles remaining until EOL threshold
- **Degradation State:** Healthy (≥90%), Moderate (80–90%), Degraded (70–80%), End-of-Life (<70%)
