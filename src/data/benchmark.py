"""Build aligned tabular and sequence benchmarks from normalized cycles."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.adapters import DischargeCycle
from src.data.partial_cycle import (
    PartialCycleConfig,
    extract_partial_cycle_features,
    make_partial_cycle_sequence,
    prior_equivalent_full_cycles,
    reference_capacity_from_characterization,
)
from src.utils.config import FEATURE_COLS_V3, RATED_CAPACITY_AH, SEQUENCE_FEATURE_COLS_V3


@dataclass
class BenchmarkBundle:
    features: pd.DataFrame
    sequences: np.ndarray
    sequence_index: pd.DataFrame
    exclusions: pd.DataFrame


def build_benchmark_bundle(
    cycles: Iterable[DischargeCycle],
    *,
    config: PartialCycleConfig = PartialCycleConfig(),
    excluded_batteries: set[str] | None = None,
    rated_capacity_by_dataset: dict[str, float] | None = None,
) -> BenchmarkBundle:
    """Build leakage-safe inputs and SOH targets from normalized cycles."""
    excluded_batteries = excluded_batteries or set()
    rated_capacity_by_dataset = rated_capacity_by_dataset or RATED_CAPACITY_AH
    grouped: dict[tuple[str, str], list[DischargeCycle]] = defaultdict(list)
    pre_exclusions: list[dict[str, object]] = []
    for cycle in cycles:
        if cycle.battery_id in excluded_batteries:
            pre_exclusions.append({
                "dataset": cycle.dataset,
                "battery_id": cycle.battery_id,
                "cycle_number": cycle.cycle_number,
                "source": cycle.source,
                "reason": "Battery excluded by the pre-registered data-quality list",
            })
            continue
        grouped[(cycle.dataset, cycle.battery_id)].append(cycle)

    feature_rows: list[dict[str, object]] = []
    sequences: list[np.ndarray] = []
    sequence_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = pre_exclusions
    for (dataset, battery_id), battery_cycles in sorted(grouped.items()):
        battery_cycles.sort(key=lambda item: item.cycle_number)
        valid_caps = [cycle.capacity_ah for cycle in battery_cycles if np.isfinite(cycle.capacity_ah) and cycle.capacity_ah > 0]
        if len(valid_caps) < config.reference_cycles:
            continue
        reference = reference_capacity_from_characterization(
            valid_caps, reference_cycles=config.reference_cycles
        )
        if dataset not in rated_capacity_by_dataset:
            raise KeyError(f"Rated capacity is not configured for dataset: {dataset}")
        rated_capacity = rated_capacity_by_dataset[dataset]
        prior_efc_values = prior_equivalent_full_cycles(
            pd.Series([cycle.capacity_ah for cycle in battery_cycles]),
            rated_capacity,
        ).to_numpy(dtype=float)
        full_capacity_threshold = 0.85 * float(np.quantile(valid_caps, 0.95))
        characterization_start = next(
            i for i, cycle in enumerate(battery_cycles)
            if np.isfinite(cycle.capacity_ah) and cycle.capacity_ah >= full_capacity_threshold
        )
        for position, cycle in enumerate(battery_cycles):
            if position < characterization_start:
                exclusion_rows.append({
                    "dataset": dataset,
                    "battery_id": battery_id,
                    "cycle_number": cycle.cycle_number,
                    "source": cycle.source,
                    "reason": "Pre-characterization partial discharge; not a full-capacity SOH label",
                })
                continue
            if not np.isfinite(cycle.capacity_ah) or cycle.capacity_ah <= 0:
                exclusion_rows.append({
                    "dataset": dataset,
                    "battery_id": battery_id,
                    "cycle_number": cycle.cycle_number,
                    "source": cycle.source,
                    "reason": "Non-positive or non-finite full-cycle capacity label",
                })
                continue
            soh = 100.0 * cycle.capacity_ah / reference
            if soh > 120.0:
                exclusion_rows.append({
                    "dataset": dataset,
                    "battery_id": battery_id,
                    "cycle_number": cycle.cycle_number,
                    "source": cycle.source,
                    "reason": "Capacity characterization exceeds 120% of the robust initial reference",
                })
                continue
            prior_efc = float(prior_efc_values[position])
            try:
                features = extract_partial_cycle_features(
                    cycle.measurements,
                    cycle_index=cycle.cycle_number,
                    prior_efc=prior_efc,
                    ambient_temperature=cycle.ambient_temperature_c,
                    rated_capacity_ah=rated_capacity,
                    config=config,
                )
                sequence = make_partial_cycle_sequence(
                    cycle.measurements,
                    ambient_temperature=cycle.ambient_temperature_c,
                    rated_capacity_ah=rated_capacity,
                    config=config,
                )
            except (KeyError, ValueError, TypeError) as exc:
                exclusion_rows.append({
                    "dataset": dataset,
                    "battery_id": battery_id,
                    "cycle_number": cycle.cycle_number,
                    "source": cycle.source,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            row: dict[str, object] = {
                "dataset": dataset,
                "battery_id": battery_id,
                "cycle_number": cycle.cycle_number,
                "capacity_ah": cycle.capacity_ah,
                "reference_capacity_ah": reference,
                "SoH": soh,
                "source": cycle.source,
            }
            row.update(features)
            feature_rows.append(row)
            sequences.append(sequence.to_numpy(dtype=np.float32))
            sequence_rows.append({
                "dataset": dataset,
                "battery_id": battery_id,
                "cycle_number": cycle.cycle_number,
                "SoH": row["SoH"],
            })

    columns = [
        "dataset", "battery_id", "cycle_number", "capacity_ah",
        "reference_capacity_ah", "SoH", "source", *FEATURE_COLS_V3,
    ]
    features_frame = pd.DataFrame(feature_rows, columns=columns)
    sequence_array = (
        np.stack(sequences)
        if sequences
        else np.empty((0, config.sequence_bins, len(SEQUENCE_FEATURE_COLS_V3)), dtype=np.float32)
    )
    return BenchmarkBundle(
        features=features_frame,
        sequences=sequence_array,
        sequence_index=pd.DataFrame(sequence_rows),
        exclusions=pd.DataFrame(
            exclusion_rows,
            columns=["dataset", "battery_id", "cycle_number", "source", "reason"],
        ),
    )


def save_benchmark_bundle(bundle: BenchmarkBundle, output_dir: str | Path) -> None:
    """Persist one dataset bundle using portable CSV and compressed NumPy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.features.to_csv(output_dir / "features.csv", index=False)
    bundle.sequence_index.to_csv(output_dir / "sequence_index.csv", index=False)
    bundle.exclusions.to_csv(output_dir / "exclusions.csv", index=False)
    np.savez_compressed(
        output_dir / "sequences.npz",
        X=bundle.sequences,
        feature_names=np.asarray(SEQUENCE_FEATURE_COLS_V3),
    )
