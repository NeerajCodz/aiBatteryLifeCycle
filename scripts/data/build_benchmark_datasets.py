"""Normalize all raw datasets and build leakage-safe benchmark bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.adapters import (
    iter_calce_xlsx_cycles,
    iter_nasa_mat_cycles,
    iter_oxford_mat_cycles,
)
from src.data.benchmark import build_benchmark_bundle, save_benchmark_bundle
from src.utils.config import EXCLUDED_BATTERIES


def load_all_cycles(raw_root: str | Path) -> dict[str, list]:
    raw_root = Path(raw_root)
    nasa_paths = sorted((raw_root / "nasa" / "5_Battery_Data_Set").rglob("B*.mat"))
    if not nasa_paths:
        raise FileNotFoundError("NASA MATLAB files are missing; run the benchmark downloader")
    oxford_path = raw_root / "oxford" / "Oxford_Battery_Degradation_Dataset_1.mat"
    if not oxford_path.exists():
        raise FileNotFoundError("Oxford MATLAB file is missing; run the benchmark downloader")
    return {
        "NASA": list(iter_nasa_mat_cycles(nasa_paths)),
        "CALCE": list(iter_calce_xlsx_cycles(raw_root / "calce")),
        "Oxford": list(iter_oxford_mat_cycles(oxford_path)),
    }


def build_all_benchmarks(raw_root: str | Path, output_root: str | Path) -> list[dict[str, object]]:
    output_root = Path(output_root)
    summaries = []
    for name, cycles in load_all_cycles(raw_root).items():
        bundle = build_benchmark_bundle(
            cycles,
            excluded_batteries=EXCLUDED_BATTERIES if name == "NASA" else set(),
        )
        destination = output_root / name.lower()
        save_benchmark_bundle(bundle, destination)
        summaries.append({
            "dataset": name,
            "cycles_loaded": len(cycles),
            "cycles_retained": len(bundle.features),
            "cycles_excluded": len(bundle.exclusions),
            "batteries": bundle.features["battery_id"].nunique(),
            "features": bundle.features.shape[1],
            "sequence_shape": tuple(bundle.sequences.shape),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/v3/features"))
    args = parser.parse_args()
    for row in build_all_benchmarks(args.raw_root, args.output_root):
        print(row)


if __name__ == "__main__":
    main()
