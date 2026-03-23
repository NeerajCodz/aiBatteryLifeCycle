from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def run_async():
    def _run(coro):
        return asyncio.run(coro)

    return _run


@pytest.fixture
def sample_features():
    return {
        "battery_id": "B0005",
        "cycle_number": 120,
        "ambient_temperature": 24.0,
        "peak_voltage": 4.19,
        "min_voltage": 2.61,
        "avg_current": 1.82,
        "avg_temp": 24.0,
        "temp_rise": 14.7,
        "cycle_duration": 3690.0,
        "Re": 0.045,
        "Rct": 0.069,
        "delta_capacity": -0.005,
    }
