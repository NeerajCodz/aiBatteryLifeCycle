from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import plotting


def test_plotting_save_fig(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = plotting.save_fig(fig, "test_plot", directory=tmp_path)
    assert path.exists()
    plt.close(fig)


def test_plotting_functions_smoke():
    cap_df = pd.DataFrame(
        {
            "battery_id": ["B1", "B1", "B2", "B2"],
            "cycle_number": [0, 1, 0, 1],
            "Capacity": [2.0, 1.9, 2.0, 1.95],
            "SoH": [100.0, 95.0, 100.0, 97.5],
            "ambient_temperature": [24, 24, 43, 43],
        }
    )
    fig1 = plotting.plot_capacity_fade(cap_df, save_name=None)
    fig2 = plotting.plot_soh_degradation(cap_df, battery_id="B1", save_name=None)
    fig3 = plotting.plot_capacity_by_temperature(cap_df, save_name=None)
    assert fig1 is not None and fig2 is not None and fig3 is not None
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
