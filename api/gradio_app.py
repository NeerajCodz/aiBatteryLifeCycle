"""
api.gradio_app
==============
Gradio interface for interactive battery SOH / RUL prediction.
Mounted at /gradio inside the FastAPI application.
"""

from __future__ import annotations

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from api.model_registry import registry, classify_degradation, soh_to_color

# ── Prediction function ──────────────────────────────────────────────────────
def predict_soh(
    cycle_number: int,
    ambient_temperature: float,
    peak_voltage: float,
    min_voltage: float,
    avg_current: float,
    avg_temp: float,
    temp_rise: float,
    cycle_duration: float,
    Re: float,
    Rct: float,
    delta_capacity: float,
    model_name: str,
):
    features = {
        "cycle_number": cycle_number,
        "ambient_temperature": ambient_temperature,
        "peak_voltage": peak_voltage,
        "min_voltage": min_voltage,
        "voltage_range": peak_voltage - min_voltage,
        "avg_current": avg_current,
        "avg_temp": avg_temp,
        "temp_rise": temp_rise,
        "cycle_duration": cycle_duration,
        "Re": Re,
        "Rct": Rct,
        "delta_capacity": delta_capacity,
    }

    name = model_name if model_name != "auto" else None
    result = registry.predict(features, model_name=name)

    soh = result["soh_pct"]
    rul = result["rul_cycles"]
    state = result["degradation_state"]
    model_used = result["model_used"]
    ci_lo = result.get("confidence_lower", soh - 2)
    ci_hi = result.get("confidence_upper", soh + 2)

    # Summary text
    summary = (
        f"## Prediction Result\n\n"
        f"- **SOH:** {soh:.1f}%\n"
        f"- **RUL:** {rul:.0f} cycles\n"
        f"- **State:** {state}\n"
        f"- **95% CI:** [{ci_lo:.1f}%, {ci_hi:.1f}%]\n"
        f"- **Model:** {model_used}\n"
    )

    # SOH gauge figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=soh,
        title={"text": "State of Health (%)"},
        delta={"reference": 100, "decreasing": {"color": "red"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": soh_to_color(soh)},
            "steps": [
                {"range": [0, 70], "color": "#fee2e2"},
                {"range": [70, 80], "color": "#fef3c7"},
                {"range": [80, 90], "color": "#fef9c3"},
                {"range": [90, 100], "color": "#dcfce7"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": 70,
            },
        },
    ))
    fig.update_layout(height=350)

    return summary, fig


# ── Capacity trajectory ──────────────────────────────────────────────────────
def plot_capacity_trajectory(battery_id: str):
    from pathlib import Path
    meta_path = Path(__file__).resolve().parents[1] / "cleaned_dataset" / "metadata.csv"
    if not meta_path.exists():
        return None
    meta = pd.read_csv(meta_path)
    sub = meta[meta["battery_id"] == battery_id].sort_values("start_time")
    if sub.empty:
        return None

    caps = sub["Capacity"].dropna().values
    cycles = np.arange(1, len(caps) + 1)
    soh = (caps / 2.0) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cycles, y=soh, mode="lines+markers",
        marker=dict(size=3), line=dict(width=2),
        name=battery_id,
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red",
                  annotation_text="EOL (70%)")
    fig.update_layout(
        title=f"SOH Trajectory — {battery_id}",
        xaxis_title="Cycle", yaxis_title="SOH (%)",
        template="plotly_white", height=400,
    )
    return fig


# ── Build Gradio app ─────────────────────────────────────────────────────────
def create_gradio_app() -> gr.Blocks:
    model_choices = ["auto"] + [m["name"] for m in registry.list_models()]

    with gr.Blocks(
        title="Battery Lifecycle Predictor",
    ) as demo:
        gr.Markdown(
            "# AI Battery Lifecycle Predictor\n"
            "Predict **State of Health (SOH)** and **Remaining Useful Life (RUL)** "
            "using machine-learning models trained on the NASA PCoE Li-ion Battery Dataset."
        )

        with gr.Tab("Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    cycle_number = gr.Number(label="Cycle Number", value=100, precision=0)
                    ambient_temp = gr.Slider(0, 60, value=24, label="Ambient Temperature (°C)")
                    peak_v = gr.Number(label="Peak Voltage (V)", value=4.2)
                    min_v = gr.Number(label="Min Voltage (V)", value=2.7)
                    avg_curr = gr.Number(label="Avg Discharge Current (A)", value=2.0)
                    avg_t = gr.Number(label="Avg Cell Temp (°C)", value=25.0)
                    temp_rise = gr.Number(label="Temp Rise (°C)", value=3.0)
                    duration = gr.Number(label="Cycle Duration (s)", value=3600)
                    re = gr.Number(label="Re (Ω)", value=0.04)
                    rct = gr.Number(label="Rct (Ω)", value=0.02)
                    delta_cap = gr.Number(label="ΔCapacity (Ah)", value=-0.005)
                    model_dd = gr.Dropdown(choices=model_choices, value="auto", label="Model")
                    btn = gr.Button("Predict", variant="primary")

                with gr.Column(scale=1):
                    result_md = gr.Markdown()
                    gauge = gr.Plot(label="SOH Gauge")

            btn.click(
                fn=predict_soh,
                inputs=[cycle_number, ambient_temp, peak_v, min_v, avg_curr,
                        avg_t, temp_rise, duration, re, rct, delta_cap, model_dd],
                outputs=[result_md, gauge],
            )

        with gr.Tab("Battery Explorer"):
            bid_input = gr.Textbox(label="Battery ID", value="B0005", placeholder="e.g., B0005")
            explore_btn = gr.Button("Load Trajectory")
            cap_plot = gr.Plot(label="Capacity Trajectory")
            explore_btn.click(fn=plot_capacity_trajectory, inputs=[bid_input], outputs=[cap_plot])

        with gr.Tab("About"):
            gr.Markdown(
                "## About\n\n"
                "This application predicts Li-ion battery degradation using models trained on the "
                "**NASA Prognostics Center of Excellence (PCoE)** Battery Dataset.\n\n"
                "### Models\n"
                "- Classical ML: Ridge, Lasso, ElasticNet, KNN, SVR, Random Forest, XGBoost, LightGBM\n"
                "- Deep Learning: LSTM (4 variants), BatteryGPT, TFT, iTransformer (3 variants), VAE-LSTM\n"
                "- Ensemble: Stacking, Weighted Average\n\n"
                "### Dataset\n"
                "- 36 Li-ion 18650 cells (nominal 2.0Ah)\n"
                "- Charge/discharge/impedance cycles at three temperature regimes\n"
                "- End-of-Life: 30% capacity fade (1.4Ah)\n\n"
                "### Reference\n"
                "B. Saha and K. Goebel (2007). *Battery Data Set*, NASA Prognostics Data Repository."
            )

    return demo
