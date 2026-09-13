"""Compact Gradio client for the leakage-safe v3 SOH estimator."""

from __future__ import annotations

import gradio as gr
import plotly.graph_objects as go

from api.model_registry import registry, soh_to_color


def predict_soh(
    cycle_number: int,
    prior_efc: float,
    ambient_temperature: float,
    segment_duration_s: float,
    segment_energy_wh: float,
    voltage_start_v: float,
    voltage_end_v: float,
    voltage_mean_v: float,
    voltage_std_v: float,
    voltage_slope: float,
    voltage_curvature: float,
    current_mean_c_rate: float,
    current_std_c_rate: float,
    temperature_mean_c: float,
    temperature_std_c: float,
    temperature_rise_c: float,
    temperature_slope: float,
    temperature_observed: bool,
    model_name: str,
):
    """Estimate SOH from one observable partial-discharge segment."""
    features = {
        "cycle_index": int(cycle_number) - 1,
        "prior_equivalent_full_cycles": prior_efc,
        "ambient_temperature": ambient_temperature,
        "segment_duration_s": segment_duration_s,
        "segment_energy_wh": segment_energy_wh,
        "voltage_start_v": voltage_start_v,
        "voltage_end_v": voltage_end_v,
        "voltage_mean_v": voltage_mean_v,
        "voltage_std_v": voltage_std_v,
        "voltage_linear_slope_v_per_s": voltage_slope,
        "voltage_curvature": voltage_curvature,
        "current_mean_c_rate": current_mean_c_rate,
        "current_std_c_rate": current_std_c_rate,
        "temperature_mean_c": temperature_mean_c,
        "temperature_std_c": temperature_std_c,
        "temperature_rise_c": temperature_rise_c,
        "temperature_slope_c_per_s": temperature_slope,
        "temperature_observed": float(temperature_observed),
    }
    name = model_name if model_name != "auto" else None
    result = registry.predict(features, model_name=name)
    soh = result["soh_pct"]
    summary = (
        "## Current-cycle SOH estimate\n\n"
        f"- **SOH:** {soh:.1f}%\n"
        f"- **State:** {result['degradation_state']}\n"
        f"- **Model:** {result['model_used']}\n\n"
        "No v3 RUL, confidence interval, charging benefit, or safety claim is "
        "produced. This laboratory-data model is a research prototype."
    )
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh,
        title={"text": "State of Health (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": soh_to_color(soh)},
            "steps": [
                {"range": [0, 70], "color": "#fee2e2"},
                {"range": [70, 80], "color": "#fef3c7"},
                {"range": [80, 90], "color": "#fef9c3"},
                {"range": [90, 100], "color": "#dcfce7"},
            ],
        },
    ))
    figure.update_layout(height=350)
    return summary, figure


def create_gradio_app() -> gr.Blocks:
    """Build the v3 partial-discharge form."""
    model_choices = ["auto"] + [model["name"] for model in registry.list_models()]
    with gr.Blocks(title="Leakage-safe battery SOH estimator") as demo:
        gr.Markdown(
            "# Leakage-safe battery SOH estimator\n"
            "Enter summaries from the **observable 4.0–3.6 V discharge segment**. "
            "Full-cycle capacity and current-cycle SOH are intentionally absent."
        )
        with gr.Row():
            with gr.Column():
                cycle = gr.Number(label="Cycle number (1-based)", value=100, precision=0)
                prior = gr.Number(label="Prior equivalent full cycles", value=90)
                ambient = gr.Number(label="Ambient temperature (°C)", value=24)
                duration = gr.Number(label="Segment duration (s)", value=800)
                energy = gr.Number(label="Segment energy (Wh)", value=0.9)
                v_start = gr.Number(label="Voltage start (V)", value=4.0)
                v_end = gr.Number(label="Voltage end (V)", value=3.6)
                v_mean = gr.Number(label="Voltage mean (V)", value=3.8)
                v_std = gr.Number(label="Voltage standard deviation (V)", value=0.1)
            with gr.Column():
                v_slope = gr.Number(label="Voltage slope (V/s)", value=-0.0005)
                v_curve = gr.Number(label="Voltage curvature", value=0)
                i_mean = gr.Number(label="Mean current (C-rate)", value=1.0)
                i_std = gr.Number(label="Current standard deviation (C-rate)", value=0.02)
                t_mean = gr.Number(label="Temperature mean (°C)", value=30)
                t_std = gr.Number(label="Temperature standard deviation (°C)", value=2)
                t_rise = gr.Number(label="Temperature rise (°C)", value=6)
                t_slope = gr.Number(label="Temperature slope (°C/s)", value=0.007)
                t_observed = gr.Checkbox(label="Temperature sensor observed", value=True)
                model = gr.Dropdown(choices=model_choices, value="auto", label="Deployable v3 model")
        button = gr.Button("Estimate current SOH", variant="primary")
        with gr.Row():
            result = gr.Markdown()
            gauge = gr.Plot(label="SOH gauge")
        button.click(
            fn=predict_soh,
            inputs=[
                cycle, prior, ambient, duration, energy, v_start, v_end, v_mean,
                v_std, v_slope, v_curve, i_mean, i_std, t_mean, t_std, t_rise,
                t_slope, t_observed, model,
            ],
            outputs=[result, gauge],
        )
        gr.Markdown(
            "### Scope\n"
            "Research prototype only. The model was evaluated for current-cycle "
            "SOH on NASA, CALCE, and Oxford laboratory datasets. It is not a "
            "validated RUL estimator or charging-control policy."
        )
    return demo
