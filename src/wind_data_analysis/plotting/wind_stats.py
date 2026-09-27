"""Wind-statistics plot adapted from the original v2 runner."""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_wind_statistics(
    avg: pd.DataFrame, maximum: pd.DataFrame, minimum: pd.DataFrame,
    std: pd.DataFrame, height: float = 120.0,
) -> tuple[go.Figure, float]:
    """Return mean/max/min/std time series and the selected measurement height.

    Inputs are concatenated frames from ``compute_lidar_stats``. The
    nearest height available in all four frames is used (lower height on
    a tie), and is shown in the title and returned to the caller. Each
    frame retains its own time index. No interpolation is performed.
    """
    height = float(height)
    if not np.isfinite(height):
        raise ValueError("height must be finite.")
    frames = [avg, maximum, minimum, std]
    common = set.intersection(*(set(frame.columns) for frame in frames))
    columns = {}
    for col in common:
        match = re.fullmatch(r"Horizontal Wind Speed \(m/s\) at (\d+(?:\.\d+)?)m", col)
        if match:
            columns[float(match.group(1))] = col
    if not columns:
        raise ValueError("No common wind-speed height found in the statistics frames.")
    selected = min(sorted(columns), key=lambda h: abs(h - height))
    labels = ["Mean", "Maximum", "Minimum", "Standard deviation"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=labels)
    for row, (label, frame) in enumerate(zip(labels, frames), start=1):
        fig.add_trace(go.Scatter(x=frame.index, y=frame[columns[selected]],
                                 mode="lines", name=label), row=row, col=1)
        fig.update_yaxes(title_text=f"{label} [m/s]", row=row, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_layout(title=f"LiDAR wind statistics at {selected:g} m", height=950)


    fig2 = go.Figure()
    for row, (label, frame) in enumerate(zip(labels, frames), start=1):
        fig2.add_trace(go.Scatter(x=frame.index, y=frame[columns[selected]],
                                 mode="lines", name=label))
        # fig.update_yaxes(title_text=f"{label} [m/s]", row=row, col=1)
    # fig.update_xaxes(title_text="Time", row=4, col=1)
    fig2.update_layout(title=f"LiDAR wind statistics at {selected:g} m", height=950)
    fig2.show()
    return fig, fig2, selected