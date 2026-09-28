"""Wind-statistics plot adapted from the original v2 runner."""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_wind_statistics(
    avg: pd.DataFrame,
    maximum: pd.DataFrame,
    minimum: pd.DataFrame,
    std: pd.DataFrame,
    height: float = 120.0,
) -> tuple[go.Figure, go.Figure, float]:
    """Return subplot and single-panel statistics figures at one height.

    Parameters
    ----------
    avg, maximum, minimum, std : pandas.DataFrame
        Statistics frames produced by ``compute_lidar_stats``.
    height : float, default 120.0
        Requested height; the nearest height common to all frames is used.

    Returns
    -------
    tuple of plotly.graph_objects.Figure, plotly.graph_objects.Figure, float
        Four-row subplot, overlaid single-panel figure, and selected height.

    Example
    -------
    ``subplot, single, used_height = plot_wind_statistics(avg, max_, min_, std)``
    builds both figures without displaying or saving either one.
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
    subplot_fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, subplot_titles=labels
    )
    for row, (label, frame) in enumerate(zip(labels, frames), start=1):
        subplot_fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[columns[selected]],
                mode="lines",
                name=label,
            ),
            row=row,
            col=1,
        )
        subplot_fig.update_yaxes(title_text=f"{label} [m/s]", row=row, col=1)
    subplot_fig.update_xaxes(title_text="Time", row=4, col=1)
    subplot_fig.update_layout(
        title=f"LiDAR wind statistics at {selected:g} m", height=950
    )

    single_fig = go.Figure()
    for label, frame in zip(labels, frames):
        single_fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[columns[selected]],
                mode="lines",
                name=label,
            )
        )
    single_fig.update_layout(
        title=f"LiDAR wind statistics at {selected:g} m",
        xaxis_title="Time",
        yaxis_title="Wind speed [m/s]",
        height=950,
    )
    return subplot_fig, single_fig, selected
