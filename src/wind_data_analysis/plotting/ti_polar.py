"""TI polar plots adapted from the original wind_data_analysis repository."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _bin_center(label: str) -> float:
    left, right = str(label).split("-")
    return (float(left) + float(right)) / 2


def plot_ti_polar_by_height(
    ti_raw: pd.DataFrame,
    heights: list[float] | None = None,
    ti_stat: str = "median",
    ncols: int = 2,
) -> go.Figure:
    """Return a polar TI figure without displaying or saving it.

    ``ti_raw`` is the output of ``calc_ti(...).ti_raw``. Its ``wsp_bin``
    uses wind speed at the reference height selected by ``calc_ti``;
    ``wdir_bin`` uses direction at each plotted height. Bin labels must
    have the form ``'0-10'``. TI is dimensionless.

    ``heights=None`` plots all heights with usable binned data. Explicit
    heights must be present. ``ti_stat`` is ``'median'`` or ``'mean'``;
    ``ncols`` is a positive integer. All panels share radial and colour
    ranges. Missing bins are omitted, not interpreted as zero TI.
    """
    if ti_stat not in {"median", "mean"}:
        raise ValueError("ti_stat must be 'median' or 'mean'.")
    if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
        raise ValueError("ncols must be a positive integer.")
    required = ["height", "ti", "wsp_bin", "wdir_bin"]
    missing = set(required) - set(ti_raw.columns)
    if missing:
        raise ValueError(f"Missing polar plot columns: {sorted(missing)}")
    data = ti_raw[required].copy().replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)
    data = data[data["ti"] >= 0]
    if data.empty:
        raise ValueError("No valid binned TI data to plot.")
    available = sorted(data["height"].unique())
    heights = available if heights is None else list(dict.fromkeys(heights))
    if not heights:
        raise ValueError("Select at least one height.")
    missing_heights = set(heights) - set(available)
    if missing_heights:
        raise ValueError(
            f"No valid binned TI data at heights {sorted(missing_heights)}. "
            f"Available heights: {available}"
        )
    data = data[data["height"].isin(heights)]
    binned = data.groupby(
        ["height", "wsp_bin", "wdir_bin"], observed=True
    )["ti"].agg(ti_stat).reset_index()
    binned["direction"] = binned["wdir_bin"].astype(str).map(_bin_center)
    binned["speed"] = binned["wsp_bin"].astype(str).map(_bin_center)
    ncols = min(ncols, len(heights))
    nrows = math.ceil(len(heights) / ncols)
    specs = [
        [{"type": "polar"} if r * ncols + c < len(heights) else None
         for c in range(ncols)]
        for r in range(nrows)
    ]
    fig = make_subplots(
        rows=nrows, cols=ncols, specs=specs,
        subplot_titles=[f"TI at {h:g} m" for h in heights],
    )
    for i, height in enumerate(heights):
        part = binned[binned["height"] == height]
        fig.add_trace(
            go.Scatterpolar(
                theta=part["direction"], r=part["speed"], mode="markers",
                marker=dict(size=10, color=part["ti"], coloraxis="coloraxis"),
                name=f"{height:g} m",
                hovertemplate=(
                    "Direction bin centre: %{theta:.0f}°<br>"
                    "Reference wind-speed bin centre: %{r:.1f} m/s<br>"
                    "TI: %{marker.color:.3f}<extra>%{fullData.name}</extra>"
                ),
            ), row=i // ncols + 1, col=i % ncols + 1,
        )
    fig.update_polars(
        angularaxis=dict(rotation=90, direction="clockwise"),
        radialaxis=dict(range=[0, float(binned["speed"].max()) + 0.5]),
    )
    fig.update_layout(
        title=f"{ti_stat.capitalize()} TI by direction and reference wind speed [m/s]",
        height=450 * nrows, showlegend=False,
        coloraxis=dict(colorscale="Viridis", cmin=float(binned["ti"].min()),
                       cmax=float(binned["ti"].max()), colorbar=dict(title="TI [-]")),
    )
    return fig
