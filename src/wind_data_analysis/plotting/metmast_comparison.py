"""Reusable Plotly figures for LiDAR and met-mast wind-speed comparisons."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wind_data_analysis.process.metmast_comparison import MetmastComparisonResult


def _pair_groups(result: MetmastComparisonResult):
    """Return matched data grouped in metric-table height-pair order.

    Parameters
    ----------
    result : MetmastComparisonResult
        Completed shared comparison result.

    Returns
    -------
    list of tuple
        ``(LiDAR height, mast height, matched frame)`` entries.

    Example
    -------
    ``_pair_groups(result)[0][2]`` is the first pair's matched data.
    """

    groups = []
    for row in result.metrics.itertuples(index=False):
        data = result.matched_data[
            result.matched_data["lidar_height_m"].eq(row.lidar_height_m)
            & result.matched_data["mast_height_m"].eq(row.mast_height_m)
        ]
        groups.append((row.lidar_height_m, row.mast_height_m, data))
    if not groups:
        raise ValueError("The comparison result contains no successful height pairs.")
    return groups


def _subplot_shape(pair_count: int, ncols: int) -> tuple[int, int]:
    """Calculate a compact subplot grid for a number of height pairs.

    Parameters
    ----------
    pair_count : int
        Number of successful height pairs.
    ncols : int
        Requested maximum number of subplot columns.

    Returns
    -------
    tuple of int
        Number of rows and columns.

    Example
    -------
    ``_subplot_shape(6, 2)`` returns ``(3, 2)``.
    """

    if int(ncols) < 1:
        raise ValueError("ncols must be a positive integer.")
    columns = min(int(ncols), pair_count)
    return math.ceil(pair_count / columns), columns


def plot_metmast_time_series(
    result: MetmastComparisonResult, ncols: int = 2
) -> go.Figure:
    """Plot overlaid LiDAR and met-mast speeds for every matched pair.

    Parameters
    ----------
    result : MetmastComparisonResult
        Completed shared comparison result.
    ncols : int, default 2
        Maximum number of subplot columns.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive multi-pair time-series figure.

    Example
    -------
    ``fig = plot_metmast_time_series(result)`` creates but does not show a plot.
    """

    groups = _pair_groups(result)
    rows, columns = _subplot_shape(len(groups), ncols)
    titles = [
        f"LiDAR {lidar:g} m and met mast {mast:g} m"
        for lidar, mast, _ in groups
    ]
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=titles)
    for index, (lidar_height, mast_height, data) in enumerate(groups):
        row, column = divmod(index, columns)
        figure.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["lidar_wind_speed"],
                mode="lines",
                name=f"LiDAR {lidar_height:g} m",
                legendgroup=f"pair-{index}",
            ),
            row=row + 1,
            col=column + 1,
        )
        figure.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["mast_wind_speed"],
                mode="lines",
                name=f"Met mast {mast_height:g} m",
                legendgroup=f"pair-{index}",
            ),
            row=row + 1,
            col=column + 1,
        )
        figure.update_xaxes(title_text="Time", row=row + 1, col=column + 1)
        figure.update_yaxes(
            title_text="Wind speed [m/s]", row=row + 1, col=column + 1
        )
    figure.update_layout(
        title="LiDAR and met-mast 10-minute mean wind speed",
        height=max(450, 360 * rows),
    )
    return figure


def plot_metmast_scatter(
    result: MetmastComparisonResult, ncols: int = 2
) -> go.Figure:
    """Plot LiDAR against met-mast speed with a 1:1 line per pair.

    Parameters
    ----------
    result : MetmastComparisonResult
        Completed shared comparison result.
    ncols : int, default 2
        Maximum number of subplot columns.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive multi-pair scatter figure.

    Example
    -------
    ``fig = plot_metmast_scatter(result)`` creates but does not show a plot.
    """

    groups = _pair_groups(result)
    rows, columns = _subplot_shape(len(groups), ncols)
    titles = [
        f"LiDAR {lidar:g} m vs met mast {mast:g} m"
        for lidar, mast, _ in groups
    ]
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=titles)
    for index, (lidar_height, mast_height, data) in enumerate(groups):
        row, column = divmod(index, columns)
        values = np.concatenate(
            [data["lidar_wind_speed"].to_numpy(), data["mast_wind_speed"].to_numpy()]
        )
        lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
        figure.add_trace(
            go.Scatter(
                x=data["mast_wind_speed"],
                y=data["lidar_wind_speed"],
                mode="markers",
                name=f"{lidar_height:g}/{mast_height:g} m",
                legendgroup=f"pair-{index}",
            ),
            row=row + 1,
            col=column + 1,
        )
        figure.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                line={"dash": "dash", "color": "black"},
                name="1:1 reference",
                legendgroup="one-to-one",
                showlegend=index == 0,
            ),
            row=row + 1,
            col=column + 1,
        )
        figure.update_xaxes(
            title_text=f"Met mast {mast_height:g} m [m/s]",
            row=row + 1,
            col=column + 1,
        )
        figure.update_yaxes(
            title_text=f"LiDAR {lidar_height:g} m [m/s]",
            row=row + 1,
            col=column + 1,
        )
    figure.update_layout(
        title="LiDAR versus met-mast 10-minute mean wind speed",
        height=max(450, 360 * rows),
    )
    return figure


def plot_metmast_difference(
    result: MetmastComparisonResult, ncols: int = 2
) -> go.Figure:
    """Plot the LiDAR-minus-met-mast difference for every height pair.

    Parameters
    ----------
    result : MetmastComparisonResult
        Completed shared comparison result.
    ncols : int, default 2
        Maximum number of subplot columns.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive multi-pair difference time series.

    Example
    -------
    ``fig = plot_metmast_difference(result)`` creates the signed-error plots.
    """

    groups = _pair_groups(result)
    rows, columns = _subplot_shape(len(groups), ncols)
    titles = [
        f"LiDAR {lidar:g} m − met mast {mast:g} m"
        for lidar, mast, _ in groups
    ]
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=titles)
    for index, (lidar_height, mast_height, data) in enumerate(groups):
        row, column = divmod(index, columns)
        figure.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["difference_lidar_minus_mast_m_s"],
                mode="lines+markers",
                name=f"{lidar_height:g}/{mast_height:g} m",
            ),
            row=row + 1,
            col=column + 1,
        )
        figure.add_hline(
            y=0,
            line_dash="dash",
            line_color="black",
            row=row + 1,
            col=column + 1,
        )
        figure.update_xaxes(title_text="Time", row=row + 1, col=column + 1)
        figure.update_yaxes(
            title_text="LiDAR − met mast [m/s]", row=row + 1, col=column + 1
        )
    figure.update_layout(
        title="Wind-speed difference: LiDAR − met mast",
        height=max(450, 360 * rows),
        showlegend=False,
    )
    return figure


def plot_metmast_metric_summary(result: MetmastComparisonResult) -> go.Figure:
    """Compare bias, MAE, and RMSE across all successful height pairs.

    Parameters
    ----------
    result : MetmastComparisonResult
        Completed shared comparison result.

    Returns
    -------
    plotly.graph_objects.Figure
        Grouped bar chart of signed bias and absolute error metrics.

    Example
    -------
    ``fig = plot_metmast_metric_summary(result)`` creates the compact summary.
    """

    if result.metrics.empty:
        raise ValueError("The comparison result contains no metrics to plot.")
    labels = [
        f"LiDAR {row.lidar_height_m:g} m / mast {row.mast_height_m:g} m"
        for row in result.metrics.itertuples(index=False)
    ]
    figure = go.Figure()
    for column, label in (
        ("bias_lidar_minus_mast_m_s", "Bias (LiDAR − met mast)"),
        ("mae_m_s", "MAE"),
        ("rmse_m_s", "RMSE"),
    ):
        figure.add_trace(go.Bar(x=labels, y=result.metrics[column], name=label))
    figure.update_layout(
        title="LiDAR/met-mast wind-speed comparison by height pair",
        xaxis_title="Actual measurement-height pair",
        yaxis_title="Wind-speed error [m/s]",
        barmode="group",
    )
    return figure
