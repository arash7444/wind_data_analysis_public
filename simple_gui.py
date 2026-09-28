from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from wind_data_analysis.data_reader import (
    find_KNMI_LiDAR_files,
    met_finder,
    read_KNMI_LiDAR,
    read_met,
)
from wind_data_analysis.process import (
    calc_shear,
    concatenate_wind_stats,
    compute_lidar_stats,
    compare_lidar_to_metmast,
    determine_comparison_period,
    wind_height_profile,
)
from wind_data_analysis.process.calc_turb import calc_ti
from wind_data_analysis.utils import lidar_height
from wind_data_analysis.plotting import (
    plot_metmast_difference,
    plot_metmast_metric_summary,
    plot_metmast_scatter,
    plot_metmast_time_series,
    plot_ti_polar_by_height,
    plot_wind_statistics,
)


def load_and_process_lidar_data(
    data_folder: str | Path,
    start_date=None,
    end_date=None,
    min_lidar_raw_coverage_percent: float = 80.0,
):
    """
    This function finds LiDAR files, reads them, calculates statistics
    file by file, and concatenates the results.

    Parameters
    ----------
    data_folder : str | Path
        Folder containing LiDAR files.
    start_date : str or None
        Start date in YYYY-MM-DD format.
    end_date : str or None
        End date in YYYY-MM-DD format.
    min_lidar_raw_coverage_percent : float, default 80.0
        Minimum valid raw-sample coverage for a ten-minute LiDAR bin.

    Returns
    -------
    lidar_avg_all : pd.DataFrame
        Concatenated average wind speed data from all files.
    lidar_max_all : pd.DataFrame
        Concatenated maximum wind speed data from all files.
    lidar_min_all : pd.DataFrame
        Concatenated minimum wind speed data from all files.
    lidar_std_all : pd.DataFrame
        Concatenated standard deviation wind speed data from all files.
    height_lidar_all : np.ndarray
        Array of all unique LiDAR heights.
    wsp_profiles : pd.DataFrame
        Wind speed profiles where each row is one timestamp and columns are heights.
    lidar_csv_files : list
        List of LiDAR files found in the selected folder and date range.
    lidar_raw_coverage_all : pandas.DataFrame or None
        Per-bin valid raw-sample percentages when raw files are loaded.
    lidar_raw_invalid_all : pandas.DataFrame or None
        Per-bin rejected raw-sample counts when raw files are loaded.

    Example
    -------
    ``load_and_process_lidar_data("tests/lidar_data", min_lidar_raw_coverage_percent=80)``
    loads the bundled LiDAR data using the default coverage threshold.
    """

    # ------------------------------------------------------------------
    # find LiDAR files in the selected folder and date range
    # ------------------------------------------------------------------
    lidar_csv_files = find_KNMI_LiDAR_files(
        Path(data_folder),
        start_date=start_date,
        end_date=end_date,
    )

    # stop if no files are found
    if len(lidar_csv_files) == 0:
        raise ValueError("No LiDAR files found in the selected folder and date range.")

    # create empty lists to store results for each file
    per_file_stats = []
    heights_all = []

    # ------------------------------------------------------------------
    # read each file, calculate statistics, and store heights
    # ------------------------------------------------------------------
    for file_name in lidar_csv_files:
        # read one LiDAR file
        data_lidar = read_KNMI_LiDAR(file_name)

        # calculate statistics for this file
        lidar_stats = compute_lidar_stats(
            data_lidar,
            min_lidar_raw_coverage_percent=min_lidar_raw_coverage_percent,
        )

        # extract available heights from the column names
        heights = lidar_height(data_lidar)

        # store statistics and heights in lists
        per_file_stats.append(lidar_stats)
        heights_all.append(np.asarray(heights, dtype=float))

    # ------------------------------------------------------------------
    # concatenate all file statistics into single dataframes
    # ------------------------------------------------------------------
    lidar_avg_all = concatenate_wind_stats([item.avg for item in per_file_stats])
    lidar_max_all = concatenate_wind_stats([item.max for item in per_file_stats])
    lidar_min_all = concatenate_wind_stats([item.min for item in per_file_stats])
    lidar_std_all = concatenate_wind_stats([item.std for item in per_file_stats])
    coverage_items = [
        item.raw_sample_coverage_percent
        for item in per_file_stats
        if item.raw_sample_coverage_percent is not None
    ]
    invalid_raw_items = [
        item.raw_invalid_sample_count
        for item in per_file_stats
        if item.raw_invalid_sample_count is not None
    ]
    lidar_raw_coverage_all = (
        concatenate_wind_stats(coverage_items) if coverage_items else None
    )
    lidar_raw_invalid_all = (
        concatenate_wind_stats(invalid_raw_items) if invalid_raw_items else None
    )

    if start_date is not None:
        start = pd.Timestamp(start_date)
        lidar_avg_all = lidar_avg_all[lidar_avg_all.index >= start]
        lidar_max_all = lidar_max_all[lidar_max_all.index >= start]
        lidar_min_all = lidar_min_all[lidar_min_all.index >= start]
        lidar_std_all = lidar_std_all[lidar_std_all.index >= start]
        if lidar_raw_coverage_all is not None:
            lidar_raw_coverage_all = lidar_raw_coverage_all[
                lidar_raw_coverage_all.index >= start
            ]
            lidar_raw_invalid_all = lidar_raw_invalid_all[
                lidar_raw_invalid_all.index >= start
            ]
    if end_date is not None:
        end = pd.Timestamp(end_date)
        lidar_avg_all = lidar_avg_all[lidar_avg_all.index < end]
        lidar_max_all = lidar_max_all[lidar_max_all.index < end]
        lidar_min_all = lidar_min_all[lidar_min_all.index < end]
        lidar_std_all = lidar_std_all[lidar_std_all.index < end]
        if lidar_raw_coverage_all is not None:
            lidar_raw_coverage_all = lidar_raw_coverage_all[
                lidar_raw_coverage_all.index < end
            ]
            lidar_raw_invalid_all = lidar_raw_invalid_all[
                lidar_raw_invalid_all.index < end
            ]
    if lidar_avg_all.empty:
        raise ValueError("No LiDAR observations remain inside the selected period.")

    # collect all unique heights from all files
    height_lidar_all = np.unique(np.concatenate(heights_all))

    # make wind speed profiles where columns are heights
    wsp_profiles = wind_height_profile(lidar_avg_all, height_lidar_all)

    return (
        lidar_avg_all,
        lidar_max_all,
        lidar_min_all,
        lidar_std_all,
        height_lidar_all,
        wsp_profiles,
        lidar_csv_files,
        lidar_raw_coverage_all,
        lidar_raw_invalid_all,
    )


def plot_ti_main(ti_values):
    """Make the main boxplot of TI versus height."""

    # make boxplot of TI values for each height
    fig_ti = px.box(
        ti_values.ti_raw,
        x="height",
        y="ti",
        points=False,
        title="TI Distribution per Height",
    )

    # update axis labels
    fig_ti.update_xaxes(title="Height [m]")
    fig_ti.update_yaxes(title="TI [-]")

    return fig_ti


def plot_ti_timeseries_at_hub(ti_values, hub_height: float):
    """Plot TI time series at the selected hub height."""

    # select TI values only for the chosen hub height
    ti_hub = ti_values.ti_raw[ti_values.ti_raw["height"] == hub_height].copy()

    # sometimes height may be stored as integer instead of float
    if len(ti_hub) == 0:
        ti_hub = ti_values.ti_raw[ti_values.ti_raw["height"] == int(hub_height)].copy()

    # return nothing if no matching data is found
    if len(ti_hub) == 0:
        return None

    fig = go.Figure()

    # add TI time series
    fig.add_trace(
        go.Scatter(
            x=ti_hub["Time"],
            y=ti_hub["ti"],
            mode="lines+markers",
            name=f"TI at {hub_height}m",
        )
    )

    # update figure layout
    fig.update_layout(
        title=f"TI time series at hub height = {hub_height}m",
        xaxis_title="Time",
        yaxis_title="TI [-]",
    )

    return fig


def plot_ti_mean_vs_height(ti_values):
    """Plot mean TI as a function of height."""

    # calculate mean TI for each height
    ti_mean_by_height = (
        ti_values.ti_raw.groupby("height")["ti"]
        .mean()
        .reset_index()
        .sort_values("height")
    )

    fig = go.Figure()

    # add mean TI profile
    fig.add_trace(
        go.Scatter(
            x=ti_mean_by_height["height"],
            y=ti_mean_by_height["ti"],
            mode="lines+markers",
            name="Mean TI",
        )
    )

    # update figure layout
    fig.update_layout(
        title="Mean TI vs height",
        xaxis_title="Height [m]",
        yaxis_title="Mean TI [-]",
    )

    return fig


def plot_ti_vs_wsp(ti_values, lidar_avg_all, hub_height: float):
    """Plot TI versus wind speed at hub height."""

    # make the column name corresponding to hub-height wind speed
    hub_col = f"Horizontal Wind Speed (m/s) at {int(hub_height)}m"

    # stop if the selected height is not available
    if hub_col not in lidar_avg_all.columns:
        return None

    # select TI values for the selected hub height
    ti_hub = ti_values.ti_raw[ti_values.ti_raw["height"] == hub_height].copy()

    # sometimes height may be stored as integer instead of float
    if len(ti_hub) == 0:
        ti_hub = ti_values.ti_raw[ti_values.ti_raw["height"] == int(hub_height)].copy()

    # return nothing if no matching data is found
    if len(ti_hub) == 0:
        return None

    # add wind speed values at the same timestamps
    ti_hub["wsp"] = ti_hub["wind_speed"]

    fig = go.Figure()

    # add scatter plot of TI versus wind speed
    fig.add_trace(
        go.Scatter(
            x=ti_hub["wsp"],
            y=ti_hub["ti"],
            mode="markers",
            name="TI vs wind speed",
        )
    )

    # update figure layout
    fig.update_layout(
        title=f"TI vs wind speed at {hub_height}m",
        xaxis_title="Wind speed [m/s]",
        yaxis_title="TI [-]",
    )

    return fig


def plot_ti_wsp_and_ti_time_series(ti_values, lidar_avg_all, hub_height: float):
    """Plot hub-height wind speed and TI in two subplots."""

    # make the column name corresponding to hub-height wind speed
    hub_col = f"Horizontal Wind Speed (m/s) at {int(hub_height)}m"

    # stop if the selected height is not available
    if hub_col not in lidar_avg_all.columns:
        return None

    # select TI values for the selected hub height
    ti_hub = ti_values.ti_raw[ti_values.ti_raw["height"] == hub_height].copy()

    # sometimes height may be stored as integer instead of float
    if len(ti_hub) == 0:
        ti_hub = ti_values.ti_raw[ti_values.ti_raw["height"] == int(hub_height)].copy()

    # return nothing if no matching data is found
    if len(ti_hub) == 0:
        return None

    # make two subplots with shared x-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"Wind speed at {hub_height}m",
            f"TI at {hub_height}m",
        ),
    )

    # add hub-height wind speed time series
    fig.add_trace(
        go.Scatter(
            x=ti_hub["Time"],
            y=ti_hub["wind_speed"],
            mode="lines+markers",
            name="Wind speed",
        ),
        row=1,
        col=1,
    )

    # add TI time series
    fig.add_trace(
        go.Scatter(
            x=ti_hub["Time"],
            y=ti_hub["ti"],
            mode="lines+markers",
            name="TI",
        ),
        row=2,
        col=1,
    )

    # update axes labels
    fig.update_yaxes(title_text="Wind speed [m/s]", row=1, col=1)
    fig.update_yaxes(title_text="TI [-]", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    # update figure layout
    fig.update_layout(
        title=f"Wind speed and TI at {hub_height}m",
        height=700,
    )

    return fig


def plot_shear_main(shear_values):
    """Plot raw shear, rolling median, and rolling mean."""

    # make 3 subplots for different versions of alpha
    fig_shear = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Raw shear slope",
            "Rolling median",
            "Rolling mean",
        ),
    )

    # add raw alpha values
    fig_shear.add_trace(
        go.Scatter(
            x=shear_values.alpha.index,
            y=shear_values.alpha.values,
            mode="markers",
            name="shear slope for LiDAR data",
        ),
        row=1,
        col=1,
    )

    # add rolling median values
    fig_shear.add_trace(
        go.Scatter(
            x=shear_values.alpha_roll_med.index,
            y=shear_values.alpha_roll_med.values,
            mode="lines+markers",
            name="LiDAR - rolling median",
        ),
        row=2,
        col=1,
    )

    # add rolling mean values
    fig_shear.add_trace(
        go.Scatter(
            x=shear_values.alpha_roll_mean.index,
            y=shear_values.alpha_roll_mean.values,
            mode="lines+markers",
            name="LiDAR - rolling mean",
        ),
        row=3,
        col=1,
    )

    # update axis labels
    fig_shear.update_yaxes(title_text="Shear slope [-]", row=1, col=1)
    fig_shear.update_yaxes(title_text="Shear slope [-]", row=2, col=1)
    fig_shear.update_yaxes(title_text="Shear slope [-]", row=3, col=1)
    fig_shear.update_xaxes(title_text="Time", row=3, col=1)

    # update figure layout
    fig_shear.update_layout(
        title="Shear slope vs time",
        height=900,
    )

    return fig_shear


def plot_shear_histogram(shear_values):
    """Plot histogram of shear exponent alpha."""

    # remove missing alpha values before plotting
    alpha_clean = shear_values.alpha.dropna()

    fig = go.Figure()

    # add histogram of alpha
    fig.add_trace(
        go.Histogram(
            x=alpha_clean.values,
            nbinsx=30,
            name="Alpha",
        )
    )

    # update figure layout
    fig.update_layout(
        title="Histogram of shear exponent alpha",
        xaxis_title="Alpha [-]",
        yaxis_title="Count",
    )

    return fig


def plot_shear_by_hour(shear_values):
    """Plot boxplot of shear exponent alpha by hour of day."""

    # make a dataframe of alpha values and corresponding hour
    alpha_hour = shear_values.alpha.dropna().to_frame(name="alpha")
    alpha_hour["hour"] = alpha_hour.index.hour

    # make boxplot by hour
    fig = px.box(
        alpha_hour,
        x="hour",
        y="alpha",
        points=False,
        title="Shear exponent alpha by hour of day",
    )

    # update axis labels
    fig.update_xaxes(title="Hour of day")
    fig.update_yaxes(title="Alpha [-]")

    return fig


def plot_shear_alpha_vs_wsp(shear_values, lidar_avg_all, hub_height: float):
    """Plot shear exponent alpha versus hub-height wind speed."""

    # make the column name corresponding to hub-height wind speed
    hub_col = f"Horizontal Wind Speed (m/s) at {int(hub_height)}m"

    # stop if the selected height is not available
    if hub_col not in lidar_avg_all.columns:
        return None

    # combine alpha values and hub-height wind speed
    alpha_wsp = shear_values.alpha.dropna().to_frame(name="alpha")
    alpha_wsp["wsp"] = lidar_avg_all.loc[alpha_wsp.index, hub_col].values

    fig = go.Figure()

    # add scatter plot of alpha versus wind speed
    fig.add_trace(
        go.Scatter(
            x=alpha_wsp["wsp"],
            y=alpha_wsp["alpha"],
            mode="markers",
            name="alpha vs wind speed",
        )
    )

    # update figure layout
    fig.update_layout(
        title=f"Shear exponent alpha vs wind speed at {hub_height}m",
        xaxis_title="Wind speed [m/s]",
        yaxis_title="Alpha [-]",
    )

    return fig


def plot_wind_profiles_selected_times(wsp_profiles):
    """Plot wind speed profiles at a few selected timestamps."""

    # select a few timestamps across the whole period
    selected_times = wsp_profiles.index[:: max(1, len(wsp_profiles) // 5)]

    fig = go.Figure()

    # plot wind speed profile for each selected time
    for time_i in selected_times:
        fig.add_trace(
            go.Scatter(
                x=wsp_profiles.loc[time_i].values,
                y=wsp_profiles.columns.astype(float),
                mode="lines+markers",
                name=str(time_i),
            )
        )

    # update figure layout
    fig.update_layout(
        title="Wind speed profiles at selected times",
        xaxis_title="Wind speed [m/s]",
        yaxis_title="Height [m]",
    )

    return fig


def main():
    """Main Streamlit GUI."""

    # set page title and layout
    st.set_page_config(page_title="Wind Data Analysis", layout="wide")

    # main title and short description
    st.title("Wind Data Analysis Tool Demo")
    st.write("Simple GUI for demonstrating TI and shear analysis.")
    st.write("Run TI and shear analysis from KNMI LiDAR data.")

    # ------------------------------------------------------------------
    # sidebar input settings
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Input settings")

        # feature selection
        features = st.multiselect(
            "Features",
            options=[
                "ti",
                "shear",
                "stats",
                "ti_polar",
                "metmast_data",
                "metmast_comparison",
            ],
            default=["ti"],
        )

        lidar_features = {"ti", "shear", "stats", "ti_polar", "metmast_comparison"}
        mast_features = {"metmast_data", "metmast_comparison"}
        needs_lidar = bool(set(features) & lidar_features)
        needs_mast = bool(set(features) & mast_features)

        data_folder_lidar = ""
        start_date_lidar = None
        end_date_lidar = None
        min_lidar_raw_coverage_percent = 80.0
        if needs_lidar:
            st.subheader("LiDAR input")
            data_folder_lidar = st.text_input(
                "LiDAR data folder or CSV file",
                value="tests/lidar_data",
            )
            start_date_lidar_text = st.text_input(
                "LiDAR start date (optional)", value="2020-06-07"
            )
            end_date_lidar_text = st.text_input(
                "LiDAR end date (optional)", value="2020-06-08"
            )
            start_date_lidar = start_date_lidar_text.strip() or None
            end_date_lidar = end_date_lidar_text.strip() or None
            min_lidar_raw_coverage_percent = st.number_input(
                "Minimum raw LiDAR coverage [%]",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
            )

        data_folder_metmast = ""
        start_date_metmast = None
        end_date_metmast = None
        max_height_difference_m = 2.0
        if needs_mast:
            st.subheader("Met-mast input")
            data_folder_metmast = st.text_input(
                "Met-mast NetCDF file or folder",
                value=(
                    "tests/metmast_data/"
                    "cesar_tower_meteo_lb1_t10_v1.2_202006.nc"
                ),
            )
            start_date_metmast_text = st.text_input(
                "Met-mast start date (optional)", value="2020-06-07"
            )
            end_date_metmast_text = st.text_input(
                "Met-mast end date (optional)", value="2020-06-08"
            )
            start_date_metmast = start_date_metmast_text.strip() or None
            end_date_metmast = end_date_metmast_text.strip() or None

        if "metmast_comparison" in features:
            max_height_difference_m = st.number_input(
                "Maximum height difference [m]",
                min_value=0.0,
                value=2.0,
                step=0.5,
            )

        hub_height = 120.0
        shear_window = 6
        if needs_lidar:
            hub_height = st.number_input(
                "Hub height [m]",
                value=120.0,
                step=1.0,
            )
            if "shear" in features:
                shear_window = st.number_input(
                    "Shear rolling window",
                    value=6,
                    step=1,
                )

        stats_height = hub_height
        if "stats" in features:
            stats_height = st.number_input(
                "Statistics height [m] (nearest measured height)", value=120.0, step=1.0,
            )
        polar_heights_text = ""
        polar_stat = "median"
        if "ti_polar" in features:
            polar_heights_text = st.text_input(
                "Polar heights [m], comma separated (blank = all)", value="",
            )
            polar_stat = st.selectbox("TI statistic for polar plots", ["median", "mean"])

        # run button
        run_button = st.button("Run analysis", type="primary")

    # ------------------------------------------------------------------
    # start analysis when user clicks the button
    # ------------------------------------------------------------------
    if run_button:
        # make sure at least one feature is selected
        if len(features) == 0:
            st.warning("Please select at least one feature.")
            return

        try:
            if needs_lidar:
                with st.spinner("Reading LiDAR files and calculating statistics..."):
                    (
                        lidar_avg_all,
                        lidar_max_all,
                        lidar_min_all,
                        lidar_std_all,
                        height_lidar_all,
                        wsp_profiles,
                        lidar_csv_files,
                        lidar_raw_coverage_all,
                        lidar_raw_invalid_all,
                    ) = load_and_process_lidar_data(
                        data_folder=data_folder_lidar,
                        start_date=start_date_lidar,
                        end_date=end_date_lidar,
                        min_lidar_raw_coverage_percent=(
                            min_lidar_raw_coverage_percent
                        ),
                    )
                with st.expander("Found LiDAR files", expanded=False):
                    for file_name in lidar_csv_files:
                        st.write(file_name)

            if needs_mast:
                mast_files = met_finder(
                    data_folder_metmast,
                    start_date=start_date_metmast,
                    end_date=end_date_metmast,
                )
                if not mast_files:
                    raise ValueError(
                        "No met-mast NetCDF files were found for the selected period."
                    )
                with st.spinner("Reading met-mast observations..."):
                    mast_data_all = pd.concat(
                        [read_met(path) for path in mast_files]
                    ).sort_index()
                    mast_data_selected = mast_data_all
                    if start_date_metmast is not None:
                        mast_data_selected = mast_data_selected[
                            mast_data_selected.index >= pd.Timestamp(start_date_metmast)
                        ]
                    if end_date_metmast is not None:
                        mast_data_selected = mast_data_selected[
                            mast_data_selected.index < pd.Timestamp(end_date_metmast)
                        ]
                    if mast_data_selected.empty:
                        raise ValueError(
                            "No met-mast observations remain inside the selected period."
                        )
                with st.expander("Found met-mast files", expanded=False):
                    for file_name in mast_files:
                        st.write(file_name)

            st.success("Requested data loaded successfully.")

            if "metmast_data" in features:
                st.header("Met-mast data")
                mast_summary = (
                    mast_data_selected.groupby("height")["wind_speed"]
                    .agg(
                        source_rows="size",
                        valid_wind_speeds="count",
                        mean_wind_speed="mean",
                    )
                    .reset_index()
                )
                st.write(
                    f"Period: {mast_data_selected.index.min()} to "
                    f"{mast_data_selected.index.max()}"
                )
                st.dataframe(mast_summary, use_container_width=True)
                st.download_button(
                    "Download tidy met-mast CSV",
                    data=mast_data_selected.reset_index().to_csv(index=False),
                    file_name="metmast_data.csv",
                    mime="text/csv",
                )

            # ----------------------------------------------------------
            # Met-mast comparison
            # ----------------------------------------------------------
            if "metmast_comparison" in features:
                with st.spinner("Aligning LiDAR and met-mast wind speeds..."):
                    comparison_start, comparison_end = determine_comparison_period(
                        lidar_avg_all.index,
                        mast_data_all.index,
                        start_date_lidar=start_date_lidar,
                        end_date_lidar=end_date_lidar,
                        start_date_mast=start_date_metmast,
                        end_date_mast=end_date_metmast,
                        timestamp_tolerance="30s",
                    )
                    comparison = compare_lidar_to_metmast(
                        lidar_avg_all,
                        mast_data_all,
                        start_date=comparison_start,
                        end_date=comparison_end,
                        max_height_difference_m=max_height_difference_m,
                        timestamp_tolerance="30s",
                        lidar_raw_sample_coverage=lidar_raw_coverage_all,
                        lidar_raw_invalid_sample_count=lidar_raw_invalid_all,
                        min_lidar_raw_coverage_percent=(
                            min_lidar_raw_coverage_percent
                        ),
                    )

                st.header("LiDAR and met-mast wind-speed comparison")
                st.caption(
                    f"Comparison overlap: {comparison_start} to "
                    f"{comparison_end} (exclusive)"
                )
                st.write(
                    "Discovered LiDAR heights [m]:",
                    list(comparison.lidar_heights_m),
                )
                st.write(
                    "Discovered met-mast heights [m]:",
                    list(comparison.mast_heights_m),
                )
                st.write(
                    "Unmatched LiDAR heights [m]:",
                    list(comparison.unmatched_lidar_heights_m),
                )
                st.write(
                    "Unmatched met-mast heights [m]:",
                    list(comparison.unmatched_mast_heights_m),
                )
                st.subheader("Automatically selected height pairs")
                st.dataframe(comparison.pairing_report, use_container_width=True)
                failed_pairs = comparison.pairing_report[
                    comparison.pairing_report["status"].ne("matched")
                ]
                if not failed_pairs.empty:
                    st.warning(
                        "Some height pairs could not be aligned. See their messages "
                        "in the pairing table; valid pairs are retained."
                    )
                st.subheader("Per-height comparison metrics")
                st.caption(
                    "Bias and differences are LiDAR − met mast. "
                    "availability_percent is paired-bin availability: valid paired "
                    "10-minute bins divided by expected bins. Raw LiDAR coverage "
                    f"is separate; the configured minimum is "
                    f"{min_lidar_raw_coverage_percent:g}%."
                )
                st.dataframe(comparison.metrics, use_container_width=True)
                st.plotly_chart(
                    plot_metmast_metric_summary(comparison),
                    use_container_width=True,
                )
                st.plotly_chart(
                    plot_metmast_time_series(comparison),
                    use_container_width=True,
                )
                st.plotly_chart(
                    plot_metmast_scatter(comparison),
                    use_container_width=True,
                )
                st.plotly_chart(
                    plot_metmast_difference(comparison),
                    use_container_width=True,
                )

            # ----------------------------------------------------------
            # TI
            # ----------------------------------------------------------
            if "ti" in features or "ti_polar" in features:
                # calculate TI values
                with st.spinner("Calculating TI..."):
                    ti_values = calc_ti(
                        lidar_avg_all,
                        lidar_std_all,
                        hub_height=hub_height,
                    )

            if "ti_polar" in features:
                st.header("TI by direction and reference wind speed")
                polar_heights = (
                    [float(h.strip()) for h in polar_heights_text.split(",")]
                    if polar_heights_text.strip() else None
                )
                st.plotly_chart(
                    plot_ti_polar_by_height(ti_values.ti_raw, polar_heights, polar_stat),
                    use_container_width=True,
                )

            if "stats" in features:
                st.header("Wind statistics")
                subplot_fig, single_fig, selected_height = plot_wind_statistics(
                    lidar_avg_all, lidar_max_all, lidar_min_all, lidar_std_all,
                    height=stats_height,
                )
                st.caption(f"Using measured height {selected_height:g} m (requested {stats_height:g} m).")
                st.plotly_chart(subplot_fig, use_container_width=True)
                st.plotly_chart(single_fig, use_container_width=True)

            if "ti" in features:
                st.header("Turbulence Intensity (TI)")
                # main TI plot
                st.plotly_chart(plot_ti_main(ti_values), use_container_width=True)

                # TI time series at hub height
                fig = plot_ti_timeseries_at_hub(ti_values, hub_height)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

                # mean TI versus height
                st.plotly_chart(
                    plot_ti_mean_vs_height(ti_values),
                    use_container_width=True,
                )

                # TI versus wind speed at hub height
                fig = plot_ti_vs_wsp(ti_values, lidar_avg_all, hub_height)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

                # hub-height wind speed and TI time series together
                fig = plot_ti_wsp_and_ti_time_series(
                    ti_values,
                    lidar_avg_all,
                    hub_height,
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

            # ----------------------------------------------------------
            # Shear
            # ----------------------------------------------------------
            if "shear" in features:
                st.header("Wind Shear")

                # calculate shear values
                with st.spinner("Calculating shear..."):
                    shear_values = calc_shear(
                        wsp_profiles,
                        window=int(shear_window),
                    )

                # main shear plot
                st.plotly_chart(plot_shear_main(shear_values), use_container_width=True)

                # histogram of alpha
                st.plotly_chart(
                    plot_shear_histogram(shear_values),
                    use_container_width=True,
                )

                # boxplot of alpha by hour of day
                st.plotly_chart(
                    plot_shear_by_hour(shear_values),
                    use_container_width=True,
                )

                # alpha versus wind speed at hub height
                fig = plot_shear_alpha_vs_wsp(
                    shear_values,
                    lidar_avg_all,
                    hub_height,
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

                # wind speed profiles at selected times
                st.plotly_chart(
                    plot_wind_profiles_selected_times(wsp_profiles),
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
