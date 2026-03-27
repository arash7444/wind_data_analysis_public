from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from wind_data_analysis.data_reader import (
    find_KNMI_LiDAR_files,
    read_KNMI_LiDAR,
)
from wind_data_analysis.process import (
    calc_shear,
    concatenate_wind_stats,
    compute_lidar_stats,
    wind_height_profile,
)
from wind_data_analysis.process.calc_turb import calc_ti
from wind_data_analysis.utils import lidar_height


def load_and_process_lidar_data(
    data_folder: str | Path,
    start_date=None,
    end_date=None,
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
        lidar_stats = compute_lidar_stats(data_lidar)

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
            x=ti_hub.index,
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
    ti_hub["wsp"] = lidar_avg_all.loc[ti_hub.index, hub_col].values

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
            x=ti_hub.index,
            y=lidar_avg_all.loc[ti_hub.index, hub_col],
            mode="lines+markers",
            name="Wind speed",
        ),
        row=1,
        col=1,
    )

    # add TI time series
    fig.add_trace(
        go.Scatter(
            x=ti_hub.index,
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

        # folder containing LiDAR data files
        data_folder = st.text_input(
            "Data folder",
            value="tests/lidar_data",
        )

        # start and end date for file selection
        start_date = st.text_input("Start date", value="2020-05-01")
        end_date = st.text_input("End date", value="2020-05-03")

        # feature selection
        features = st.multiselect(
            "Features",
            options=["ti", "shear"],
            default=["ti"],
        )

        # hub height for TI and some extra plots
        hub_height = st.number_input(
            "Hub height [m]",
            value=120.0,
            step=1.0,
        )

        # rolling window used in shear calculation
        shear_window = st.number_input(
            "Shear rolling window",
            value=6,
            step=1,
        )

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
            # read LiDAR files and calculate statistics
            with st.spinner("Reading LiDAR files and calculating statistics..."):
                (
                    lidar_avg_all,
                    lidar_max_all,
                    lidar_min_all,
                    lidar_std_all,
                    height_lidar_all,
                    wsp_profiles,
                    lidar_csv_files,
                ) = load_and_process_lidar_data(
                    data_folder=data_folder,
                    start_date=start_date,
                    end_date=end_date,
                )

            st.success("Data loaded successfully.")

            # show list of found files
            with st.expander("Found files", expanded=False):
                for file_name in lidar_csv_files:
                    st.write(file_name)

            # ----------------------------------------------------------
            # TI
            # ----------------------------------------------------------
            if "ti" in features:
                st.header("Turbulence Intensity (TI)")

                # calculate TI values
                with st.spinner("Calculating TI..."):
                    ti_values = calc_ti(
                        lidar_avg_all,
                        lidar_std_all,
                        hub_height=hub_height,
                    )

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
