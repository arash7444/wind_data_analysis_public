import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from wind_data_analysis.data_reader import (
    find_KNMI_LiDAR_files,
    read_KNMI_LiDAR,
)
from wind_data_analysis.utils import lidar_height
from wind_data_analysis.process import (
    concatenate_wind_stats,
    wind_height_profile,
    compute_lidar_stats,
    calc_shear,
)
from wind_data_analysis.process.calc_turb import calc_ti


def run_program_from_input(input_file: str | Path):
    """
    This function reads a json input file and based on that input
    it runs different features of the wind_data_analysis tool.

    Parameters
    ----------
    input_file : str | Path
        Path to the json input file.

    Returns
    -------
    None
    """

    # read input json file
    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # read general settings from input file
    data_folder = input_data["data_folder"]
    start_date = input_data.get("start_date", None)
    end_date = input_data.get("end_date", None)
    features = input_data.get("features", [])
    hub_height = input_data.get("hub_height", 120.0)
    shear_window = input_data.get("shear_window", 6)
    show_plot = input_data.get("show_plot", True)
    save_dir = input_data.get("save_dir", "outputs")
    extra_plots = input_data.get("extra_plots", True)

    # make output folder
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if len(features) == 0:
        raise ValueError(
            "The input file must contain at least one feature in 'features'."
        )

    # convert all feature names to lower case
    features = [item.lower() for item in features]

    # ------------------------------------------------------------------
    # find lidar files
    # ------------------------------------------------------------------
    lidar_csv_files = find_KNMI_LiDAR_files(
        Path(data_folder),
        start_date=start_date,
        end_date=end_date,
    )

    if len(lidar_csv_files) == 0:
        raise ValueError("No LiDAR files found in the selected folder and date range.")

    print("The following files were found:")
    for file_name in lidar_csv_files:
        print(file_name)

    # ------------------------------------------------------------------
    # read lidar files and calculate statistics file by file
    # ------------------------------------------------------------------
    per_file_stats = []
    heights_all = []

    for file_name in lidar_csv_files:
        # read the LiDAR data
        data_lidar = read_KNMI_LiDAR(file_name)

        # compute the statistics for the LiDAR data
        lidar_stats = compute_lidar_stats(data_lidar)

        # extract the heights from the column names
        heights = lidar_height(data_lidar)

        # save the statistics and heights for each file in a list
        per_file_stats.append(lidar_stats)
        heights_all.append(np.asarray(heights, dtype=float))

    # ------------------------------------------------------------------
    # concatenate the statistics from all files
    # ------------------------------------------------------------------
    lidar_avg_all = concatenate_wind_stats([item.avg for item in per_file_stats])
    lidar_max_all = concatenate_wind_stats([item.max for item in per_file_stats])
    lidar_min_all = concatenate_wind_stats([item.min for item in per_file_stats])
    lidar_std_all = concatenate_wind_stats([item.std for item in per_file_stats])

    # extract the unique heights from all files and sort them
    height_lidar_all = np.unique(np.concatenate(heights_all))

    print("\nAvailable LiDAR heights are:")
    print(height_lidar_all)

    # ------------------------------------------------------------------
    # make wind speed profile once
    # ------------------------------------------------------------------
    wsp_profiles = wind_height_profile(lidar_avg_all, height_lidar_all)

    # ------------------------------------------------------------------
    # feature : TI
    # ------------------------------------------------------------------
    if "ti" in features:
        ti_values = calc_ti(lidar_avg_all, lidar_std_all, hub_height=hub_height)

        # main TI boxplot
        fig_ti = px.box(
            ti_values.ti_raw,
            x="height",
            y="ti",
            points=False,
            title="TI Distribution per Height",
        )
        fig_ti.update_xaxes(title="Height [m]")
        fig_ti.update_yaxes(title="TI [-]")

        html_name = Path(save_dir) / "TI_boxplot.html"
        fig_ti.write_html(html_name)
        print(f"TI plot is saved in: {html_name}")

        if show_plot:
            fig_ti.show()

        # --------------------------------------------------------------
        # extra TI plots
        # --------------------------------------------------------------
        if extra_plots:
            # TI time series at hub height
            ti_hub = ti_values.ti_raw[ti_values.ti_raw["height"] == hub_height].copy()

            if len(ti_hub) > 0:
                fig_ti_hub = go.Figure()

                fig_ti_hub.add_trace(
                    go.Scatter(
                        x=ti_hub.index,
                        y=ti_hub["ti"],
                        mode="lines+markers",
                        name=f"TI at {hub_height}m",
                    )
                )

                fig_ti_hub.update_layout(
                    title=f"TI time series at hub height = {hub_height}m",
                    xaxis_title="Time",
                    yaxis_title="TI [-]",
                )

                html_name = Path(save_dir) / f"TI_timeseries_{int(hub_height)}m.html"
                fig_ti_hub.write_html(html_name)
                print(f"TI time series plot is saved in: {html_name}")

                if show_plot:
                    fig_ti_hub.show()

            # Mean TI vs height
            ti_mean_by_height = (
                ti_values.ti_raw.groupby("height")["ti"]
                .mean()
                .reset_index()
                .sort_values("height")
            )

            fig_ti_mean = go.Figure()

            fig_ti_mean.add_trace(
                go.Scatter(
                    x=ti_mean_by_height["height"],
                    y=ti_mean_by_height["ti"],
                    mode="lines+markers",
                    name="Mean TI",
                )
            )

            fig_ti_mean.update_layout(
                title="Mean TI vs height",
                xaxis_title="Height [m]",
                yaxis_title="Mean TI [-]",
            )

            html_name = Path(save_dir) / "TI_mean_vs_height.html"
            fig_ti_mean.write_html(html_name)
            print(f"Mean TI vs height plot is saved in: {html_name}")

            if show_plot:
                fig_ti_mean.show()

            # TI vs wind speed at hub height
            hub_col = f"Horizontal Wind Speed (m/s) at {int(hub_height)}m"

            if hub_col in lidar_avg_all.columns:
                ti_hub = ti_values.ti_raw[
                    ti_values.ti_raw["height"] == hub_height
                ].copy()

                if len(ti_hub) > 0:
                    ti_hub["wsp"] = lidar_avg_all.loc[ti_hub.index, hub_col].values

                    fig_ti_scatter = go.Figure()

                    fig_ti_scatter.add_trace(
                        go.Scatter(
                            x=ti_hub["wsp"],
                            y=ti_hub["ti"],
                            mode="markers",
                            name="TI vs wind speed",
                        )
                    )

                    fig_ti_scatter.update_layout(
                        title=f"TI vs wind speed at {hub_height}m",
                        xaxis_title="Wind speed [m/s]",
                        yaxis_title="TI [-]",
                    )

                    html_name = Path(save_dir) / f"TI_vs_wsp_{int(hub_height)}m.html"
                    fig_ti_scatter.write_html(html_name)
                    print(f"TI vs wind speed plot is saved in: {html_name}")

                    if show_plot:
                        fig_ti_scatter.show()

        print("TI is calculated and plotted successfully.")

    # ------------------------------------------------------------------
    # feature : shear
    # ------------------------------------------------------------------
    if "shear" in features:
        shear_values = calc_shear(wsp_profiles, window=shear_window)

        # main shear plot
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

        fig_shear.update_yaxes(title_text="Shear slope [-]", row=1, col=1)
        fig_shear.update_yaxes(title_text="Shear slope [-]", row=2, col=1)
        fig_shear.update_yaxes(title_text="Shear slope [-]", row=3, col=1)
        fig_shear.update_xaxes(title_text="Time", row=3, col=1)

        fig_shear.update_layout(
            title="Shear slope vs time",
            height=900,
        )

        html_name = Path(save_dir) / "shear_plot.html"
        fig_shear.write_html(html_name)
        print(f"Shear plot is saved in: {html_name}")

        if show_plot:
            fig_shear.show()

        # --------------------------------------------------------------
        # extra shear plots
        # --------------------------------------------------------------
        if extra_plots:
            # Histogram of alpha
            alpha_clean = shear_values.alpha.dropna()

            fig_alpha_hist = go.Figure()

            fig_alpha_hist.add_trace(
                go.Histogram(
                    x=alpha_clean.values,
                    nbinsx=30,
                    name="Alpha",
                )
            )

            fig_alpha_hist.update_layout(
                title="Histogram of shear exponent alpha",
                xaxis_title="Alpha [-]",
                yaxis_title="Count",
            )

            html_name = Path(save_dir) / "shear_alpha_histogram.html"
            fig_alpha_hist.write_html(html_name)
            print(f"Alpha histogram is saved in: {html_name}")

            if show_plot:
                fig_alpha_hist.show()

            # Boxplot of alpha by hour of day
            alpha_hour = shear_values.alpha.dropna().to_frame(name="alpha")
            alpha_hour["hour"] = alpha_hour.index.hour

            fig_alpha_hour = px.box(
                alpha_hour,
                x="hour",
                y="alpha",
                points=False,
                title="Shear exponent alpha by hour of day",
            )

            fig_alpha_hour.update_xaxes(title="Hour of day")
            fig_alpha_hour.update_yaxes(title="Alpha [-]")

            html_name = Path(save_dir) / "shear_alpha_by_hour.html"
            fig_alpha_hour.write_html(html_name)
            print(f"Alpha by hour plot is saved in: {html_name}")

            if show_plot:
                fig_alpha_hour.show()

            # Alpha vs hub-height wind speed
            hub_col = f"Horizontal Wind Speed (m/s) at {int(hub_height)}m"

            if hub_col in lidar_avg_all.columns:
                alpha_wsp = shear_values.alpha.dropna().to_frame(name="alpha")
                alpha_wsp["wsp"] = lidar_avg_all.loc[alpha_wsp.index, hub_col].values

                fig_alpha_wsp = go.Figure()

                fig_alpha_wsp.add_trace(
                    go.Scatter(
                        x=alpha_wsp["wsp"],
                        y=alpha_wsp["alpha"],
                        mode="markers",
                        name="alpha vs wind speed",
                    )
                )

                fig_alpha_wsp.update_layout(
                    title=f"Shear exponent alpha vs wind speed at {hub_height}m",
                    xaxis_title="Wind speed [m/s]",
                    yaxis_title="Alpha [-]",
                )

                html_name = (
                    Path(save_dir) / f"shear_alpha_vs_wsp_{int(hub_height)}m.html"
                )
                fig_alpha_wsp.write_html(html_name)
                print(f"Alpha vs wind speed plot is saved in: {html_name}")

                if show_plot:
                    fig_alpha_wsp.show()

            # Wind speed profile for selected timestamps
            selected_times = wsp_profiles.index[:: max(1, len(wsp_profiles) // 5)]

            fig_profile = go.Figure()

            for time_i in selected_times:
                fig_profile.add_trace(
                    go.Scatter(
                        x=wsp_profiles.loc[time_i].values,
                        y=wsp_profiles.columns.astype(float),
                        mode="lines+markers",
                        name=str(time_i),
                    )
                )

            fig_profile.update_layout(
                title="Wind speed profiles at selected times",
                xaxis_title="Wind speed [m/s]",
                yaxis_title="Height [m]",
            )

            html_name = Path(save_dir) / "wind_speed_profiles_selected_times.html"
            fig_profile.write_html(html_name)
            print(f"Wind speed profile plot is saved in: {html_name}")

            if show_plot:
                fig_profile.show()

        print("Shear is calculated and plotted successfully.")


if __name__ == "__main__":
    input_file = Path(".", "input_files", "input_config.json")
    run_program_from_input(input_file)
