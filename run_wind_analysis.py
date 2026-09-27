import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from wind_data_analysis.data_reader import (
    find_KNMI_LiDAR_files,
    met_finder,
    read_KNMI_LiDAR,
    read_met,
)
from wind_data_analysis.utils import lidar_height
from wind_data_analysis.process import (
    concatenate_wind_stats,
    wind_height_profile,
    compute_lidar_stats,
    calc_shear,
    compare_lidar_to_metmast,
    determine_comparison_period,
)
from wind_data_analysis.process.calc_turb import calc_ti
from wind_data_analysis.plotting import (
    plot_metmast_difference,
    plot_metmast_metric_summary,
    plot_metmast_scatter,
    plot_metmast_time_series,
    plot_ti_polar_by_height,
    plot_wind_statistics,
)


def _load_lidar_data(
    data_folder,
    start_date=None,
    end_date=None,
    min_lidar_raw_coverage_percent=80.0,
):
    """Load LiDAR files and calculate the statistics used by runner features.

    Parameters
    ----------
    data_folder : str or pathlib.Path
        LiDAR CSV file or folder.
    start_date, end_date : datetime-like or None
        Optional inclusive start and exclusive end for LiDAR observations.
    min_lidar_raw_coverage_percent : float, default 80.0
        Minimum valid raw-sample coverage for a ten-minute LiDAR bin.

    Returns
    -------
    tuple
        Concatenated statistics, heights, wind profile, and source file list.

    Example
    -------
    ``_load_lidar_data("tests/lidar_data", "2020-06-07", "2020-06-08")``
    loads one bundled day.
    """

    lidar_csv_files = find_KNMI_LiDAR_files(
        Path(data_folder), start_date=start_date, end_date=end_date
    )
    if not lidar_csv_files:
        raise ValueError("No LiDAR files found in the selected folder and date range.")
    print("The following LiDAR files were found:")
    for file_name in lidar_csv_files:
        print(file_name)

    per_file_stats = []
    heights_all = []
    for file_name in lidar_csv_files:
        data_lidar = read_KNMI_LiDAR(file_name)
        lidar_stats = compute_lidar_stats(
            data_lidar,
            min_lidar_raw_coverage_percent=min_lidar_raw_coverage_percent,
        )
        per_file_stats.append(lidar_stats)
        heights_all.append(np.asarray(lidar_height(data_lidar), dtype=float))

    frames = [
        concatenate_wind_stats([getattr(item, attribute) for item in per_file_stats])
        for attribute in ("avg", "max", "min", "std")
    ]
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
        frames = [frame[frame.index >= start] for frame in frames]
        if lidar_raw_coverage_all is not None:
            lidar_raw_coverage_all = lidar_raw_coverage_all[
                lidar_raw_coverage_all.index >= start
            ]
            lidar_raw_invalid_all = lidar_raw_invalid_all[
                lidar_raw_invalid_all.index >= start
            ]
    if end_date is not None:
        end = pd.Timestamp(end_date)
        frames = [frame[frame.index < end] for frame in frames]
        if lidar_raw_coverage_all is not None:
            lidar_raw_coverage_all = lidar_raw_coverage_all[
                lidar_raw_coverage_all.index < end
            ]
            lidar_raw_invalid_all = lidar_raw_invalid_all[
                lidar_raw_invalid_all.index < end
            ]
    if frames[0].empty:
        raise ValueError("No LiDAR observations remain inside the selected period.")
    height_lidar_all = np.unique(np.concatenate(heights_all))
    wsp_profiles = wind_height_profile(frames[0], height_lidar_all)
    return (
        *frames,
        height_lidar_all,
        wsp_profiles,
        lidar_csv_files,
        lidar_raw_coverage_all,
        lidar_raw_invalid_all,
    )


def _load_metmast_data(data_folder, start_date=None, end_date=None):
    """Load met-mast files and create unfiltered and selected tidy frames.

    Parameters
    ----------
    data_folder : str or pathlib.Path
        Met-mast NetCDF file or folder.
    start_date, end_date : datetime-like or None
        Optional inclusive start and exclusive end for met-mast observations.

    Returns
    -------
    tuple
        Unfiltered selected-file data, date-filtered data, and source file list.

    Example
    -------
    ``_load_metmast_data("tests/metmast_data", "2020-06-07", "2020-06-08")``
    returns the bundled one-day mast observations.
    """

    mast_files = met_finder(data_folder, start_date=start_date, end_date=end_date)
    if not mast_files:
        raise ValueError(
            "No met-mast NetCDF files were found in the selected path and date range."
        )
    mast_all = pd.concat([read_met(path) for path in mast_files]).sort_index()
    mast_selected = mast_all
    if start_date is not None:
        mast_selected = mast_selected[mast_selected.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        mast_selected = mast_selected[mast_selected.index < pd.Timestamp(end_date)]
    if mast_selected.empty:
        raise ValueError("No met-mast observations remain inside the selected period.")
    return mast_all, mast_selected.copy(), mast_files


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
    data_folder_lidar = input_data.get(
        "data_folder_lidar", input_data.get("data_folder")
    )
    start_date_lidar = input_data.get(
        "start_date_lidar", input_data.get("start_date")
    )
    end_date_lidar = input_data.get(
        "end_date_lidar", input_data.get("end_date")
    )
    data_folder_metmast = input_data.get(
        "data_folder_Metmast", input_data.get("metmast_data")
    )
    start_date_metmast = input_data.get(
        "start_date_Metmast", input_data.get("start_date")
    )
    end_date_metmast = input_data.get(
        "end_date_Metmast", input_data.get("end_date")
    )
    features = input_data.get("features", [])
    hub_height = input_data.get("hub_height", 120.0)
    shear_window = input_data.get("shear_window", 6)
    show_plot = input_data.get("show_plot", True)
    save_dir = input_data.get("save_dir", "outputs")
    extra_plots = input_data.get("extra_plots", True)
    min_lidar_raw_coverage_percent = input_data.get(
        "min_lidar_raw_coverage_percent", 80.0
    )

    # make output folder
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if len(features) == 0:
        raise ValueError(
            "The input file must contain at least one feature in 'features'."
        )

    # convert all feature names to lower case
    if isinstance(features, str):
        features = [features]
    features = [item.lower() for item in features]
    unknown = set(features) - {
        "ti",
        "shear",
        "stats",
        "ti_polar",
        "metmast_data",
        "metmast_comparison",
    }
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")

    lidar_features = {"ti", "shear", "stats", "ti_polar", "metmast_comparison"}
    mast_features = {"metmast_data", "metmast_comparison"}
    needs_lidar = bool(set(features) & lidar_features)
    needs_mast = bool(set(features) & mast_features)

    if needs_lidar:
        if not data_folder_lidar:
            raise ValueError(
                "LiDAR features require 'data_folder_lidar' (legacy 'data_folder' is also accepted)."
            )
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
        ) = _load_lidar_data(
            data_folder_lidar,
            start_date=start_date_lidar,
            end_date=end_date_lidar,
            min_lidar_raw_coverage_percent=min_lidar_raw_coverage_percent,
        )
        print("\nAvailable LiDAR heights are:")
        print(height_lidar_all)

    if needs_mast:
        if not data_folder_metmast:
            raise ValueError(
                "Met-mast features require 'data_folder_Metmast'."
            )
        mast_data_all, mast_data_selected, mast_files = _load_metmast_data(
            data_folder_metmast,
            start_date=start_date_metmast,
            end_date=end_date_metmast,
        )
        print("The following met-mast files were found:")
        for file_name in mast_files:
            print(file_name)

    if "metmast_data" in features:
        metmast_csv_path = Path(save_dir) / "metmast_data.csv"
        mast_export = mast_data_selected.reset_index()
        mast_export.to_csv(metmast_csv_path, index=False)
        mast_summary = (
            mast_data_selected.groupby("height")["wind_speed"]
            .agg(source_rows="size", valid_wind_speeds="count", mean_wind_speed="mean")
            .reset_index()
        )
        print("\nMet-mast data summary:")
        print("Period:", mast_data_selected.index.min(), "to", mast_data_selected.index.max())
        print("Rows:", len(mast_data_selected))
        print(mast_summary.to_string(index=False))
        print("Tidy met-mast observations are saved in:", metmast_csv_path)

    # ------------------------------------------------------------------
    # feature : met-mast comparison
    # ------------------------------------------------------------------
    if "metmast_comparison" in features:
        timestamp_tolerance = pd.Timedelta(
            seconds=input_data.get("timestamp_tolerance_seconds", 30.0)
        )
        comparison_start, comparison_end = determine_comparison_period(
            lidar_avg_all.index,
            mast_data_all.index,
            start_date_lidar=start_date_lidar,
            end_date_lidar=end_date_lidar,
            start_date_mast=start_date_metmast,
            end_date_mast=end_date_metmast,
            timestamp_tolerance=timestamp_tolerance,
        )
        comparison = compare_lidar_to_metmast(
            lidar_avg_all,
            mast_data_all,
            start_date=comparison_start,
            end_date=comparison_end,
            max_height_difference_m=input_data.get(
                "max_height_difference_m", 2.0
            ),
            timestamp_tolerance=timestamp_tolerance,
            lidar_raw_sample_coverage=lidar_raw_coverage_all,
            lidar_raw_invalid_sample_count=lidar_raw_invalid_all,
            min_lidar_raw_coverage_percent=min_lidar_raw_coverage_percent,
        )

        matched_path = Path(save_dir) / "metmast_matched_data.csv"
        metrics_path = Path(save_dir) / "metmast_metrics.csv"
        pairs_path = Path(save_dir) / "metmast_height_pairs.csv"
        comparison.matched_data.to_csv(matched_path, index=False)
        comparison.metrics.to_csv(metrics_path, index=False)
        comparison.pairing_report.to_csv(pairs_path, index=False)

        figures = {
            "metmast_timeseries.html": plot_metmast_time_series(comparison),
            "metmast_scatter.html": plot_metmast_scatter(comparison),
            "metmast_difference.html": plot_metmast_difference(comparison),
            "metmast_summary.html": plot_metmast_metric_summary(comparison),
        }
        for filename, figure in figures.items():
            output_path = Path(save_dir) / filename
            figure.write_html(output_path)
            if show_plot:
                figure.show()

        summary = comparison.metrics[
            [
                "lidar_height_m",
                "mast_height_m",
                "matched_observation_count",
                "availability_percent",
                "invalid_lidar_bin_count",
                "invalid_mast_bin_count",
                "invalid_paired_bin_count",
                "lidar_raw_invalid_sample_count",
                "lidar_mean_raw_sample_coverage_percent",
                "lidar_min_raw_sample_coverage_percent",
                "lidar_below_minimum_coverage_bin_count",
                "bias_lidar_minus_mast_m_s",
                "mae_m_s",
                "rmse_m_s",
                "pearson_correlation",
            ]
        ].copy()
        summary = summary.rename(
            columns={"availability_percent": "paired_bin_availability_percent"}
        )
        summary.insert(2, "period_start", str(comparison_start))
        summary.insert(3, "period_end_exclusive", str(comparison_end))
        print("\nLiDAR/met-mast comparison:")
        print(
            "Minimum raw LiDAR coverage threshold:",
            f"{float(min_lidar_raw_coverage_percent):g}%",
        )
        print("Availability column is paired-bin availability.")
        print(summary.to_string(index=False))
        print("\nDiscovered LiDAR heights [m]:", comparison.lidar_heights_m)
        print("Discovered met-mast heights [m]:", comparison.mast_heights_m)
        print("Unmatched LiDAR heights [m]:", comparison.unmatched_lidar_heights_m)
        print("Unmatched met-mast heights [m]:", comparison.unmatched_mast_heights_m)
        failed_pairs = comparison.pairing_report[
            comparison.pairing_report["status"].ne("matched")
        ]
        if not failed_pairs.empty:
            print("Pairs that could not be aligned:")
            print(
                failed_pairs[
                    ["lidar_height_m", "mast_height_m", "message"]
                ].to_string(index=False)
            )
        print(
            "Comparison outputs are saved in:",
            Path(save_dir),
        )

    # ------------------------------------------------------------------
    # feature : TI
    # ------------------------------------------------------------------
    if "ti" in features or "ti_polar" in features:
        ti_values = calc_ti(lidar_avg_all, lidar_std_all, hub_height=hub_height)

    if "ti_polar" in features:
        fig_polar = plot_ti_polar_by_height(
            ti_values.ti_raw,
            heights=input_data.get("polar_heights"),
            ti_stat=input_data.get("polar_stat", "median"),
            ncols=input_data.get("polar_ncols", 2),
        )
        html_name = Path(save_dir) / "TI_polar_by_height.html"
        fig_polar.write_html(html_name)
        print(f"TI polar plot is saved in: {html_name}")
        if show_plot:
            fig_polar.show()

    if "stats" in features:
        fig_stats, selected_height = plot_wind_statistics(
            lidar_avg_all, lidar_max_all, lidar_min_all, lidar_std_all,
            height=input_data.get("stats_height", hub_height),
        )
        html_name = Path(save_dir) / f"stats_{selected_height:g}m.html"
        fig_stats.write_html(html_name)
        print(f"Wind statistics at {selected_height:g} m are saved in: {html_name}")
        if show_plot:
            fig_stats.show()

    if "ti" in features:
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
                        x=ti_hub["Time"],
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
                    ti_hub["wsp"] = ti_hub["wind_speed"]

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
    import argparse

    parser = argparse.ArgumentParser(description="Run wind analysis from a JSON config.")
    parser.add_argument("config", nargs="?", default="input_files/input_config.json")
    run_program_from_input(parser.parse_args().config)
