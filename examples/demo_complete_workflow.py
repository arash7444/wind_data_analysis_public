"""Run a coherent end-to-end analysis with the repository's public functions."""

import json

import plotly.express as px
import plotly.graph_objects as go

from wind_data_analysis.plotting import plot_ti_polar_by_height, plot_wind_statistics
from wind_data_analysis.process import calc_shear, calc_ti

from demo_support import REPOSITORY_ROOT, load_lidar_statistics, make_output_directory


def main() -> None:
    """Load, validate, analyze, plot, and report bundled LiDAR measurements.

    Parameters
    ----------
    None

    Returns
    -------
    None
        CSV, JSON, and interactive HTML outputs are saved under
        ``outputs/examples/complete_workflow``.

    Example
    -------
    Run from the repository root with
    ``python examples/demo_complete_workflow.py``.
    """
    output_directory = make_output_directory("complete_workflow")
    results = load_lidar_statistics(
        REPOSITORY_ROOT / "tests" / "lidar_data", "2020-05-01", "2020-05-03"
    )

    average = results["average"]
    standard_deviation = results["standard_deviation"]
    profiles = results["profiles"]
    if average.empty or profiles.empty:
        raise RuntimeError("The selected LiDAR data did not produce usable profiles.")

    ti_values = calc_ti(average, standard_deviation, hub_height=139.0)
    shear_values = calc_shear(profiles, window=6)

    statistics_figure_subplot, statistics_figure, statistics_height = plot_wind_statistics(
        average,
        results["maximum"],
        results["minimum"],
        standard_deviation,
        height=139.0,
    )
    statistics_figure_subplot.write_html(output_directory / "wind_statistics_subplot.html")
    statistics_figure.write_html(output_directory / "wind_statistics.html")
    plot_ti_polar_by_height(
        ti_values.ti_raw, heights=[19.0, 59.0, 139.0, 199.0]
    ).write_html(output_directory / "ti_polar.html")

    ti_boxplot = px.box(
        ti_values.ti_raw,
        x="height",
        y="ti",
        points=False,
        title="Turbulence intensity distribution by height",
    )
    ti_boxplot.write_html(output_directory / "ti_boxplot.html")

    shear_figure = go.Figure(
        go.Scatter(
            x=shear_values.alpha.index,
            y=shear_values.alpha,
            mode="markers",
            name="alpha",
        )
    )
    shear_figure.update_layout(
        title="Power-law shear exponent", xaxis_title="Time", yaxis_title="Alpha [-]"
    )
    shear_figure.write_html(output_directory / "shear_alpha.html")

    ti_values.ti_median.to_csv(output_directory / "median_ti_by_height.csv", index=False)
    shear_values.alpha.to_csv(output_directory / "shear_alpha.csv", header=True)

    summary = {
        "input_files": [str(path) for path in results["files"]],
        "period_start": str(average.index.min()),
        "period_end": str(average.index.max()),
        "ten_minute_records": len(average),
        "measurement_heights_m": results["heights"].tolist(),
        "statistics_height_m": statistics_height,
        "valid_ti_values": int(ti_values.ti_raw["ti"].notna().sum()),
        "valid_shear_values": int(shear_values.alpha.notna().sum()),
        "machine_learning": "Skipped: ML exists only on the separate ML branch.",
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved the complete workflow outputs in: {output_directory}")


if __name__ == "__main__":
    main()
