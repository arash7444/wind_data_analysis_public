"""Create non-interactive Plotly demonstrations from bundled LiDAR data."""

from wind_data_analysis.plotting import plot_ti_polar_by_height, plot_wind_statistics
from wind_data_analysis.process import calc_ti

from demo_support import REPOSITORY_ROOT, load_lidar_statistics, make_output_directory

SHOW_PLOTS = False
SAVE_PLOTS = True


def main() -> None:
    """Build wind-statistics and TI-polar figures and save them as HTML.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Interactive HTML files are saved below ``outputs/examples/visualization``.

    Example
    -------
    Run from the repository root with
    ``python examples/demo_visualization.py``.
    """
    output_directory = make_output_directory("visualization")
    results = load_lidar_statistics(
        REPOSITORY_ROOT / "tests" / "lidar_data_10min",
        "2020-05-01",
        "2020-05-03",
    )

    statistics_subplot, statistics_single, selected_height = plot_wind_statistics(
        results["average"],
        results["maximum"],
        results["minimum"],
        results["standard_deviation"],
        height=120.0,
    )
    ti_values = calc_ti(
        results["average"], results["standard_deviation"], hub_height=139.0
    )
    polar_figure = plot_ti_polar_by_height(
        ti_values.ti_raw, heights=[19.0, 59.0, 139.0, 199.0]
    )

    statistics_path = output_directory / f"wind_statistics_{selected_height:g}m.html"
    statistics_single_path = (
        output_directory / f"wind_statistics_single_{selected_height:g}m.html"
    )
    polar_path = output_directory / "ti_polar_by_height.html"
    figures = {
        statistics_path: statistics_subplot,
        statistics_single_path: statistics_single,
        polar_path: polar_figure,
    }
    for path, figure in figures.items():
        if SAVE_PLOTS:
            figure.write_html(path)
        if SHOW_PLOTS:
            figure.show()

    print(f"Statistics figure uses the nearest common height: {selected_height:g} m")
    if SAVE_PLOTS:
        for path in figures:
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
