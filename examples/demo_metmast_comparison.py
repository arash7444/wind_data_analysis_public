"""Run the shared multi-height LiDAR/met-mast comparison on bundled data."""

from pathlib import Path

from wind_data_analysis.data_reader import read_KNMI_LiDAR, read_met
from wind_data_analysis.plotting import (
    plot_metmast_difference,
    plot_metmast_metric_summary,
    plot_metmast_scatter,
    plot_metmast_time_series,
)
from wind_data_analysis.process import compare_lidar_to_metmast, compute_lidar_stats
from rich.console import Console
from rich.markdown import Markdown
from rich.traceback import install
install()
console = Console()

SHOW_PLOTS = False
SAVE_PLOTS = True

def main() -> None:
    """Compare the bundled 7 June 2020 LiDAR and met-mast observations.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Results are printed and written below ``outputs/examples/metmast``.

    Example
    -------
    Run ``uv run python examples/demo_metmast_comparison.py`` from the root.
    """

    lidar_path = Path(
        "tests/lidar_data/ZephIR_Cabauw_ZP738_raw_20200607_v1.CSV"
    )
    mast_path = Path(
        "tests/metmast_data/cesar_tower_meteo_lb1_t10_v1.2_202006.nc"
    )
    output = Path("outputs/examples/metmast")
    output.mkdir(parents=True, exist_ok=True)

    lidar_mean = compute_lidar_stats(read_KNMI_LiDAR(lidar_path)).avg
    mast_data = read_met(mast_path)
    result = compare_lidar_to_metmast(
        lidar_mean,
        mast_data,
        start_date="2020-06-07",
        end_date="2020-06-08",
        max_height_difference_m=2.0,
    )

    result.matched_data.to_csv(output / "metmast_matched_data.csv", index=False)
    result.metrics.to_csv(output / "metmast_metrics.csv", index=False)
    result.pairing_report.to_csv(output / "metmast_height_pairs.csv", index=False)
    figures = {
        "metmast_timeseries.html": plot_metmast_time_series(result),
        "metmast_scatter.html": plot_metmast_scatter(result),
        "metmast_difference.html": plot_metmast_difference(result),
        "metmast_summary.html": plot_metmast_metric_summary(result),
    }
    for filename, figure in figures.items():
        if SAVE_PLOTS:
            figure.write_html(output / filename)
        if SHOW_PLOTS:
            figure.show()

    # console.print(result.pairing_report.to_string(index=False))
    # console.print(result.metrics.to_string(index=False))
    if SAVE_PLOTS:
        console.print(f"Outputs written to {output}")


if __name__ == "__main__":
    main()
