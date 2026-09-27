"""Shared orchestration helpers for the runnable examples."""

from pathlib import Path

import numpy as np

from wind_data_analysis.data_reader import find_KNMI_LiDAR_files, read_KNMI_LiDAR
from wind_data_analysis.process import (
    compute_lidar_stats,
    concatenate_wind_stats,
    wind_height_profile,
)
from wind_data_analysis.utils import lidar_height


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "examples"


def load_lidar_statistics(
    data_folder: str | Path,
    start_date: str,
    end_date: str,
) -> dict[str, object]:
    """Load LiDAR files and assemble the statistics used by the demos.

    Parameters
    ----------
    data_folder : str | Path
        Folder containing KNMI LiDAR CSV files.
    start_date : str
        Inclusive start date in a format accepted by pandas.
    end_date : str
        Exclusive end date in a format accepted by pandas.

    Returns
    -------
    dict[str, object]
        Files, concatenated statistics, measured heights, and wind profiles.

    Example
    -------
    >>> results = load_lidar_statistics(
    ...     REPOSITORY_ROOT / "tests" / "lidar_data",
    ...     "2020-05-01",
    ...     "2020-05-02",
    ... )
    >>> len(results["files"])
    1
    """
    files = find_KNMI_LiDAR_files(data_folder, start_date, end_date)
    if not files:
        raise RuntimeError("No LiDAR files matched the requested period.")

    statistics = []
    heights = []
    for file_path in files:
        lidar_data = read_KNMI_LiDAR(file_path)
        statistics.append(compute_lidar_stats(lidar_data))
        heights.append(np.asarray(lidar_height(lidar_data), dtype=float))

    average = concatenate_wind_stats([item.avg for item in statistics])
    maximum = concatenate_wind_stats([item.max for item in statistics])
    minimum = concatenate_wind_stats([item.min for item in statistics])
    standard_deviation = concatenate_wind_stats([item.std for item in statistics])
    measured_heights = np.unique(np.concatenate(heights))

    return {
        "files": files,
        "average": average,
        "maximum": maximum,
        "minimum": minimum,
        "standard_deviation": standard_deviation,
        "heights": measured_heights,
        "profiles": wind_height_profile(average, measured_heights),
    }


def make_output_directory(name: str) -> Path:
    """Create and return a named output directory for an example.

    Parameters
    ----------
    name : str
        Directory name below ``outputs/examples``.

    Returns
    -------
    Path
        Absolute path to the created directory.

    Example
    -------
    >>> directory = make_output_directory("data_loading")
    >>> directory.name
    'data_loading'
    """
    output_directory = OUTPUT_ROOT / name
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory
