"""Demonstrate LiDAR discovery, CSV reading, cleaning, and met-mast reading."""

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.traceback import install
install()
console = Console()


import pandas as pd

from wind_data_analysis.data_reader import (
    clean_data,
    find_KNMI_LiDAR_files,
    met_finder,
    read_KNMI_LiDAR,
    read_met,
)
from wind_data_analysis.utils import NA_cols, lidar_height

from demo_support import REPOSITORY_ROOT, make_output_directory

from wind_data_analysis.process.wind_speed_validity import (
    valid_wind_speed_mask,
    validate_coverage_threshold,
)
def main() -> None:
    """Run the data-loading demonstration and save small CSV previews.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Results are printed and written below ``outputs/examples/data_loading``.

    Example
    -------
    Run from the repository root with
    ``python examples/demo_data_loading.py``.
    """
    output_directory = make_output_directory("data_loading")

    lidar_folder = REPOSITORY_ROOT / "tests" / "lidar_data"
    
    lidar_files = find_KNMI_LiDAR_files(
        lidar_folder, start_date="2020-05-01", end_date="2020-05-02"
    )
    lidar_data = read_KNMI_LiDAR(lidar_files[0])

    console.print(f"Found {len(lidar_files)} LiDAR file for the selected day.")
    console.print(f"Loaded {len(lidar_data):,} measurements.")
    console.print(f"Measurement heights: {lidar_height(lidar_data).tolist()}")
    console.print(f"Columns containing missing values: {len(NA_cols(lidar_data))}")

    speed_columns = [column for column in lidar_data if "Wind Speed" in column]
    cleaning_example = lidar_data[speed_columns[:2]].head(5).copy()
    cleaning_example.iloc[0, 0] = -1 # injecting an invalid value
    cleaning_example.iloc[1, 0] = 100 # injecting an invalid value
    # cleaned = clean_data(cleaning_example) # this is old function and it detects and cleans the invalid values, then it returns the cleaned data
    cleaned_mask = valid_wind_speed_mask(cleaning_example) # detecting the invalid values, it returns a boolean dataframe with True for valid values and False for invalid values
    cleaned = cleaning_example.where(cleaned_mask) # cleaning the invalid values, it returns a dataframe with invalid values replaced by NaN
    
    cleaned.to_csv(output_directory / "cleaned_lidar_preview.csv")
    console.print("Injected -1 and 100 m/s values were converted to NaN:")
    console.print(cleaned.head(2).to_string())

    metmast_folder = REPOSITORY_ROOT / "tests" / "metmast_data"
    metmast_files = met_finder(
        metmast_folder, start_date="2020-05-01", end_date="2020-06-01"
    )
    metmast_data = read_met(
        metmast_files[0], start_date="2020-05-01", end_date="2020-05-02"
    )
    metmast_data.head(20).to_csv(output_directory / "metmast_preview.csv")
    console.print(
        f"Loaded {len(metmast_data):,} tidy met-mast rows at "
        f"{metmast_data['height'].nunique()} heights."
    )
    console.print(f"Saved previews in: {output_directory}")


if __name__ == "__main__":
    main()
