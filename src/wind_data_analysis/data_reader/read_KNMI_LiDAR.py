import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import re

from wind_data_analysis.utils.lidar_file_parsing import extract_date_from_filename


def find_KNMI_LiDAR_files(
    file_folder: str | Path, start_date=None, end_date=None
) -> list:
    """
    Finds all KNMI LiDAR files in a directory and returns a list of file paths.

    Parameters
    ----------
    directory : str | Path
                Path to the directory containing the LiDAR files.
    start_date: pandas.Timestamp or datetime, optional. the format is YYYY-MM-DD

    end_date: pandas.Timestamp or datetime, optional. the format is YYYY-MM-DD

    Returns
    -------
    file_list : list
                List of CSV file paths in the directory.
    """

    if start_date is not None:
        start_date = pd.Timestamp(start_date)

    if end_date is not None:
        end_date = pd.Timestamp(end_date)

    CSV_files = []
    if os.path.isfile(file_folder):
        print("The path includes a file.")
        CSV_files.append(file_folder)
    else:
        for root, dirs, files in os.walk(os.path.join(file_folder)):
            for file in files:
                if file.endswith(".CSV") or file.endswith(".csv"):
                    if start_date is None:
                        tmp = os.path.join(root, file)
                        CSV_files.append(tmp)
                    else:
                        filename = file

                        file_day = extract_date_from_filename(filename)
                        if file_day is None:
                            print(
                                f"Could not extract date from filename: {filename}. Skipping this file."
                            )
                            continue

                        if (file_day >= start_date) and (file_day < end_date):
                            tmp = os.path.join(root, file)
                            CSV_files.append(tmp)

    return CSV_files


def read_KNMI_LiDAR(file_path: str | Path) -> pd.DataFrame:
    """
    Reads a KNMI LiDAR file and returns a pandas DataFrame with the data.

    Parameters
    ----------
    file_path : str | Path
                Path to the CSV file containing the LiDAR data.
                CSV file is expected to have columns for Time and Date, wind speed, and wind direction

    Returns
    -------
    lidar_data : pd.DataFrame
                 DataFrame containing the LiDAR data

    Example
    -------
    >>> data_lidar = read_KNMI_LiDAR('lidar_data.csv')
    >>> data_lidar = read_KNMI_LiDAR(Path(".","tests","lidar_data","ZephIR_Cabauw_ZP738_raw_20200501_v1.CSV"))
    >>>     lidar_CSV_files = find_KNMI_LiDAR_files(
        Path(".","tests","lidar_data"), start_date="2020-05-01", end_date="2020-05-02")
    """

    lidar_data = pd.read_csv(file_path, skiprows=1)

    print(lidar_data.columns)
    print(lidar_data.head())

    lidar_data.columns = [
        col.strip() for col in lidar_data.columns
    ]  # remove whitespace from beginning and end of column names

    lidar_data["Time"] = pd.to_datetime(
        lidar_data["Time and Date"], format="%d/%m/%Y %H:%M:%S"
    )

    # I use  lidar_data['Time'] as index so that it would be easy to resample and compare it with the mast data
    lidar_data = lidar_data.set_index("Time")

    return lidar_data


if __name__ == "__main__":
    # lidar_test = Path(".", "tests", "lidar_data")
    # lidar_test = Path(".", "tests", "lidar_data_10min")
    # lidar_CSV_files = find_KNMI_LiDAR_files(
    #     lidar_test, start_date="2020-05-01", end_date="2020-05-02"
    # )

    lidar_CSV_files = find_KNMI_LiDAR_files(
        Path(".", "tests", "lidar_data"), start_date="2020-05-01", end_date="2020-05-02"
    )

    print(lidar_CSV_files)

    for file_name in lidar_CSV_files:
        data_lidar = read_KNMI_LiDAR(file_name)
        print(data_lidar)
