import pandas as pd
import numpy as np
import scipy.stats as sci_stats
import matplotlib.pyplot as plt

from pathlib import Path
import re
import os

import seaborn as sns
import warnings

from wind_data_analysis.data_reader import (
    find_KNMI_LiDAR_files,
    read_KNMI_LiDAR,
    clean_data,
)
from wind_data_analysis.utils import lidar_height


from wind_data_analysis.process.concatenate_wind_stats import concatenate_wind_stats
from wind_data_analysis.process.wind_height_profile import wind_height_profile


from dataclasses import dataclass


@dataclass
class LidarStats:
    """Container for LiDAR summary statistics."""

    avg: pd.DataFrame
    max: pd.DataFrame
    min: pd.DataFrame
    std: pd.DataFrame


def compute_lidar_stats(
    data: pd.DataFrame,
) -> LidarStats:
    """
    This function checks if the LiDAR data is high frequency or low frequency based on the presence of standard deviation columns.
    If the standard deviation columns are present, it is considered low frequency data and the function will call "compute_lidar_stats_lowres" to calculate the statistics.
    If the standard deviation columns are missing, it is considered high frequency data and the function will call "compute_lidar_stats_highres" to calculate the statistics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data from LiDAR measurements.

    Returns
    -------
    LidarStats
        Object containing average, maximum, minimum, and standard deviation
        dataframes.
    it contains the following attributes:
    avg: pd.DataFrame
            DataFrame containing the average wind speed and direction at each height and time.
    max: pd.DataFrame
            DataFrame containing the maximum wind speed and direction at each height and time.
    min: pd.DataFrame
            DataFrame containing the minimum wind speed and direction at each height and time.
    std: pd.DataFrame
            DataFrame containing the standard deviation of wind speed at each height and time.

    """

    lidar_data = data.copy()

    std_cols = [
        cols for cols in lidar_data.columns if "Horizontal Wind Speed Std." in cols
    ]

    if len(std_cols) == 0:
        print("Lidar data is High frequency because all std columns are missing \n")

        LidarStats = compute_lidar_stats_highres(lidar_data)
        return LidarStats
    else:
        print(" Lidar data is Low frequency because std columns are present \n")
        LidarStats = compute_lidar_stats_lowres(lidar_data)
        return LidarStats


def compute_lidar_stats_highres(
    data: pd.DataFrame,
) -> LidarStats:
    """
    Calculate basic statistics fo a LiDAR dataset, including mean, median, standard deviation

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data from LiDAR (high-resolution) measurements.

    returns
    -------
    LidarStats
        Object containing average, maximum, minimum, and standard deviation
        dataframes.
    it contains the following attributes:
    avg: pd.DataFrame
            DataFrame containing the average wind speed and direction at each height and time.
    max: pd.DataFrame
            DataFrame containing the maximum wind speed and direction at each height and time.
    min: pd.DataFrame
            DataFrame containing the minimum wind speed and direction at each height and time.
    std: pd.DataFrame
            DataFrame containing the standard deviation of wind speed at each height and time.
    """
    lidar_data = data.copy()

    # # Take only wind related data and covert them to numbers
    wind_cols = [
        cols
        for cols in lidar_data.columns
        if "Wind Speed" in cols or "Wind Direction" in cols
    ]

    lidar_numeric = lidar_data[wind_cols].copy()

    # --- TODO: I need to clean up data more effectively.
    # lidar_numeric = clean_data(lidar_numeric)

    lidar_numeric["Time_seconds"] = (
        lidar_numeric.index.hour * 3600  # convert hours to seconds
        + lidar_numeric.index.minute * 60  # convert minutes to seconds
        + lidar_numeric.index.second
    )

    lidar_avg = lidar_numeric.resample("10min").mean()
    lidar_max = lidar_numeric.resample("10min").max()
    lidar_min = lidar_numeric.resample("10min").min()

    lidar_std = lidar_numeric.resample("10min").std()

    if lidar_std.isnull().any().any():
        warnings.warn(
            "Lidar data is not high frequency because all std values are missing\n"
            "Use a high frequency lidar file to get the correct statistics"
        )

    print(lidar_max.columns)  # check the column names of max dataframe
    print(lidar_min.columns)  # check the column names of min dataframe
    print(lidar_std.columns)  # check the column names of std dataframe
    print(lidar_avg.columns)  # check the column names of avg dataframe

    return LidarStats(
        avg=lidar_avg,
        max=lidar_max,
        min=lidar_min,
        std=lidar_std,
    )


def compute_lidar_stats_lowres(data: pd.DataFrame) -> LidarStats:
    """
    Low resolution LiDAR data is already averaged over 10 minutes, so we only select the interesting columns and return them as the statistics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data from LiDAR (low-resolution) measurements.

    Returns
    -------
    LidarStats
        Object containing average, maximum, minimum, and standard deviation
        dataframes

    """

    lidar_data = data.copy()

    ## Take only wind related data and save them in different dataframe for average, max, min, and std.
    #  Since the data is already averaged over 10 minutes.

    # # Take only wind related data and covert them to numbers
    wind_cols = [
        cols
        for cols in lidar_data.columns
        if "Wind Speed" in cols or "Wind Direction" in cols
    ]
    lidar_numeric = lidar_data[wind_cols].copy()

    # --- TODO: I need to clean up data more effectively.
    # lidar_numeric = clean_data(lidar_numeric)

    avg_cols = [
        cols
        for cols in lidar_data.columns
        if "Horizontal Wind Speed (m/s) at" in cols or "Wind Direction (deg) at" in cols
    ]
    lidar_avg = lidar_numeric[avg_cols].copy()

    std_cols = [
        cols for cols in lidar_data.columns if "Horizontal Wind Speed Std." in cols
    ]

    if len(std_cols) == 0:
        raise ValueError(
            "Lidar data is not low frequency because all std columns are missing \n"
        )
    else:
        lidar_std = lidar_data[std_cols].copy()

    max_cols = [
        cols for cols in lidar_data.columns if "Horizontal Wind Speed Max" in cols
    ]
    lidar_max = lidar_data[max_cols].copy()

    min_cols = [
        cols for cols in lidar_data.columns if "Horizontal Wind Speed Min" in cols
    ]
    lidar_min = lidar_data[min_cols].copy()

    assert lidar_max.shape == lidar_min.shape == lidar_std.shape, (
        "Max, min, and std dataframes should have the same shape"
    )

    # Remove the "Max", "Min", and "Std. Dev." from the column names to make them consistent with the high-resolution data columns for easier comparison and plotting later on.

    lidar_max.columns = lidar_max.columns.str.replace(" Max", "")
    lidar_min.columns = lidar_min.columns.str.replace(" Min", "")
    lidar_std.columns = lidar_std.columns.str.replace(" Std. Dev.", "")

    return LidarStats(
        avg=lidar_avg,
        max=lidar_max,
        min=lidar_min,
        std=lidar_std,
    )


if __name__ == "__main__":
    lidar_csv_files = find_KNMI_LiDAR_files(
        Path(".", "tests", "lidar_data_10min"),
        start_date="2020-05-01",
        end_date="2020-05-03",
    )

    per_file_stats = []  # empty list to store the statistics for each file to concatenate them later
    heights_all = []  # empty list to store the heights for each file to concatenate them later

    for file_name in lidar_csv_files:
        # read the LiDAR data
        data_lidar = read_KNMI_LiDAR(file_name)

        # compute the statistics for the LiDAR data
        lidar_stats = compute_lidar_stats(data_lidar)

        # extract the heights from the column names and save them in a list
        heights = lidar_height(data_lidar)

        # save the statistics and heights for each file in a list to concatenate them later
        per_file_stats.append(lidar_stats)
        heights_all.append(np.asarray(heights, dtype=float))

    # concatenate the statistics from all files into a single dataframe for each statistic type (avg, max, min, std) and sort them by time index
    lidar_avg_all = concatenate_wind_stats([item.avg for item in per_file_stats])
    lidar_max_all = concatenate_wind_stats([item.max for item in per_file_stats])
    lidar_min_all = concatenate_wind_stats([item.min for item in per_file_stats])
    lidar_std_all = concatenate_wind_stats([item.std for item in per_file_stats])

    # extract the unique heights from all files and sort them
    height_lidar_all = np.unique(np.concatenate(heights_all))

    print(lidar_avg_all)

    # Build a Dataframe of wind speed values where the index is time and the columns are the heights. The values are the wind speed at that height and time.
    wsp_profiles = wind_height_profile(lidar_avg_all, height_lidar_all)

    plt.figure(figsize=(12, 8))
    plt.plot(
        lidar_avg_all.index,
        lidar_avg_all["Horizontal Wind Speed (m/s) at 299m"],
        label="Average Wind Speed at 299m",
        marker="o",
        color="blue",
    )
