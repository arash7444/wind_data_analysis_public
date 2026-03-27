import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from pathlib import Path
import re
import os


def lidar_height(data: pd.DataFrame) -> list:
    """
    Extracts the heights of the LiDAR measurements from the column names of the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data from LiDAR measurements.

    Returns
    -------
    heights : list
        List of heights corresponding to the LiDAR measurements.
    """
    lidar_data = data.copy()

    # Take only wind related data and covert them to numbers
    wind_cols = [
        cols
        for cols in lidar_data.columns
        if "Wind Speed" in cols or "Wind Direction" in cols
    ]

    height_all = []
    for col in wind_cols:
        if "Wind Speed" in col or "Wind Direction" in col:
            if re.search(
                "at", col
            ):  # check at in column name so that I can extract the height
                h_match = re.search(r"at \d+m", col)
                if h_match:
                    height = float(
                        h_match.group().replace("at ", "").replace("m", "")
                    )  # extract height from column name
                    height_all.append(height)
    height_lidar = np.sort(np.unique(height_all))

    return height_lidar


def NA_cols(df: pd.DataFrame) -> list:
    """
    This function take a dataframe as input
    and returns a list of columns with missing values

    Parameters
    ----------
    df : pd.DataFrame
        the Dataframe to check for missing values

    Returns
    -------
    missing_columns : list
        List of column names with missing values

    """
    missing_columns = df.isna().any()
    missing_columns = missing_columns[missing_columns].index.tolist()

    return missing_columns


# if __name__ == "__main__":
#     lidar_test = Path(".", "tests", "lidar_data_10min")

#     lidar_CSV_files = find_KNMI_LiDAR_files(lidar_test)

#     print(lidar_CSV_files)

#     data_lidar = read_KNMI_LiDAR(lidar_CSV_files[0])
#     heights = lidar_height(data_lidar)

#     print(heights)
