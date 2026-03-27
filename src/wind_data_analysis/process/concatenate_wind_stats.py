import pandas as pd
import numpy as np


def concatenate_wind_stats(stat_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate a list of dataframes and sort by time index.

    Parameters
    ----------
    stat_list : list[pd.DataFrame]
        A list of dataframes containing wind statistics (e.g., average, max, min, std) for different LiDAR files.
    Returns
    -------
    pd.DataFrame
        A single dataframe containing the concatenated and sorted wind statistics.

    Example
    -------
    lidar_avg_all = concatenate_wind_stats(avg_list)

    """
    if not stat_list:
        return pd.DataFrame()
    return pd.concat(stat_list).sort_index()
