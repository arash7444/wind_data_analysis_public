import pandas as pd
import xarray as xr
import numpy as np

import re


def bin_wdir(df):
    """
    This function bins the wind direction data into specified bins and counts the number of observations in each bin.
    To simplify the process, the range of wind directions is hardcoded to 360 degrees, and the bins are defined in 10 degree intervals (0-10, 10-20, ..., 350-360).
    It can be changed to be more flexible in the future if needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the wind direction data with a column named "wind_direction" for the hub height wind direction.

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame with additional columns for wind direction bins and their counts.

    bin_counts : pd.Series
        Series containing the counts of observations in each wind direction bin.


    """
    # ==== Now Compute binned TI by wind speed ---
    bins = np.arange(0, 360.0, 10)
    labels = [
        f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)
    ]  # e.g., 0-10, 10-20, etc. so i can add them as columns easily
    df["wdir_bin"] = pd.cut(
        df["wind_direction"], bins=bins, labels=labels, include_lowest=True
    )

    bin_counts = df.groupby("wdir_bin", observed=True).size() / len(
        df["height"].unique()
    )  # Divide by num heights since data is repeated per height

    bin_counts = bin_counts.astype(int)  # Since counts should be integers

    return df, bin_counts
