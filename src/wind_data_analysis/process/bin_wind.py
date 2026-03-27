import pandas as pd
import xarray as xr
import numpy as np

import re


def bin_wind(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function bins the wind speed data into specified bins and counts the number of observations in each bin.
    To simplify the process, the range of wind speeds is hardcoded to 30 m/s, and the bins are defined in 1 m/s intervals (0-1, 1-2, ..., 29-30).
    It can be changed to be more flexible in the future if needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the wind speed data with a column named "hub_wsp" for the hub height wind speed.

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame with additional columns for w
    """

    # ==== Now Compute binned TI by wind speed ---
    bins = np.arange(0, 31.0, 1)  # range is hardcoded to 30 m/s
    labels = [
        f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)
    ]  # e.g., 0-1, 1-2, 2-3, etc. so i can add them as columns easily

    df["wsp_bin"] = pd.cut(df["hub_wsp"], bins=bins, labels=labels, include_lowest=True)

    # count the number of observations per bin

    bin_counts = df.groupby("wsp_bin", observed=True).size() / len(
        df["height"].unique()
    )  # Divide by num heights since data is repeated per height

    bin_counts = bin_counts.astype(int)  # Since counts should be integers
    df["wsp_bincount"] = df["wsp_bin"].map(
        bin_counts
    )  # assign each bin count to corresponding wind speed bin

    return df, bin_counts
