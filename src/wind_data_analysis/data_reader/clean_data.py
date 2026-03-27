import pandas as pd
import numpy as np

from wind_data_analysis.utils import lidar_height


def clean_data(data_in: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by removing invalid values and outliers.

    Parameters:
    - data: DataFrame containing the raw  data.

    Returns:
    - data_clean: DataFrame containing the cleaned LiDAR data.
    """

    ws_cols = [col for col in data_in.columns if "Wind Speed" in col]
    data_2 = data_in.copy()

    # Convert wind speed columns to numeric, coercing errors to NaN
    data_2[ws_cols] = data_2[ws_cols].apply(pd.to_numeric, errors="coerce")

    valid_mask = (data_2[ws_cols] >= 0) & (data_2[ws_cols] <= 50)

    # bad values become NaN, good values stay
    data_2[ws_cols] = data_2[ws_cols].where(valid_mask)

    data_clean = data_2.copy()

    return data_clean
