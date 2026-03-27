import numpy as np
import pandas as pd


def wind_height_profile(lidar_data: pd.DataFrame, height_lidar: list) -> pd.DataFrame:
    """
    Build a Dataframe of wind speed values where the index is time and the columns are the heights. The values are the wind speed at that height and time.

    Parameters
    ----------
    lidar_data: pd.DataFrame
        The raw lidar data as a DataFrame.
        height_lidar: list
        A list of heights corresponding to the lidar measurements.

    Returns
    -------
    wsp_profiles: pd.DataFrame
        A DataFrame where the index is time and the columns are the heights. The values are the wind speed at that height and time.


    example
    -------
    wsp_profiles = _build_lidar_profiles(lidar_data, height_lidar)
    """
    # Columns that contain wind speed
    speed_cols = [c for c in lidar_data.columns if "Wind Speed" in c]

    mapping = {}
    for h in np.unique(np.asarray(height_lidar, float)):
        pattern = f"Horizontal Wind Speed (m/s) at {int(h)}m"
        cols_h = [c for c in speed_cols if pattern in c]
        if not cols_h:
            continue
        if len(cols_h) == 1:
            series = lidar_data[cols_h[0]]
        else:
            # BUGG !!!!! this is a bug because it detects  vertical and horizontal wind speeds and you cannot average them !!!!!!!
            # If there are multiple columns for the same height, average them
            series = lidar_data[cols_h].mean(axis=1)
        mapping[h] = series

    wsp_profiles = pd.DataFrame(mapping)
    wsp_profiles = wsp_profiles.sort_index()
    wsp_profiles = wsp_profiles.reindex(sorted(wsp_profiles.columns), axis=1)
    wsp_profiles.columns = wsp_profiles.columns.astype(float)

    return wsp_profiles
