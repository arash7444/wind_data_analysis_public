import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

from wind_data_analysis.data_reader import (
    find_KNMI_LiDAR_files,
    read_KNMI_LiDAR,
)

from wind_data_analysis.utils import lidar_height

from wind_data_analysis.process import (
    concatenate_wind_stats,
    wind_height_profile,
    compute_lidar_stats,
)
from wind_data_analysis.process.bin_wind import bin_wind
from wind_data_analysis.process.bin_wdir import bin_wdir

from dataclasses import dataclass


@dataclass
class TurbValues:
    """
    Container for TI values.

    """

    ti_raw: pd.DataFrame  # raw TI values for each time and height
    ti_median: pd.DataFrame  # median TI profile across heights
    ti_median_binned: (
        pd.DataFrame
    )  # median TI profile across heights and wind speed bins


def calc_ti(avg_val, std_val, hub_height=120.0) -> TurbValues:

    wind_cols = [
        cols for cols in avg_val.columns if "Horizontal Wind Speed (m/s) at" in cols
    ]

    wdir_cols = [cols for cols in avg_val.columns if "Wind Direction (deg) at" in cols]

    data_avg = avg_val[wind_cols].copy()
    data_std = std_val[wind_cols].copy()

    lidar_ti = pd.DataFrame(index=data_avg.index, columns=data_avg.columns)

    lidar_wsp = pd.DataFrame(index=data_avg.index, columns=data_avg.columns)

    lidar_wdir = pd.DataFrame(index=data_avg.index, columns=data_avg.columns)

    lidar_ti = data_std.div(data_avg).where(data_avg != 0, np.nan)
    lidar_wsp = data_avg[wind_cols].copy()
    lidar_wdir = avg_val[wdir_cols].copy()

    heights = lidar_height(data_avg)

    lidar_ti.columns = [
        col.replace("Horizontal Wind Speed (m/s)", "TI") for col in lidar_ti.columns
    ]

    if lidar_ti.isna().any().any():
        print("Warning: Some TI values are NaN, check what is the reason")

    # remove unrealistic/invalid values
    lidar_ti = lidar_ti.mask(
        (data_avg.reindex(columns=lidar_ti.columns) <= 0)
        | (data_avg.reindex(columns=lidar_ti.columns) >= 999)
    )

    # reset index to have Time as a column for melting
    lidar_ti = lidar_ti.reset_index()

    # melt the DataFrame to have a long format suitable for grouping by height
    lidar_ti_tidy = lidar_ti.melt(id_vars="Time", var_name="col", value_name="ti")

    # extract height from column names and create a new column for it
    lidar_ti_tidy["height"] = (
        lidar_ti_tidy["col"].str.extract(r"at (\d+)m").astype(float)
    )

    # Now let's also extract wind speed and wind direction values and merge them to "lidar_ti_tidy"

    # for Wind Speed:
    lidar_wsp = lidar_wsp.reset_index()
    lidar_wsp_tidy = lidar_wsp.melt(
        id_vars="Time", var_name="col_wsp", value_name="wind_speed"
    )
    lidar_wsp_tidy["height"] = (
        lidar_wsp_tidy["col_wsp"].str.extract(r"at (\d+)m").astype(float)
    )

    lidar_ti_tidy = pd.merge(
        lidar_ti_tidy, lidar_wsp_tidy, on=["Time", "height"], how="left"
    )  # merge lidar_ti_tidy and lidar_wsp_tidy

    # for Wind Direction:
    lidar_wdir = lidar_wdir.reset_index()
    lidar_wdir_tidy = lidar_wdir.melt(
        id_vars="Time", var_name="col_dir", value_name="wind_direction"
    )
    lidar_wdir_tidy["height"] = (
        lidar_wdir_tidy["col_dir"].str.extract(r"at (\d+)m").astype(float)
    )

    lidar_ti_tidy.drop(
        columns=["col", "col_wsp"], inplace=True
    )  # drop the columns we don't need anymore
    lidar_wdir_tidy.drop(
        columns=["col_dir"], inplace=True
    )  # drop the columns we don't need anymore

    lidar_ti_tidy = pd.merge(
        lidar_ti_tidy, lidar_wdir_tidy, on=["Time", "height"], how="left"
    )  # merge lidar_ti_tidy and lidar_wdir_tidy

    # find the closest height to the hub height and use it as reference for wind speed binning
    if hub_height not in lidar_ti_tidy["height"].unique():
        hub_height = min(
            lidar_ti_tidy["height"].unique(), key=lambda x: abs(x - hub_height)
        )
    print(
        f"Using closest LiDAR height {hub_height}m as reference for wind speed binning."
    )

    LiDAR_ref = lidar_ti_tidy[lidar_ti_tidy["height"] == hub_height][
        ["Time", "wind_speed"]
    ].rename(columns={"wind_speed": "hub_wsp"})

    lidar_ti_tidy = lidar_ti_tidy.merge(LiDAR_ref, on="Time", how="left")

    # include wind speed bin and number of samples in each bin as new columns in lidar_ti_tidy
    lidar_ti_tidy, bins_counts = bin_wind(lidar_ti_tidy)

    # include wind direction bin and number of samples in each bin as new columns in lidar_ti_tidy
    lidar_ti_tidy, bins_counts_wdir = bin_wdir(lidar_ti_tidy)

    # clean up lidar_ti_tidy
    lidar_ti_tidy = lidar_ti_tidy[lidar_ti_tidy["ti"] < 1]

    # group by height and calculate the median TI for each height
    ti_profile_lidar = lidar_ti_tidy.groupby("height", as_index=False)["ti"].median()

    # group by height and wind speed bin and calculate the median TI for each height and wind speed bin
    ti_profile_lidar_binned = lidar_ti_tidy.groupby(
        ["height", "wsp_bin"], as_index=False, observed=True
    )["ti"].median()

    return TurbValues(
        ti_raw=lidar_ti_tidy,  # TI values for each time and height with wind speed and direction information
        ti_median=ti_profile_lidar,  # median TI profile across heights
        ti_median_binned=ti_profile_lidar_binned,  # median TI profile across heights and wind speed bins
    )


# if __name__ == "__main__":
#     lidar_csv_files = find_KNMI_LiDAR_files(
#         Path(".", "tests", "lidar_data"),
#         start_date="2020-05-01",
#         end_date="2020-05-03",
#     )

#     per_file_stats = []  # empty list to store the statistics for each file to concatenate them later
#     heights_all = []  # empty list to store the heights for each file to concatenate them later

#     for file_name in lidar_csv_files:
#         # read the LiDAR data
#         data_lidar = read_KNMI_LiDAR(file_name)

#         # compute the statistics for the LiDAR data
#         lidar_stats = compute_lidar_stats(data_lidar)

#         # extract the heights from the column names and save them in a list
#         heights = lidar_height(data_lidar)

#         # save the statistics and heights for each file in a list to concatenate them later
#         per_file_stats.append(lidar_stats)
#         heights_all.append(np.asarray(heights, dtype=float))

#     # concatenate the statistics from all files into a single dataframe for each statistic type (avg, max, min, std) and sort them by time index
#     lidar_avg_all = concatenate_wind_stats([item.avg for item in per_file_stats])
#     lidar_std_all = concatenate_wind_stats([item.std for item in per_file_stats])

#     print(lidar_avg_all.head())
#     print(lidar_std_all.head())

#     ti_values = calc_ti(lidar_avg_all, lidar_std_all, hub_height=120.0)

#     plt.figure(figsize=(7, 5))
#     sns.boxplot(x="height", y="ti", data=ti_values.ti_raw)
#     plt.xlabel("Height [m]")
#     plt.ylabel("TI [-]")
#     plt.title("TI Distribution per Height (Mast)")
#     plt.grid(True)
#     plt.show()

#     print("TI is calculated and plotted successfully.")
