from pathlib import Path

from wind_data_analysis.data_reader import find_KNMI_LiDAR_files, read_KNMI_LiDAR

from wind_data_analysis.utils import lidar_height

from wind_data_analysis.process import (
    concatenate_wind_stats,
    wind_height_profile,
)

from wind_data_analysis.process import (
    compute_lidar_stats,
    wind_height_profile,
)

from matplotlib import pyplot as plt
import warnings
import pandas as pd
import numpy as np


lidar_test_high = Path(".", "tests", "lidar_data")
lidar_test_low = Path(".", "tests", "lidar_data_10min")

lidar_CSV_files_high = find_KNMI_LiDAR_files(lidar_test_high)

lidar_CSV_files_low = find_KNMI_LiDAR_files(lidar_test_low)

lidar_file_high = Path(
    ".", "tests", "lidar_data", "ZephIR_Cabauw_ZP738_raw_20200501_v1.CSV"
)
lidar_file_low = Path(
    ".", "tests", "lidar_data_10min", "ZephIR_Cabauw_ZP738_10min_20200501_v1.CSV"
)

if len(lidar_CSV_files_high) > 0:
    per_file_stats = []  # empty list to store the statistics for each file to concatenate them later
    heights_all = []  # empty list to store the heights for each file to concatenate them later

    for file_name in lidar_CSV_files_high:
        # read the LiDAR data
        data_lidar = read_KNMI_LiDAR(lidar_file_high)

        # compute the statistics for the LiDAR data
        lidar_stats_high = compute_lidar_stats(data_lidar)

        # extract the heights from the column names and save them in a list
        heights_highres = lidar_height(data_lidar)

        print(lidar_stats_high)

        # save the statistics and heights for each file in a list to concatenate them later
        per_file_stats.append(lidar_stats_high)
        heights_all.append(np.asarray(heights_highres, dtype=float))

    # concatenate the statistics from all files into a single dataframe for each statistic type (avg, max, min, std) and sort them by time index
    lidar_avg_all = concatenate_wind_stats([item.avg for item in per_file_stats])
    lidar_max_all = concatenate_wind_stats([item.max for item in per_file_stats])
    lidar_min_all = concatenate_wind_stats([item.min for item in per_file_stats])
    lidar_std_all = concatenate_wind_stats([item.std for item in per_file_stats])

    height_lidar_all = np.unique(np.concatenate(heights_all))

    print(lidar_avg_all)

    wsp_profiles = wind_height_profile(lidar_avg_all, height_lidar_all)

if len(lidar_CSV_files_low) > 0:
    per_file_stats = []  # empty list to store the statistics for each file to concatenate them later
    heights_all = []  # empty list to store the heights for each file to concatenate them later

    for file_name in lidar_CSV_files_low:
        # read the LiDAR data for low resolution
        data_lidar = read_KNMI_LiDAR(lidar_file_low)

        # compute the statistics for the LiDAR data
        lidar_stats_low = compute_lidar_stats(data_lidar)

        # extract the heights from the column names and save them in a list
        heights_lowres = lidar_height(data_lidar)

        print(lidar_stats_low)

        per_file_stats.append(lidar_stats_low)
        heights_all.append(np.asarray(heights_lowres, dtype=float))

    # concatenate the statistics from all files into a single dataframe for each statistic type (avg, max, min, std) and sort them by time index
    lidar_avg_all = concatenate_wind_stats([item.avg for item in per_file_stats])
    lidar_max_all = concatenate_wind_stats([item.max for item in per_file_stats])
    lidar_min_all = concatenate_wind_stats([item.min for item in per_file_stats])
    lidar_std_all = concatenate_wind_stats([item.std for item in per_file_stats])

    height_lidar_all = np.unique(np.concatenate(heights_all))

    print(lidar_avg_all)

    wsp_profiles = wind_height_profile(lidar_avg_all, height_lidar_all)


# %% Plotting the results
# plot them:
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(
    lidar_stats_high.avg.index,
    lidar_stats_high.avg["Horizontal Wind Speed (m/s) at 299m"],
    label="Average Wind Speed at 299m",
    marker="o",
    color="blue",
)

plt.plot(
    lidar_stats_low.avg.index,
    lidar_stats_low.avg["Horizontal Wind Speed (m/s) at 299m"],
    label="Average Wind Speed at 299m (10min)",
    marker="x",
    color="orange",
)
# plt.xlabel("Time")
plt.ylabel("Wind Speed (m/s)")

plt.legend()
plt.grid()

plt.title(
    "Statistics of Wind Speed at 299m from High-Resolution and Low-Resolution LiDAR Data"
)


# plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 2)
plt.plot(
    lidar_stats_high.max.index,
    lidar_stats_high.max["Horizontal Wind Speed (m/s) at 299m"],
    label="Maximum Wind Speed at 299m",
    marker="o",
    color="blue",
)

plt.plot(
    lidar_stats_low.max.index,
    lidar_stats_low.max["Horizontal Wind Speed (m/s) at 299m"],
    label="Maximum Wind Speed at 299m (10min)",
    marker="x",
    color="orange",
)
# plt.xlabel("Time")
plt.ylabel("Wind Speed (m/s)")
plt.legend()
plt.grid()

# plt.title(
#     "Maximum Wind Speed at 299m from High-Resolution and Low-Resolution LiDAR Data"
# )

# plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 3)
plt.plot(
    lidar_stats_high.std.index,
    lidar_stats_high.std["Horizontal Wind Speed (m/s) at 299m"],
    label="Standard Deviation Wind Speed at 299m",
    marker="o",
    color="blue",
)
plt.plot(
    lidar_stats_low.std.index,
    lidar_stats_low.std["Horizontal Wind Speed (m/s) at 299m"],
    label="Standard Deviation Wind Speed at 299m (10min)",
    marker="x",
    color="orange",
)
plt.xlabel("Time")
plt.ylabel("Wind Speed (m/s)")
plt.legend()
# plt.title(
#     "Standard Deviation Wind Speed at 299m from High-Resolution and Low-Resolution LiDAR Data"
# )
plt.grid()
plt.tight_layout()
plt.show()
