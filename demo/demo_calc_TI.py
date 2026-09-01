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
from wind_data_analysis.process.calc_turb import calc_ti


# This is a demo script to show how to calculate the turbulence intensity (TI) from the LiDAR data and plot the TI distribution per height.

lidar_csv_files = find_KNMI_LiDAR_files(
    Path(".", "tests", "lidar_data"),
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
lidar_std_all = concatenate_wind_stats([item.std for item in per_file_stats])

print(lidar_avg_all.head())
print(lidar_std_all.head())

ti_values = calc_ti(lidar_avg_all, lidar_std_all, hub_height=120.0)

plt.figure(figsize=(7, 5))
sns.boxplot(x="height", y="ti", data=ti_values.ti_raw)
plt.xlabel("Height [m]")
plt.ylabel("TI [-]")
plt.title("TI Distribution per Height (Mast)")
plt.grid(True)
plt.show(block=True)

print("TI is calculated and plotted successfully.")
