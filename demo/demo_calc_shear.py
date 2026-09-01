import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from wind_data_analysis.data_reader import (
    find_KNMI_LiDAR_files,
    read_KNMI_LiDAR,
)
from wind_data_analysis.utils import lidar_height
from wind_data_analysis.process import (
    concatenate_wind_stats,
    wind_height_profile,
    compute_lidar_stats,
    calc_shear,
)


# This demo script calculates the power-law shear exponent alpha from LiDAR wind speed profiles and plots the results.

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

# Build a Dataframe of wind speed values where the index is time and the columns are the heights. The values are the wind speed at that height and time.
wsp_profiles = wind_height_profile(lidar_avg_all, height_lidar_all)

# Calculate the power-law shear exponent alpha and its uncertainty from the wind speed profiles.
ShearValues = calc_shear(wsp_profiles, window=6)

# ------- plot shear
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
# 1) raw alpha + error bars
ax = axes[0]
ax.plot(
    ShearValues.alpha.index,
    ShearValues.alpha.values,
    "o",
    label="shear slope for LiDAR data",
)
ax.set_ylabel("Shear slope [-]")
ax.legend()
ax.grid(True)

# 2) rolling median
ax = axes[1]
ax.plot(
    ShearValues.alpha_roll_med.index,
    ShearValues.alpha_roll_med.values,
    "o-",
    label="LiDAR  - rolling median",
)
ax.set_ylabel("Shear slope [-]")
ax.legend()
ax.grid(True)

# 3) rolling mean
ax = axes[2]
ax.plot(
    ShearValues.alpha_roll_mean.index,
    ShearValues.alpha_roll_mean.values,
    "o-",
    label="LiDAR  - rolling mean",
)
ax.set_ylabel("Shear slope [-]")
ax.set_xlabel("Time [hour]")
ax.set_title("Shear slope vs hour")
ax.legend()
ax.grid(True)

fig.tight_layout()
plt.show(block=True)
