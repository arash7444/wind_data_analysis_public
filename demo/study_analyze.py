from pathlib import Path
import matplotlib.pyplot as plt

from wind_data_analysis.data_reader import read_KNMI_LiDAR


from wind_data_analysis.process.stats_func import compute_lidar_stats

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import xarray as xr
import re


from wind_data_analysis.plotting import (
    plot_ti_polar_by_height,
    plot_wind_statistics,
)
from wind_data_analysis.process import bin_wdir, bin_wind


from wind_data_analysis.data_reader import (
    find_KNMI_LiDAR_files,
    read_KNMI_LiDAR,
    met_finder,
    read_met,
)

from wind_data_analysis.utils import lidar_height

from wind_data_analysis.process import (
    concatenate_wind_stats,
    wind_height_profile,
    compute_lidar_stats,
)



from rich.console import Console
from rich.markdown import Markdown
from rich.traceback import install
install()
console = Console()



if __name__ == "__main__":

    # console.print("lidar")

    lidar_file = r".\tests\lidar_data\ZephIR_Cabauw_ZP738_raw_20200607_v1.CSV"
    lidar_data = read_KNMI_LiDAR(lidar_file)
    # console.print(lidar_data.describe())
    
    # wind_col = [col for col in lidar_data.columns if "Horizontal Wind Speed" in col]
    # console.print(wind_col)

    # wdir_col = [col for col in lidar_data.columns if "Wind Direction (deg)" in col]
    # console.print(wdir_col)

    # wsp_lidar = lidar_data[wind_col]
    # wdir_lidar = lidar_data[wdir_col]

    



    # console.print("met mast")

    mast_file = r".\tests\metmast_data\cesar_tower_meteo_lb1_t10_v1.2_202006.nc"
    # mast_data = read_met(mast_file)
    mast_data = read_met(mast_file, start_date="2020-06-01", end_date="2020-06-30")
    # console.print(mast_data.describe())





    #--------------------------------------------------
    # 1. Basic structure
    # --------------------------------------------------
    console.print(Markdown("## Dataset structure"))

    console.print(f"Shape: {lidar_data.shape}")
    console.print(f"Number of rows: {len(lidar_data)}")
    console.print(f"Number of columns: {lidar_data.shape[1]}")

    console.print(f"First timestamp: {lidar_data.index.min()}")
    console.print(f"Last timestamp:  {lidar_data.index.max()}")

    console.print(
        f"Datetime index: "
        f"{isinstance(lidar_data.index, pd.DatetimeIndex)}"
    )



    # --------------------------------------------------
    # 2. Determine the sampling interval
    # --------------------------------------------------
    time_difference = lidar_data.index.to_series().diff().dropna()

    console.print(Markdown("## Most common sampling intervals"))
    console.print(time_difference.value_counts().head(10))

    console.print(
        f"Timestamps are sorted: "
        f"{lidar_data.index.is_monotonic_increasing}"
    )

    console.print(
        f"Duplicate timestamps: "
        f"{lidar_data.index.duplicated().sum()}"
    )

# --------------------------------------------------
    # 3. Identify important columns
    # --------------------------------------------------
    wind_speed_columns = [
        column
        for column in lidar_data.columns
        if column.startswith("Horizontal Wind Speed (m/s) at")
    ]

    wind_direction_columns = [
        column
        for column in lidar_data.columns
        if column.startswith("Wind Direction (deg) at")
    ]

    console.print(Markdown("## Wind-speed columns"))
    console.print(wind_speed_columns)

    console.print(Markdown("## Wind-direction columns"))
    console.print(wind_direction_columns)

    # --------------------------------------------------
    # 4. Inspect a few actual observations
    # --------------------------------------------------
    selected_columns = wind_speed_columns + wind_direction_columns

    console.print(Markdown("## First three wind observations"))

    # Transpose makes many columns easier to inspect.
    console.print(lidar_data[selected_columns].head(3).T)

    # --------------------------------------------------
    # 5. Check the data types
    # --------------------------------------------------
    console.print(Markdown("## Data types"))

    console.print(
        lidar_data[selected_columns]
        .dtypes
        .value_counts()
    )
    print("done")




    # ============================================================
    # Step 2: Data-quality checks
    # ============================================================

    lidar_wsp_column = "Horizontal Wind Speed (m/s) at 139m"
    lidar_wdir_column = "Wind Direction (deg) at 139m"

    lidar_selected = lidar_data[
        [lidar_wsp_column, lidar_wdir_column, "Status Flags"]
    ].copy()


    console.print(Markdown("## Lidar data quality"))

    console.print(f"Number of rows: {len(lidar_selected)}")
    console.print(f"Duplicate timestamps: {lidar_selected.index.duplicated().sum()}")
    console.print(f"Timestamps are sorted: {lidar_selected.index.is_monotonic_increasing}")

    console.print("\nMissing values:")
    console.print(lidar_selected.isna().sum())

    console.print("\nStatus flags:")
    console.print(lidar_selected["Status Flags"].value_counts(dropna=False))

    console.print("\nWind-speed summary:")
    console.print(lidar_selected[lidar_wsp_column].describe())

    console.print("\nWind-direction summary:")
    console.print(lidar_selected[lidar_wdir_column].describe())




    negative_wind_speed = lidar_selected[
        lidar_selected[lidar_wsp_column] < 0
    ]

    invalid_wind_direction = lidar_selected[
        ~lidar_selected[lidar_wdir_column].between(0, 360)
    ]

    console.print(f"\nNegative wind speeds: {len(negative_wind_speed)}")
    console.print(f"Wind directions outside 0–360°: {len(invalid_wind_direction)}")


    ## Check the Gap ###
    time_difference = lidar_selected.index.to_series().diff()

    console.print("\nMost common time intervals:")
    console.print(time_difference.value_counts().head(10))

    console.print("\nIntervals longer than 30 seconds:")
    console.print(time_difference[time_difference > pd.Timedelta(seconds=30)].describe())


    height_lidar = lidar_height(lidar_data)
    console.print("\nLidar heights:")
    console.print(height_lidar)

    

    # lidar_files = find_KNMI_LiDAR_files(
    #     file_folder=r"d:\Projects\002_WindAnalyzer\KNMI_Data\ZP738_1s", start_date="2020-06-01", end_date="2020-06-30" 
    # )

    # console.print("lidar files:")
    # console.print(lidar_files)

    # for file in lidar_files:
    #     data = read_KNMI_LiDAR(file)
    #     console.print(data.head())
    #     stats = compute_lidar_stats(data)
    #     console.print(stats)




#------------------ Metmast check

mast_data = read_met(
    mast_file,
    start_date="2020-06-07",
    end_date="2020-06-08",
)

mast_140m = mast_data[mast_data["height"] == 140].copy()

console.print(Markdown("## Met-mast data quality at 140 m"))

console.print(f"Number of rows: {len(mast_140m)}")
console.print(f"Duplicate timestamps: {mast_140m.index.duplicated().sum()}")
console.print(f"Timestamps are sorted: {mast_140m.index.is_monotonic_increasing}")

console.print("\nMissing values:")
console.print(mast_140m.isna().sum())

console.print("\nWind-speed summary:")
console.print(mast_140m["wind_speed"].describe())

console.print("\nMost common time intervals:")
console.print(
    mast_140m.index
    .to_series()
    .diff()
    .value_counts()
    .head()
)



expected_mast_rows = 24 * 6
actual_mast_rows = len(mast_140m)

completeness = actual_mast_rows / expected_mast_rows

console.print(f"Expected rows: {expected_mast_rows}")
console.print(f"Actual rows: {actual_mast_rows}")
console.print(f"Completeness: {completeness:.1%}")


#-----------------------------------------
# 2B: Visualize the raw lidar measurements
#-------------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=lidar_selected.index,
        y=lidar_selected[lidar_wsp_column],
        mode="lines",
        name="Lidar wind speed at 139 m",
    )
)

flagged_lidar = lidar_selected[
    lidar_selected["Status Flags"] != "Fully Operational"
]

fig.add_trace(
    go.Scatter(
        x=flagged_lidar.index,
        y=flagged_lidar[lidar_wsp_column],
        mode="markers",
        name="Flagged records",
        marker=dict(color="red", size=5),
    )
)

fig.update_layout(
    title="Raw lidar wind speed at 139 m",
    xaxis_title="Time (UTC)",
    yaxis_title="Wind speed (m/s)",
    template="plotly_white",
)

fig.show()



#---Distribution of raw lidar wind speed#----
fig = go.Figure()

fig.add_trace(
    go.Histogram(
        x=lidar_selected[lidar_wsp_column],
        nbinsx=40,
        name="Lidar 139 m",
    )
)

fig.update_layout(
    title="Distribution of raw lidar wind speed at 139 m",
    xaxis_title="Wind speed (m/s)",
    yaxis_title="Count",
    template="plotly_white",
)

fig.show()

# import matplotlib.pyplot as plt

# plt.hist(lidar_selected[lidar_wsp_column], bins=20)
# plt.title("Distribution of raw lidar wind speed at 139 m")
# plt.xlabel("Wind speed (m/s)")
# plt.ylabel("Count")
# plt.show()

fig = go.Figure()

fig.add_trace(
    go.Box(
        x=lidar_selected[lidar_wsp_column],
        name="Lidar 139 m",
        boxpoints="outliers",
    )
)#     sns.boxplot(x="height", y="ti", data=ti_values.ti_raw)

fig.update_layout(
    title="Raw lidar wind speed at 139 m",
    xaxis_title="Wind speed (m/s)",
    template="plotly_white",
)

fig.show()




#------------------------
# # Convert lidar to 10-minute observations
#---------------------

lidar_stats = compute_lidar_stats(lidar_data)

lidar_10min = (
    lidar_stats.avg[[lidar_wsp_column]]
    .rename(columns={lidar_wsp_column: "lidar_139m"})
)



# mast_140m_wsp = (
#     mast_data[["wind_speed"]]
#     .rename(columns={"wind_speed": "mast_140m"})
# )
# # mast_140m = mast_140m_wsp.index.round('s')
# # lidar_10min = lidar_10min.index.round('s')
# mast_140m.index = mast_140m.index.round("10min") # because they have some milisecond error

# common_idx = mast_140m_wsp.index.intersection(lidar_10min.index)
# mast_align = mast_140m_wsp.loc[common_idx]
# lidar_align = lidar_10min.loc[common_idx]

# comparison = pd.concat(
#     [lidar_align, mast_align],
#     axis=1,
# ).dropna()

comparison = pd.concat(
    [lidar_10min, mast_140m],
    axis=1,
    join="inner",
).dropna()
comparison = comparison.rename(columns={"wind_speed": "wsp_mast_140m"})


console.print(Markdown("## Matched 10-minute observations"))
console.print(comparison.head())
console.print(comparison.describe())
console.print(f"Number of matched periods: {len(comparison)}")



fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=comparison.index,
        y=comparison["lidar_139m"],
        mode="lines",
        name="Lidar 139 m",
    )
)

fig.add_trace(
    go.Scatter(
        x=comparison.index,
        y=comparison["wsp_mast_140m"],
        mode="markers",
        name="Met mast 140 m",
    )
)

fig.update_layout(
    title="10-minute wind speed comparison",
    xaxis_title="Time (UTC)",
    yaxis_title="Wind speed (m/s)",
    template="plotly_white",
)

fig.show()




#Step 3: Statistical question

diff = comparison["lidar_139m"] - comparison["wsp_mast_140m"]
plt.plot(comparison.index,diff)