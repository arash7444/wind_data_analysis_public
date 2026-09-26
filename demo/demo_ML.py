import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             r2_score)    

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

from wind_data_analysis.ML import (
    prepare_ti_regression_data,
    chronological_split,
    create_ml_inputs,
    train_baseline_model,
    train_knn_model,
)


from wind_data_analysis.ML.ti_data import (
    prepare_ti_regression_data,
    chronological_split,
    create_ml_inputs,
)

from wind_data_analysis.ML.ti_regression import (
    train_baseline_model,
    train_knn_model,
)

from wind_data_analysis.ML.evaluation import (
    calculate_regression_metrics,
)




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

ml_data, selected_height = prepare_ti_regression_data(
    ti_values.ti_raw,
    target_height=120.0,
)

print("Selected height:", selected_height)
print(ml_data.head())
print("Dataset shape:", ml_data.shape)
print(ml_data.isna().sum())


# split the data into training and testing sets chronologically

from sklearn.model_selection import train_test_split

# Split the complete DataFrame so the timestamps remain available.
# shuffle=False preserves chronological order:
# earlier observations go into training, and later observations go into testing.
train_data, test_data = train_test_split(
    ml_data,
    test_size=0.30,
    shuffle=False,
)

# this is manual split, but I prefer to use sklearn's train_test_split with shuffle=False
train_data_2, test_data_2 = chronological_split(
    ml_data,
    test_fraction=0.30,
)


print("Training shape:", train_data.shape)
print("Test shape:", test_data.shape)

print("Training period:")
print(train_data["Time"].min(), "to", train_data["Time"].max())

print("Test period:")
print(test_data["Time"].min(), "to", test_data["Time"].max())

print(
    "Chronological separation:",
    train_data["Time"].max() < test_data["Time"].min(),
)


X_train, X_test, y_train, y_test = create_ml_inputs(
    train_data,
    test_data,
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("Training features:")
print(X_train.head())

print("Training target:")
print(y_train.head())





baseline_model, baseline_predictions = train_baseline_model(
    X_train,
    y_train,
    X_test,
)

print("Mean training TI:", y_train.mean())
print("Baseline model value:", baseline_model.constant_[0])
print("Number of predictions:", len(baseline_predictions))
print("First five predictions:", baseline_predictions[:5])
print("Unique prediction values:", np.unique(baseline_predictions))



baseline_metrics = calculate_regression_metrics(
    y_test,
    baseline_predictions,
)


print("Baseline metrics")
print(f"MAE:  {baseline_metrics.mae:.6f}")
print(f"RMSE: {baseline_metrics.rmse:.6f}") 
print(f"R²:   {baseline_metrics.r2:.6f}")   

# plotting the results

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Measured TI", alpha=0.7)
plt.plot(
    y_test.index, 
    baseline_predictions, 
    label="Baseline TI", 
    alpha=0.7
)

plt.xlabel("Time")
plt.ylabel("TI")
plt.title("TI Prediction - Baseline Model")
plt.legend()
plt.grid(True)

plt.savefig("baseline_model_ti_prediction.png", dpi=300, bbox_inches="tight")






knn_model, knn_predictions = train_knn_model(
    X_train,
    y_train,
    X_test,
    n_neighbors=5,
)

knn_metrics = calculate_regression_metrics(
    y_test,
    knn_predictions,
)

print("Baseline metrics")
print(f"MAE:  {baseline_metrics.mae:.6f}")
print(f"RMSE: {baseline_metrics.rmse:.6f}")
print(f"R²:   {baseline_metrics.r2:.6f}")

print("\nKNN metrics")
print(f"MAE:  {knn_metrics.mae:.6f}")
print(f"RMSE: {knn_metrics.rmse:.6f}")
print(f"R²:   {knn_metrics.r2:.6f}")

print("\nFirst five KNN predictions:")
print(knn_predictions[:5])

print("-------------------------")

