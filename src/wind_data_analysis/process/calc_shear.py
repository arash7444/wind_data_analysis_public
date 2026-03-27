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
)

from dataclasses import dataclass


@dataclass
class ShearValues:
    """
    Container for shear values and their uncertainties.

    """

    alpha: pd.DataFrame  # power-law shear exponent alpha
    alpha_err: pd.DataFrame  # standard error of alpha
    alpha_roll_med: pd.DataFrame  # rolling median of alpha
    alpha_roll_mean: pd.DataFrame  # rolling mean of alpha


def fit_alpha_with_uncertainty(
    heights_m: list | np.ndarray, wsp: list | np.ndarray
) -> tuple[float, float, int]:
    """
    Fit power-law shear exponent alpha from log(U) vs log(z)
    and estimate standard error of alpha from regression residuals.

    Parameters
    ----------
    heights_m : list | np.ndarray
        Heights in meters corresponding to the wind speed measurements.
    wsp : list | np.ndarray
        Wind speed measurements at the corresponding heights.


    Returns
    -------
    alpha : float
        Estimated power-law shear exponent. NaN if fit is not possible.
    se_alpha : float
        Standard error of the estimated alpha. NaN if not enough data points.
    n : int
        Number of valid data points used in the fit.

    examples
    --------
    >>> heights = [10, 20, 40, 80]
    >>> wsp = [5, 7, 10, 14]
    >>> alpha, se_alpha, n = _fit_alpha_with_uncertainty(heights, wsp)
    >>> print(f"Estimated alpha: {alpha:.3f}, Standard error: {se_alpha:.3f}, Data points: {n}")
    Estimated alpha: 0.585, Standard error: 0.015, Data points: 4

    """
    heights_m = np.asarray(heights_m, float)
    wsp = np.asarray(wsp, float)

    mask = (
        (heights_m > 0.0) & np.isfinite(wsp) & (wsp > 0.0)
    )  # only consider positive heights and wind speeds for the log-log fit

    x = np.log(heights_m[mask])  # log of heights
    y = np.log(wsp[mask])  # log of wind speeds
    n = x.size  # number of valid data points

    # if there are less than 2 valid data points or if all x values are the same, return NaN for alpha and its standard error
    if n < 2 or np.allclose(x, x.mean()):
        alpha = np.nan
        se_alpha = np.nan

        return alpha, se_alpha, n

    else:
        x_mean = x.mean()
        y_mean = y.mean()
        Sxx = np.sum((x - x_mean) ** 2)  # sum of squares of x deviations
        Sxy = np.sum(
            (x - x_mean) * (y - y_mean)
        )  # sum of products of x and y deviations

        alpha = (
            Sxy / Sxx
        )  # slope of the log-log fit, which is the power-law shear exponent

        b = y_mean - alpha * x_mean  # intercept of the log-log fit

        y_pred = alpha * x + b  # predicted log(wind speed) from the fit
        resid = y - y_pred  # residuals of the fit

        sigma2 = np.sum(resid**2) / (n - 2)
        se_alpha = np.sqrt(sigma2 / Sxx)

        if alpha < 0:
            alpha = (
                np.nan
            )  # set negative alpha to NaN, as it is not physically meaningful
            se_alpha = np.nan  # set standard error to NaN if alpha is not valid

        return float(alpha), float(se_alpha), int(n)


def calc_shear(wsp_profiles: pd.DataFrame, window: int = 6) -> ShearValues:
    """
    Calculate the power-law shear exponent alpha and its uncertainty from LiDAR wind speed profiles.

    Parameters
    ----------
    wsp_profiles : pd.DataFrame
        DataFrame of wind speed values where the index is time and the columns are the heights. The values are the wind speed at that height and time.
    window : int, optional
        Window size for rolling statistics (median and mean) of alpha. Default is 6 (1 hour for  10-minute intervals).

    returns
    -------
        ShearValues
            Object containing alpha, its standard error, and rolling statistics.
            it contains the following attributes:
        - alpha: the power-law shear exponent
        - alpha_err: the standard error of alpha
        - alpha_roll_med: the rolling median of alpha with the specified window size
        - alpha_roll_mean: the rolling mean of alpha with the specified window size

    """
    heights = np.asarray(wsp_profiles.columns, float)

    alpha = pd.Series(
        index=wsp_profiles.index, dtype=float, name="alpha"
    )  # empty series to store the alpha values with the same index as wsp_profiles

    alpha_err = pd.Series(
        index=wsp_profiles.index, dtype=float, name="alpha_err"
    )  # empty series to store the standard error of alpha with the same index as wsp_profiles

    for t, row in wsp_profiles.iterrows():
        # fit alpha
        alpha_tmp, se_alpha_tmp, n_valid = fit_alpha_with_uncertainty(
            heights, row.values.astype(float)
        )

        alpha.loc[t] = alpha_tmp  # store the alpha value
        alpha_err.loc[t] = se_alpha_tmp  # store the standard error of alpha

    alpha = alpha.sort_index()
    alpha_err = alpha_err.sort_index()

    alpha_roll_med = alpha.rolling(window, center=True, min_periods=3).median()
    alpha_roll_mean = alpha.rolling(window, center=True, min_periods=3).mean()
    alpha_roll_med.name = "alpha_roll_med"
    alpha_roll_mean.name = "alpha_roll_mean"

    return ShearValues(
        alpha=alpha,
        alpha_err=alpha_err,
        alpha_roll_med=alpha_roll_med,
        alpha_roll_mean=alpha_roll_mean,
    )


# if __name__ == "__main__":
#     lidar_csv_files = find_KNMI_LiDAR_files(
#         Path(".", "tests", "lidar_data_10min"),
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
#     lidar_max_all = concatenate_wind_stats([item.max for item in per_file_stats])
#     lidar_min_all = concatenate_wind_stats([item.min for item in per_file_stats])
#     lidar_std_all = concatenate_wind_stats([item.std for item in per_file_stats])

#     # extract the unique heights from all files and sort them
#     height_lidar_all = np.unique(np.concatenate(heights_all))

#     print(lidar_avg_all)

#     # ---- plot stats

#     plt.figure(figsize=(12, 8))
#     plt.plot(
#         lidar_avg_all.index,
#         lidar_avg_all["Horizontal Wind Speed (m/s) at 299m"],
#         label="Average Wind Speed at 299m",
#         marker="o",
#         color="blue",
#     )

#     # Build a Dataframe of wind speed values where the index is time and the columns are the heights. The values are the wind speed at that height and time.
#     wsp_profiles = wind_height_profile(lidar_avg_all, height_lidar_all)

#     # Calculate the power-law shear exponent alpha and its uncertainty from the wind speed profiles.
#     ShearValues = calc_shear(wsp_profiles, window=6)

#     # ------- plot shear
#     fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

#     # 1) raw alpha + error bars
#     ax = axes[0]
#     ax.plot(
#         ShearValues.alpha.index,
#         ShearValues.alpha.values,
#         "o",
#         label="shear slope for LiDAR data",
#     )
#     ax.set_ylabel("Shear slope [-]")
#     ax.legend()
#     ax.grid(True)

#     # 2) rolling median
#     ax = axes[1]
#     ax.plot(
#         ShearValues.alpha_roll_med.index,
#         ShearValues.alpha_roll_med.values,
#         "o-",
#         label="LiDAR  - rolling median",
#     )
#     ax.set_ylabel("Shear slope [-]")
#     ax.legend()
#     ax.grid(True)

#     # 3) rolling mean
#     ax = axes[2]
#     ax.plot(
#         ShearValues.alpha_roll_mean.index,
#         ShearValues.alpha_roll_mean.values,
#         "o-",
#         label="LiDAR  - rolling mean",
#     )
#     ax.set_ylabel("Shear slope [-]")
#     ax.set_xlabel("Time [hour]")
#     ax.set_title("Shear slope vs hour")
#     ax.legend()
#     ax.grid(True)

#     fig.tight_layout()
