"""Tests for wind-speed validity and raw LiDAR coverage diagnostics."""

import numpy as np
import pandas as pd
import pytest

from wind_data_analysis.process import compare_lidar_to_metmast, compute_lidar_stats


def _raw_lidar_frame(values, interval_seconds=150) -> pd.DataFrame:
    """Build raw LiDAR data with production-style speed and direction columns.

    Parameters
    ----------
    values : array-like
        Horizontal wind-speed samples.
    interval_seconds : int, default 150
        Separation between consecutive samples.

    Returns
    -------
    pandas.DataFrame
        Raw LiDAR-like observations indexed by time.

    Example
    -------
    ``_raw_lidar_frame([1, 2, 3, 4])`` creates one ten-minute bin.
    """

    index = pd.date_range(
        "2020-01-01", periods=len(values), freq=f"{interval_seconds}s", name="Time"
    )
    return pd.DataFrame(
        {
            "Horizontal Wind Speed (m/s) at 10m": values,
            "Wind Direction (deg) at 10m": [180.0] * len(values),
        },
        index=index,
    )


def _mast_frame(index, values) -> pd.DataFrame:
    """Build tidy 10 m met-mast observations.

    Parameters
    ----------
    index : array-like
        Observation timestamps.
    values : array-like
        Wind-speed values.

    Returns
    -------
    pandas.DataFrame
        Tidy met-mast frame.

    Example
    -------
    ``_mast_frame(times, [1, 2])`` creates two 10 m observations.
    """

    return pd.DataFrame(
        {"height": 10.0, "wind_speed": values}, index=pd.DatetimeIndex(index)
    )


def test_comparison_excludes_invalid_values_and_reports_union_counts() -> None:
    """Verify invalid values cannot affect metrics and exclusions are explicit.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate both instrument masks and three exclusion counts.

    Example
    -------
    Run with ``pytest -k comparison_excludes_invalid``.
    """

    times = pd.date_range("2020-01-01", periods=9, freq="10min")
    lidar = pd.DataFrame(
        {
            "Horizontal Wind Speed (m/s) at 10m": [
                0.0,
                99.0,
                -1.0,
                100.0,
                np.nan,
                np.inf,
                -np.inf,
                8.0,
                9.0,
            ]
        },
        index=times,
    )
    mast = _mast_frame(
        times,
        [0.0, 99.0, 3.0, 4.0, 5.0, 6.0, 7.0, -1.0, np.inf],
    )
    result = compare_lidar_to_metmast(
        lidar, mast, "2020-01-01", "2020-01-01 01:30"
    )

    metric = result.metrics.iloc[0]
    assert result.matched_data["lidar_wind_speed"].tolist() == [0.0, 99.0]
    assert result.matched_data["mast_wind_speed"].tolist() == [0.0, 99.0]
    assert metric["invalid_lidar_bin_count"] == 5
    assert metric["invalid_mast_bin_count"] == 2
    assert metric["invalid_paired_bin_count"] == 7
    assert metric["availability_percent"] == pytest.approx(200 / 9)
    assert metric["bias_lidar_minus_mast_m_s"] == pytest.approx(0.0)


def test_raw_lidar_masks_invalid_samples_before_averaging() -> None:
    """Verify invalid raw speeds are removed before means and coverage checks.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate valid-only means and raw rejection diagnostics.

    Example
    -------
    Run with ``pytest -k masks_invalid_samples``.
    """

    stats = compute_lidar_stats(
        _raw_lidar_frame([0.0, 99.0, -1.0, np.inf]),
        min_lidar_raw_coverage_percent=50.0,
    )
    column = "Horizontal Wind Speed (m/s) at 10m"
    assert stats.avg.iloc[0][column] == pytest.approx(49.5)
    assert stats.raw_sample_coverage_percent.iloc[0][column] == pytest.approx(50.0)
    assert stats.raw_invalid_sample_count.iloc[0][column] == 2
    assert stats.raw_sampling_interval_seconds == pytest.approx(150.0)
    assert stats.expected_raw_samples_per_bin == 4


def test_raw_lidar_coverage_default_custom_and_equality_boundary() -> None:
    """Verify default/custom thresholds and equality-pass behavior.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate 80% default and strict below-threshold masking.

    Example
    -------
    Run with ``pytest -k coverage_default_custom``.
    """

    column = "Horizontal Wind Speed (m/s) at 10m"
    frame = _raw_lidar_frame([1.0, 2.0, 3.0, 4.0, np.nan], interval_seconds=120)
    default_stats = compute_lidar_stats(frame)
    strict_stats = compute_lidar_stats(
        frame, min_lidar_raw_coverage_percent=80.01
    )
    permissive_stats = compute_lidar_stats(
        frame, min_lidar_raw_coverage_percent=0.0
    )

    assert default_stats.raw_sample_coverage_percent.iloc[0][column] == 80.0
    assert default_stats.avg.iloc[0][column] == pytest.approx(2.5)
    assert np.isnan(strict_stats.avg.iloc[0][column])
    assert permissive_stats.min_lidar_raw_coverage_percent == 0.0


@pytest.mark.parametrize("threshold", [-0.1, 100.1, np.nan, np.inf, "bad"])
def test_invalid_raw_lidar_coverage_threshold_raises(threshold) -> None:
    """Verify raw-coverage thresholds must be finite numeric percentages.

    Parameters
    ----------
    threshold : object
        Invalid threshold supplied by the parametrized test.

    Returns
    -------
    None
        Assertion validates a clear configuration error.

    Example
    -------
    Run with ``pytest -k invalid_raw_lidar_coverage_threshold``.
    """

    with pytest.raises(ValueError, match="between 0 and 100"):
        compute_lidar_stats(
            _raw_lidar_frame([1.0, 2.0, 3.0, 4.0]),
            min_lidar_raw_coverage_percent=threshold,
        )


def test_expected_samples_use_median_interval_and_round_half_up() -> None:
    """Verify median positive interval and round-half-up expected sample count.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate a 160-second median produces four expected samples.

    Example
    -------
    Run with ``pytest -k round_half_up``.
    """

    index = pd.DatetimeIndex(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 00:02:30",
            "2020-01-01 00:05:10",
            "2020-01-01 00:08:00",
        ],
        name="Time",
    )
    frame = pd.DataFrame(
        {
            "Horizontal Wind Speed (m/s) at 10m": [1.0, 2.0, 3.0, 4.0],
            "Wind Direction (deg) at 10m": [180.0] * 4,
        },
        index=index,
    )
    stats = compute_lidar_stats(frame)

    assert stats.raw_sampling_interval_seconds == pytest.approx(160.0)
    assert stats.expected_raw_samples_per_bin == 4


def test_comparison_exports_raw_coverage_and_aggregate_diagnostics() -> None:
    """Verify matched rows and pair summaries expose raw LiDAR diagnostics.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate row-level and aggregate coverage metadata.

    Example
    -------
    Run with ``pytest -k exports_raw_coverage``.
    """

    raw = _raw_lidar_frame(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, np.nan],
        interval_seconds=150,
    )
    stats = compute_lidar_stats(raw, min_lidar_raw_coverage_percent=80.0)
    mast = _mast_frame(stats.avg.index, [2.5, 5.5])
    result = compare_lidar_to_metmast(
        stats.avg,
        mast,
        "2020-01-01",
        "2020-01-01 00:20",
        lidar_raw_sample_coverage=stats.raw_sample_coverage_percent,
        lidar_raw_invalid_sample_count=stats.raw_invalid_sample_count,
        min_lidar_raw_coverage_percent=stats.min_lidar_raw_coverage_percent,
    )

    metric = result.metrics.iloc[0]
    assert result.matched_data["lidar_raw_sample_coverage_percent"].tolist() == [
        100.0
    ]
    assert result.matched_data["lidar_raw_invalid_sample_count"].tolist() == [0]
    assert metric["lidar_raw_invalid_sample_count"] == 2
    assert metric["lidar_mean_raw_sample_coverage_percent"] == pytest.approx(75.0)
    assert metric["lidar_min_raw_sample_coverage_percent"] == pytest.approx(50.0)
    assert metric["lidar_below_minimum_coverage_bin_count"] == 1
    assert metric["invalid_lidar_bin_count"] == 1


def test_preaveraged_lidar_applies_bin_validity_without_raw_coverage() -> None:
    """Verify low-resolution bins use the rule and expose no raw diagnostics.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate 0/99 acceptance and unavailable raw coverage.

    Example
    -------
    Run with ``pytest -k preaveraged_lidar``.
    """

    index = pd.date_range("2020-01-01", periods=4, freq="10min", name="Time")
    frame = pd.DataFrame(
        {
            "Horizontal Wind Speed (m/s) at 10m": [0.0, 99.0, -1.0, 100.0],
            "Horizontal Wind Speed Std. Dev. (m/s) at 10m": [1.0] * 4,
            "Horizontal Wind Speed Max (m/s) at 10m": [1.0] * 4,
            "Horizontal Wind Speed Min (m/s) at 10m": [1.0] * 4,
            "Wind Direction (deg) at 10m": [180.0] * 4,
        },
        index=index,
    )
    stats = compute_lidar_stats(frame)
    column = "Horizontal Wind Speed (m/s) at 10m"

    assert stats.avg[column].iloc[:2].tolist() == [0.0, 99.0]
    assert stats.avg[column].iloc[2:].isna().all()
    assert stats.raw_sample_coverage_percent is None
    assert stats.raw_invalid_sample_count is None
    assert stats.raw_sampling_interval_seconds is None
    assert stats.expected_raw_samples_per_bin is None
