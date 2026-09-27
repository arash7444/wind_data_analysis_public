"""Align LiDAR and met-mast wind speeds and calculate comparison metrics."""

from dataclasses import dataclass
from functools import lru_cache
import re

import numpy as np
import pandas as pd


_LIDAR_WIND_SPEED_PATTERN = re.compile(
    r"^Horizontal Wind Speed \(m/s\) at\s*([-+]?\d+(?:\.\d+)?)m$"
)
_PAIR_COLUMNS = ["lidar_height_m", "mast_height_m", "height_difference_m"]


@dataclass(frozen=True)
class MetmastComparisonResult:
    """Contain aligned observations, metrics, and height-pair diagnostics.

    Parameters
    ----------
    matched_data : pandas.DataFrame
        Tidy, non-missing observations for all successful height pairs.
    metrics : pandas.DataFrame
        One row of comparison metrics per successful height pair.
    pairing_report : pandas.DataFrame
        Selected height pairs, counts, and success or failure status.
    lidar_heights_m : tuple of float
        Numeric LiDAR heights discovered from wind-speed columns.
    mast_heights_m : tuple of float
        Numeric met-mast heights discovered from the ``height`` column.
    unmatched_lidar_heights_m : tuple of float
        LiDAR heights not selected by the one-to-one matcher.
    unmatched_mast_heights_m : tuple of float
        Met-mast heights not selected by the one-to-one matcher.

    Returns
    -------
    MetmastComparisonResult
        Immutable container holding the complete comparison result.

    Example
    -------
    ``result.metrics`` returns the tidy per-height metric table.
    """

    matched_data: pd.DataFrame
    metrics: pd.DataFrame
    pairing_report: pd.DataFrame
    lidar_heights_m: tuple[float, ...]
    mast_heights_m: tuple[float, ...]
    unmatched_lidar_heights_m: tuple[float, ...]
    unmatched_mast_heights_m: tuple[float, ...]


def extract_lidar_wind_speed_heights(lidar_mean: pd.DataFrame) -> tuple[float, ...]:
    """Extract sorted numeric heights from LiDAR mean wind-speed columns.

    Parameters
    ----------
    lidar_mean : pandas.DataFrame
        Processed LiDAR 10-minute means with standard KNMI column names.

    Returns
    -------
    tuple of float
        Unique measurement heights in ascending order.

    Example
    -------
    ``extract_lidar_wind_speed_heights(lidar_stats.avg)`` may return
    ``(10.0, 19.0, 38.0)``.
    """

    heights = {
        float(match.group(1))
        for column in lidar_mean.columns
        if (match := _LIDAR_WIND_SPEED_PATTERN.match(str(column).strip()))
    }
    if not heights:
        raise ValueError("No LiDAR horizontal wind-speed height columns were found.")
    if not all(np.isfinite(height) for height in heights):
        raise ValueError("LiDAR measurement heights must be finite numbers.")
    return tuple(sorted(heights))


def extract_mast_wind_speed_heights(mast_data: pd.DataFrame) -> tuple[float, ...]:
    """Extract sorted numeric heights from tidy met-mast observations.

    Parameters
    ----------
    mast_data : pandas.DataFrame
        Tidy met-mast data containing ``height`` and ``wind_speed``.

    Returns
    -------
    tuple of float
        Unique measurement heights in ascending order.

    Example
    -------
    ``extract_mast_wind_speed_heights(mast_data)`` may return
    ``(10.0, 20.0, 40.0)``.
    """

    missing = {"height", "wind_speed"} - set(mast_data.columns)
    if missing:
        raise ValueError(f"Met-mast data is missing columns: {sorted(missing)}")
    numeric = pd.to_numeric(mast_data["height"], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError("Met-mast measurement heights must be finite numbers.")
    heights = tuple(sorted(float(value) for value in numeric.unique()))
    if not heights:
        raise ValueError("No met-mast measurement heights were found.")
    return heights


def _select_better_matching(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Select a matching by cardinality, distance, then lower-height order.

    Parameters
    ----------
    first : tuple of tuple of float
        First candidate sequence of ``(LiDAR height, mast height)`` pairs.
    second : tuple of tuple of float
        Second candidate sequence of height pairs.

    Returns
    -------
    tuple of tuple of float
        The deterministic better candidate.

    Example
    -------
    ``_select_better_matching(((10, 10),), ())`` returns ``((10, 10),)``.
    """

    def ranking(pairs):
        """Build the minimization key used to compare two matchings.

        Parameters
        ----------
        pairs : tuple of tuple of float
            Candidate height-pair sequence.

        Returns
        -------
        tuple
            Key prioritizing more pairs, lower total distance, and low heights.

        Example
        -------
        ``ranking(((10, 11),))`` starts with ``(-1, 1.0)``.
        """

        return (
            -len(pairs),
            float(sum(abs(lidar - mast) for lidar, mast in pairs)),
            pairs,
        )

    return min((first, second), key=ranking)


def pair_nearest_heights(
    lidar_heights_m,
    mast_heights_m,
    max_height_difference_m: float = 2.0,
) -> tuple[tuple[float, float], ...]:
    """Create an optimal deterministic one-to-one height matching.

    The matching first maximizes the number of valid pairs, then minimizes
    total absolute height difference. If both objectives tie, the
    lexicographically lower sequence of LiDAR/mast heights is selected.

    Parameters
    ----------
    lidar_heights_m : iterable of float
        Available LiDAR measurement heights.
    mast_heights_m : iterable of float
        Available met-mast measurement heights.
    max_height_difference_m : float, default 2.0
        Inclusive maximum absolute height difference for a valid pair.

    Returns
    -------
    tuple of tuple of float
        Selected ``(LiDAR height, mast height)`` pairs in ascending order.

    Example
    -------
    ``pair_nearest_heights([19, 38], [20, 40])`` returns
    ``((19.0, 20.0), (38.0, 40.0))``.
    """

    limit = float(max_height_difference_m)
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("max_height_difference_m must be a finite non-negative number.")
    lidar = tuple(sorted({float(value) for value in lidar_heights_m}))
    mast = tuple(sorted({float(value) for value in mast_heights_m}))
    if not all(np.isfinite(value) for value in lidar + mast):
        raise ValueError("Measurement heights must be finite numbers.")

    @lru_cache(maxsize=None)
    def solve(lidar_index: int, mast_index: int):
        """Solve the ordered matching subproblem from two index positions.

        Parameters
        ----------
        lidar_index : int
            Current position in the sorted LiDAR-height sequence.
        mast_index : int
            Current position in the sorted mast-height sequence.

        Returns
        -------
        tuple of tuple of float
            Optimal matching for the remaining heights.

        Example
        -------
        ``solve(0, 0)`` computes the complete matching.
        """

        if lidar_index >= len(lidar) or mast_index >= len(mast):
            return ()
        best = _select_better_matching(
            solve(lidar_index + 1, mast_index),
            solve(lidar_index, mast_index + 1),
        )
        if abs(lidar[lidar_index] - mast[mast_index]) <= limit:
            paired = (
                (lidar[lidar_index], mast[mast_index]),
            ) + solve(lidar_index + 1, mast_index + 1)
            best = _select_better_matching(best, paired)
        return best

    pairs = solve(0, 0)
    if not pairs:
        raise ValueError(
            "No LiDAR/met-mast height pair is within "
            f"max_height_difference_m={limit:g} m."
        )
    return pairs


def normalize_timestamps_to_grid(
    index,
    frequency: str = "10min",
    tolerance: str | pd.Timedelta = "30s",
) -> pd.DatetimeIndex:
    """Validate and round timestamps to a regular comparison grid.

    Parameters
    ----------
    index : array-like
        Source timestamps to normalize.
    frequency : str, default "10min"
        Pandas-compatible target-grid frequency.
    tolerance : str or pandas.Timedelta, default "30s"
        Maximum allowed absolute offset from the nearest grid point.

    Returns
    -------
    pandas.DatetimeIndex
        Sorted-order-preserving timestamps rounded to the requested grid.

    Example
    -------
    ``normalize_timestamps_to_grid(["2020-01-01 00:10:00.018"])`` returns
    the exact timestamp ``2020-01-01 00:10:00``.
    """

    timestamps = pd.DatetimeIndex(pd.to_datetime(index))
    if timestamps.hasnans:
        raise ValueError("Timestamps must not contain missing values.")
    allowed = pd.Timedelta(tolerance)
    if allowed < pd.Timedelta(0):
        raise ValueError("Timestamp tolerance must be non-negative.")
    rounded = timestamps.round(frequency)
    offsets = pd.Series(timestamps - rounded).abs()
    if (offsets > allowed).any():
        largest = offsets.max()
        raise ValueError(
            f"Timestamp offset {largest} exceeds the allowed {allowed} "
            f"distance from the {frequency} grid."
        )
    duplicates = rounded[rounded.duplicated(keep=False)]
    if len(duplicates):
        examples = sorted({str(value) for value in duplicates[:5]})
        raise ValueError(
            "Duplicate timestamps exist after normalization: " + ", ".join(examples)
        )
    return rounded


def determine_comparison_period(
    lidar_index,
    mast_index,
    start_date_lidar=None,
    end_date_lidar=None,
    start_date_mast=None,
    end_date_mast=None,
    timestamp_tolerance: str | pd.Timedelta = "30s",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Determine the exclusive-end overlap of two optional source periods.

    Explicit instrument bounds take precedence. An omitted start is derived
    from that source's earliest normalized observation; an omitted end is one
    10-minute interval after its latest normalized observation.

    Parameters
    ----------
    lidar_index : array-like
        LiDAR observation timestamps.
    mast_index : array-like
        Met-mast observation timestamps, which may repeat across heights.
    start_date_lidar, end_date_lidar : datetime-like or None
        Optional inclusive start and exclusive end for LiDAR.
    start_date_mast, end_date_mast : datetime-like or None
        Optional inclusive start and exclusive end for met-mast data.
    timestamp_tolerance : str or pandas.Timedelta, default "30s"
        Maximum offset accepted while normalizing source extents.

    Returns
    -------
    tuple of pandas.Timestamp
        Inclusive comparison start and exclusive comparison end.

    Example
    -------
    ``determine_comparison_period(lidar_times, mast_times)`` returns their
    common data period on the validated 10-minute grid.
    """

    lidar_unique = pd.DatetimeIndex(pd.to_datetime(lidar_index)).unique()
    mast_unique = pd.DatetimeIndex(pd.to_datetime(mast_index)).unique()
    if not len(lidar_unique) or not len(mast_unique):
        raise ValueError("Both instruments need observations for comparison.")
    lidar_normalized = normalize_timestamps_to_grid(
        lidar_unique, tolerance=timestamp_tolerance
    )
    mast_normalized = normalize_timestamps_to_grid(
        mast_unique, tolerance=timestamp_tolerance
    )
    lidar_start = (
        pd.Timestamp(start_date_lidar)
        if start_date_lidar is not None
        else lidar_normalized.min()
    )
    lidar_end = (
        pd.Timestamp(end_date_lidar)
        if end_date_lidar is not None
        else lidar_normalized.max() + pd.Timedelta("10min")
    )
    mast_start = (
        pd.Timestamp(start_date_mast)
        if start_date_mast is not None
        else mast_normalized.min()
    )
    mast_end = (
        pd.Timestamp(end_date_mast)
        if end_date_mast is not None
        else mast_normalized.max() + pd.Timedelta("10min")
    )
    if lidar_start >= lidar_end:
        raise ValueError("LiDAR start date must be before its end date.")
    if mast_start >= mast_end:
        raise ValueError("Met-mast start date must be before its end date.")
    overlap_start = max(lidar_start, mast_start)
    overlap_end = min(lidar_end, mast_end)
    if overlap_start >= overlap_end:
        raise ValueError("The LiDAR and met-mast periods do not overlap.")
    return overlap_start, overlap_end


def _lidar_column_for_height(lidar_mean: pd.DataFrame, height_m: float) -> str:
    """Return the unique LiDAR wind-speed column for a numeric height.

    Parameters
    ----------
    lidar_mean : pandas.DataFrame
        Processed LiDAR mean data.
    height_m : float
        Requested measurement height.

    Returns
    -------
    str
        Matching source column name.

    Example
    -------
    ``_lidar_column_for_height(frame, 10)`` returns a column ending in ``10m``.
    """

    matches = []
    for column in lidar_mean.columns:
        match = _LIDAR_WIND_SPEED_PATTERN.match(str(column).strip())
        if match and float(match.group(1)) == float(height_m):
            matches.append(column)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one LiDAR wind-speed column at {height_m:g} m; found {len(matches)}."
        )
    return matches[0]


def _normalized_series_frame(
    series: pd.Series,
    value_name: str,
    source_time_name: str,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    """Prepare one source series for a normalized-time join.

    Parameters
    ----------
    series : pandas.Series
        Wind-speed values indexed by source timestamps.
    value_name : str
        Output name for the wind-speed values.
    source_time_name : str
        Output name preserving original timestamps.
    tolerance : pandas.Timedelta
        Maximum rounding offset accepted by timestamp normalization.

    Returns
    -------
    pandas.DataFrame
        Values and source timestamps indexed by normalized comparison time.

    Example
    -------
    ``_normalized_series_frame(series, "speed", "source_time", pd.Timedelta("30s"))``
    creates a two-column alignment frame.
    """

    source_times = pd.DatetimeIndex(series.index)
    normalized = normalize_timestamps_to_grid(source_times, tolerance=tolerance)
    frame = pd.DataFrame(
        {
            value_name: pd.to_numeric(series.to_numpy(), errors="coerce"),
            source_time_name: source_times,
        },
        index=normalized,
    )
    frame.index.name = "time"
    return frame.sort_index()


def _expected_period_count(
    start_date,
    end_date,
    normalized_indexes: list[pd.DatetimeIndex],
) -> int:
    """Calculate the scheduled 10-minute denominator for availability.

    Parameters
    ----------
    start_date : datetime-like or None
        Inclusive requested comparison start.
    end_date : datetime-like or None
        Exclusive requested comparison end.
    normalized_indexes : list of pandas.DatetimeIndex
        Normalized source indexes used when explicit bounds are omitted.

    Returns
    -------
    int
        Number of expected 10-minute periods.

    Example
    -------
    ``_expected_period_count("2020-01-01", "2020-01-02", [])`` returns 144.
    """

    if (start_date is None) != (end_date is None):
        raise ValueError("Provide both start_date and end_date, or neither.")
    if start_date is not None:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start >= end:
            raise ValueError("start_date must be before end_date.")
        return len(pd.date_range(start=start, end=end, freq="10min", inclusive="left"))
    nonempty = [index for index in normalized_indexes if len(index)]
    if not nonempty:
        return 0
    start = min(index.min() for index in nonempty)
    end = max(index.max() for index in nonempty) + pd.Timedelta("10min")
    return len(pd.date_range(start=start, end=end, freq="10min", inclusive="left"))


def _comparison_metrics(
    matched: pd.DataFrame,
    lidar_height_m: float,
    mast_height_m: float,
    expected_count: int,
    lidar_source_count: int,
    mast_source_count: int,
    shared_timestamp_count: int,
) -> dict:
    """Calculate per-pair wind-speed comparison metrics.

    Parameters
    ----------
    matched : pandas.DataFrame
        Non-missing paired observations with the two wind-speed columns.
    lidar_height_m : float
        Actual LiDAR measurement height.
    mast_height_m : float
        Actual met-mast measurement height.
    expected_count : int
        Scheduled 10-minute periods in the comparison interval.
    lidar_source_count : int
        LiDAR rows available before alignment and missing-value removal.
    mast_source_count : int
        Mast rows available before alignment and missing-value removal.
    shared_timestamp_count : int
        Common timestamps before pairwise missing-value removal.

    Returns
    -------
    dict
        One tidy metrics-table record. Wind-speed metrics use m/s.

    Example
    -------
    ``_comparison_metrics(frame, 10, 10, 144, 144, 144, 144)`` returns a
    record whose bias is defined as LiDAR minus met mast.
    """

    lidar_values = matched["lidar_wind_speed"].astype(float)
    mast_values = matched["mast_wind_speed"].astype(float)
    difference = lidar_values - mast_values
    count = len(matched)
    correlation = np.nan
    if count >= 2 and lidar_values.nunique() > 1 and mast_values.nunique() > 1:
        correlation = float(lidar_values.corr(mast_values))
    availability = 100.0 * count / expected_count if expected_count else np.nan
    return {
        "lidar_height_m": float(lidar_height_m),
        "mast_height_m": float(mast_height_m),
        "height_difference_m": abs(float(lidar_height_m) - float(mast_height_m)),
        "expected_observation_count": int(expected_count),
        "lidar_source_observation_count": int(lidar_source_count),
        "mast_source_observation_count": int(mast_source_count),
        "shared_timestamp_count": int(shared_timestamp_count),
        "matched_observation_count": int(count),
        "lidar_mean_wind_speed_m_s": float(lidar_values.mean()),
        "mast_mean_wind_speed_m_s": float(mast_values.mean()),
        "bias_lidar_minus_mast_m_s": float(difference.mean()),
        "mae_m_s": float(difference.abs().mean()),
        "rmse_m_s": float(np.sqrt(np.mean(np.square(difference)))),
        "pearson_correlation": correlation,
        "availability_percent": availability,
    }


def compare_lidar_to_metmast(
    lidar_mean: pd.DataFrame,
    mast_data: pd.DataFrame,
    start_date=None,
    end_date=None,
    max_height_difference_m: float = 2.0,
    timestamp_tolerance: str | pd.Timedelta = "30s",
) -> MetmastComparisonResult:
    """Align all optimal height pairs and compare 10-minute wind speeds.

    Parameters
    ----------
    lidar_mean : pandas.DataFrame
        Processed LiDAR 10-minute mean data with a DatetimeIndex.
    mast_data : pandas.DataFrame
        Tidy met-mast data with a DatetimeIndex and ``height``/``wind_speed``.
    start_date : datetime-like or None
        Inclusive start used for filtering and availability calculation.
    end_date : datetime-like or None
        Exclusive end used for filtering and availability calculation.
    max_height_difference_m : float, default 2.0
        Inclusive maximum height separation for one-to-one pairing.
    timestamp_tolerance : str or pandas.Timedelta, default "30s"
        Largest offset allowed before rounding to the 10-minute grid.

    Returns
    -------
    MetmastComparisonResult
        Matched tidy data, metrics, pair diagnostics, and unmatched heights.

    Example
    -------
    ``compare_lidar_to_metmast(lidar_stats.avg, mast, "2020-06-07", "2020-06-08")``
    compares every automatically selected height pair for one day.
    """

    if not isinstance(lidar_mean.index, pd.DatetimeIndex):
        raise ValueError("LiDAR mean data must use a DatetimeIndex.")
    if not isinstance(mast_data.index, pd.DatetimeIndex):
        raise ValueError("Met-mast data must use a DatetimeIndex.")
    tolerance = pd.Timedelta(timestamp_tolerance)
    if tolerance < pd.Timedelta(0):
        raise ValueError("Timestamp tolerance must be non-negative.")
    if (start_date is None) != (end_date is None):
        raise ValueError("Provide both start_date and end_date, or neither.")
    lidar = lidar_mean.sort_index().copy()
    mast = mast_data.sort_index().copy()
    if start_date is not None:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start >= end:
            raise ValueError("start_date must be before end_date.")
        lidar = lidar[
            (lidar.index >= start - tolerance) & (lidar.index < end + tolerance)
        ]
        mast = mast[
            (mast.index >= start - tolerance) & (mast.index < end + tolerance)
        ]

    lidar_heights = extract_lidar_wind_speed_heights(lidar)
    mast_heights = extract_mast_wind_speed_heights(mast)
    pairs = pair_nearest_heights(
        lidar_heights, mast_heights, max_height_difference_m=max_height_difference_m
    )
    paired_lidar = {pair[0] for pair in pairs}
    paired_mast = {pair[1] for pair in pairs}
    unmatched_lidar = tuple(value for value in lidar_heights if value not in paired_lidar)
    unmatched_mast = tuple(value for value in mast_heights if value not in paired_mast)

    matched_parts = []
    metric_records = []
    report_records = []
    for lidar_height, mast_height in pairs:
        report = {
            "lidar_height_m": lidar_height,
            "mast_height_m": mast_height,
            "height_difference_m": abs(lidar_height - mast_height),
            "status": "failed",
            "message": "",
        }
        lidar_series = lidar[_lidar_column_for_height(lidar, lidar_height)]
        mast_series = mast.loc[
            pd.to_numeric(mast["height"], errors="coerce").eq(mast_height),
            "wind_speed",
        ]
        report["lidar_source_observation_count"] = len(lidar_series)
        report["mast_source_observation_count"] = len(mast_series)
        try:
            lidar_frame = _normalized_series_frame(
                lidar_series,
                "lidar_wind_speed",
                "lidar_source_time",
                tolerance,
            )
            mast_frame = _normalized_series_frame(
                mast_series,
                "mast_wind_speed",
                "mast_source_time",
                tolerance,
            )
            if start_date is not None:
                lidar_frame = lidar_frame[
                    (lidar_frame.index >= start) & (lidar_frame.index < end)
                ]
                mast_frame = mast_frame[
                    (mast_frame.index >= start) & (mast_frame.index < end)
                ]
            report["lidar_source_observation_count"] = len(lidar_frame)
            report["mast_source_observation_count"] = len(mast_frame)
            joined = lidar_frame.join(mast_frame, how="inner")
            report["shared_timestamp_count"] = len(joined)
            matched = joined.dropna(
                subset=["lidar_wind_speed", "mast_wind_speed"]
            ).copy()
            report["matched_observation_count"] = len(matched)
            if matched.empty:
                report["message"] = "No matched non-missing observations."
                report_records.append(report)
                continue
            expected_count = _expected_period_count(
                start_date, end_date, [lidar_frame.index, mast_frame.index]
            )
            matched = matched.reset_index()
            matched.insert(1, "lidar_height_m", lidar_height)
            matched.insert(2, "mast_height_m", mast_height)
            matched.insert(3, "height_difference_m", abs(lidar_height - mast_height))
            matched["difference_lidar_minus_mast_m_s"] = (
                matched["lidar_wind_speed"] - matched["mast_wind_speed"]
            )
            matched_parts.append(matched)
            metric_records.append(
                _comparison_metrics(
                    matched,
                    lidar_height,
                    mast_height,
                    expected_count,
                    len(lidar_frame),
                    len(mast_frame),
                    len(joined),
                )
            )
            report["status"] = "matched"
            report["message"] = ""
        except ValueError as error:
            report["shared_timestamp_count"] = 0
            report["matched_observation_count"] = 0
            report["message"] = str(error)
        report_records.append(report)

    if not matched_parts:
        details = "; ".join(
            f"{row['lidar_height_m']:g}/{row['mast_height_m']:g} m: {row['message']}"
            for row in report_records
        )
        raise ValueError("No selected height pair has matched observations. " + details)

    matched_data = pd.concat(matched_parts, ignore_index=True).sort_values(
        ["lidar_height_m", "mast_height_m", "time"]
    )
    metrics = pd.DataFrame(metric_records).sort_values(_PAIR_COLUMNS).reset_index(drop=True)
    report_frame = pd.DataFrame(report_records).sort_values(_PAIR_COLUMNS).reset_index(drop=True)
    return MetmastComparisonResult(
        matched_data=matched_data.reset_index(drop=True),
        metrics=metrics,
        pairing_report=report_frame,
        lidar_heights_m=lidar_heights,
        mast_heights_m=mast_heights,
        unmatched_lidar_heights_m=unmatched_lidar,
        unmatched_mast_heights_m=unmatched_mast,
    )
