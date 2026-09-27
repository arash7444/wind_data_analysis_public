"""Shared scientific validity rules for horizontal wind speed."""

import numpy as np
import pandas as pd


MIN_VALID_WIND_SPEED_M_S = 0.0
MAX_VALID_WIND_SPEED_M_S = 99.0


def valid_wind_speed_mask(
    values: pd.Series | pd.DataFrame, *, allow_zero: bool = True
) -> pd.Series | pd.DataFrame:
    """Return the approved finite 0--99 m/s wind-speed validity mask.

    Parameters
    ----------
    values : pandas.Series or pandas.DataFrame
        Wind-speed values to validate; non-numeric values are invalid.
    allow_zero : bool, default True
        Whether the inclusive lower boundary is zero. Set to ``False`` for TI,
        where division requires a strictly positive mean speed.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Boolean mask aligned with ``values``.

    Example
    -------
    ``valid_wind_speed_mask(pd.Series([0, 99, 100])).tolist()`` returns
    ``[True, True, False]``.
    """

    if isinstance(values, pd.Series):
        numeric = pd.to_numeric(values, errors="coerce")
    elif isinstance(values, pd.DataFrame):
        numeric = values.apply(pd.to_numeric, errors="coerce")
    else:
        raise TypeError("values must be a pandas Series or DataFrame.")
    lower_bound = numeric.gt(MIN_VALID_WIND_SPEED_M_S)
    if allow_zero:
        lower_bound = numeric.ge(MIN_VALID_WIND_SPEED_M_S)
    return lower_bound & numeric.le(MAX_VALID_WIND_SPEED_M_S) & np.isfinite(numeric)


def validate_coverage_threshold(value: float) -> float:
    """Validate and normalize a raw LiDAR coverage percentage.

    Parameters
    ----------
    value : float
        Requested minimum coverage percentage.

    Returns
    -------
    float
        Finite percentage between 0 and 100 inclusive.

    Example
    -------
    ``validate_coverage_threshold(80)`` returns ``80.0``.
    """

    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "min_lidar_raw_coverage_percent must be finite and between 0 and 100."
        ) from error
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 100.0:
        raise ValueError(
            "min_lidar_raw_coverage_percent must be finite and between 0 and 100."
        )
    return threshold
