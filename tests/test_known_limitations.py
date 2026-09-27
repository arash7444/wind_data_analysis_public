"""Executable documentation for scientific behavior that needs a future decision."""

import pandas as pd
import pytest

from wind_data_analysis.process import calc_ti


@pytest.mark.xfail(
    strict=True,
    reason=(
        "calc_ti's invalid-speed mask is aligned against renamed TI columns, so "
        "negative wind speeds currently survive as negative TI values."
    ),
)
def test_calc_ti_rejects_negative_wind_speed() -> None:
    """Specify that non-positive wind speed must not yield a retained TI value.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The assertion documents the intended behavior and is marked xfail until
        the scientific correction is reviewed.

    Example
    -------
    Run with ``pytest tests/test_known_limitations.py``.
    """
    index = pd.DatetimeIndex(["2020-01-01"], name="Time")
    average = pd.DataFrame(
        {
            "Horizontal Wind Speed (m/s) at 100m": [-5.0],
            "Wind Direction (deg) at 100m": [180.0],
        },
        index=index,
    )
    standard_deviation = pd.DataFrame(
        {"Horizontal Wind Speed (m/s) at 100m": [1.0]}, index=index
    )

    result = calc_ti(average, standard_deviation, hub_height=100.0)

    assert result.ti_raw.empty
