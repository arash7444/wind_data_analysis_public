"""Executable documentation for scientific behavior that needs a future decision."""

import pandas as pd
import pytest

from wind_data_analysis.process import calc_ti


def test_calc_ti_uses_aligned_valid_mean_wind_speeds() -> None:
    """Verify TI rejects invalid means while accepting the 99 m/s boundary.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions document the approved finite, positive, at-most-99 rule.

    Example
    -------
    Run with ``pytest tests/test_known_limitations.py``.
    """
    index = pd.date_range("2020-01-01", periods=7, freq="10min", name="Time")
    average = pd.DataFrame(
        {
            "Horizontal Wind Speed (m/s) at 100m": [
                -5.0,
                0.0,
                99.0,
                100.0,
                float("nan"),
                float("inf"),
                float("-inf"),
            ],
            "Wind Direction (deg) at 100m": [180.0] * 7,
        },
        index=index,
    )
    standard_deviation = pd.DataFrame(
        {"Horizontal Wind Speed (m/s) at 100m": [1.0] * 7}, index=index
    )

    result = calc_ti(average, standard_deviation, hub_height=100.0)

    assert result.ti_raw["wind_speed"].tolist() == [99.0]
    assert result.ti_raw["ti"].tolist() == pytest.approx([1.0 / 99.0])
