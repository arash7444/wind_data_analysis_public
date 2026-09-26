"""Experimental KNMI met-mast reader adapted from the original private project.

This returns tidy measurements for Python use. The LiDAR runner and GUI do
not yet accept met-mast inputs. Validate against your real NetCDF files.
"""

from pathlib import Path
import re

import pandas as pd
import xarray as xr


def _date_bounds(start_date, end_date):
    start = pd.Timestamp(start_date) if start_date is not None else None
    end = pd.Timestamp(end_date) if end_date is not None else None
    if start is not None and end is not None and start >= end:
        raise ValueError("start_date must be before end_date.")
    return start, end


def met_finder(pth_met_base: str | Path, start_date=None, end_date=None) -> list[str]:
    """Find monthly KNMI ``*_YYYYMM.nc`` files overlapping [start, end).

    Bounds may be ISO date strings or pandas timestamps and either may be
    omitted. A partial month includes that month's file. Files with other
    names are retained so useful data is not silently skipped; ``read_met``
    applies exact measurement-time filtering. A single file is also accepted.
    """
    start, end = _date_bounds(start_date, end_date)
    base = Path(pth_met_base)
    if not base.exists():
        raise FileNotFoundError(base)
    candidates = [base] if base.is_file() else sorted(base.rglob("*.nc"))
    selected = []
    for path in candidates:
        match = re.search(r"_(\d{4})(\d{2})\.nc$", path.name)
        if match:
            month_start = pd.Timestamp(year=int(match[1]), month=int(match[2]), day=1)
            month_end = month_start + pd.offsets.MonthBegin(1)
            if start is not None and month_end <= start:
                continue
            if end is not None and month_start >= end:
                continue
        selected.append(str(path))
    return selected


def read_met(met_nc_file: str | Path, start_date=None, end_date=None) -> pd.DataFrame:
    """Read KNMI measurements, retaining missing optional weather values.

    Required variables: ``z`` (height), ``F`` (wind speed), ``SF`` (speed
    standard deviation), and ``D`` (direction). Optional ``SD``, ``TA``,
    ``Q`` become direction standard deviation, air temperature, humidity.
    Returns a DataFrame indexed by ``time``, filtered to [start, end).
    Values and units are retained as stored in the input; no quality-flag
    filtering or interpolation is applied. The file is closed before return.
    """
    start, end = _date_bounds(start_date, end_date)
    names = {"z": "height", "F": "wind_speed", "SF": "wind_speed_std",
             "D": "wind_direction", "SD": "wind_direction_std",
             "TA": "air_temp", "Q": "humidity"}
    with xr.open_dataset(met_nc_file) as dataset:
        missing = {"z", "F", "SF", "D", "time"} - set(dataset.variables)
        if missing:
            raise ValueError(f"Missing KNMI met-mast variables: {sorted(missing)}")
        variables = [name for name in names if name in dataset.variables]
        frame = dataset[variables].to_dataframe().reset_index()
    frame = frame.rename(columns=names)
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.set_index("time").sort_index()
    if start is not None:
        frame = frame[frame.index >= start]
    if end is not None:
        frame = frame[frame.index < end]
    return frame.dropna(subset=["height", "wind_speed"])
