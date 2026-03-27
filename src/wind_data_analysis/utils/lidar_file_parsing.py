import os
from pathlib import Path
import re
import pandas as pd


def extract_date_from_filename(filename: str) -> pd.Timestamp | None:
    """
    Extract a date from filename in multiple formats:
    YYYYMMDD, YYYY-MM-DD, YYYY_MM_DD
    """
    patterns = [
        (r"(?<!\d)(\d{8})(?!\d)", "%Y%m%d"),  # 20200501
        (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),  # 2020-05-01
        (r"(\d{4}_\d{2}_\d{2})", "%Y_%m_%d"),  # 2020_05_01
    ]

    for pattern, date_format in patterns:
        match = re.search(pattern, filename)
        if match:
            date_str = match.group(1)
            try:
                return pd.to_datetime(date_str, format=date_format)
            except ValueError:
                continue

    return None
