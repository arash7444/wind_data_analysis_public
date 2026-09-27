from pathlib import Path

from wind_data_analysis.data_reader import find_KNMI_LiDAR_files, read_KNMI_LiDAR


def test_read_knmi_lidar() -> None:
    file_path = Path("tests", "lidar_data", "ZephIR_Cabauw_ZP738_raw_20200501_v1.CSV")

    df = read_KNMI_LiDAR(file_path)

    assert not df.empty
    assert "Met Wind Speed (m/s)" in df.columns
    print("read_KNMI_LiDAR test passed successfully!")


def test_lidar_file_dates_can_be_filtered_independently() -> None:
    """Verify start-only and end-only LiDAR discovery date filters.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate independently optional date bounds.

    Example
    -------
    Run with ``pytest -k lidar_file_dates_can_be_filtered``.
    """

    folder = Path("tests", "lidar_data")
    start_only = find_KNMI_LiDAR_files(folder, start_date="2020-06-01")
    end_only = find_KNMI_LiDAR_files(folder, end_date="2020-05-02")
    assert [Path(path).name for path in start_only] == [
        "ZephIR_Cabauw_ZP738_raw_20200607_v1.CSV"
    ]
    assert [Path(path).name for path in end_only] == [
        "ZephIR_Cabauw_ZP738_raw_20200501_v1.CSV"
    ]
