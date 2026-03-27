from pathlib import Path

from wind_data_analysis.data_reader import read_KNMI_LiDAR


def test_read_knmi_lidar() -> None:
    file_path = Path("tests", "lidar_data", "ZephIR_Cabauw_ZP738_raw_20200501_v1.CSV")

    df = read_KNMI_LiDAR(file_path)

    assert not df.empty
    assert "Met Wind Speed (m/s)" in df.columns
    print("read_KNMI_LiDAR test passed successfully!")
