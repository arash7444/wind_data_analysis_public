from wind_data_analysis.process.stats_func import compute_lidar_stats


from wind_data_analysis.data_reader import read_KNMI_LiDAR

from pathlib import Path


def test_LiDAR_stats():

    Lidar_file = Path(
        ".", "tests", "lidar_data", "ZephIR_Cabauw_ZP738_raw_20200501_v1.CSV"
    )
    data_lidar = read_KNMI_LiDAR(Lidar_file)

    lidar_stats = compute_lidar_stats(data_lidar)

    wind_speed = lidar_stats.avg["Met Wind Speed (m/s)"]
    assert int(wind_speed.iloc[0]) == 5, (
        "Expected average wind speed to be 5 m/s, but it is not correct"
    )

    print("LiDAR stats test passed successfully!")


if __name__ == "__main__":
    test_LiDAR_stats()
