"""Demonstrate statistics for raw and pre-averaged LiDAR inputs."""

from demo_support import REPOSITORY_ROOT, load_lidar_statistics, make_output_directory


def main() -> None:
    """Calculate LiDAR statistics and height profiles for both input formats.

    Parameters
    ----------
    None

    Returns
    -------
    None
        CSV summaries are saved below ``outputs/examples/statistics_profiles``.

    Example
    -------
    Run from the repository root with
    ``python examples/demo_statistics_and_profiles.py``.
    """
    output_directory = make_output_directory("statistics_profiles")
    folders = {
        "raw": REPOSITORY_ROOT / "tests" / "lidar_data",
        "ten_minute": REPOSITORY_ROOT / "tests" / "lidar_data_10min",
    }

    for label, folder in folders.items():
        results = load_lidar_statistics(folder, "2020-05-01", "2020-05-02")
        average = results["average"]
        profiles = results["profiles"]
        average.head(12).to_csv(output_directory / f"{label}_average_preview.csv")
        profiles.head(12).to_csv(output_directory / f"{label}_profiles_preview.csv")

        print(f"{label}: {len(average)} ten-minute rows")
        print(f"{label}: profile shape {profiles.shape}")
        print(f"{label}: heights {results['heights'].tolist()}")

    print(f"Saved statistics and profile previews in: {output_directory}")


if __name__ == "__main__":
    main()
