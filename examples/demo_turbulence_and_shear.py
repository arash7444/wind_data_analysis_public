"""Demonstrate turbulence-intensity and power-law shear calculations."""

from wind_data_analysis.process import calc_shear, calc_ti

from demo_support import REPOSITORY_ROOT, load_lidar_statistics, make_output_directory


def main() -> None:
    """Calculate TI and shear from the bundled high-frequency LiDAR sample.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Tidy TI and shear CSV files are written below ``outputs/examples``.

    Example
    -------
    Run from the repository root with
    ``python examples/demo_turbulence_and_shear.py``.
    """
    output_directory = make_output_directory("turbulence_shear")
    results = load_lidar_statistics(
        REPOSITORY_ROOT / "tests" / "lidar_data", "2020-05-01", "2020-05-03"
    )

    ti_values = calc_ti(
        results["average"], results["standard_deviation"], hub_height=139.0
    )
    shear_values = calc_shear(results["profiles"], window=6)

    ti_values.ti_raw.to_csv(output_directory / "turbulence_intensity.csv", index=False)
    ti_values.ti_median.to_csv(
        output_directory / "median_turbulence_by_height.csv", index=False
    )
    shear_values.alpha.to_csv(output_directory / "shear_alpha.csv", header=True)

    print("Median turbulence intensity by height:")
    print(ti_values.ti_median.to_string(index=False))
    print(
        f"Computed {shear_values.alpha.notna().sum()} valid shear exponents "
        f"from {len(shear_values.alpha)} profiles."
    )
    print(f"Saved calculated values in: {output_directory}")


if __name__ == "__main__":
    main()
