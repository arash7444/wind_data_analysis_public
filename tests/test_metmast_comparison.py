"""Tests for shared LiDAR and met-mast wind-speed comparison behavior."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from wind_data_analysis.data_reader import read_KNMI_LiDAR, read_met
from wind_data_analysis.process import (
    compare_lidar_to_metmast,
    compute_lidar_stats,
    determine_comparison_period,
    normalize_timestamps_to_grid,
    pair_nearest_heights,
)


def _lidar_frame(index, values_by_height) -> pd.DataFrame:
    """Build a synthetic processed LiDAR mean frame.

    Parameters
    ----------
    index : array-like
        Timestamps for the synthetic observations.
    values_by_height : dict
        Mapping from numeric height to wind-speed values.

    Returns
    -------
    pandas.DataFrame
        LiDAR frame using production-style wind-speed column names.

    Example
    -------
    ``_lidar_frame(times, {10: [1, 2]})`` creates a 10 m series.
    """

    return pd.DataFrame(
        {
            f"Horizontal Wind Speed (m/s) at {height:g}m": values
            for height, values in values_by_height.items()
        },
        index=pd.DatetimeIndex(index),
    )


def _mast_frame(index, values_by_height) -> pd.DataFrame:
    """Build synthetic tidy met-mast wind-speed observations.

    Parameters
    ----------
    index : array-like
        Timestamps repeated for each measurement height.
    values_by_height : dict
        Mapping from numeric height to wind-speed values.

    Returns
    -------
    pandas.DataFrame
        Tidy met-mast frame indexed by source time.

    Example
    -------
    ``_mast_frame(times, {10: [1, 2]})`` creates a 10 m mast series.
    """

    parts = [
        pd.DataFrame(
            {"height": float(height), "wind_speed": values},
            index=pd.DatetimeIndex(index),
        )
        for height, values in values_by_height.items()
    ]
    return pd.concat(parts).sort_index()


def test_timestamp_normalization_accepts_small_offsets_and_rejects_large_ones():
    """Verify controlled nearest-grid rounding and its 30-second boundary.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate accepted and rejected timestamp offsets.

    Example
    -------
    Run with ``pytest tests/test_metmast_comparison.py``.
    """

    source = pd.DatetimeIndex(
        ["2020-01-01 00:10:00.018310546", "2020-01-01 00:19:59.981689453"]
    )
    normalized = normalize_timestamps_to_grid(source)
    expected = pd.DatetimeIndex(
        np.array(
            ["2020-01-01T00:10:00", "2020-01-01T00:20:00"],
            dtype="datetime64[ns]",
        )
    )
    pd.testing.assert_index_equal(normalized, expected)
    with pytest.raises(ValueError, match="exceeds"):
        normalize_timestamps_to_grid(["2020-01-01 00:10:31"])


def test_height_pairing_is_optimal_one_to_one_and_stable():
    """Verify cardinality, total-distance, reuse, and lower-height tie rules.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions specify deterministic height matching.

    Example
    -------
    Run with ``pytest -k height_pairing``.
    """

    assert pair_nearest_heights([10, 12], [11], 1) == ((10.0, 11.0),)
    pairs = pair_nearest_heights([0, 2], [1, 3], 2)
    assert pairs == ((0.0, 1.0), (2.0, 3.0))
    assert len({lidar for lidar, _ in pairs}) == len(pairs)
    assert len({mast for _, mast in pairs}) == len(pairs)


def test_height_pairing_respects_configurable_limit():
    """Verify that distant pairs are excluded as the configured limit changes.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate inclusive maximum-height behavior.

    Example
    -------
    Run with ``pytest -k configurable_limit``.
    """

    with pytest.raises(ValueError, match="No LiDAR/met-mast height pair"):
        pair_nearest_heights([10, 20], [12, 22], 1.9)
    assert pair_nearest_heights([10, 20], [12, 22], 2.0) == (
        (10.0, 12.0),
        (20.0, 22.0),
    )


def test_metrics_are_calculated_per_height_pair():
    """Verify means, signed bias, MAE, RMSE, correlation, and availability.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate metrics against a small analytic example.

    Example
    -------
    Run with ``pytest -k metrics_are_calculated``.
    """

    times = pd.date_range("2020-01-01", periods=3, freq="10min")
    result = compare_lidar_to_metmast(
        _lidar_frame(times, {10: [2.0, 4.0, 6.0]}),
        _mast_frame(times, {10: [1.0, 5.0, 5.0]}),
        "2020-01-01 00:00",
        "2020-01-01 00:30",
    )
    metric = result.metrics.iloc[0]
    assert metric["matched_observation_count"] == 3
    assert metric["lidar_mean_wind_speed_m_s"] == pytest.approx(4.0)
    assert metric["mast_mean_wind_speed_m_s"] == pytest.approx(11 / 3)
    assert metric["bias_lidar_minus_mast_m_s"] == pytest.approx(1 / 3)
    assert metric["mae_m_s"] == pytest.approx(1.0)
    assert metric["rmse_m_s"] == pytest.approx(1.0)
    assert metric["pearson_correlation"] == pytest.approx(np.sqrt(3) / 2)
    assert metric["availability_percent"] == pytest.approx(100.0)


def test_missing_values_are_removed_independently_for_each_pair():
    """Verify that missing data at one height does not reduce another pair.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions compare the independently matched per-pair counts.

    Example
    -------
    Run with ``pytest -k independently``.
    """

    times = pd.date_range("2020-01-01", periods=3, freq="10min")
    result = compare_lidar_to_metmast(
        _lidar_frame(times, {10: [1, np.nan, 3], 20: [4, 5, 6]}),
        _mast_frame(times, {10: [1, 2, 3], 20: [4, np.nan, 6]}),
        "2020-01-01 00:00",
        "2020-01-01 00:30",
    )
    counts = result.metrics.set_index("lidar_height_m")[
        "matched_observation_count"
    ].to_dict()
    assert counts == {10.0: 2, 20.0: 2}
    assert len(result.matched_data) == 4


def test_no_overlapping_timestamps_raises_a_useful_error():
    """Verify complete temporal separation fails instead of returning empties.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertion validates the no-observation error path.

    Example
    -------
    Run with ``pytest -k no_overlapping``.
    """

    lidar_times = pd.date_range("2020-01-01", periods=2, freq="10min")
    mast_times = pd.date_range("2020-01-02", periods=2, freq="10min")
    with pytest.raises(ValueError, match="No selected height pair"):
        compare_lidar_to_metmast(
            _lidar_frame(lidar_times, {10: [1, 2]}),
            _mast_frame(mast_times, {10: [1, 2]}),
        )


def test_duplicate_timestamps_after_normalization_raise_an_error():
    """Verify normalization never silently discards duplicate observations.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertion validates duplicate detection after rounding.

    Example
    -------
    Run with ``pytest -k duplicate_timestamps``.
    """

    source = pd.DatetimeIndex(
        ["2020-01-01 00:09:50", "2020-01-01 00:10:10"]
    )
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        normalize_timestamps_to_grid(source)


def test_failed_pair_is_reported_while_valid_pair_is_retained():
    """Verify a pair-specific duplicate does not discard another valid pair.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate partial failure reporting and retained metrics.

    Example
    -------
    Run with ``pytest -k failed_pair_is_reported``.
    """

    lidar_times = pd.DatetimeIndex(["2020-01-01 00:10", "2020-01-01 00:20"])
    bad_mast = pd.DataFrame(
        {"height": [10.0, 10.0], "wind_speed": [1.0, 2.0]},
        index=pd.DatetimeIndex(
            ["2020-01-01 00:09:50", "2020-01-01 00:10:10"]
        ),
    )
    good_mast = pd.DataFrame(
        {"height": [20.0, 20.0], "wind_speed": [3.0, 4.0]},
        index=lidar_times,
    )
    result = compare_lidar_to_metmast(
        _lidar_frame(lidar_times, {10: [1, 2], 20: [3, 4]}),
        pd.concat([bad_mast, good_mast]).sort_index(),
        "2020-01-01 00:00",
        "2020-01-01 00:30",
    )
    statuses = result.pairing_report.set_index("lidar_height_m")["status"].to_dict()
    assert statuses == {10.0: "failed", 20.0: "matched"}
    assert result.metrics["lidar_height_m"].tolist() == [20.0]
    assert "Duplicate timestamps" in result.pairing_report.iloc[0]["message"]


def test_insufficient_correlation_data_returns_nan():
    """Verify Pearson correlation is missing rather than raising an exception.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertion validates the documented insufficient-data behavior.

    Example
    -------
    Run with ``pytest -k insufficient_correlation``.
    """

    times = pd.date_range("2020-01-01", periods=2, freq="10min")
    result = compare_lidar_to_metmast(
        _lidar_frame(times, {10: [2.0, 2.0]}),
        _mast_frame(times, {10: [1.0, 3.0]}),
        "2020-01-01 00:00",
        "2020-01-01 00:20",
    )
    assert np.isnan(result.metrics.iloc[0]["pearson_correlation"])


def test_bundled_day_discovers_six_pairs_with_144_matches_each():
    """Verify dynamic discovery and alignment against the bundled real files.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions enforce the fixture-based scientific acceptance criteria.

    Example
    -------
    Run with ``pytest -k bundled_day``.
    """

    lidar = read_KNMI_LiDAR(
        Path("tests/lidar_data/ZephIR_Cabauw_ZP738_raw_20200607_v1.CSV")
    )
    lidar_stats = compute_lidar_stats(lidar)
    mast = read_met(
        Path("tests/metmast_data/cesar_tower_meteo_lb1_t10_v1.2_202006.nc"),
        "2020-06-07",
        "2020-06-08",
    )
    result = compare_lidar_to_metmast(
        lidar_stats.avg,
        mast,
        "2020-06-07",
        "2020-06-08",
        2.0,
        lidar_raw_sample_coverage=lidar_stats.raw_sample_coverage_percent,
        lidar_raw_invalid_sample_count=lidar_stats.raw_invalid_sample_count,
        min_lidar_raw_coverage_percent=(
            lidar_stats.min_lidar_raw_coverage_percent
        ),
    )
    actual_pairs = list(
        result.metrics[["lidar_height_m", "mast_height_m"]].itertuples(
            index=False, name=None
        )
    )
    assert actual_pairs == [
        (10.0, 10.0),
        (19.0, 20.0),
        (38.0, 40.0),
        (79.0, 80.0),
        (139.0, 140.0),
        (199.0, 200.0),
    ]
    assert result.metrics["matched_observation_count"].tolist() == [144] * 6
    assert result.metrics["availability_percent"].tolist() == [100.0] * 6
    assert result.metrics["lidar_below_minimum_coverage_bin_count"].tolist() == [
        0
    ] * 6
    assert result.metrics["lidar_min_raw_sample_coverage_percent"].min() >= 80.0


def test_runner_writes_complete_metmast_comparison_outputs(tmp_path, monkeypatch):
    """Run the JSON workflow and verify its pair data, metrics, and plots.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary output directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to create lightweight HTML output files.

    Returns
    -------
    None
        Assertions validate runner integration and generated artifacts.

    Example
    -------
    Run with ``pytest -k runner_writes_complete``.
    """

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("comparison_runner", root / "run_wind_analysis.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def lightweight_html(figure, path, *args, **kwargs):
        """Write a small marker file in place of full Plotly HTML.

        Parameters
        ----------
        figure : plotly.graph_objects.Figure
            Figure whose write method was invoked.
        path : str or pathlib.Path
            Requested HTML output path.
        *args : tuple
            Unused positional Plotly arguments.
        **kwargs : dict
            Unused keyword Plotly arguments.

        Returns
        -------
        None
            The marker file is written for artifact assertions.

        Example
        -------
        ``lightweight_html(fig, "plot.html")`` writes a minimal HTML file.
        """

        Path(path).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(go.Figure, "write_html", lightweight_html)
    monkeypatch.setattr(
        go.Figure,
        "show",
        lambda *args, **kwargs: pytest.fail("show_plot=False ignored"),
    )
    config = {
        "data_folder_lidar": str(root / "tests/lidar_data"),
        "data_folder_Metmast": str(
            root
            / "tests/metmast_data/cesar_tower_meteo_lb1_t10_v1.2_202006.nc"
        ),
        "start_date_lidar": "2020-06-07",
        "end_date_lidar": "2020-06-08",
        "start_date_Metmast": "2020-06-07",
        "end_date_Metmast": "2020-06-08",
        "features": ["metmast_comparison"],
        "max_height_difference_m": 2.0,
        "min_lidar_raw_coverage_percent": 80.0,
        "show_plot": False,
        "save_dir": str(tmp_path / "outputs"),
    }
    config_path = tmp_path / "comparison.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    runner.run_program_from_input(config_path)

    output = tmp_path / "outputs"
    expected_files = {
        "metmast_matched_data.csv",
        "metmast_metrics.csv",
        "metmast_height_pairs.csv",
        "metmast_timeseries.html",
        "metmast_scatter.html",
        "metmast_difference.html",
        "metmast_summary.html",
    }
    assert expected_files <= {path.name for path in output.iterdir()}
    metrics = pd.read_csv(output / "metmast_metrics.csv")
    assert metrics["matched_observation_count"].tolist() == [144] * 6
    assert metrics["availability_percent"].tolist() == [100.0] * 6
    matched = pd.read_csv(output / "metmast_matched_data.csv")
    assert len(matched) == 6 * 144
    assert {
        "lidar_raw_sample_coverage_percent",
        "lidar_raw_invalid_sample_count",
    } <= set(matched.columns)


def test_comparison_period_uses_instrument_range_overlap():
    """Verify independent source bounds produce their exclusive-end overlap.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Assertions validate configured and data-derived overlap behavior.

    Example
    -------
    Run with ``pytest -k instrument_range_overlap``.
    """

    lidar_times = pd.date_range("2020-01-01", periods=12, freq="10min")
    mast_times = pd.date_range("2020-01-01 01:00", periods=12, freq="10min")
    start, end = determine_comparison_period(
        lidar_times,
        mast_times,
        start_date_lidar="2020-01-01 00:30",
        end_date_lidar="2020-01-01 01:50",
        start_date_mast="2020-01-01 01:00",
        end_date_mast="2020-01-01 02:00",
    )
    assert start == pd.Timestamp("2020-01-01 01:00")
    assert end == pd.Timestamp("2020-01-01 01:50")
    with pytest.raises(ValueError, match="do not overlap"):
        determine_comparison_period(
            lidar_times,
            mast_times,
            end_date_lidar="2020-01-01 00:30",
            start_date_mast="2020-01-01 01:00",
        )


def test_runner_supports_metmast_only_export(tmp_path, capsys):
    """Verify met-mast data can be loaded and exported without LiDAR input.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory for config and CSV output.
    capsys : pytest.CaptureFixture
        Fixture used to inspect the printed met-mast summary.

    Returns
    -------
    None
        Assertions validate standalone loading, summary, and tidy export.

    Example
    -------
    Run with ``pytest -k metmast_only_export``.
    """

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "metmast_only_runner", root / "run_wind_analysis.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / "mast-output"
    config = {
        "data_folder_Metmast": str(
            root
            / "tests/metmast_data/cesar_tower_meteo_lb1_t10_v1.2_202006.nc"
        ),
        "features": ["metmast_data"],
        "save_dir": str(output),
        "show_plot": False,
    }
    config_path = tmp_path / "metmast-only.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    runner.run_program_from_input(config_path)
    exported = pd.read_csv(output / "metmast_data.csv")
    assert len(exported) > 7 * 144
    assert sorted(exported["height"].unique()) == [2, 10, 20, 40, 80, 140, 200]
    terminal_output = capsys.readouterr().out
    assert "Met-mast data summary" in terminal_output
    assert "Tidy met-mast observations are saved" in terminal_output
