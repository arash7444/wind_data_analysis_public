"""Regression checks for the features brought over from the private project."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import xarray as xr

from wind_data_analysis.data_reader import met_finder, read_met
from wind_data_analysis.plotting import plot_ti_polar_by_height, plot_wind_statistics
from wind_data_analysis.process.bin_wdir import bin_wdir


def test_polar_aggregation_and_compass_orientation():
    data = pd.DataFrame({
        "height": [19, 19, 19, 59], "ti": [0.1, 0.2, 0.6, 0.4],
        "wsp_bin": ["5-6"] * 4, "wdir_bin": ["350-360"] * 4,
    })
    original = data.copy(deep=True)
    median = plot_ti_polar_by_height(data, [19, 59])
    mean = plot_ti_polar_by_height(data, [19], ti_stat="mean")
    assert median.data[0].marker.color[0] == pytest.approx(0.2)
    assert mean.data[0].marker.color[0] == pytest.approx(0.3)
    assert median.data[0].theta[0] == 355
    assert median.data[0].r[0] == 5.5
    assert median.layout.polar.angularaxis.rotation == 90
    assert median.layout.polar.angularaxis.direction == "clockwise"
    assert median.data[0].marker.coloraxis == median.data[1].marker.coloraxis
    pd.testing.assert_frame_equal(data, original)
    with pytest.raises(ValueError, match="heights"):
        plot_ti_polar_by_height(data, [120])
    with pytest.raises(ValueError, match="No valid"):
        plot_ti_polar_by_height(data.iloc[:0])


def test_north_sector_is_not_dropped():
    data = pd.DataFrame({"wind_direction": [0, 355, 360, -1, 361], "height": [19] * 5})
    binned, _ = bin_wdir(data)
    assert list(binned["wdir_bin"].iloc[:3]) == ["0-10", "350-360", "350-360"]
    assert binned["wdir_bin"].iloc[3:].isna().all()


def test_statistics_selects_nearest_height_and_preserves_values():
    times = pd.date_range("2020-05-01", periods=2, freq="10min")
    frames = [pd.DataFrame({"Horizontal Wind Speed (m/s) at 139m": [i, i+1]},
                           index=times) for i in [5, 8, 3, 1]]
    fig, height = plot_wind_statistics(*frames, height=120)
    assert height == 139
    assert "139 m" in fig.layout.title.text
    for trace, frame in zip(fig.data, frames):
        np.testing.assert_array_equal(trace.y, frame.iloc[:, 0])
        np.testing.assert_array_equal(pd.to_datetime(trace.x), times)


def test_metmast_month_overlap_and_measurement_filtering(tmp_path):
    times = pd.date_range("2020-05-14", periods=4, freq="D")
    ds = xr.Dataset(
        {"F": (("time", "z"), [[5], [6], [7], [8]]),
         "SF": (("time", "z"), [[1]] * 4),
         "D": (("time", "z"), [[180]] * 4),
         "TA": (("time", "z"), [[np.nan]] * 4)},
        coords={"time": times, "z": [100.0]},
    )
    path = tmp_path / "mast_202005.nc"
    ds.to_netcdf(path, engine="scipy")
    files = met_finder(tmp_path, "2020-05-15", "2020-05-17")
    assert files == [str(path)]
    assert met_finder(tmp_path, "2020-06-01") == []
    assert met_finder(tmp_path, end_date="2020-05-01") == []
    frame = read_met(path, "2020-05-15", "2020-05-17")
    assert list(frame["wind_speed"]) == [6, 7]
    assert list(frame["height"]) == [100, 100]
    assert frame["air_temp"].isna().all()


@pytest.mark.parametrize("folder", ["lidar_data", "lidar_data_10min"])
def test_runner_with_sample_data(tmp_path, monkeypatch, folder):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("runner", root / "run_wind_analysis.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = json.loads((root / "input_files/input_config_extended.json").read_text())
    config.update(data_folder=str(root / "tests" / folder), save_dir=str(tmp_path))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    captured = {}

    def capture_html(fig, path, *args, **kwargs):
        captured[Path(path).name] = fig

    monkeypatch.setattr(go.Figure, "write_html", capture_html)
    monkeypatch.setattr(go.Figure, "show", lambda *a, **kw: pytest.fail("show_plot=False ignored"))
    runner.run_program_from_input(config_path)
    assert {"TI_polar_by_height.html", "stats_139m.html", "TI_boxplot.html",
            "shear_plot.html", "TI_vs_wsp_139m.html"} <= captured.keys()
    polar = captured["TI_polar_by_height.html"]
    assert len(polar.data) == 4
    assert all(len(trace.r) > 0 for trace in polar.data)
    times = pd.to_datetime(captured["TI_timeseries_139m.html"].data[0].x)
    assert times.min() >= pd.Timestamp("2020-05-01")
    assert times.max() < pd.Timestamp("2020-05-03")
