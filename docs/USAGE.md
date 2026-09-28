# Usage guide

Run all commands from the repository root after completing the
[installation](../README.md#installation).

## 1. Use the Streamlit app

1. Start the app:

   ```bash
   uv run streamlit run simple_gui.py
   ```

2. Open the local URL printed by Streamlit if a browser does not open
   automatically.
3. Select one or more features in the sidebar:
   - `ti`: turbulence-intensity plots.
   - `shear`: shear plots and wind profiles.
   - `stats`: mean, maximum, minimum, and standard-deviation plots at the
     nearest common measured height.
   - `ti_polar`: TI grouped by measurement-height direction and
     reference-height wind-speed bin.
   - `metmast_data`: a met-mast summary and tidy CSV download.
   - `metmast_comparison`: paired-height LiDAR/met-mast diagnostics, metrics,
     and plots.
4. Complete the input controls that appear for the selected features:
   - LiDAR features require a LiDAR CSV file or folder. LiDAR start and end
     dates are optional; the start is inclusive and the end is exclusive.
   - Met-mast features require a NetCDF file or folder. Its date bounds are
     also optional, inclusive-start, and exclusive-end.
   - Raw LiDAR uses the minimum coverage percentage shown in the sidebar; the
     default is 80%.
   - `stats` accepts a requested height and uses the nearest height common to
     all four statistic frames.
   - `ti_polar` accepts comma-separated exact measured heights. Leave the
     field blank to use all available heights.
   - `metmast_comparison` accepts the maximum difference between paired
     LiDAR and mast heights; the default is 2 m.
5. Select **Run analysis**. Results appear in the page. Met-mast-only data can
   be downloaded as a tidy CSV; the other Streamlit results are rendered as
   tables and Plotly charts.

The app currently fixes the met-mast timestamp normalization tolerance at 30
seconds. Use the JSON runner when a different tolerance is required.

## 2. Run an analysis from JSON

### Start from an example

The runner accepts one optional positional config path. Without one it uses
`input_files/input_config.json`.

```bash
uv run python run_wind_analysis.py
uv run python run_wind_analysis.py input_files/input_config_extended.json
uv run python run_wind_analysis.py input_files/input_config_metmast_data.json
uv run python run_wind_analysis.py input_files/input_config_metmast_comparison.json
```

To create a custom LiDAR run, copy an example and edit it:

```json
{
  "data_folder_lidar": "tests/lidar_data",
  "start_date_lidar": "2020-05-01",
  "end_date_lidar": "2020-05-03",
  "min_lidar_raw_coverage_percent": 80.0,
  "features": ["ti", "shear", "stats", "ti_polar"],
  "hub_height": 139.0,
  "stats_height": 139.0,
  "polar_heights": [19, 59, 139, 199],
  "polar_stat": "median",
  "polar_ncols": 2,
  "shear_window": 6,
  "extra_plots": true,
  "show_plots": false,
  "save_plots": true,
  "save_dir": "outputs/my_analysis"
}
```

Then pass its path to the runner:

```bash
uv run python run_wind_analysis.py path/to/config.json
```

### Plot display and saving are independent

`show_plots` and `save_plots` are read separately and both default to `true`:

| `show_plots` | `save_plots` | Result |
|---|---|---|
| `true` | `true` | Display figures and write HTML files. |
| `true` | `false` | Display figures without writing HTML files. |
| `false` | `true` | Write HTML files without opening figures. |
| `false` | `false` | Build figures without displaying or saving them. |

The legacy `show_plot` key is accepted only when `show_plots` is absent. New
configs should use `show_plots`.

The runner creates `save_dir` even when `save_plots` is `false`. The
`metmast_data` feature always writes `metmast_data.csv`, and
`metmast_comparison` always writes its matched-data, metrics, and height-pair
CSV files; `save_plots` controls Plotly HTML only.

### Supported settings

| Setting | Meaning | Default |
|---|---|---|
| `features` | One or more of `ti`, `shear`, `stats`, `ti_polar`, `metmast_data`, `metmast_comparison`. | Required, no default feature |
| `data_folder_lidar` | LiDAR CSV file or folder; required by LiDAR features. | — |
| `start_date_lidar`, `end_date_lidar` | Optional LiDAR interval `[start, end)`. | Unbounded |
| `min_lidar_raw_coverage_percent` | Required valid raw samples per 10-minute bin, from 0 to 100. | `80.0` |
| `data_folder_Metmast` | Met-mast NetCDF file or folder; required by met-mast features. | — |
| `start_date_Metmast`, `end_date_Metmast` | Optional met-mast interval `[start, end)`. | Unbounded |
| `hub_height` | Reference height used by TI and some TI/shear plots. | `120.0` |
| `shear_window` | Centered rolling window used for shear mean and median. | `6` |
| `stats_height` | Requested statistics height; the nearest common height is used. | `hub_height` |
| `polar_heights` | Exact measured heights for TI polar panels. | All available heights |
| `polar_stat` | `median` or `mean` within each TI polar bin. | `median` |
| `polar_ncols` | Positive number of TI polar subplot columns. | `2` |
| `max_height_difference_m` | Largest allowed LiDAR/mast pairing difference. | `2.0` |
| `timestamp_tolerance_seconds` | Largest mast timestamp offset normalized to the 10-minute grid. | `30.0` |
| `extra_plots` | Build the runner's additional TI and shear figures. | `true` |
| `show_plots` | Display generated Plotly figures. | `true` |
| `save_plots` | Save generated Plotly figures as HTML. | `true` |
| `save_dir` | Output directory. | `outputs` |

Legacy LiDAR keys `data_folder`, `start_date`, and `end_date` remain accepted,
but instrument-specific keys avoid ambiguity.

### Met-mast comparison quality checks

Comparison pairs discovered heights one-to-one, normalizes acceptable mast
timestamp offsets to the 10-minute grid, and aligns exact timestamps within
the common period. Before metrics are calculated, LiDAR and mast wind speeds
are independently required to be numeric, finite, and within 0–99 m/s
inclusive. The comparison does not apply unit conversion, instrument-specific
quality flags, vertical interpolation, or calibration.

## 3. Use the Python API

The following example uses the bundled pre-averaged LiDAR files. It finds and
loads the data, calculates four statistics, combines files, builds a
time-by-height wind profile, and creates both supported statistics layouts.

```python
from pathlib import Path

import numpy as np

from wind_data_analysis.data_reader import (
    find_KNMI_LiDAR_files,
    read_KNMI_LiDAR,
)
from wind_data_analysis.plotting import plot_wind_statistics
from wind_data_analysis.process import (
    compute_lidar_stats,
    concatenate_wind_stats,
    wind_height_profile,
)
from wind_data_analysis.utils import lidar_height

files = find_KNMI_LiDAR_files(
    Path("tests/lidar_data_10min"),
    start_date="2020-05-01",
    end_date="2020-05-03",
)

per_file_stats = []
per_file_heights = []
for file_path in files:
    lidar_data = read_KNMI_LiDAR(file_path)
    per_file_stats.append(compute_lidar_stats(lidar_data))
    per_file_heights.append(lidar_height(lidar_data))

average = concatenate_wind_stats([stats.avg for stats in per_file_stats])
maximum = concatenate_wind_stats([stats.max for stats in per_file_stats])
minimum = concatenate_wind_stats([stats.min for stats in per_file_stats])
standard_deviation = concatenate_wind_stats(
    [stats.std for stats in per_file_stats]
)

heights = np.unique(np.concatenate(per_file_heights))
profiles = wind_height_profile(average, heights)

subplot_figure, single_figure, selected_height = plot_wind_statistics(
    average,
    maximum,
    minimum,
    standard_deviation,
    height=120.0,
)

print(files)
print(profiles.head())
print(f"Plots use the nearest common height: {selected_height:g} m")
subplot_figure.show()
single_figure.write_html("wind_statistics.html")
```

Run the saved example as a script with:

```bash
uv run python path/to/example.py
```

`compute_lidar_stats` returns a `LidarStats` object. Its primary attributes are
`avg`, `max`, `min`, and `std`; raw input can additionally populate coverage
and invalid-sample diagnostics. `plot_wind_statistics` does not display or save
anything itself. It returns exactly three values in this order:

1. a four-row Plotly subplot figure;
2. a single-panel Plotly figure with all four statistics overlaid;
3. the numeric height selected as nearest among the heights common to all four
   input frames.

For more API details and known limitations, see the
[project guide](PROJECT_GUIDE.md) and the [runnable examples](../examples/README.md).
