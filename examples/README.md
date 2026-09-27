# Runnable examples

These examples exercise the capabilities available on the current `dev` branch. They use the real package functions and the small KNMI sample files under `tests/`. Machine-learning work is intentionally excluded because it exists only on the separate `ML` branch.

All commands below assume the current directory is the repository root. Generated files are written below `outputs/examples/`, which is ignored by Git.

## Installation

The current branch uses standard Python packaging with `pip`, `requirements.txt`, and `pyproject.toml`. Python 3.10 or newer is declared.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

An editable install is important: the scripts live outside `src/` and import `wind_data_analysis` as an installed package.

## Examples

### Data loading and cleaning

```bash
python examples/demo_data_loading.py
```

Uses one raw LiDAR CSV and one included KNMI met-mast NetCDF file. It demonstrates file discovery, timestamp parsing, height discovery, missing-value inspection, the currently standalone wind-speed cleaner, and the experimental met-mast reader.

Expected output:

- a short summary in the terminal;
- `outputs/examples/data_loading/cleaned_lidar_preview.csv`;
- `outputs/examples/data_loading/metmast_preview.csv`.

The included NetCDF files work with the installed xarray environment. Other NetCDF encodings may require an optional backend such as `netCDF4`.

### Statistics and wind profiles

```bash
python examples/demo_statistics_and_profiles.py
```

Uses both bundled LiDAR formats: raw/high-frequency data and pre-averaged 10-minute data. It shows how `compute_lidar_stats` dispatches between them and how average wind speed is reshaped into time-by-height profiles.

Expected output: raw and 10-minute CSV previews under `outputs/examples/statistics_profiles/`.

### Turbulence intensity and shear

```bash
python examples/demo_turbulence_and_shear.py
```

Uses two days of raw LiDAR samples, aggregates them to 10-minute statistics, calculates TI and wind-speed/direction bins, and fits the power-law shear exponent at every timestamp.

Expected output: TI and shear CSV files under `outputs/examples/turbulence_shear/`.

### Visualization

```bash
python examples/demo_visualization.py
```

Uses the bundled pre-averaged LiDAR sample. It creates a four-panel wind-statistics plot and multi-height TI polar plots without opening browser windows.

Expected output: interactive HTML files under `outputs/examples/visualization/`.

### Complete workflow

```bash
python examples/demo_complete_workflow.py
```

Runs the main reusable workflow: file discovery, reading, validation, statistics, height profiles, TI, shear, tabular exports, and Plotly visualization. It does not run ML because ML is not present on the current branch.

Expected output under `outputs/examples/complete_workflow/`:

- `summary.json`;
- `median_ti_by_height.csv` and `shear_alpha.csv`;
- wind-statistics, TI-polar, TI-boxplot, and shear HTML figures.

## Existing application entry points

The examples complement rather than replace the existing applications:

```bash
python run_wind_analysis.py input_files/input_config_extended.json
streamlit run simple_gui.py
```

The first command writes the configured Plotly figures to `outputs/extended/`. The second starts the interactive Streamlit application and requires a browser session.

## Verification

Run the automated suite with:

```bash
pytest
```

One test in `tests/test_known_limitations.py` is deliberately marked `xfail`. It records the known negative-wind-speed TI issue without changing the current scientific behavior.
