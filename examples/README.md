# Runnable examples

These examples exercise the current public package capabilities. They use the real package functions and the small KNMI sample files under `tests/`. Machine-learning work is intentionally outside this repository's supported workflow.

All commands below assume the current directory is the repository root. Generated files are written below `outputs/examples/`, which is ignored by Git.

## Installation

The current branch uses `uv`, `pyproject.toml`, and the committed `uv.lock`. Python 3.10 or newer is declared. `uv` creates and manages `.venv` automatically.

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync --locked --dev
```

On Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --locked --dev
```

Run these commands from the repository root. `uv sync` installs the project itself so scripts outside `src/` can import `wind_data_analysis`.

## Examples

### Data loading and cleaning

```bash
uv run python examples/demo_data_loading.py
```

Uses one raw LiDAR CSV and one included KNMI met-mast NetCDF file. It demonstrates file discovery, timestamp parsing, height discovery, missing-value inspection, the currently standalone wind-speed cleaner, and the met-mast reader used by the comparison workflow.

Expected output:

- a short summary in the terminal;
- `outputs/examples/data_loading/cleaned_lidar_preview.csv`;
- `outputs/examples/data_loading/metmast_preview.csv`.

The included NetCDF files work with the installed xarray environment. Other NetCDF encodings may require an optional backend such as `netCDF4`.

### Statistics and wind profiles

```bash
uv run python examples/demo_statistics_and_profiles.py
```

Uses both bundled LiDAR formats: raw/high-frequency data and pre-averaged 10-minute data. It shows how `compute_lidar_stats` dispatches between them and how average wind speed is reshaped into time-by-height profiles.

Expected output: raw and 10-minute CSV previews under `outputs/examples/statistics_profiles/`.

### LiDAR and met-mast comparison

```bash
uv run python examples/demo_metmast_comparison.py
```

Uses the bundled 7 June 2020 measurements and the shared production functions
to discover nearby one-to-one height pairs, validate the 10-minute timestamps,
calculate metrics, and write tidy CSV and interactive HTML outputs under
`outputs/examples/metmast/`. Set the example's `SHOW_PLOTS` and `SAVE_PLOTS`
constants independently to control display and HTML output.

### Turbulence intensity and shear

```bash
uv run python examples/demo_turbulence_and_shear.py
```

Uses two days of raw LiDAR samples, aggregates them to 10-minute statistics, calculates TI and wind-speed/direction bins, and fits the power-law shear exponent at every timestamp.

Expected output: TI and shear CSV files under `outputs/examples/turbulence_shear/`.

### Visualization

```bash
uv run python examples/demo_visualization.py
```

Uses the bundled pre-averaged LiDAR sample. It creates subplot and single-panel
wind-statistics figures plus multi-height TI polar plots. Set `SHOW_PLOTS` and
`SAVE_PLOTS` independently; the defaults save HTML without opening windows.

Expected output: interactive HTML files under `outputs/examples/visualization/`.

### Complete workflow

```bash
uv run python examples/demo_complete_workflow.py
```

Runs the main reusable workflow: file discovery, reading, validation, statistics, height profiles, TI, shear, tabular exports, and Plotly visualization. Machine learning is outside the supported workflow.
Its `SHOW_PLOTS` and `SAVE_PLOTS` constants independently control Plotly
display and HTML saving; tabular outputs remain part of the workflow.

Expected output under `outputs/examples/complete_workflow/`:

- `summary.json`;
- `median_ti_by_height.csv` and `shear_alpha.csv`;
- wind-statistics, TI-polar, TI-boxplot, and shear HTML figures.

## Existing application entry points

The examples complement rather than replace the existing applications:

```bash
uv run python run_wind_analysis.py input_files/input_config_extended.json
uv run python run_wind_analysis.py input_files/input_config_metmast_data.json
uv run python run_wind_analysis.py input_files/input_config_metmast_comparison.json
uv run streamlit run simple_gui.py
```

The runner examples demonstrate LiDAR-only, met-mast-only, and combined inputs.
The Streamlit command starts the interactive application and requires a browser
session.

## Verification

Run the automated suite with:

```bash
uv run --locked pytest
```

`tests/test_known_limitations.py` now contains a passing regression for aligned
TI mean-speed validation: mean speed must be finite, greater than zero, and at
most 99 m/s.
