# Wind Data Analysis

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![Status](https://img.shields.io/badge/status-active--development-orange)
![CI](https://github.com/arash7444/wind_data_analysis_public/actions/workflows/ci-pipeline.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

`wind_data_analysis` processes KNMI LiDAR and met-mast measurements for wind
engineering workflows. It provides a JSON-configured batch runner, a Streamlit
app, and reusable Python functions.

## Key features

- Discover and read KNMI LiDAR CSV files and met-mast NetCDF files.
- Convert raw LiDAR observations to 10-minute statistics or use existing
  10-minute statistics.
- Calculate turbulence intensity and power-law vertical wind shear.
- Create wind profiles, time-series statistics, and TI polar plots with Plotly.
- Pair nearby LiDAR and met-mast heights and compare aligned 10-minute wind
  speeds.
- Apply a shared numeric, finite, inclusive 0–99 m/s wind-speed validity rule;
  raw LiDAR additionally supports a configurable minimum sample coverage.

## Installation

Python 3.10 or newer is required. Install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/arash7444/wind_data_analysis_public.git
cd wind_data_analysis_public
uv sync --locked --dev
uv run --locked pytest tests
```

## Quick start

From the repository root, run the bundled configuration:

```bash
uv run python run_wind_analysis.py input_files/input_config.json
```

It analyzes the bundled LiDAR data and, with the supplied settings, displays
and saves interactive Plotly figures under `outputs/`.

## Guides

- [Usage guide](docs/USAGE.md): Streamlit, JSON configuration, and Python API
  walkthroughs.
- [Project guide](docs/PROJECT_GUIDE.md): detailed modules, functions, data
  flow, limitations, and development notes.
- [Runnable examples](examples/README.md): focused demonstrations using the
  bundled data.

## Screenshots

### Streamlit app

![Streamlit app showing LiDAR inputs and a turbulence-intensity plot](docs/images/streamlit_gui.png)

### Interactive Plotly output

![Plotly wind-speed profiles at selected times](docs/images/plotly_output.png)
