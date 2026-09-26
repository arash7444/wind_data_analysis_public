# Wind Data Analysis


![Python](https://img.shields.io/badge/python-3.10+-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![Status](https://img.shields.io/badge/status-active--development-orange)
![CI](https://github.com/arash7444/wind_data_analysis_public/actions/workflows/ci-pipeline.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey)


`Wind Data Analysis` is a Python tool for analyzing wind measurement data, with a current focus on **LiDAR-based wind data processing** (met-mast support planned for future versions). 

This tool is designed to support wind engineering workflows such as load validation, site assessment, and LiDAR-based analysis.

The package helps you:

- read KNMI LiDAR CSV files,
- filter files by date range,
- compute wind statistics from raw or 10-minute LiDAR data,
- build wind-speed profiles across heights,
- calculate **vertical wind shear**,
- calculate **turbulence intensity (TI)**,
- prepare data for plotting and further analysis.

The repository is organized as a Python package with a `src/` layout and includes tests and example datasets for development and validation.  

## Current status

At the moment, the tool is mainly centered on **KNMI LiDAR workflows**.  
**Please note** that this project is under active development and new features are continuously being added.


## Main features

### 1. Read KNMI LiDAR files
The package can:

- search a folder recursively for KNMI LiDAR CSV files,
- filter files between a start date and end date,
- read the files into pandas DataFrames with a time index.

### 2. Compute LiDAR statistics
The package can detect whether the LiDAR data is:

- **high-frequency** raw data, or
- **low-frequency / 10-minute** data,

and then compute or extract:

- mean wind speed,
- max wind speed,
- min wind speed,
- standard deviation.

### 3. Build wind profiles
The tool can reorganize multi-height wind-speed measurements into profiles, where:

- the index is time,
- the columns are heights,
- the values are wind speeds.

This is useful for shear analysis and profile-based visualization.

### 4. Calculate vertical wind shear
The package includes power-law shear fitting:


`U(z) = U_ref * (z / z_ref)^α`

It estimates:

- the shear exponent `alpha`,
- the standard error of `alpha`,
- rolling median and rolling mean values.

### 5. Calculate turbulence intensity (TI)
The package calculates turbulence intensity using:

`TI = σ / U`

It also supports:

- TI by height,
- TI in wind-speed bins,
- TI with wind-direction context.


## Requirements

- Python 3.10+
- OS: Windows, Linux, or macOS

Main dependencies:

- `numpy`, `pandas`, `scipy`, `xarray`
- `matplotlib`, `seaborn`, `plotly`, `kaleido`
- `streamlit`
- `pytest` (for tests)



## Installation

### 1. Create and activate a virtual environment

Windows (PowerShell or CMD):

```bash
python -m venv .venv
.venv\Scripts\activate
```

Windows (Anaconda Prompt):

```bash
conda create -n wind_analysis python=3.10
conda activate wind_analysis
```

Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies and package

```bash
git clone https://github.com/arash7444/wind_data_analysis_public.git
cd wind_data_analysis_public
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. Test the installation

```bash
pytest tests
```

## Quick Start

Use the JSON config in `input_files/input_config.json` and run:

```bash
python run_wind_analysis.py
```

This script reads `input_files/input_config.json` by default and writes plots to `outputs/`.

## Configuration File

The input JSON controls what analysis runs.

Example (`input_files/input_config.json`):

```json
{
    "data_folder": "tests/lidar_data",
    "start_date": "2020-05-01",
    "end_date": "2020-05-03",
    "features": ["shear", "ti"],
    "hub_height": 120.0,
    "shear_window": 6,
    "show_plot": true,
    "save_dir": "outputs"
}
```


## Usage
### Option A: Main runner (default config path)

```bash
python run_wind_analysis.py
```

Behavior:

- Reads `input_files/input_config.json`
- Supports `ti` and `shear`
- Saves interactive Plotly HTML outputs to `save_dir`

### Option B: Streamlit App

You can explore inputs interactively via Streamlit:

```bash
streamlit run simple_gui.py
```

## Polar plots and wind statistics

The runner and `simple_gui.py` now support two additional feature choices:

- `ti_polar`: TI by wind direction and reference wind-speed bin at selected heights.
- `stats`: mean, maximum, minimum and standard deviation of wind speed over time.

Try all features with the bundled LiDAR data, from the repository root:

```bash
python run_wind_analysis.py input_files/input_config_extended.json
```

This saves interactive HTML plots in `outputs/extended/`, including
`TI_polar_by_height.html` and `stats_139m.html`. Open them in a browser.
The example uses `show_plot: false`, so no browser windows open automatically.
Running `python run_wind_analysis.py` still uses the original configuration.

Add the feature names to your JSON `features` list and set these options:

| Setting | Meaning | Default when omitted |
|---|---|---|
| `stats_height` | Requested height for the four statistics; the nearest common measured height is reported in the plot and output filename | `hub_height` |
| `polar_heights` | List of exact measured heights, such as `[19, 59, 139, 199]`; unavailable heights produce an error | All heights with valid binned TI data |
| `polar_stat` | `median` or `mean` TI within each speed/direction bin | `median` |
| `polar_ncols` | Positive integer giving the number of polar subplot columns | `2` |

Polar angle is wind direction at each plotted height, with north at the top
and angles increasing clockwise. Radius is the centre of the **reference-height
wind-speed bin**, in m/s; it is not necessarily the wind speed at the plotted
height. `calc_ti` selects the nearest available reference height to `hub_height`.
Colour represents dimensionless TI with a shared scale across panels. Empty
bins are omitted. The existing speed bins cover 0–30 m/s and direction bins
now cover the full 0–360° range.

In Streamlit, select `stats` and/or `ti_polar` under **Features**. Additional
controls appear in the sidebar. Polar heights are entered as comma-separated
numbers; leave the field blank to plot all available heights.

For Python use, both plotting functions return figures without displaying them:

```python
from wind_data_analysis.plotting import plot_ti_polar_by_height

# ti_values comes from calc_ti(...).
fig = plot_ti_polar_by_height(ti_values.ti_raw, heights=[19, 139])
fig.show()  # Or fig.write_html("ti_polar.html")
```

## Experimental met-mast reader

The earlier KNMI NetCDF reader is available for Python use. It is **not yet
connected to the LiDAR runner or GUI** and has only been tested with synthetic
NetCDF data; validate it with a representative KNMI file before engineering use.

```python
from wind_data_analysis.data_reader import met_finder, read_met

files = met_finder("path/to/metmast", "2020-05-15", "2020-06-01")
frames = [read_met(path, "2020-05-15", "2020-06-01") for path in files]
```

Date bounds include the start and exclude the end. Monthly files overlapping
the requested period are included, then `read_met` filters the actual measurement
times. The reader expects `z`, `F`, `SF`, `D` and `time`; it also reads `SD`, `TA`
and `Q` when present. Missing optional weather values do not discard valid wind
measurements. Values retain their input units; no quality-flag filtering is
applied. Some NetCDF formats may require an additional xarray backend such as
`netCDF4` installed in your environment.

These features were adapted from the original `wind_data_analysis` project's
polar plot, v2 runner and met-mast reader. The additional Streamlit prototypes
with generated data are not used by this application.

## Demo

### Streamlit Interface

![Streamlit GUI](docs/images/streamlit_gui.png)

### Interactive Plot (Plotly)

![Plotly Output](docs/images/plotly_output.png)
