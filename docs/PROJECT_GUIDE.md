# Wind Data Analysis project guide

## Project purpose and current scope

`wind_data_analysis` is a Python package and pair of applications for processing KNMI wind measurements, primarily ZephIR LiDAR CSV files. It discovers measurements by date, converts raw or already averaged inputs into a common set of 10-minute statistics, organizes measurements into height profiles, calculates turbulence intensity (TI) and power-law vertical shear, and creates interactive Plotly figures.

This guide describes the current public package without relying on a stale branch name or commit identifier. Machine-learning code is intentionally outside the supported workflow.

Met-mast support reads KNMI monthly NetCDF files and compares 10-minute mean horizontal wind speed with dynamically paired LiDAR heights through the runner, Streamlit app, and reusable package functions.

## Main capabilities

1. Recursively find LiDAR CSV files and select files whose filename dates fall in an inclusive-start, exclusive-end range.
2. Read KNMI LiDAR CSV files, strip column whitespace, parse `Time and Date`, and use parsed time as the DataFrame index.
3. Detect raw/high-frequency versus pre-averaged/low-frequency LiDAR layouts.
4. Resample raw wind speed and direction to 10-minute mean, maximum, minimum, and standard deviation; extract corresponding columns from 10-minute inputs.
5. Concatenate per-file results and reshape horizontal wind speeds into time-by-height profiles.
6. Fit a power-law shear exponent and calculate centered rolling mean and median series.
7. Calculate TI as wind-speed standard deviation divided by mean wind speed, reshape it into tidy form, attach speed/direction context, and bin it.
8. Plot wind statistics and TI by direction/reference-speed bin.
9. Run the workflow through a JSON-configured command-line script or a Streamlit GUI.
10. Read KNMI met-mast NetCDF data, align validated 10-minute timestamps, dynamically pair nearby heights, and calculate per-height wind-speed comparison metrics.

## End-to-end data flow

```text
LiDAR folder + [start, end)
        |
        v
find_KNMI_LiDAR_files
        |
        v
read_KNMI_LiDAR (one DataFrame per CSV, DatetimeIndex named Time)
        |
        v
compute_lidar_stats
    | raw input                 | pre-averaged input
    v                           v
10-minute resampling       select/rename statistic columns
        \                       /
         v                     v
       LidarStats(avg, max, min, std), one per file
                        |
                        v
             concatenate_wind_stats
                        |
            +-----------+-------------+
            |                         |
            v                         v
 wind_height_profile               calc_ti
            |                         |
            v                         v
       calc_shear             tidy and binned TI
            |                         |
            +-----------+-------------+
                        v
           Plotly HTML / Streamlit charts / CSV reports
```

For comparison, `met_finder` selects monthly `.nc` files and `read_met` creates tidy measurements. `compare_lidar_to_metmast` discovers heights, performs deterministic one-to-one pairing, validates and normalizes the 10-minute timestamps, aligns each pair independently, and returns matched data, metrics, and diagnostics used by both applications.

## Repository structure

| Path | Role |
|---|---|
| `src/wind_data_analysis/` | Reusable package code. |
| `src/wind_data_analysis/data_reader/` | LiDAR/met-mast input and the standalone cleaner. |
| `src/wind_data_analysis/process/` | Statistics, reshaping, binning, TI, and shear calculations. |
| `src/wind_data_analysis/plotting/` | Reusable Plotly figure builders. |
| `src/wind_data_analysis/utils/` | Column/filename parsing helpers. |
| `run_wind_analysis.py` | JSON-configured batch application; saves Plotly HTML. |
| `simple_gui.py` | Streamlit application; duplicates some batch orchestration and plotting. |
| `input_files/` | Example runner configurations. |
| `tests/` | Pytest regression tests plus representative LiDAR and met-mast data. |
| `examples/` | Small runnable demonstrations added for learning and verification. |
| `demo/` | Untracked exploratory study code that predated this guide; it is not part of the verified example suite. |
| `docs/` | Documentation and screenshots. |
| `.github/workflows/ci-pipeline.yml` | uv-based CI test definition for Python 3.10–3.14 on Windows and Ubuntu. |
| `pyproject.toml`, `uv.lock` | Project metadata, dependency declarations, and the reproducible uv lockfile. |

There are no notebooks in the repository.

## Module dependencies

- Readers use `utils.lidar_file_parsing` for filename dates.
- Statistics use the reader exports and `utils.lidar_height`; their long example block also imports concatenation/profile helpers.
- TI uses `lidar_height`, `bin_wind`, and `bin_wdir`.
- Shear consumes the output of `wind_height_profile` and otherwise has unnecessary application-level imports.
- Plotting consumes the DataFrames/dataclasses produced by the process modules but does not save or show figures itself.
- Both applications independently orchestrate reader, statistics, profile, TI, shear, and plotting functions.

Several package modules import plotting, filesystem, or scientific names that are unused. These do not change behavior but make the intended module boundaries less clear.

## Source-file reference

### `data_reader/read_KNMI_LiDAR.py`

Responsibility: locate and read KNMI LiDAR CSV files.

#### `find_KNMI_LiDAR_files(file_folder, start_date=None, end_date=None) -> list`

- Parameters: a `str`/`Path` file or directory; optional pandas-compatible date bounds.
- Returns: paths as strings (or the original file object when a single file is passed), in filesystem traversal order.
- Behavior: recursively finds case-insensitive `.csv` suffixes. When `start_date` is supplied, it extracts a date from the filename and keeps `start_date <= date < end_date`.
- Called by: both applications, examples, tests, and commented module demonstrations.
- Side effects: prints when passed a file or when a filename has no recognized date.
- Assumptions: dated filtering expects `YYYYMMDD`, `YYYY-MM-DD`, or `YYYY_MM_DD` somewhere in the filename.
- Limitations: returned order is not explicitly sorted, and a nonexistent path silently produces an empty list. Start-only and end-only filtering are supported independently.
- Example: `find_KNMI_LiDAR_files("tests/lidar_data", "2020-05-01", "2020-05-03")`.

#### `read_KNMI_LiDAR(file_path) -> pd.DataFrame`

- Parameters: LiDAR CSV path as `str` or `Path`.
- Returns: all CSV columns with stripped names and a parsed `DatetimeIndex` named `Time`; the original `Time and Date` column remains.
- Called by: statistics tests, applications, examples, and module demonstrations.
- Side effects: reads disk; pandas can emit a fragmentation performance warning for wide low-resolution files.
- Assumptions: line one is metadata, line two is the header, and timestamps exactly match `%d/%m/%Y %H:%M:%S`.
- Limitations: no schema validation, timezone attachment, sorting, duplicate handling, quality-flag filtering, or numeric conversion.
- Example: `read_KNMI_LiDAR("tests/lidar_data/ZephIR_Cabauw_ZP738_raw_20200501_v1.CSV")`.

### `data_reader/clean_data.py`

#### `clean_data(data_in) -> pd.DataFrame`

- Responsibility: copy a DataFrame, coerce every column containing `Wind Speed` to numeric, and replace values outside 0–50 m/s with `NaN`.
- Parameters/return: input and copied output are pandas DataFrames.
- Called by: the data-loading example; imported but disabled in `stats_func.py`.
- Side effects: none on the caller's object.
- Assumptions: substring matching identifies the intended speed columns.
- Limitations: not part of the production runner, does not inspect quality flags or direction, and the 0–50 m/s thresholds are hard-coded.
- Example: `cleaned = clean_data(raw_frame)`.

### `data_reader/read_KNMI_metmast.py`

Responsibility: experimental monthly NetCDF discovery and tidy conversion.

#### `_date_bounds(start_date, end_date)`

- Parameters: optional pandas-compatible bounds.
- Returns: two `pd.Timestamp` objects or `None` values.
- Called by: `met_finder` and `read_met` only.
- Raises: `ValueError` when both bounds exist and start is not before end.
- Example: `_date_bounds("2020-05-01", "2020-06-01")`.

#### `met_finder(pth_met_base, start_date=None, end_date=None) -> list[str]`

- Parameters: a NetCDF path/folder and optional time bounds.
- Returns: sorted string paths for matching `.nc` files.
- Behavior: understands filename suffixes `_YYYYMM.nc` and selects months overlapping `[start, end)`; unrecognized names are retained for later time filtering.
- Called by: tests, the data-loading example, and direct Python users.
- Side effects: recursive filesystem scan.
- Limitations: selection uses filenames, not file metadata, and only lower-case `.nc` is found.
- Example: `met_finder("tests/metmast_data", "2020-05-01", "2020-06-01")`.

#### `read_met(met_nc_file, start_date=None, end_date=None) -> pd.DataFrame`

- Parameters: NetCDF path and optional exact measurement bounds.
- Returns: a time-indexed tidy DataFrame with `height`, `wind_speed`, `wind_speed_std`, `wind_direction`, and optional direction/weather columns.
- Required variables: `z`, `F`, `SF`, `D`, and `time`; optional `SD`, `TA`, and `Q` are included when present.
- Called by: tests, the data-loading example, and direct users.
- Side effects: opens and closes a NetCDF dataset.
- Assumptions: xarray can decode the file and its variables follow the KNMI names.
- Limitations: no unit conversion, quality filtering, interpolation, or runner/GUI integration. Some files need an optional xarray backend.
- Example: `read_met("mast_202005.nc", "2020-05-01", "2020-05-02")`.

### `utils/lidar_file_parsing.py`

#### `extract_date_from_filename(filename) -> pd.Timestamp | None`

- Responsibility: find the first valid date in `YYYYMMDD`, `YYYY-MM-DD`, or `YYYY_MM_DD` form.
- Called by: `find_KNMI_LiDAR_files`.
- Side effects: none.
- Limitations: accepts a string rather than a declared `Path`; ignores other date forms and continues past invalid matches.
- Example: `extract_date_from_filename("lidar_20200501.csv")` returns `Timestamp('2020-05-01')`.

### `utils/utils.py`

#### `lidar_height(data) -> list`

- Responsibility: scan wind speed/direction column names for `at <integer>m` and return sorted unique heights.
- Parameters/return: pandas DataFrame to a NumPy array despite the `list` annotation.
- Called by: applications, statistics/TI modules, and examples.
- Side effects: copies the complete input DataFrame unnecessarily.
- Assumptions: integer heights and exact English column fragments.
- Limitations: decimal heights are not parsed and unrelated columns matching the pattern are included.
- Example: `lidar_height(lidar_data)`.

#### `NA_cols(df) -> list`

- Responsibility: return names of columns containing at least one missing value.
- Called by: the data-loading example; otherwise unused.
- Side effects: none.
- Example: `NA_cols(frame)`.

### `process/stats_func.py`

Defines `LidarStats`, a dataclass with `avg`, `max`, `min`, and `std` DataFrames
plus optional raw-sample coverage, invalid-sample counts, inferred interval,
expected samples per bin, and the applied coverage threshold.

#### `compute_lidar_stats(data, min_lidar_raw_coverage_percent=80.0) -> LidarStats`

- Responsibility: dispatch to high- or low-resolution processing based on whether any column contains `Horizontal Wind Speed Std.`.
- Called by: applications, tests, examples, and module demonstrations.
- Side effects: prints the detected input type.
- Limitation: format detection depends on one exact column-name fragment rather than explicit metadata/schema validation.
- Validity rule: horizontal wind speed must be numeric, finite, and within
  0--99 m/s inclusive.
- Example: `stats = compute_lidar_stats(read_KNMI_LiDAR(path), 80.0)`.

#### `compute_lidar_stats_highres(data) -> LidarStats`

- Parameters: time-indexed raw LiDAR DataFrame.
- Returns: 10-minute mean/max/min/std plus raw-speed diagnostics. Invalid
  speeds are masked before resampling. The per-file interval is the median
  positive timestamp difference; expected samples use round-half-up on
  `600 / interval_seconds`; coverage equal to the threshold passes.
- Called by: `compute_lidar_stats`.
- Side effects: prints all output column indexes and can warn when resampled standard deviations contain missing values.
- Assumptions: a valid `DatetimeIndex`; pandas numeric operations work for all selected wind columns.
- Limitations: arithmetic mean of circular wind direction is scientifically
  questionable near 0/360 degrees; direction behavior is unchanged pending a
  separate scientific policy. Maximum/minimum/std of direction and
  `Time_seconds` are not necessarily meaningful.
- Example: `compute_lidar_stats_highres(raw_data)`.

#### `compute_lidar_stats_lowres(data) -> LidarStats`

- Parameters: pre-averaged LiDAR DataFrame.
- Returns: selected mean/direction columns and speed max/min/std columns
  renamed to the mean speed convention. Pre-averaged mean-speed bins receive
  the 0--99 finite check; raw coverage is unavailable.
- Called by: `compute_lidar_stats`.
- Side effects: assertion failure when max/min/std shapes differ.
- Assumptions: exact KNMI column labels and matching statistic columns for every height.
- Limitations: no timestamp resampling; directions occur only in `avg`; the
  shape assertion does not verify height-by-height label correspondence.
- Example: `compute_lidar_stats_lowres(ten_minute_data)`.

### `process/concatenate_wind_stats.py`

#### `concatenate_wind_stats(stat_list) -> pd.DataFrame`

- Responsibility: concatenate per-file statistic DataFrames row-wise and sort the time index.
- Called by: both applications and examples.
- Returns: an empty DataFrame for an empty list.
- Limitations: duplicates are retained and differing schemas are unioned with missing values.
- Example: `all_average = concatenate_wind_stats([day1.avg, day2.avg])`.

### `process/wind_height_profile.py`

#### `wind_height_profile(lidar_data, height_lidar) -> pd.DataFrame`

- Responsibility: select horizontal-speed columns and map them to numeric height columns.
- Parameters: statistics DataFrame and iterable of heights.
- Returns: time-indexed, height-column DataFrame sorted on both axes.
- Called by: applications, examples, and statistics demonstration code.
- Side effects: none.
- Assumptions: columns use `Horizontal Wind Speed (m/s) at <integer>m`.
- Limitations: silently skips missing heights. The multi-column fallback averages matches; the source itself flags this as dangerous because overly broad matching could mix measurements. With the current exact pattern, normal KNMI horizontal columns generally produce one match.
- Example: `profiles = wind_height_profile(stats.avg, [19, 59, 139])`.

### `process/bin_wind.py`

#### `bin_wind(df) -> tuple[pd.DataFrame, pd.Series]`

- Responsibility: mutate a tidy TI DataFrame by adding 1 m/s reference-speed bins and per-bin counts.
- Inputs: DataFrame with `hub_wsp` and `height`.
- Returns: the same mutated DataFrame plus bin counts.
- Called by: `calc_ti` and exploratory code.
- Side effects: adds `wsp_bin` and `wsp_bincount` to the caller's object.
- Assumptions: each timestamp repeats once per height, so dividing row counts by the number of heights estimates sample counts.
- Limitations: hard-coded 0–30 m/s range; missing heights or uneven observations can make counts inaccurate; the return annotation incorrectly claims only a DataFrame.
- Example: `binned, counts = bin_wind(tidy_ti)`.

### `process/bin_wdir.py`

#### `bin_wdir(df) -> tuple[pd.DataFrame, pd.Series]`

- Responsibility: mutate a tidy TI DataFrame with 10-degree direction bins and return counts.
- Inputs: DataFrame with `wind_direction` and `height`.
- Called by: `calc_ti`, tests, and exploratory code.
- Side effects: adds `wdir_bin`.
- Assumptions/limitations: hard-coded 0–360 range and the same repeated-height count assumption as `bin_wind`; values outside the range become missing. With pandas' interval convention, 360 is included in `350-360`.
- Example: `binned, counts = bin_wdir(tidy_ti)`.

### `process/calc_turb.py`

Defines `TurbValues` with tidy `ti_raw`, median TI by height, and median TI by height/reference-speed bin.

#### `calc_ti(avg_val, std_val, hub_height=120.0) -> TurbValues`

- Parameters: mean and standard-deviation DataFrames sharing a named `Time` index; desired reference height as a number.
- Behavior: validates aligned mean-speed columns as finite, strictly positive,
  and at most 99 m/s before calculating `std / mean`; then reshapes TI/speed/
  direction to tidy rows, selects the nearest measured reference height, adds
  reference wind speed, applies speed/direction bins, retains `ti < 1`, and
  aggregates medians.
- Called by: both applications, plotting tests indirectly through the runner, and examples.
- Side effects: prints the selected reference height and a missing-TI warning.
- Assumptions: matching horizontal-speed columns in average/std frames; matching direction columns; integer height labels; index name exactly `Time` for `melt(id_vars="Time")`.
- Important meaning: `wsp_bin` is based on wind speed at the single selected reference height, while `wdir_bin` is based on direction at each row's own height.
- Limitations: rows with `ti >= 1` are discarded without retaining a rejection
  reason, and no minimum bin count is enforced. Wind-direction validity remains
  outside this rule and requires a separate scientific policy.
- Example: `ti = calc_ti(stats.avg, stats.std, hub_height=139)`.

### `process/calc_shear.py`

Defines `ShearValues` with raw alpha, standard error, rolling median, and rolling mean series. Type comments say DataFrame, but actual values are pandas Series.

#### `fit_alpha_with_uncertainty(heights_m, wsp) -> tuple[float, float, int]`

- Responsibility: regress `log(U)` on `log(z)` and return slope alpha, its residual standard error, and valid point count.
- Parameters: list/array heights and wind speeds.
- Called by: `calc_shear`.
- Assumptions: positive heights/speeds describe a power law; negative alpha is considered invalid and replaced by `NaN`.
- Limitations: does not explicitly validate equal array lengths; standard error is undefined at exactly two valid points and the calculation can emit divide-by-zero warnings; negative shear is removed even though it can occur physically under some atmospheric conditions.
- Example: `fit_alpha_with_uncertainty([10, 20, 40], [5, 6, 7])`.

#### `calc_shear(wsp_profiles, window=6) -> ShearValues`

- Parameters: time-by-height speed profile and integer-like rolling window.
- Returns: four time-indexed pandas Series in a dataclass.
- Called by: both applications and examples.
- Behavior: fits each row and computes centered rolling mean/median with `min_periods=3`.
- Side effects: none besides possible NumPy warnings from the fit.
- Limitations: Python-level row iteration may be slow for large datasets; `n_valid` is discarded; no confidence level or fit-quality metric is returned; window validation is delegated to pandas.
- Example: `shear = calc_shear(profiles, window=6)`.

### `plotting/wind_stats.py`

#### `plot_wind_statistics(avg, maximum, minimum, std, height=120.0) -> tuple[go.Figure, go.Figure, float]`

- Responsibility: create a four-row subplot and a single-panel overlaid Plotly
  figure, then select the nearest height common to all inputs.
- Parameters: four statistic DataFrames and requested finite height.
- Returns: subplot figure, single-panel figure, and selected height; does not
  display or save.
- Called by: runner, GUI, tests, and examples.
- Assumptions: exact normalized horizontal-speed column labels.
- Limitations: tie-breaking chooses the lower sorted height; traces retain independent time indexes and are not explicitly aligned.
- Example: `subplot, single, used_height = plot_wind_statistics(avg, max_, min_, std, 120)`.

### `plotting/ti_polar.py`

#### `_bin_center(label) -> float`

- Responsibility: parse a `left-right` bin label and return its midpoint.
- Called by: `plot_ti_polar_by_height`.
- Limitation: malformed labels raise normal conversion/unpacking errors.

#### `plot_ti_polar_by_height(ti_raw, heights=None, ti_stat="median", ncols=2) -> go.Figure`

- Parameters: `calc_ti(...).ti_raw`, optional exact heights, aggregation (`median`/`mean`), and positive subplot column count.
- Returns: a Plotly figure without showing/saving it.
- Called by: runner, GUI, tests, and examples.
- Behavior: removes non-finite/negative TI, aggregates by height/speed bin/direction bin, uses compass orientation, and shares radial/color scales.
- Assumptions: speed/direction bin labels are `number-number` strings.
- Limitations: explicitly requested heights require exact numeric matches; marker size is fixed; empty bins and sample counts are not visualized.
- Example: `plot_ti_polar_by_height(ti.ti_raw, [19, 139], "median", 2)`.

### `run_wind_analysis.py`

#### `run_program_from_input(input_file) -> None`

- Responsibility: complete batch application driven by JSON.
- Instrument inputs use `data_folder_lidar` with optional `start_date_lidar`/`end_date_lidar`, and `data_folder_Metmast` with optional `start_date_Metmast`/`end_date_Metmast`. A folder is required only when a selected feature needs that instrument; legacy LiDAR keys remain accepted.
- Supported features: `ti`, `shear`, `stats`, `ti_polar`, `metmast_data`, and `metmast_comparison`.
- Outputs: creates `save_dir`, prints progress, writes Plotly HTML files only
  when `save_plots` is enabled, and displays figures only when `show_plots` is
  enabled. The flags are independent and both default to `true`; legacy
  `show_plot` is accepted as a fallback.
- Called by: its CLI block and runner regression tests.
- Assumptions: execution starts from a directory where config-relative data/output paths resolve correctly.
- Limitations: data-loading orchestration and many plots duplicate GUI code; no result object is returned; output files are overwritten; hub-height extras require an exact measurement height even though `calc_ti` itself chooses the nearest height; configuration is not schema-validated before output-directory creation.
- Example: `uv run python run_wind_analysis.py input_files/input_config_extended.json`.

### `simple_gui.py`

This is an application rather than reusable package code.

- `load_and_process_lidar_data(data_folder, start_date=None, end_date=None, min_lidar_raw_coverage_percent=80.0)` duplicates the runner's load/statistics/profile path and returns the four statistics, heights, profiles, source files, raw coverage, and raw invalid counts. It raises when no file is found.
- `plot_ti_main`, `plot_ti_timeseries_at_hub`, `plot_ti_mean_vs_height`, `plot_ti_vs_wsp`, and `plot_ti_wsp_and_ti_time_series` build GUI-specific TI figures.
- `plot_shear_main`, `plot_shear_histogram`, `plot_shear_by_hour`, `plot_shear_alpha_vs_wsp`, and `plot_wind_profiles_selected_times` build GUI-specific shear figures.
- `main()` creates Streamlit controls, runs selected features, and catches all exceptions for display.

Parameters are the relevant process dataclasses/DataFrames plus numeric hub height; figure functions return a Plotly figure or `None` when an exact hub-height series is unavailable. Their primary caller is `main`; equivalent plotting logic also exists inline in the runner. Streamlit calls are the main side effect. Run with `uv run streamlit run simple_gui.py`.

Limitations include code duplication in the older LiDAR-only plots, broad exception handling that hides tracebacks from users, and exact-height behavior in some TI extras that differs from TI's nearest-height reference selection. Met-mast scientific calculations and plots are shared rather than duplicated.

### Package `__init__.py` files

The data-reader, process, plotting, and utility `__init__.py` files only re-export selected public functions. The top-level package `__init__.py` is empty. They add no calculations or side effects beyond importing their submodules.

## Scripts, reusable modules, tests, and applications

- Reusable modules live under `src/wind_data_analysis`. They should calculate or build figures without choosing user paths or launching an interface.
- `run_wind_analysis.py` is a batch application: it owns configuration, paths, saving, and optional display.
- `simple_gui.py` is an interactive application: it owns Streamlit state and rendering.
- `examples/` contains small learning programs. They deliberately call public project functions and save reproducible results but are not a second implementation.
- `tests/` defines automated expectations and supplies compact real/synthetic data. Tests are not user-facing demonstrations.
- `demo/study_analyze.py` is exploratory scratch work and is not considered a supported application. The concise supported comparison demonstration is `examples/demo_metmast_comparison.py`.

## Normal user workflow

1. Install `uv`, ensure Python 3.10+ is available, and run `uv sync --locked --dev` from the repository root.
2. Put compatible KNMI LiDAR CSV files in a folder, retaining dates in filenames.
3. Copy and edit an `input_files/` example with the applicable instrument
   folder, optional instrument-specific date bounds, desired features, and
   independent `show_plots`/`save_plots` choices.
4. Run `uv run python run_wind_analysis.py your_config.json`.
5. Open the generated HTML files in `save_dir`, or use `uv run streamlit run simple_gui.py` for interactive exploration.
6. For programmatic work, call readers, statistics, TI/shear, and plotting functions separately as shown in `examples/`.
7. Validate assumptions and quality flags before using results for engineering decisions; the package does not currently perform comprehensive measurement-quality control.

## Unfinished, duplicated, deprecated, or experimental areas

- Machine learning is outside the supported workflow.
- Met-mast comparison is limited to uncorrected horizontal wind speed at nearby measured heights.
- `clean_data` exists but calls are commented out in statistics processing.
- Runner and GUI duplicate the complete load path and much Plotly construction.
- Several modules retain unused imports and long commented-out `__main__` prototypes.
- `demo/study_analyze.py` remains exploratory; supported runnable demonstrations live under `examples/`.
- The historical patch is not runtime code and may become stale relative to the repository.
- Runtime dependencies are declared in `pyproject.toml`; `pytest` is isolated in the `dev` dependency group, and exact resolutions are committed in `uv.lock`.
- CI uses the committed lockfile across Python 3.10–3.14 on both Windows and Ubuntu.
- The bundled real KNMI files exercise the met-mast comparison end to end; engineering validation and source quality review remain necessary.

## Known issues and effects

1. **Direction averaging is linear.** Raw samples around 359° and 1° average near 180°, not north. Effect: 10-minute direction and direction-binned TI can be wrong near wraparound. Follow-up: define circular averaging, invalid/sentinel handling, direction coverage, and whether direction comparison is required before changing behavior.
2. **LiDAR filename dates are day-resolution.** Discovery supports independent start-only and end-only bounds, while the runner applies exact timestamp filtering after reading. Effect: custom filenames still need a parseable date for folder discovery.
3. **Two-point shear uncertainty is undefined.** The residual degrees of freedom are zero. Effect: warnings and non-finite uncertainty, while alpha may still be returned. Decide whether at least three points should be required for uncertainty or whether alpha-with-missing-error is acceptable.
4. **Negative shear is discarded.** Stable/inverted profiles can produce negative alpha, but the implementation converts it to missing. Effect: the shear distribution is biased if negative shear is meaningful for the intended engineering workflow. This needs a domain decision, not a silent code change.
5. **Instrument quality flags are not yet interpreted.** The approved numeric 0--99 m/s rule filters speeds, but instrument-specific flags need a separate reviewed policy.
6. **Duplicate timestamps are retained across files.** Effect: overlapping files can double-count observations and create ambiguous index selection.

## Recommended next development steps

1. Define and test the remaining scientific quality-control policy: instrument
   flags, circular direction handling/coverage/comparison, and negative shear.
2. Keep the shared wind-speed validity and TI alignment regression tests in the
   locked suite as scientific acceptance criteria.
3. Extract one reusable LiDAR workflow service returning a typed result, and make both runner and GUI call it.
4. Add focused unit tests for every process function, especially raw/low-resolution equivalence, date-bound edge cases, circular direction, two-point shear, and duplicate timestamps.
5. Add explicit configuration validation and consistent nearest-height selection/reporting across batch and GUI paths.
6. Extend met-mast analysis only after defining reviewed units and quality-flag policies; the current comparison deliberately leaves source values uncorrected.
7. Separate runtime, plotting/UI, and development dependencies, remove duplicates/unused imports, and add a configured formatter/linter.
8. Decide later whether to merge, redesign, or retire the separate `ML` branch; it should not be presented as a current capability until merged and tested.
