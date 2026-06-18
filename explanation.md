# BuildStock Weather Regression Tool — Detailed Explanation

This document explains how the repository’s A/B/C/D/E pipeline works end-to-end, based on:

- `A_start_building_stock_parallel_agg.sh`
- `B_building_stock_parallel_agg.py`
- `C_run_bldg_chunk_agg.sh`
- `D_process_chunk_agg.py`
- `E_combine_nonHVAC.py`
- `E_combine_nonHVAC.sh`
- `README.md`
- `switches_agg.json`

---

## 1) What this tool does

At a high level, this tool:

1. Pulls hourly HVAC and annual metadata from ResStock/ComStock (via `buildstock_query`/Athena).
2. Trains county-level weather-response regressions from a **base weather year** (default 2018).
3. Predicts HVAC electricity (and natural-gas heating) for one or more **target weather years**.
4. Writes county-level hourly HVAC output files and county-level diagnostic metadata.
5. Optionally combines regressed HVAC with non-HVAC 2018 profiles into scenario/state totals.

The regression engine is primarily **Random Forest**; when weather extrapolation is required and enabled, it uses a **hybrid RF + Neural Network** approach.

---

## 2) Pipeline architecture (A → B → C → D → E)

### A) `A_start_building_stock_parallel_agg.sh` (top-level HPC launcher)

Purpose:
- Submit one SLURM job that starts the full parallel workflow.

What it does:
- Sets SLURM resources (`--time`, `--mem`, `--qos`, etc.).
- Ensures `uv` is on `PATH`.
- Runs `aws sso login`.
- Executes `uv run B_building_stock_parallel_agg.py`.

Use case:
- Recommended for large/national runs on Kestrel or other SLURM systems.

---

### B) `B_building_stock_parallel_agg.py` (orchestrator)

Purpose:
- Build run manifest and split work into county chunks.
- Create reproducible run folder structure.
- Dispatch chunk jobs locally or through SLURM.

Core flow:

1. **Detect runtime mode**
   - HPC mode if `REEDS_USE_SLURM=1`.
   - Otherwise local multiprocessing mode.

2. **Create timestamped output directory**
   - `outputs/outputs_<YYYY-mm-dd-HH-MM-SS>/`
   - Copies repository inputs into `outputs/.../inputs/` for reproducibility.
   - Writes `commit_hash.txt` in copied inputs.

3. **Load switches** (`switches_agg.json`)
   - Determines ResStock vs ComStock, upgrades, chunk size, etc.

4. **Load/prepare metadata**
   - For ComStock 2025.2: uses custom per-state/per-county parquet logic.
   - Else: reads standard national metadata parquet by upgrade.
   - Removes AK/HI.
   - Applies weighting to sqft/electric/gas annual values.
   - Aggregates to building grouping columns (`res_bsq_cols` or `com_bsq_cols`).
   - Creates a `bldg_id` tuple-string key.
   - Writes `res_meta_master_upgrade<id>.csv` or `com_meta_master_upgrade<id>.csv`.

5. **Split counties into chunks**
   - Uses `chunk_size` from switches.
   - For each chunk:
     - HPC: submit `C_run_bldg_chunk_agg.sh ...` via `sbatch`.
     - Local: queue command to run `D_process_chunk_agg.py` in multiprocessing pool.

6. **Finish when all chunks complete**

Outputs from B:
- One master metadata CSV per upgrade.
- A full copied run snapshot under `outputs/.../inputs`.
- Many chunk jobs dispatched to C/D.

---

### C) `C_run_bldg_chunk_agg.sh` (per-chunk HPC wrapper)

Purpose:
- Run one chunk processing job under SLURM.

What it does:
- Accepts chunk arguments from B:
  - start/end index, metadata path, upgrade, prefix, output dir, script dir, counties string.
- Caps thread counts (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.) to reduce oversubscription.
- Runs:
  - `uv run python <output_dir>/inputs/D_process_chunk_agg.py ...args...`

Why separate C exists:
- Clean resource isolation per chunk.
- Easy rerun of failed chunk with the same argument list.

---

### D) `D_process_chunk_agg.py` (core engine)

Purpose:
- For one chunk of counties and one upgrade:
  - Query BuildStock hourly + annual data,
  - optionally apply weather-year regression,
  - write chunk-level outputs.

This is the main technical core.

#### D.1 Inputs

From CLI args (passed by B/C):
- `start_index`, `end_index`
- `meta_path`
- `upgrade`
- `prefix` (`res_` or `com_`)
- `output_dir`, `script_dir`
- `counties_str`

From copied switches (`outputs/.../inputs/switches_agg.json`):
- Model/control flags (`apply_regression`, `hybrid_model`, `cross_val`, etc.)
- Data source/run configuration (`base_run`, `target_run`, `run_types`)
- Years (`base_year`, `target_year`)
- EPW location (`weather_data_base`)

#### D.2 Year handling

`target_year` accepts:
- single int/string (e.g., `2018` or `"2018"`)
- range string (e.g., `"2007-2024"`)
- list mix (e.g., `["2007-2013", 2016, "2018"]`)

These are expanded to a sorted unique list.

#### D.3 BuildStockQuery extraction (`process_chunk_agg`)

For each chunk, D queries hourly data from AWS/Athena via `BuildStockQuery`.

Key behaviors:
- Supports OEDI schemas and non-OEDI schemas.
- Supports `aggregate_timeseries` and `savings_shape` branches.
- Applies restrictions by state/county/upgrade/applicability.
- Includes retry wrapper (`query_execution`) for transient failures.
- Contains custom query-string rewrites for:
  - OEDI table naming differences,
  - ComStock 2025.2 table names,
  - timestamp conversion (`from_unixtime_nanos` + hour truncation),
  - partition filtering by state/upgrade/county.

Post-query processing:
- Normalizes end-use columns.
- Sums HVAC electric end uses into `HVAC.elec`.
- Sums natural gas heating end uses.
- Converts kWh→MWh (and kbtu→kWh where needed, then MWh).
- Forms `bldg_id` index.

#### D.4 Weather ingestion (`weather_data`)

For each county and year, D reads local EPW:
- Path pattern: `<weather_data_base>/FIPS_<year>/<county>_<year>.epw`
- Uses EPW columns for:
  - dry bulb temp,
  - RH,
  - wind speed/direction,
  - GHI/DNI/DHI.

Feature engineering adds:
- `Time of Day`
- `Weekend`
- temperature lag features from `lag_hours_temperature` (default `[-1,1,3,6,12]`)

Then:
- fill lag NaNs,
- keep first 8760 rows for year consistency.

#### D.5 Regression model (`prediction`) — detailed inputs/outputs

The regression is run **per `bldg_id`** and **per energy target**:

- `HVAC.elec` (MWh hourly)
- `natural_gas.heating.energy_consumption` (MWh hourly)

So each `bldg_id` gets two independent models (or one model executed twice with different targets).

##### D.5.1 Exact model inputs

For one `bldg_id`, the `prediction()` call receives:

1. **Base-year hourly target series** `Y`
   - Source: `ts_agg.loc[bldg_id][energy_type]`
   - Unit at this point: MWh/hour
   - Important alignment step: first EULP row is dropped (`df_eulp = df_eulp.iloc[1:]`) to align with weather timestamps.

2. **Base-year weather feature matrix** `X`
   - Source: `weather_data(weather_data_base, base_year, county_id)`
   - Then last row is dropped (`X = X.iloc[:-1]`) to match `Y` length.
   - Feature columns are:
     - `Dry Bulb Temperature [°C]`
     - `Relative Humidity [%]`
     - `Wind Speed [m/s]`
     - `Wind Direction [Deg]`
     - `Global Horizontal Radiation [W/m2]`
     - `Direct Normal Radiation [W/m2]`
     - `Diffuse Horizontal Radiation [W/m2]`
     - `Time of Day`
     - `Weekend`
     - lagged dry-bulb features from `lag_hours_temperature`

3. **Target-year weather matrix** `X_Predict`
   - Built by concatenating weather frames for all requested `target_years`.
   - Ordering is year order from parsed `target_years`, then hourly order within year.
   - A parallel vector `target_year_by_row` tracks which year each prediction row belongs to.

##### D.5.2 Training behavior

- Primary estimator:
  - `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)`
- If `cross_val=true`, script computes 5-fold CV MSE before final fit on full `X, Y`.
- If base-year load is effectively zero (`Y.sum() <= 0.01`), model training is skipped and output is all zeros.

##### D.5.3 Hybrid extrapolation behavior

After RF predicts `X_Predict`, code checks if target dry-bulb exceeds training range:

- Train range: $[T_{\min}^{train}, T_{\max}^{train}]$
- Predict range: $[T_{\min}^{pred}, T_{\max}^{pred}]$

If `hybrid_model=true` **and** extrapolation is needed:

1. Train a small NN on standardized `X`.
2. Predict both RF and NN on target weather.
3. Select per hour:

$$
\hat{y}_t =
\begin{cases}
\hat{y}^{RF}_t & T_t \in [T_{\min}^{train}, T_{\max}^{train}] \\
\hat{y}^{NN}_t & T_t \notin [T_{\min}^{train}, T_{\max}^{train}]
\end{cases}
$$

If extrapolation is not needed (or hybrid disabled), final predictions are RF-only.

##### D.5.4 Timestamp/output shaping

Raw model predictions are post-processed as follows:

1. **Hour-end alignment shift by year block**
   - For each target year separately, predictions are shifted forward by 1 hour within that year block.
   - First hour of year is filled with that year’s first modeled value.

2. **Timestamp construction**
   - Creates continuous hourly timestamps starting at `YYYY-01-01 01:00:00` for each target year.

3. **Physical constraint**
   - Negative predictions are clipped: $\hat{y}_t = \max(0, \hat{y}_t)$.

4. **Return object**
   - DataFrame with columns:
     - `timestamp`
     - `<energy_type>` (either `HVAC.elec` or `natural_gas.heating.energy_consumption`)

##### D.5.5 From per-building output to final file output

For HVAC output files, D then:

1. Renames each building’s prediction column to its `bldg_id`.
2. Concatenates all `bldg_id` columns into one hourly table.
3. Collapses multiple `bldg_id`s to county by summing columns with same county label.
4. Keeps first 8760 rows per modeled year (after internal year assignment logic).
5. Writes CSV:
   - `res_eulp_hvac_elec_MWh_upgrade<id>_<start>-<end>.csv` or
   - `com_eulp_hvac_elec_MWh_upgrade<id>_<start>-<end>.csv`

So the **primary regression output** consumed downstream is:

- Index: hourly timestamps (potentially multi-year)
- Columns: counties
- Values: regressed HVAC electric load in MWh

Natural-gas regression output is not written as a separate hourly CSV in current D flow; it is used to populate annual diagnostics in chunk metadata files.

#### D.6 Parallelism in D

- On HPC: process buildings in a process pool (`ProcessPoolExecutor`) for that chunk.
- Local mode: building loop is serial in D (overall chunk parallelism is handled by B).

#### D.7 Diagnostics & metadata reconciliation

D calculates county-level comparisons among:
- metadata annual totals (`meta_*`),
- queried AWS hourly sums (`AWS_*`),
- regressed annual sums (`HVAC.elec`, `natural_gas...`).

It writes ratios and percent differences such as:
- `ratio_HVAC_AWS_meta`
- `ratio_HVAC_reg_meta`
- `diff_HVAC_reg_AWS`
- and gas equivalents.

#### D.8 D outputs per chunk

- `res_eulp_hvac_elec_MWh_upgrade<id>_<start>-<end>.csv` (or `com_...`)
  - hourly rows, county columns, MWh.
- `res_meta_upgrade<id>_<start>-<end>.csv` (or `com_meta_...`)
  - county-level annual diagnostics and metadata.

---

### E) `E_combine_nonHVAC.py` (post-processing combiner)

Purpose:
- Merge regressed HVAC with non-HVAC 2018 load profiles and produce final scenario/state and national-sector outputs.

Core functions:

1. **Load non-HVAC profiles**
   - ResStock and ComStock from configured external directories.
   - ComStock can include an added “gap model” profile by state.

2. **Load HVAC outputs**
   - Reads county-level HVAC CSVs from one or more run directories.
   - Aggregates county columns to state columns via GISJOIN state FIPS mapping.

3. **Align non-HVAC to HVAC calendar**
   - If HVAC index differs from non-HVAC 2018 index, performs day-of-week alignment by year.
   - Handles leap-year padding.

4. **Combine profiles**
   - Total = aligned non-HVAC + HVAC (state hourly series).

5. **Scenario mapping**
   - Uses `SCENARIO_MAPPING` to map upgrade combinations into labels like:
     - `Baseline`, `ASHP`, `GHP`, `GHP + Envelope`.
   - Supports mutually-exclusive bundled upgrades (e.g., ComStock `1` + `14`) by synthetic combination logic:

$$
\text{combined} = \text{baseline} + \sum_i (u_i - \text{baseline})
$$

6. **Write outputs**
   - One CSV per scenario with state columns.
   - National sector totals (res/com/total) by scenario under `sector_totals/`.

---

### E shell wrapper: `E_combine_nonHVAC.sh`

Purpose:
- SLURM wrapper to run E on HPC.

---

## 3) `switches_agg.json` — control plane

`switches_agg.json` controls nearly all behavior.

Major groups:

1. **Run size and scope**
   - `testmode`, `upgrades`, `chunk_size`, `target_year`, `base_year`

2. **Dataset branch**
   - `comstock` (`false` = ResStock path, `true` = ComStock path)
   - `base_run`, `target_run`, `run_types`
   - version tags (`version_resstock`, `version_comstock`)

3. **Regression settings**
   - `apply_regression`
   - `cross_val`
   - `hybrid_model`
   - `lag_hours_temperature`
   - diagnostics flags (`test_base`, `save_metrics`, `show_fit`, `save_fit`, `test_target`)

4. **Query behavior**
   - `savings_shape` vs aggregate path
   - `applied_only`
   - `sleep_seconds` (random startup delay to reduce AWS auth contention)

5. **Weather source**
   - `weather_data_base` (root directory containing `FIPS_<year>` subfolders)

6. **BuildStockQuery schema config**
   - `run_types` entries define Athena workgroup/db/schema/table names.

---

## 4) Data flow summary

1. **A** launches **B**.
2. **B** snapshots inputs and builds master metadata by upgrade.
3. **B** chunks counties and runs many **C/D** jobs.
4. **D** queries hourly data, regresses to target weather years, writes chunk files.
5. Optional downstream aggregation (outside A–D):
   - `agg_buildings.py` can combine chunk files.
6. **E** combines HVAC outputs with non-HVAC and emits scenario files.

---

## 5) Output structure and semantics

In each timestamped run folder (`outputs/outputs_<timestamp>/`):

- `inputs/` — frozen copy of scripts + switches used for reproducibility.
- `res_meta_master_upgradeX.csv` / `com_meta_master_upgradeX.csv`
  - pre-chunk grouped annual metadata.
- `res_meta_upgradeX_####-####.csv` / `com_meta_...`
  - per-chunk county diagnostics.
- `res_eulp_hvac_elec_MWh_upgradeX_####-####.csv` / `com_eulp_...`
  - per-chunk county hourly HVAC in MWh.

From E:
- `outputs/<Scenario>.csv` (state-level total load by hour)
- `outputs/sector_totals/<Scenario>_national_sector_totals.csv`

---

## 6) Practical run modes

### Fast test mode
- `testmode: true`
- Typically VT subset in B logic.
- Good for setup verification.

### Full ResStock run
- `comstock: false`
- `testmode: false`
- `target_year`: desired range/list
- `upgrades`: one or more ids

### Full ComStock run
- `comstock: true`
- set `base_run` to ComStock run type (e.g., `comstock_2025_2`)
- often smaller `chunk_size` (e.g., 10)
- usually longer `sleep_seconds` to mitigate token contention

### No-regression extraction
- `apply_regression: false`
- set `target_year == base_year`
- tool behaves like data extraction/aggregation without weather transfer learning

---

## 7) Reliability and reproducibility features

- Full input snapshot per run (`outputs/.../inputs`).
- Git commit hash written for traceability.
- Retry loop around query execution.
- Random stagger (`sleep_seconds`) to reduce parallel auth failures.
- Explicit rerun command printed for each chunk in D.

---

## 8) Important implementation details to be aware of

1. **County identity differs by branch**
   - ResStock uses `in.county`.
   - ComStock uses NHGIS county IDs and, for 2025.2, also simulated-county IDs.

2. **Hourly alignment is intentional**
   - D applies timestamp handling so modeled load matches hour-ending conventions.

3. **Weather data must exist locally**
   - D expects local EPW file structure and will fail fast if files are missing.

4. **Modeling granularity**
   - Training/prediction is per `bldg_id` group (county-level aggregation key tuple), then collapsed back to county output columns.

5. **E assumes external non-HVAC datasets**
   - Paths in E are currently configured to project-specific absolute directories.

---

## 9) In one sentence

This repository is a chunk-parallel, AWS-query-backed weather-year transfer pipeline that builds county-level HVAC load regressions from BuildStock base-year data, projects them across target weather years, and optionally merges with non-HVAC profiles to produce state and national scenario load outputs.
