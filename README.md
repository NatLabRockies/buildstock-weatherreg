# ResStock / ComStock Weather-Regression + Projection Pipeline

End-to-end pipeline that pulls per-building EULP (End Use Load Profiles) from
ResStock / ComStock on AWS, optionally weather-regresses them to multiple
target years, optionally calibrates ResStock against ground-truth net
electricity, aggregates to county-level CSVs, projects forward to future stock
years (2027–2050) across four adoption cohorts, and emits handoff folders for
downstream consumers (ReEDS, LBL, and an Intermediate publishing/debugging view).

> **HPC-first.** Heavy compute (BSQ pulls, chunk processing, projection) runs on
> SLURM. The login node is for orchestration only — a runaway login-node load
> trips the HPC monitor and throttles the user.

---

## Pipeline at a glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          one command starts everything:                     │
│       sbatch A_start_building_stock_parallel_agg.sh switches_agg_*.json     │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
    ╔═════════════╗
    ║  A (SLURM)  ║  thin SLURM wrapper; authenticates AWS, then invokes B
    ║  launcher   ║  with the switches path.
    ╚═════════════╝
          │
          ▼
    ╔════════════════════════════╗
    ║  B  (Python orchestrator)  ║  parses switches_agg_*.json; for each run_spec:
    ║  uv run B_*.py             ║    • computes county chunks (bin-packed by weather_loc)
    ║                            ║    • submits C array (chunk pulls + per-county processing)
    ║                            ║    • submits F (chunk aggregation), depends afterok on C
    ║                            ║  at the end, submits Z, depends afterok on every F.
    ╚════════════════════════════╝
          │
          ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │                       per run_spec (in parallel)                  │
  │                                                                   │
  │   ╔═══════════════════════╗      ╔═══════════════════════════╗    │
  │   ║  C  array (one task   ║──▶   ║ D_process_chunk_agg.py    ║    │
  │   ║  per county chunk)    ║      ║  • BSQ pull from Athena   ║    │
  │   ╚═══════════════════════╝      ║  • gather ts_agg, df_meta ║    │
  │          per task                ║  • apply RES calibration* ║    │
  │      (5-7 GB RSS)                ║  • train RF/NN regression ║    │
  │                                  ║  • predict target years   ║    │
  │                                  ║  • write chunk CSV        ║    │
  │                                  ╚═══════════════════════════╝    │
  │                                            │                      │
  │                                            ▼                      │
  │                              ╔═════════════════════════════╗      │
  │                              ║  F_aggregate_chunks.sh      ║      │
  │                              ║  → agg_buildings.py         ║      │
  │                              ║  • stitch chunk CSVs        ║      │
  │                              ║  • write 4 per-enduse aggs  ║      │
  │                              ╚═════════════════════════════╝      │
  │   * RES calibration only fires when the run_type has              │
  │     `adjustment_factor` set; see "Calibration" below.             │
  └───────────────────────────────────────────────────────────────────┘
          │
          │  every F has completed
          ▼
    ╔════════════════════════════╗
    ║  Z_post_pipeline.sh        ║  meta-launcher; submits four jobs with
    ║  (single light SLURM job)  ║  the right `--dependency=afterok` chain.
    ╚════════════════════════════╝
          │
          ├──▶ G_run_projection.sh   (state res, medmem, 20 workers, 480 GB)
          │       → projections_state/*.csv
          │
          ├──▶ G_run_projection.sh   (county_group res, medmem, 16 workers, 800 GB)
          │       → projections_county_group/*.csv
          │
          ├──▶ handoff_light (debug)            deps: both projections
          │       python -m projections.reeds        → ReEDs/
          │       python -m projections.intermediate → intermediate/
          │
          └──▶ H_run_lbl.sh         (medmem)    deps: county_group projection
                  python -m projections.lbl    → LBL/

      everything below is automatic; outputs land in <output_dir>/
```

After the chain completes, the run_dir holds:

```
<output_dir>/
├── chunks_reg_b2018/                 — per-chunk regressed timeseries (D outputs)
├── chunks_meta_b2018/                — per-chunk metadata
├── agg_<stock>_eulp_<enduse>_GWh_upgrade<tag>.csv     — county-level annual hourly CSVs (F output)
├── aux_coverage_upgrade<tag>.csv     — cohort sizes per state, used by projection
├── aux_samples_upgrade<tag>.csv      — sampled bldg_ids per cohort, used by LBL
├── projections_state/                — per-component CSVs at state resolution (G)
├── projections_county_group/         — per-component CSVs at 1,038 county-group resolution (G)
├── ReEDs/                            — 24 files: wide-format MWh, lowercase state names (handoff)
├── LBL/                              — 156 files: long-format kWh county-group timeseries + samples
└── intermediate/{state, county_group}/  — symlinks: per-component, relabeled with sector/cohort/enduse
```

---

## Entry point

```bash
# resstock with calibration (recommended starting point):
sbatch A_start_building_stock_parallel_agg.sh switches_agg_resstock.json

# comstock (no res-calibration concept):
sbatch A_start_building_stock_parallel_agg.sh switches_agg_comstock.json
```

A single submission produces every output above — chunks, aggregates, projections, and all three handoff folders. No manual chaining needed.

If anything inside the chain fails, downstream stages are short-circuited via
`--dependency=afterok` so you won't see phantom handoffs derived from broken
projections.

---

## File reference

| Stage | File | Type | What it does |
|---|---|---|---|
| 0 | `switches_agg_resstock.json` / `switches_agg_comstock.json` | config | Per-stock spec list, run_type definitions, calibration path, scenario name map |
| A | `A_start_building_stock_parallel_agg.sh` | SLURM wrapper | AWS auth, renames the running job `res_/com_*`, runs B |
| B | `B_building_stock_parallel_agg.py` | Python orchestrator | For each spec: bin-pack chunks, submit C array, submit F (depends C), accumulate F IDs, finally submit Z |
| C | `C_run_bldg_chunk_agg.sh` | SLURM array task | Runs `D_process_chunk_agg.py` with the chunk's start/end indices |
| D | `D_process_chunk_agg.py` | Python worker | BSQ Athena pull, **apply calibration factors if set**, RF / NN regression, write chunk CSV |
| E | `E_aux_query.py` | Python | Auxiliary BSQ pulls — populates `aux_coverage_*.csv` (cohort sizes) and `aux_samples_*.csv` (per-cohort bldg_ids + weights) |
| F | `F_aggregate_chunks.sh` | SLURM job | Runs `agg_buildings.py` per spec — stitches per-chunk CSVs into 4 per-enduse aggregates |
| Z | `Z_post_pipeline.sh` | SLURM meta-launcher | When every F completes: submits G×2 projections + handoff_light + H_run_lbl |
| G | `G_run_projection.sh` | SLURM wrapper | Runs `python -m projections <run_dir> --resolution {state\|county\|county_group}` |
| H | `H_run_lbl.sh` | SLURM wrapper | Runs `python -m projections.lbl <run_dir>` (parallel ProcessPool) |
| — | `projections/` | package | Future-year projections + ReEDs / LBL / intermediate handoff modules (see below) |
| — | `agg_buildings.py` | Python | Stitches per-chunk hourly CSVs into one per-enduse CSV at county resolution |
| — | `epw_sync.py` | Python | One-time download of EPW weather files (off-Kestrel only) |
| — | `validation.py` / `validation_supplemental.py` | Python | Reg-vs-ref diagnostics |
| — | `res_state_adjustment_factors_amy2018.parquet` | data | Per-(state, hour) calibration factors for ResStock amy2018 |
| — | `gap_by_state.csv` | data | ComStock gap-model 2018 hourly profile, 49 states |
| — | `shell_factors_combined.csv` | data | New-construction efficiency factors used by the projection |
| — | `AEO 2025/*.csv` | data | AEO 2025 cohort splits used by `projections.growth_factors` |
| — | `county_group_mapping.csv` | data | 3,144-row map from county-FIPS to BuildStock county_group (n=1,049) |

---

## switches_agg_*.json reference (modern fields)

The two template files define everything per stock:

```json
{
  "output_dir": "/projects/geohc/radhikar/outputs/resstock_cross_val_june8_2026",
  "testmode": false,
  "comstock": false,
  "run_specs": [
    { "name": "All-Baseline",            "upgrade_id": 0, "apply_regression": true,
      "base_year": 2018, "target_year": ["2007-2024"],
      "base_run": "resstock_amy2018_2025_1", "target_run": "resstock_amy2012_2025_1",
      "chunk_size": 75 }
    /* …Upgraded-Baseline, Non-Upgraded-Baseline,
       Upgraded-Upgrade4/8/32 (res) or Upgraded-Upgrade1-14/55/59 (com)… */
  ],

  "scenario_names": {
    "All-Baseline": "Baseline",
    "Upgrade4": "ASHP",  "Upgrade8": "GHP",  "Upgrade32": "GHP+Envelope"
  },

  "run_types": {
    "resstock_amy2018_2025_1": {
      "workgroup": "ghpphase2",
      "db_name": "buildstock_sdr",
      "table_name": "resstock_amy2018_r1_2025",
      "db_schema": "resstock_oedi_new",
      "buildstock_type": "resstock",
      "adjustment_factor": "res_state_adjustment_factors_amy2018.parquet",
      "skip_reports": true
    },
    "resstock_amy2012_2025_1": { /* …no adjustment_factor — used as ground truth… */ }
  }
}
```

Field highlights added since the original pipeline:

| Field | Where it lives | Meaning |
|---|---|---|
| `output_dir` | top level | Where the entire pipeline writes — projections, handoffs, slurm-out/, everything. Bump the date to start a fresh run without overwriting old data. |
| `scenario_names` | top level | `{spec-short-id: display-name}`. Strips `Upgraded-` prefix and looks the result up; used by every projection filename and every handoff. Falls back to the raw short id if absent. |
| `run_types[<name>].adjustment_factor` | per run_type | Optional. Filename of a per-(state, hour) `[8760 × 49]` parquet of multiplicative calibration factors. Resolved against the repo dir. When set, `D_process_chunk_agg.py` multiplies every electricity column in that run_type's `ts_agg` by `factor[state, hour]`. Absent → no calibration. |
| `apply_regression` | per spec | `true` trains and predicts; `false` writes the base-year load directly. Calibration runs in both modes (it sits *before* the regression branch). |
| `target_run` (different from base) | per spec | Pulled as ground-truth for `test_target` validation. Calibration on this run is *off by design* — it's the un-calibrated reality the regressed base predicts against. |

---

## Calibration (ResStock only)

A 2018-net-electricity calibration is wired in for ResStock so that simulated
net consumption matches measured net consumption at the (state, hour) level.
The mechanism:

* The parquet at `res_state_adjustment_factors_amy2018.parquet` has 8760 rows
  (hour-ending EST, `2018-01-01 01:00` … `2019-01-01 00:00`) and 49 columns
  (CONUS state postals + DC). This file is created as a part of ComStock Gap
  modelling
* In `D_process_chunk_agg.py`, `_apply_state_adjustment` (called from inside
  `process_chunk_agg`) multiplies every **electricity** column of `ts_agg`
  uniformly by the corresponding `factor[state, hour]`:

  ```
  cooling.elec  ← cooling.elec × factor
  heating.elec  ← heating.elec × factor
  ev            ← ev           × factor
  pv            ← pv           × factor   # scaled in lockstep so net=total−pv also scales
  total         ← total        × factor
  ```

  `natural_gas.heating.energy_consumption` is *not* scaled — the parquet
  calibrates electric only.
* The `non_hvac.elec = total − cooling − heating − ev` residual computed
  immediately after picks up the same scaling automatically since every input
  was scaled by the same factor.
* Because PV scales in lockstep with everything else, the net relationship
  holds exactly: `new_net = new_total − new_pv = factor × (total − pv) = factor × old_net`.

To turn calibration *off* for a run_type, just remove the `adjustment_factor`
key from that run_type definition.

---

## projections/ package

After F lands the per-enduse aggregate CSVs, `python -m projections` projects
hourly load forward to future stock years (2027, 2030, 2035, 2040, 2045, 2050)
across four cohorts (new construction, surviving adoption, surviving
non-adoption, plus gap for commercial).

```
projections/
├── common.py          shared types, state geography, agg/aux loaders
├── factors.py         efficiency + cohort-growth multipliers
├── gap.py             ComStock gap-model loader (state CSV / per-county S3)
├── growth_factors.py  AEO 2025 cohort splits
├── projection.py      the six projection components + parallel driver
├── reeds.py           ReEDs handoff (state-aggregated long → wide MWh)
├── lbl.py             LBL handoff (county-group long-format + per-cohort samples)
├── intermediate.py    Intermediate handoff (per-component relabeled symlinks)
├── __main__.py        entry for `python -m projections`
└── __init__.py        public re-exports
```

Resolutions supported by `--resolution`:

| Choice | Output | Notes |
|---|---|---|
| `state` *(default)* | 49 state-postal cols (CONUS+DC) | Light; ~58 MB/frame. Used by ReEDs handoff. |
| `county` | ~3,107 county-FIPS cols | Heavy; per-county gap fetched from S3 (cached). |
| `county_group` | 1,038 BuildStock county-groups (CONUS) | Used by LBL + intermediate handoffs. Gap computed by collapsing cached county-level data. |

---

## Handoffs

Three folders under the run_dir, one per downstream consumer. Each handoff
follows its own spec image:

### `ReEDs/` — state-aggregated, wide, MWh, lowercase

* One CSV per `(scenario, stock_year)` = 24 files per stock.
* Wide format: `timestamp_EST` index × 48 state columns (DC merged into MD).
* Column names are full lowercase state names (`alabama, arizona, …, wyoming`).
* Values in **MWh** (GWh × 1,000).
* Sums every component (cohorts + gap for com) into a single column per state.

### `LBL/` — county-group, long format, kWh, per-cohort

* Timeseries: one CSV per `(scenario, sector, cohort, stock_year, weather_year)`
  = 132 files per stock, plus 24 `aux_samples_*` files = 156 total.
* Filenames `<scenario>_<sector>_<cohort>_<year>_amy<weather>.csv`.
* Columns: `timestamp_EST, county_group, sector, cohort, enduse, value_kwh`.
* Sectors: `residential` (res run_dir) and `commercial` (com run_dir).
* Cohorts: `NC` (new construction), `SA` (surviving adopting), `SNA` (surviving
  not adopting). Baseline emits only NC + SNA (no adoption).
* Weather years: **2012 and 2018 only** (the un-regressed actuals).
* Gap is **excluded** from LBL by spec.

### `intermediate/` — per-component view for publishing/debugging

* Two subfolders: `state/` and `county_group/`.
* Each entry is a **relative symlink** to the authoritative per-component CSV
  under `projections_<resolution>/`, named with the LBL cohort vocabulary so
  the file is publishable without disambiguation.
* Filename: `<scenario>_<sector>_<cohort>_<enduse>_y<stock_year>.csv`.
* Gap is included as cohort = `gap` (separate, not folded into commercial cohorts).
* Symlinks default for zero-disk impact; pass `--copy` to materialize real files.

---

## Manual reruns (after the chain has finished)

Any stage can be re-run by itself once the prerequisites exist on disk:

```bash
# Just refresh the handoffs (projections already exist):
python -m projections.reeds        <run_dir>
python -m projections.intermediate <run_dir>
sbatch H_run_lbl.sh                <run_dir>

# Re-project at a different resolution:
sbatch G_run_projection.sh <run_dir> res county_group

# Re-run a single failed chunk (slurm-*.out for the failed task prints the exact command):
sbatch --job-name=res_chunk_<tag> --array=<idx> ./C_run_bldg_chunk_agg.sh <manifest> <meta> <upg> res_ <out> <repo> <spec_idx>

# Re-run only Z (skip B/C/D/F entirely; useful if F's done but Z didn't fire):
sbatch Z_post_pipeline.sh <run_dir> <res|com>
```

---

## Setup (one-time)

### Access prerequisites

- **AWS Stratus Cloud account + allocation** — see the Stratus Cloud team
  ([Stratus Cloud: Home](https://stratus-cloud.thesource.nrel.gov/)).
- **AWS ResStock/ComStock Sandbox access** (likely `resbldg`) — contact the
  Buildings teams.
- **HPC (Kestrel) allocation** — see the [HPC team](https://www.nlr.gov/hpc/).

### Install uv (Python package manager)

```bash
# macOS / Linux / Kestrel
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install AWS CLI

```bash
cd ~
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install -i "$HOME/local/aws-cli" -b "$HOME/local/bin"
echo 'export PATH="$HOME/local/bin:$PATH"' >> ~/.bashrc
echo 'export REEDS_USE_SLURM=1' >> ~/.bashrc   # only on HPC
source ~/.bashrc
aws --version
```

### AWS SSO config

Follow the
[AWS SSO configuration instructions](https://github.com/NatLabRockies/buildstock-query/wiki/AWS-setup#with-sso-for-nrel-employees)
all the way through step 8. Re-authenticate at the start of each work session:

```bash
aws sso login
```

### Repo + dependencies

```bash
git clone git@github.com:NatLabRockies/buildstock-weatherreg.git
cd buildstock-weatherreg
uv sync   # creates .venv, installs everything from pyproject.toml
```

### Non-HVAC Load Queries (Aggregate Notebooks)

 Use the notebooks in `aggregates/` for queries of non-HVAC electricity loads by state and hour.


### Full Runs
Set these switches in `switches_agg.json`:
- `"testmode": false`. This deactivates the Vermont-only test run and runs full national.
- `"upgrades": [0,4]` (or any list of upgrades/measures to run). Upgrade 0 is Baseline. Note that for ComStock runs we typically only run one upgrade at a time on HPC.
- `"target_year": ["2007-2013","2016-2023"]` (or any integer or list of either integers or strings with ranges, as shown). These are the years for which regressed EULP data is output.
- Change any other switches as shown in the subsections below. In the subsections below we discuss using [ComStock](#comstock-regressed) rather than ResStock, and running the tool [without regressions](#non-regressed) to simply extract existing ResStock/ComStock data.

```bash
uv run python -c "import site; print(site.getsitepackages())"
# copy comstock_oedi.toml to <site-packages-dir>/buildstock_query/db_schema/
```

### Weather files (off-Kestrel only)

If you're not on Kestrel the EPW files have to be downloaded once:

* Edit `_BASE_ROOT` in `epw_sync.py`.
* On Windows, set `DEFAULT_MODE = "copy"` (default is `"symlink"`).
* Run `epw_sync.py`.
* Update `"weather_data_base"` in your switches file to point at the new dir.

On Kestrel this is already set to `/projects/geohc/EPW/epw_symlinks`.

---

## Outputs cheat-sheet

Per the **Pipeline at a glance** layout above. Most-asked questions:

* **Where are the per-county hourly aggregates?** `<output_dir>/agg_<stock>_eulp_<enduse>_GWh_upgrade<tag>.csv` — one file per (enduse, spec). Index is `timestamp_EST`, columns are county FIPS.
* **Where are the per-stock-year projections?** `<output_dir>/projections_state/proj_<stock>_<scenario>_<group>_<enduse>_GWh_y<year>.csv` (or `projections_county_group/` for finer geography). Wide format.
* **Where are the handoff files I send to a stakeholder?** The three folders directly under `<output_dir>/`: `ReEDs/`, `LBL/`, `intermediate/`. The naming, units, and shape of each is described in **Handoffs** above.
* **Why didn't the pipeline run end-to-end?** Check `<output_dir>/slurm-out/` and the launcher's `slurm-res_building_stock_parallel_*.out`. The `--dependency=afterok` chain short-circuits on any failure, so downstream stages will simply show `Dependency` as their reason until the upstream stage is fixed and the chain rebuilt.

---

## Validation

See regression validation outputs for ResStock and ComStock HVAC EULP here:
<https://drive.google.com/file/d/1qDy9DrraTP7Kkzk1i6_tDVStEf3fzrQn/view>

---

## Switches reference (full)

| Switch | Where | Description | Typical |
|---|---|---|---|
| `output_dir` | top-level | Run output folder. Bump date for fresh runs. | `…/outputs/<stock>_cross_val_<date>` |
| `testmode` | top-level | Vermont-only test slice in B. | `false` (production) |
| `comstock` | top-level | `true` selects ComStock branches. | `false` (resstock template) / `true` (comstock template) |
| `scenario_names` | top-level | `{short-id → display-name}` map for projection filenames + handoffs. | `{All-Baseline→Baseline, Upgrade4→ASHP, …}` |
| `version_*` | top-level | ResStock / ComStock version tuples used in source paths/tables. | `["2025", "1"]`, `["2025", "2"]` |
| `url_base` | top-level | OEDI prefix used in B for remote paths. | `s3://oedi-data-lake/nrel-pds-building-stock/…` |
| `weather_data_base` | top-level | Local EPW directory; one subdir per year. | `/projects/geohc/EPW/epw_symlinks` |
| `chunk_size` | spec / top-level | Counties per chunk. ResStock 75 / ComStock 10. | per spec |
| `sleep_seconds` | top-level | Max random startup delay in D, staggers Athena hits. | `30` / `300` |
| `res_bsq_cols` / `com_bsq_cols` | top-level | Grouping cols in BSQ pulls. | see templates |
| `lag_hours_temperature` | top-level | Weather lag offsets used in regression features. | `[-1, 1, 3, 6, 12]` |
| `run_specs[i].name` | per spec | Canonical spec identifier (`All-Baseline`, `Upgraded-Upgrade8`, …). | per spec |
| `run_specs[i].apply_regression` | per spec | `true` train+predict, `false` pull base year directly. | per spec |
| `run_specs[i].base_year` / `target_year` | per spec | Base year for training; target year spec (`"2007-2024"` etc.) parsed in D. | `2018` / `["2007-2024"]` |
| `run_specs[i].upgrade_id` | per spec | Single int or list — list signals "sum across these upgrades". | `0` for baselines |
| `run_specs[i].restrict` / `avoid` | per spec | Optional applied-upgrade predicates (`{"all_of":[…], "any_of":[…]}`). | per spec |
| `run_types[<name>]` | top-level | Per-run BSQ config — workgroup, db, schema, table, plus optional `adjustment_factor`. | see templates |
| `cross_val` / `hybrid_model` / `test_target` / `test_base` | top-level | Regression / diagnostics toggles. | `true` / `true` / `false` / `false` |
| `save_metrics` / `show_fit` / `save_fit` | top-level | Diagnostic outputs. | `true` / `false` / `false` |

