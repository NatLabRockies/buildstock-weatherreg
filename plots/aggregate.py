"""Heavy aggregation step for the BuildStock projection dashboard.

Reads three handoff folders (ReEDs/, intermediate/state/, LBL/) from one
ResStock and one ComStock run_dir, pre-aggregates everything to the smallest
shapes the dashboard's four-tab plot set actually needs, and writes:

  plots/data/main.js          ~55 MB   (CONUS payload — sets window.PAYLOAD)
  plots/data/state_<postal>.js ~10 MB × 49  (per-state — lazy-loaded via
                                            <script> injection on click)

The dashboard itself is `plots/dashboard.html` (tracked in git, edit
directly). It references `data/main.js` via <script src>. After this
script writes new data/, refresh the browser to pick it up.

Re-run aggregate.py only when the source run_dirs change. Edit
dashboard.html directly to iterate on plot design — no build step.

EXACTLY WHAT GETS COMPUTED
==========================

A. ReEDs streaming  →  panel1 (CONUS trajectory)
   For each of the 24 ReEDs CSVs per stock (scenario × stock_year):
     - read the file (timestamp_EST index, 48 lowercase-state-name columns, MWh)
     - add res + com row-wise if both run_dirs have the file
     - sum across state columns → CONUS hourly MWh series
     - group by year-of-timestamp → 18 (annual GWh, peak GW) pairs per file
   Output:
     panel1.annual_gwh[scenario][stock_year][weather_year] = GWh
     panel1.peak_gw  [scenario][stock_year][weather_year] = GW

B. panel3_peak_week  (parallel; ProcessPool, 24 tasks)
   For each (scenario, stock_year):
     - load each of the 7 cohort `total` files (res NC/SA/SNA + com NC/SA/SNA/gap)
     - sum cohorts → res / com / total CONUS hourly GWh (= GW hourly)
     - for each weather year:
         summer peak = max(total) in Jun-Sep
         winter peak = max(total) in Dec-Feb
         extract ±3-day windows from the FULL series (not wy-sliced) so
         Jan-1 boundary peaks still get all 168 hours
   Output:
     panel3[scenario][stock_year][weather_year][summer|winter] =
       {timestamps, residential, commercial, peak_iso, peak_gw,
        cohorts: {res_NC: [168 GW], ..., com_gap: [168 GW]}}

C. panel_state_by_sector  (parallel; ProcessPool, 96 tasks)
   For each (scenario, stock_year, sector) where sector ∈ {residential,
   commercial, gap, total}:
     - read the cohort `total` files matching the (scenario, sector, year)
     - sum to hourly state-x-time matrix
     - per-state per-wy: ann = matrix.groupby(year).sum(),
                        peak = matrix.groupby(year).max()  [intra-state coincident]
     - CONUS: sum hourly across states first → joint hourly, then take max
              [correct coincident peak — sum-of-state-peaks would over-count]
   For sector='total' the task additionally computes per-(wy, state)
   peak_contributions: each sub-sector's value AT the total-peak hour.
   These sum to the total peak (coincident decomposition for the
   trajectory's Peak GW breakdown stacked area).
   Output:
     state_by_sector.annual_gwh[scenario][year][wy][sector][state]    = GWh
     state_by_sector.peak_gw   [scenario][year][wy][sector][state]    = GW
     state_by_sector.peak_contributions[scenario][year][wy][state]
                              = {residential, commercial, gap}        = GW

D. panel_cohort_daily_all_wys  (parallel; ProcessPool, 168 tasks)
   For each (scenario, stock_year, sector, cohort):
     - read the cohort `total` intermediate file
     - sum across state cols → CONUS hourly GWh
     - group by year-of-timestamp → per-wy daily resample (24h→1)
   Output:
     cohort_daily[scenario][year][wy] = {dates: [iso], cohorts: {key: [daily]}}

E. panel_intermediate_annual  (parallel; ProcessPool)
   For every intermediate/state `total` file:
     - read, sum across state cols → CONUS hourly
     - group by year → annual GWh per weather year
   Used for tabs 2 + 3 reconciliation against ReEDs and LBL.
   Output:
     intermediate_annual[scenario][year][wy][sector][cohort] = GWh

F. panel_lbl  (per-file polars; ProcessPool, POLARS_MAX_THREADS=1 per worker)
   For every LBL/*.csv (long-format, ~9M rows each):
     - polars scan_csv pushes SUM(value_kwh) down into the CSV scanner
     - kWh → GWh
   LBL ships only AMY 2012 and 2018 and excludes the commercial gap by spec.
   Output:
     lbl_annual[scenario][year][wy][sector][cohort] = GWh

G. Axis pin maxes  (cheap, in build_payload)
   * annual_gwh_max / peak_gw_max (CONUS legacy pins, rounded to step)
   * trajectory_max[state][sector][metric] — per-state per-sector global max
     across all (scenario, year, wy); used to pin y-axes so sliding
     doesn't rescale.
   * choropleth_max[sector][metric] — per-sector global max across the
     non-CONUS states; used to pin the choropleth color scale.

CLI
===
  uv run python plots/aggregate.py \\
      --res-run-dir /projects/geohc/radhikar/outputs/resstock_cross_val_may13_2026 \\
      --com-run-dir /projects/geohc/radhikar/outputs/comstock_cross_val_may13_2026
  # output: plots/payload.json

  # Then turn the payload into HTML (sub-second):
  uv run python plots/build_dashboard.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Repo root on sys.path so we can `from projections.X import Y`. The dashboard
# bake is launched as `python plots/aggregate.py`, which puts plots/ (not the
# repo root) first on sys.path — without this prepend the projections package
# imports below fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_T0: float = time.time()


def _log(msg: str) -> None:
    """Print a progress line prefixed with elapsed wall time, flushed immediately
    so the SLURM .out file shows progress in real time."""
    elapsed = time.time() - _T0
    mm, ss = divmod(int(elapsed), 60)
    print(f'[{mm:02d}:{ss:02d}] {msg}', flush=True)


# === Constants ============================================================
SCENARIOS: list[str] = ['Baseline', 'ASHP', 'GHP', 'GHP+Envelope']
BASELINE_YEAR: int = 2018                          # unprojected calibration year
PROJECTION_YEARS: list[int] = [2027, 2030, 2035, 2040, 2045, 2050]
STOCK_YEARS: list[int] = [BASELINE_YEAR] + PROJECTION_YEARS
RES_COHORTS: list[str] = ['NC', 'SA', 'SNA']
COM_COHORTS: list[str] = ['NC', 'SA', 'SNA', 'gap']
LBL_COHORTS: list[str] = ['NC', 'SA', 'SNA']  # LBL excludes gap by spec
LBL_WEATHER_YEARS: list[int] = [2012, 2018]   # LBL ships only these two AMYs

# 2018 is the unprojected calibration anchor — only Baseline is meaningful
# here. Adoption (ASHP/GHP/GHP+Envelope) begins at the projection anchor
# year (2027) per projections/growth_factors.py, so for any year before
# that the scenario chips would be identical-by-construction. The dashboard
# disables non-Baseline chips when the slider sits at 2018.
#
# Source data: county-level agg_<res|com>_eulp_total_GWh_upgradeAll-Baseline_
# reg_b2018.csv at the run_dir root. The series gets *scaled* (see
# _baseline2018_scale_factors) to map ResStock/ComStock-modeled stock onto
# AEO 2018 actual totals — without it the totals are off by 16% (res) /
# 44% (com) and don't reconcile against EIA.
BASELINE_2018_SPEC = {'res': 'All-Baseline', 'com': 'All-Baseline'}

# FIPS state code → 2-letter postal. AK + HI absent from this dataset.
# DC (FIPS 11) maps to MD per the same convention used elsewhere in this
# pipeline (ReEDs merges DC into MD).
_STATE_FIPS_TO_POSTAL: dict[int, str] = {
    1: 'AL',  4: 'AZ',  5: 'AR',  6: 'CA',  8: 'CO',  9: 'CT', 10: 'DE',
    11: 'MD',  # DC → MD
    12: 'FL', 13: 'GA', 16: 'ID', 17: 'IL', 18: 'IN', 19: 'IA', 20: 'KS',
    21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD', 25: 'MA', 26: 'MI', 27: 'MN',
    28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV', 33: 'NH', 34: 'NJ',
    35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 40: 'OK', 41: 'OR',
    42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN', 48: 'TX', 49: 'UT',
    50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV', 55: 'WI', 56: 'WY',
}


def _county_fips_to_state(col: str) -> str | None:
    """Map a county FIPS column header (4 or 5 digit, no leading zero) to its
    state postal. The agg files strip leading zeros — '1001' = state 1
    (Alabama) county 001; '11001' = state 11 (DC, → MD) county 001."""
    try:
        fips = int(col)
    except ValueError:
        return None
    return _STATE_FIPS_TO_POSTAL.get(fips // 1000)


def _baseline2018_scale_factors(
    res_run_dir: Path, com_run_dir: Path
) -> tuple[float, float]:
    """Return (scale_res, gap_ratio_com) — two *different* multiplicative
    corrections needed to map the raw ResStock/ComStock 2018 output onto
    AEO 2018 actuals.

    scale_res — household occupancy correction.
        aux_coverage.units_count sums ALL housing units in the ResStock
        sampling frame (~140 M, includes vacant). AEO's residential
        total is OCCUPIED households (~118 M). Multiply the modeled
        residential load by AEO/modeled so the result reflects load from
        occupied households only.

    gap_ratio_com — coverage-gap ratio for the unmodeled commercial.
        ComStock covers ~70 % of total US commercial floor space
        (modeled = 64 B sqft, AEO total = 92 B sqft). The remaining
        ~30 % is the dashboard's `gap` sector. By construction:
              gap_load = modeled_load × gap_ratio_com
        where gap_ratio_com = (AEO_sqft − modeled_sqft) / modeled_sqft.
        The commercial sector itself stays unscaled — only the gap is
        synthesized — so the modeled commercial value remains directly
        comparable to projection-year commercial values, while
        commercial+gap matches AEO actuals.

    AEO 2025 carries no 2018 data, so the 2018 vintage CSVs in `AEO 2025/`
    are the source for both AEO totals. AEO 2018 vs AEO 2025 diverges
    5–30 % for future years, so it must NOT be used for 2027–2050.
    """
    repo_dir = Path(__file__).resolve().parent.parent
    aeo_dir = repo_dir / 'AEO 2025'
    res_aeo = pd.read_csv(aeo_dir /
        'Residential_Sector_Key_Indicators_and_Consumption_2018.csv', skiprows=4)
    com_aeo = pd.read_csv(aeo_dir /
        'Commercial_Sector_Key_Indicators_and_Consumption_2018.csv', skiprows=4)
    name_col_res = res_aeo.columns[1]
    name_col_com = com_aeo.columns[1]
    aeo_res_total = float(res_aeo.loc[res_aeo[name_col_res] ==
        'Residential: Key Indicators: Households: Total: Reference case', '2018'].iloc[0])
    aeo_com_total = float(com_aeo.loc[com_aeo[name_col_com] ==
        'Commercial: Total Floorspace: Total: Reference case', '2018'].iloc[0])
    aeo_res_raw = aeo_res_total * 1e6   # millions HH → raw HH
    aeo_com_raw = aeo_com_total * 1e9   # billions sqft → raw sqft

    res_aux = pd.read_csv(res_run_dir / 'aux_coverage_upgradeAll-Baseline_reg_b2018.csv')
    com_aux = pd.read_csv(com_run_dir / 'aux_coverage_upgradeAll-Baseline_reg_b2018.csv')
    modeled_res = float(res_aux['units_count'].sum())
    modeled_com = float(com_aux['sqft'].sum())
    scale_res = aeo_res_raw / modeled_res
    gap_ratio_com = (aeo_com_raw - modeled_com) / modeled_com
    _log(f'  2018 calibration: scale_res={scale_res:.4f} '
         f'(modeled {modeled_res/1e6:.2f}M HH → AEO {aeo_res_total:.2f}M); '
         f'gap_ratio_com={gap_ratio_com:.4f} '
         f'(modeled {modeled_com/1e9:.2f}B sqft, AEO {aeo_com_total:.2f}B → '
         f'unmodeled fraction = {gap_ratio_com / (1 + gap_ratio_com):.3f})')
    return scale_res, gap_ratio_com

# ReEDs CSVs spell out state names in lowercase ("alabama, …"). Plotly's
# USA-states choropleth wants 2-letter postals. DC is intentionally absent
# (ReEDs merged DC into MD per spec).
_LOWERNAME_TO_POSTAL: dict[str, str] = {
    'alabama': 'AL', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
    'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL',
    'georgia': 'GA', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN',
    'iowa': 'IA', 'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA',
    'maine': 'ME', 'maryland': 'MD', 'massachusetts': 'MA', 'michigan': 'MI',
    'minnesota': 'MN', 'mississippi': 'MS', 'missouri': 'MO', 'montana': 'MT',
    'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH', 'new jersey': 'NJ',
    'new mexico': 'NM', 'new york': 'NY', 'north carolina': 'NC',
    'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR',
    'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
    'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA',
    'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY',
}

_REEDS_FILE_RE = re.compile(r'(?P<scenario>.+)_y(?P<year>\d{4})\.csv')
_INTERMEDIATE_FILE_RE = re.compile(
    r'(?P<scenario>.+)_(?P<sector>residential|commercial)_(?P<cohort>NC|SA|SNA|gap)_'
    r'(?P<enduse>cooling|heating|non_hvac|total)_y(?P<year>\d{4})\.csv'
)
_LBL_FILE_RE = re.compile(
    r'(?P<scenario>.+)_(?P<sector>residential|commercial)_(?P<cohort>NC|SA|SNA)_'
    r'(?P<year>\d{4})_amy(?P<wy>\d{4})\.csv'
)

# Process worker cap. ProcessPoolExecutor for guaranteed parallelism — pandas'
# GIL release is inconsistent across read / groupby / dict-build, so threads
# can stall. On Linux fork() is COW so passing the path dicts per task is cheap.
WORKERS: int = max(1, min(16, int(os.environ.get('BAKE_WORKERS', os.cpu_count() or 4)) - 1))


# === File-system catalogers ===============================================
def _parse_reeds_files(reeds_dir: Path) -> dict[tuple[str, int], Path]:
    out: dict[tuple[str, int], Path] = {}
    if not reeds_dir.is_dir():
        return out                                # tolerate missing dir (handoff may have bundled into the sibling run_dir)
    for p in sorted(reeds_dir.iterdir()):
        m = _REEDS_FILE_RE.match(p.name)
        if not m:
            continue
        out[(m['scenario'], int(m['year']))] = p
    return out


def _parse_intermediate_files(intermediate_state_dir: Path) -> dict[tuple[str, str, str, str, int], Path]:
    out: dict[tuple[str, str, str, str, int], Path] = {}
    if not intermediate_state_dir.is_dir():
        return out
    for p in sorted(intermediate_state_dir.iterdir()):
        m = _INTERMEDIATE_FILE_RE.match(p.name)
        if not m:
            continue
        out[(m['scenario'], m['sector'], m['cohort'], m['enduse'], int(m['year']))] = p
    return out


def _read_with_index(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col='timestamp_EST', parse_dates=['timestamp_EST'])


# === Helpers ==============================================================
def _sum_intermediate_to_conus(
    paths: dict[tuple[str, str, str, str, int], Path],
    scenario: str, sector: str, year: int, enduse: str = 'total',
) -> pd.Series | None:
    """Sum every (cohort) file matching (scenario, sector, year, enduse) into a
    single CONUS hourly series in GWh. None if no files matched."""
    keys = [(s, sec, c, e, y) for (s, sec, c, e, y) in paths
            if s == scenario and sec == sector and y == year and e == enduse]
    if not keys:
        return None
    total: pd.Series | None = None
    for k in keys:
        df = _read_with_index(paths[k])
        s = df.sum(axis=1)
        total = s if total is None else total.add(s, fill_value=0.0)
    return total


def _extract_window(res_wy: pd.Series, com_wy: pd.Series, total_wy: pd.Series,
                    center_ts: pd.Timestamp,
                    cohort_series_wy: dict[str, pd.Series] | None = None) -> dict | None:
    """Build a payload dict for a 7-day window centered on center_ts.

    Always includes timestamps + residential + commercial + peak metadata.
    If cohort_series_wy is provided, also includes per-cohort hourly windows
    for the hourly peak-week cohort decomposition panel.
    """
    win_start = center_ts - pd.Timedelta(days=3)
    win_end   = center_ts + pd.Timedelta(days=4) - pd.Timedelta(hours=1)
    res_win = res_wy.loc[win_start:win_end]
    com_win = com_wy.loc[win_start:win_end]
    tot_win = total_wy.loc[win_start:win_end]
    if len(res_win) < 24:
        return None
    out: dict = {
        'timestamps':  [t.isoformat() for t in res_win.index],
        'residential': [float(v) for v in res_win.values],
        'commercial':  [float(v) for v in com_win.values],
        'peak_iso':    center_ts.isoformat(),
        'peak_gw':     float(tot_win.max()),
    }
    if cohort_series_wy:
        out['cohorts'] = {
            k: [float(v) for v in s.loc[win_start:win_end].values]
            for k, s in cohort_series_wy.items()
        }
    return out


def _scenario_color() -> dict[str, str]:
    return {
        'Baseline':      '#7f8c8d',
        'ASHP':          '#2980b9',
        'GHP':           '#27ae60',
        'GHP+Envelope':  '#e67e22',
    }


# === Top-level worker functions (must be module-level for ProcessPool) =====
def _peak_week_task(args: tuple) -> tuple[str, int, dict]:
    """One (scenario, year) → seasonal peak windows per weather year.

    For CONUS: window centered on the CONUS-summed peak hour, with cohort
    series summed across states.

    For each state: window centered on THAT STATE's own peak hour in the
    season — so the panel reflects the state's actual peak (not its value
    at the CONUS peak hour). This makes the map's "TX summer 107 GW"
    match the cohort panel's "Summer peak 107 GW" for TX.
    """
    res_inter, com_inter, scenario, year = args
    cohort_dfs: dict[str, pd.DataFrame] = {}    # state-x-time per cohort

    for paths, file_sector, cohorts in [
        (res_inter, 'residential', RES_COHORTS),
        (com_inter, 'commercial',  COM_COHORTS),
    ]:
        prefix = 'res_' if file_sector == 'residential' else 'com_'
        for cohort in cohorts:
            key = (scenario, file_sector, cohort, 'total', year)
            p = paths.get(key)
            if p is None:
                continue
            cohort_dfs[prefix + cohort] = _read_with_index(p)

    if not cohort_dfs:
        return scenario, year, {}

    # Build sector-and-total state-x-time DataFrames.
    res_df: pd.DataFrame | None = None
    com_df: pd.DataFrame | None = None
    for k, df in cohort_dfs.items():
        if k.startswith('res_'):
            res_df = df if res_df is None else res_df.add(df, fill_value=0.0)
        else:
            com_df = df if com_df is None else com_df.add(df, fill_value=0.0)
    if res_df is None or com_df is None:
        return scenario, year, {}
    total_df = res_df.add(com_df, fill_value=0.0)
    total_conus = total_df.sum(axis=1)
    summer_months = [6, 7, 8, 9]
    winter_months = [12, 1, 2]

    def _extract_window(peak_ts, series_or_col_fn):
        """Return (timestamps, cohort_values_per_key) for the ±3-day window
        around peak_ts. series_or_col_fn(df) returns the series to slice
        for a given cohort df — either df.sum(axis=1) for CONUS or df[st]
        for a state."""
        win_start = peak_ts - pd.Timedelta(days=3)
        win_end   = peak_ts + pd.Timedelta(days=4) - pd.Timedelta(hours=1)
        idx = total_df.loc[win_start:win_end].index
        if len(idx) < 24:
            return None, None
        timestamps = [t.isoformat() for t in idx]
        cohort_window = {}
        for k, df in cohort_dfs.items():
            s = series_or_col_fn(df).loc[win_start:win_end]
            cohort_window[k] = [round(float(v), 2) for v in s.values]
        return timestamps, cohort_window

    result: dict[int, dict[str, dict]] = {}
    for wy in sorted(set(int(y) for y in total_df.index.year.unique())):
        wy_mask = total_df.index.year == wy
        wy_total_conus = total_conus[wy_mask]
        seasons: dict[str, dict] = {}

        for season_name, season_months in [('summer', summer_months),
                                           ('winter', winter_months)]:
            season_mask_conus = wy_total_conus.index.month.isin(season_months)
            if not season_mask_conus.any():
                continue
            conus_peak_ts = wy_total_conus[season_mask_conus].idxmax()

            # CONUS window — cohort series summed across states.
            timestamps_conus, conus_cohorts = _extract_window(
                conus_peak_ts, lambda df: df.sum(axis=1))
            if timestamps_conus is None:
                continue

            # Res/com sector windows (CONUS-summed) for back-compat.
            conus_res = [round(float(v), 2) for v in
                         res_df.sum(axis=1)
                         .loc[conus_peak_ts - pd.Timedelta(days=3):
                              conus_peak_ts + pd.Timedelta(days=4) - pd.Timedelta(hours=1)].values]
            conus_com = [round(float(v), 2) for v in
                         com_df.sum(axis=1)
                         .loc[conus_peak_ts - pd.Timedelta(days=3):
                              conus_peak_ts + pd.Timedelta(days=4) - pd.Timedelta(hours=1)].values]

            # Per-state: each state gets ITS OWN window centered on its
            # own peak hour in this season. This is what makes the cohort
            # panel's peak match the map's per-state seasonal peak.
            per_state: dict[str, dict] = {}
            wy_total_state = total_df[wy_mask]
            for st in total_df.columns:
                st_series = wy_total_state[st]
                st_season_mask = st_series.index.month.isin(season_months)
                if not st_season_mask.any():
                    continue
                st_peak_ts = st_series[st_season_mask].idxmax()
                ts_st, cohorts_st = _extract_window(
                    st_peak_ts, lambda df, st=st: df[st])
                if ts_st is None:
                    continue
                # The state's peak_gw IS its season max — equals what the
                # map's state_by_sector.peak_gw[sector='total'].summer reports.
                st_peak_gw = float(st_series.loc[st_peak_ts])
                per_state[st] = {
                    'timestamps': ts_st,
                    'peak_iso':   st_peak_ts.isoformat(),
                    'peak_gw':    round(st_peak_gw, 2),
                    'cohorts':    cohorts_st,
                }

            seasons[season_name] = {
                'timestamps':  timestamps_conus,
                'peak_iso':    conus_peak_ts.isoformat(),
                'peak_gw':     float(wy_total_conus.loc[conus_peak_ts]),
                'residential': conus_res,
                'commercial':  conus_com,
                'cohorts':     conus_cohorts,
                'per_state':   per_state,
            }

        if seasons:
            result[wy] = seasons
    return scenario, year, result


def _intermediate_annual_task(args: tuple) -> tuple[tuple, dict[int, float]]:
    """One intermediate/state `total` file → annual CONUS GWh per weather year."""
    key, p = args
    df = _read_with_index(p)
    conus = df.sum(axis=1)
    ann_by_wy = conus.groupby(conus.index.year).sum()
    return key, {int(wy): float(v) for wy, v in ann_by_wy.items()}


def _state_sector_task(args: tuple) -> tuple[str, int, str, dict, dict, dict | None]:
    """One (scenario, year, sector) → per-state + CONUS annual+peak per wy.

    The worker reads every cohort `total` file matching the (scenario, sector,
    year) request and sums them into one hourly state-x-time DataFrame, then
    groups by year-of-timestamp to get per-state per-wy `.sum()` (annual GWh)
    and `.max()` (peak GW). For CONUS, we sum *hourly* across states first
    (giving the joint hourly series), then take the max — this is the true
    coincident peak. Summing per-state peaks would over-count because peaks
    happen at different hours in different states.

    For sector='total' the task additionally returns a `coincident_decomp`
    dict carrying each sub-sector's value at the total peak hour, per
    (wy, state). Summing across {residential, commercial, gap} for a given
    (wy, state) equals the total peak — i.e. the stack adds up correctly.
    This is what the breakdown sub-tab needs for metric=Peak GW. For other
    sectors, coincident_decomp is None.

    `sector` is a *display sector*, not the intermediate file's sector column:
      * 'residential' → residential cohorts (NC, SA, SNA)
      * 'commercial'  → commercial cohorts EXCLUDING gap (NC, SA, SNA)
      * 'gap'         → commercial gap only
      * 'total'       → all of the above (read both res and com files)
    """
    res_inter, com_inter, scenario, year, sector = args

    def _sum_files(paths, file_sector, cohort_list) -> pd.DataFrame | None:
        acc: pd.DataFrame | None = None
        for cohort in cohort_list:
            key = (scenario, file_sector, cohort, 'total', year)
            p = paths.get(key)
            if p is None:
                continue
            df = _read_with_index(p)
            acc = df if acc is None else acc.add(df, fill_value=0.0)
        return acc

    # Build the hourly state-x-time matrix per *sub-sector* when the caller
    # asked for 'total'; otherwise just produce the one matrix for the sector.
    if sector == 'total':
        res_df = _sum_files(res_inter, 'residential', RES_COHORTS)
        com_df = _sum_files(com_inter, 'commercial', ['NC', 'SA', 'SNA'])
        gap_df = _sum_files(com_inter, 'commercial', ['gap'])
        parts: list[pd.DataFrame] = [d for d in (res_df, com_df, gap_df) if d is not None]
        if not parts:
            return scenario, year, sector, {}, {}, None
        total = parts[0].copy()
        for d in parts[1:]:
            total = total.add(d, fill_value=0.0)
    elif sector == 'residential':
        total = _sum_files(res_inter, 'residential', RES_COHORTS)
    elif sector == 'commercial':
        total = _sum_files(com_inter, 'commercial', ['NC', 'SA', 'SNA'])
    elif sector == 'gap':
        total = _sum_files(com_inter, 'commercial', ['gap'])
    else:
        raise ValueError(f'bad sector: {sector}')

    if total is None:
        return scenario, year, sector, {}, {}, None

    ann_dict, peak_dict, coincident = _compute_state_xt_stats(
        total=total,
        res_df=res_df if sector == 'total' else None,
        com_df=com_df if sector == 'total' else None,
        gap_df=gap_df if sector == 'total' else None,
        compute_coincident=(sector == 'total'),
    )
    return scenario, year, sector, ann_dict, peak_dict, coincident


def _compute_state_xt_stats(
    total: pd.DataFrame,
    res_df: pd.DataFrame | None = None,
    com_df: pd.DataFrame | None = None,
    gap_df: pd.DataFrame | None = None,
    compute_coincident: bool = False,
) -> tuple[dict, dict, dict | None]:
    """Shared math: given an hourly state-x-time matrix in GWh, return the
    per-wy per-state {annual_gwh, peak_gw} dicts plus the optional coincident
    decomposition. Reused by both the projection-year path
    (_state_sector_task) and the 2018 baseline path (_baseline2018_task) so
    the output shape — and the peak semantics — stay identical."""
    summer_months = [6, 7, 8, 9]
    winter_months = [12, 1, 2]

    ann = total.groupby(total.index.year).sum()
    peak = total.groupby(total.index.year).max()
    # CONUS = joint hourly sum, then aggregate. Correct coincident peak —
    # summing per-state peaks would over-count.
    conus_hourly = total.sum(axis=1)
    conus_ann = conus_hourly.groupby(conus_hourly.index.year).sum()
    conus_peak = conus_hourly.groupby(conus_hourly.index.year).max()

    ann_dict: dict = {}
    peak_dict: dict = {}
    for wy in ann.index:
        wy_int = int(wy)
        ann_dict[wy_int] = {st: float(v) for st, v in ann.loc[wy].items()}
        ann_dict[wy_int]['CONUS'] = float(conus_ann.loc[wy])

        wy_mask = total.index.year == wy
        wy_data = total[wy_mask]
        summer = wy_data[wy_data.index.month.isin(summer_months)]
        winter = wy_data[wy_data.index.month.isin(winter_months)]
        summer_max = summer.max() if not summer.empty else None
        winter_max = winter.max() if not winter.empty else None

        wy_conus = conus_hourly[wy_mask]
        s_conus = wy_conus[wy_conus.index.month.isin(summer_months)]
        w_conus = wy_conus[wy_conus.index.month.isin(winter_months)]

        peak_dict[wy_int] = {
            st: {
                'annual': float(v),
                'summer': float(summer_max[st]) if summer_max is not None else 0.0,
                'winter': float(winter_max[st]) if winter_max is not None else 0.0,
            }
            for st, v in peak.loc[wy].items()
        }
        peak_dict[wy_int]['CONUS'] = {
            'annual': float(conus_peak.loc[wy]),
            'summer': float(s_conus.max()) if not s_conus.empty else 0.0,
            'winter': float(w_conus.max()) if not w_conus.empty else 0.0,
        }

    coincident: dict | None = None
    if compute_coincident:
        coincident = {}
        for wy in ann.index:
            wy_int = int(wy)
            wy_mask = total.index.year == wy
            wy_total = total[wy_mask]
            wy_res = res_df[wy_mask] if res_df is not None else None
            wy_com = com_df[wy_mask] if com_df is not None else None
            wy_gap = gap_df[wy_mask] if gap_df is not None else None
            coincident[wy_int] = {}
            for st in wy_total.columns:
                t = wy_total[st].idxmax()
                coincident[wy_int][st] = {
                    'residential': float(wy_res.loc[t, st]) if wy_res is not None else 0.0,
                    'commercial':  float(wy_com.loc[t, st]) if wy_com is not None else 0.0,
                    'gap':         float(wy_gap.loc[t, st]) if wy_gap is not None else 0.0,
                }
            t = wy_total.sum(axis=1).idxmax()
            coincident[wy_int]['CONUS'] = {
                'residential': float(wy_res.loc[t].sum()) if wy_res is not None else 0.0,
                'commercial':  float(wy_com.loc[t].sum()) if wy_com is not None else 0.0,
                'gap':         float(wy_gap.loc[t].sum()) if wy_gap is not None else 0.0,
            }
    return ann_dict, peak_dict, coincident


def _roll_agg_b2018_to_state_xt(
    run_dir: Path, stock: str, specs: list[str], scale: float = 1.0,
) -> pd.DataFrame | None:
    """Read each `agg_<stock>_eulp_total_GWh_upgrade<spec>_reg_b2018.csv` in
    the run_dir, roll columns county→state via FIPS prefix, sum across specs,
    and multiply by `scale`. Uses polars for the CSV scan (~3 s vs pandas'
    60 s on this 157K × 3100 county shape). Returns an hourly state-x-time
    DataFrame in GWh, or None if no file matched. CONUS column is NOT added
    here — _compute_state_xt_stats does."""
    total: pd.DataFrame | None = None
    for spec in specs:
        p = run_dir / f'agg_{stock}_eulp_total_GWh_upgrade{spec}_reg_b2018.csv'
        if not p.exists():
            continue
        # In-memory read: one file at a time consumes ~4 GB peak (157K rows ×
        # 3107 county columns × 8 bytes). The sequential caller in
        # panel_state_by_sector reads one at a time, so total peak stays under
        # ~5 GB — well within the 64 GB SLURM budget. Streaming engine was
        # ~30× slower for this workload than collect().
        df_pl = pl.read_csv(p)
        groups: dict[str, list[str]] = {}
        for col in df_pl.columns:
            if col == 'timestamp_EST':
                continue
            st = _county_fips_to_state(col)
            if st is not None:
                groups.setdefault(st, []).append(col)
        if not groups:
            continue
        agg_exprs = [pl.col('timestamp_EST')]
        for st in sorted(groups):
            agg_exprs.append(
                pl.sum_horizontal([pl.col(c) for c in groups[st]]).alias(st))
        rolled_pl = df_pl.select(agg_exprs)
        del df_pl  # release the wide DataFrame before we touch pandas
        rolled = rolled_pl.to_pandas().set_index('timestamp_EST')
        rolled.index = pd.to_datetime(rolled.index)
        total = rolled if total is None else total.add(rolled, fill_value=0.0)
    if total is not None and scale != 1.0:
        total = total * scale
    return total


def _cohort_daily_allwys_task(args: tuple) -> tuple[str, int, str, str, dict]:
    """One (scenario, year, sector_in_file, cohort) intermediate `total` file →
    daily-aggregated GWh per (state, weather year) and the CONUS sum.

    Returns: (scenario, year, file_sector, cohort, by_wy) where
      by_wy[wy] = {'dates': [iso], 'CONUS': [daily], 'AL': [daily], ...}
    """
    paths, scenario, year, file_sector, cohort = args
    key = (scenario, file_sector, cohort, 'total', year)
    p = paths.get(key)
    if p is None:
        return scenario, year, file_sector, cohort, {}
    df = _read_with_index(p)              # rows=hours, cols=states
    by_wy: dict[int, dict] = {}
    for wy, group_idx in df.groupby(df.index.year).groups.items():
        wy_df = df.loc[group_idx]
        daily = wy_df.resample('D').sum()  # rows=days, cols=states
        # Daily GWh per state, rounded to 1 decimal (0.1 GWh resolution
        # ≈ 100 MWh; well below visualization granularity). CONUS keeps
        # full precision because it lives in main.js (small footprint).
        states_data: dict = {st: [round(float(v), 1) for v in daily[st].values]
                             for st in daily.columns}
        states_data['CONUS'] = [float(v) for v in daily.sum(axis=1).values]
        by_wy[int(wy)] = {
            'dates': [d.isoformat() for d in daily.index],
            'states': states_data,
        }
    return scenario, year, file_sector, cohort, by_wy


def _lbl_worker_init() -> None:
    """ProcessPool initializer: keep each worker's polars on a single thread.
    Effective only with the 'spawn' start method — see panel_lbl. With
    fork(), polars is already imported in the parent (we use it for the 2018
    baseline path) and the thread pool is fixed at fork time; the env var
    set here would no-op."""
    os.environ['POLARS_MAX_THREADS'] = '1'


def _lbl_one_file_task(path_str: str) -> tuple[str, str, str, int, int, float] | None:
    """One LBL CSV → (scenario, sector, cohort, year, wy, annual_gwh).

    Each LBL CSV is ~9M long-format rows; pandas would take ~30 s per file.
    polars `scan_csv` on a single file pushes the `SUM(value_kwh)` down into
    its CSV scanner and finishes in ~1-2 s with bounded memory (no
    materialization of the full table). Per-file dispatch via ProcessPool
    keeps total resident memory across N workers bounded at ~N × per-file
    chunk size instead of N × full-union size."""
    p = Path(path_str)
    m = _LBL_FILE_RE.match(p.name)
    if not m:
        return None
    total_kwh = pl.scan_csv(path_str).select(pl.col('value_kwh').sum()).collect().item()
    return (m['scenario'], m['sector'], m['cohort'], int(m['year']), int(m['wy']),
            float(total_kwh) / 1e6)


# === Panel builders =======================================================
def panel3_peak_week(
    res_inter: dict[tuple[str, str, str, str, int], Path],
    com_inter: dict[tuple[str, str, str, str, int], Path],
) -> tuple[dict, dict]:
    """Parallel over (scenario, year). Returns a tuple:
      (conus_only, per_state)
    where:
      conus_only[scenario][year][wy][season] = {timestamps, residential,
        commercial, cohorts, peak_iso, peak_gw}        ← main payload
      per_state[state][scenario][year][wy][season] =
        {timestamps, cohorts, peak_iso, peak_gw}        ← lazy sidecars
    """
    # 2018 baseline has no cohort breakdown (county-level agg files aren't
    # split by NC/SA/SNA) — peak-week panel skips 2018 and the dashboard
    # shows its existing "No data" placeholder for that cell.
    tasks = [(res_inter, com_inter, s, y) for s in SCENARIOS for y in PROJECTION_YEARS]
    conus_only: dict = {s: {} for s in SCENARIOS}
    per_state: dict = {}
    _log(f'  peak-week — {len(tasks)} (scenario, year) tasks across {WORKERS} processes')
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for scenario, year, result in pool.map(_peak_week_task, tasks):
            done += 1
            tag = 'skipped (no data)' if not result else f'{len(result)} wy'
            _log(f'    ({done:>2}/{len(tasks)}) {scenario} y{year}  {tag}')
            if not result:
                continue
            # Split each (wy, season) dict into a CONUS-only entry + a
            # per-state slice. Each per-state entry now carries its OWN
            # timestamps + peak_iso (windows are state-centered, not
            # CONUS-centered), so no merge with CONUS is needed at render time.
            conus_only[scenario][year] = {}
            for wy, seasons in result.items():
                for season, w in seasons.items():
                    state_block = w.pop('per_state', {})
                    for st, st_data in state_block.items():
                        (per_state.setdefault(st, {})
                                  .setdefault(scenario, {})
                                  .setdefault(year, {})
                                  .setdefault(wy, {}))[season] = st_data
                conus_only[scenario][year][wy] = seasons
    return conus_only, per_state


def panel_state_by_sector(
    res_inter: dict[tuple[str, str, str, str, int], Path],
    com_inter: dict[tuple[str, str, str, str, int], Path],
    res_run_dir: Path,
    com_run_dir: Path,
) -> dict:
    """Per-state annual + peak per (scenario, year, wy, sector).

    Projection years (2027–2050) read from intermediate/state via
    _state_sector_task. Baseline year (2018) reads from the county-level
    agg_*_b2018.csv files via _baseline2018_task and is rolled to state by
    FIPS prefix. Both paths return identical output shapes, so the merge
    logic below doesn't branch on year.
    """
    proj_tasks = [(res_inter, com_inter, s, y, sec)
                  for s in SCENARIOS for y in PROJECTION_YEARS
                  for sec in ('residential', 'commercial', 'gap', 'total')]
    b2018_tasks = [(res_run_dir, com_run_dir, s, sec)
                   for s in SCENARIOS
                   for sec in ('residential', 'commercial', 'gap', 'total')]

    annual: dict = {s: {y: {} for y in STOCK_YEARS} for s in SCENARIOS}
    peak:   dict = {s: {y: {} for y in STOCK_YEARS} for s in SCENARIOS}
    coincident_decomp: dict = {s: {y: {} for y in STOCK_YEARS} for s in SCENARIOS}

    def _absorb(scenario, year, sector, ann_d, peak_d, coinc):
        for wy, by_state in ann_d.items():
            annual[scenario][year].setdefault(wy, {})[sector] = by_state
        for wy, by_state in peak_d.items():
            peak[scenario][year].setdefault(wy, {})[sector] = by_state
        if coinc is not None:
            for wy, by_state in coinc.items():
                coincident_decomp[scenario][year][wy] = by_state

    _log(f'  state-by-sector projection — {len(proj_tasks)} tasks across {WORKERS} processes')
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for scenario, year, sector, ann_d, peak_d, coinc in pool.map(_state_sector_task, proj_tasks):
            done += 1
            _absorb(scenario, year, sector, ann_d, peak_d, coinc)
            if done == 1 or done == len(proj_tasks) or done % 16 == 0:
                _log(f'    ({done:>2}/{len(proj_tasks)}) {scenario}/{year}/{sector}  '
                     f'states={len(next(iter(ann_d.values()), {}))}')

    # 2018 path: only Baseline scenario (adoption begins at 2027 per
    # projections/growth_factors.py, so ASHP/GHP/+Env at 2018 would be
    # identical-by-construction to Baseline; we drop them rather than render
    # four overlapping curves). Sequential pre-load to keep memory bounded
    # (each agg file is ~4 GB in pandas; parallel × 4 files OOMs).
    scale_res, gap_ratio_com = _baseline2018_scale_factors(res_run_dir, com_run_dir)
    _log('  state-by-sector baseline 2018 — Baseline only')
    # Residential: scaled to occupied households (× ~0.84).
    # Commercial: unscaled — leave modeled commercial directly comparable to
    # projection-year commercial values.
    # Gap: synthesized from commercial × gap_ratio_com (= unmodeled-sqft /
    # modeled-sqft from aux_coverage vs AEO 2018), so commercial + gap = AEO
    # total commercial. Same gap semantic as projection years.
    res_total = _roll_agg_b2018_to_state_xt(
        res_run_dir, 'res', [BASELINE_2018_SPEC['res']], scale=scale_res)
    com_total = _roll_agg_b2018_to_state_xt(
        com_run_dir, 'com', [BASELINE_2018_SPEC['com']], scale=1.0)
    if res_total is None and com_total is None:
        _log('    no 2018 data found — skipping')
    else:
        gap_total = (com_total * gap_ratio_com) if com_total is not None else None
        parts = [d for d in (res_total, com_total, gap_total) if d is not None]
        total_sum = parts[0].copy()
        for d in parts[1:]:
            total_sum = total_sum.add(d, fill_value=0.0)
        sector_to_df = {
            'residential': res_total,
            'commercial':  com_total,
            'gap':         gap_total,
            'total':       total_sum,
        }
        for sec, total_df in sector_to_df.items():
            if total_df is None:
                continue
            ann_d, peak_d, coinc = _compute_state_xt_stats(
                total=total_df,
                res_df=res_total if sec == 'total' else None,
                com_df=com_total if sec == 'total' else None,
                gap_df=gap_total if sec == 'total' else None,
                compute_coincident=(sec == 'total'),
            )
            _absorb('Baseline', BASELINE_YEAR, sec, ann_d, peak_d, coinc)
            _log(f'    Baseline/{BASELINE_YEAR}/{sec}  '
                 f'states={len(next(iter(ann_d.values()), {}))}')

    return {
        'annual_gwh': annual,
        'peak_gw': peak,
        'peak_contributions': coincident_decomp,   # {scen}{year}{wy}{state} = {res, com, gap}
    }


def panel_cohort_daily_all_wys(
    res_inter: dict[tuple[str, str, str, str, int], Path],
    com_inter: dict[tuple[str, str, str, str, int], Path],
) -> tuple[dict, dict]:
    """Cohort daily decomposition for EVERY (scenario, year, weather_year).
    Returns a tuple:
      (conus_only, per_state)
    where:
      conus_only[scenario][year][wy] = {dates, cohorts: {key: [daily]}}     ← main
      per_state[state][scenario][year][wy] = {dates, cohorts: {key: [daily]}} ← lazy

    Tasks: 4 scenarios × 6 stock_years × 7 (sector,cohort) combos = 168.
    Each task now returns per-state daily series in addition to CONUS.
    """
    sector_cohorts = ([('residential', c) for c in RES_COHORTS]
                      + [('commercial',  c) for c in COM_COHORTS])
    # 2018 baseline has no cohort split (NC/SA/SNA come from intermediate
    # files; the 2018 agg files don't carry that dimension). Skip the
    # baseline year — the dashboard shows "No data" for those cells.
    tasks = []
    for s in SCENARIOS:
        for y in PROJECTION_YEARS:
            for sec, coh in sector_cohorts:
                paths = res_inter if sec == 'residential' else com_inter
                tasks.append((paths, s, y, sec, coh))

    conus_only: dict = {s: {y: {} for y in PROJECTION_YEARS} for s in SCENARIOS}
    per_state: dict = {}
    _log(f'  cohort_daily_all_wys — {len(tasks)} (scenario, year, sector, cohort) tasks across {WORKERS} processes')
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for scenario, year, file_sector, cohort, by_wy in pool.map(_cohort_daily_allwys_task, tasks):
            done += 1
            cohort_key = ('res_' if file_sector == 'residential' else 'com_') + cohort
            for wy, payload in by_wy.items():
                dates = payload['dates']
                states_data = payload['states']
                # CONUS goes into the main bucket.
                conus_entry = conus_only[scenario][year].setdefault(wy, {'dates': dates, 'cohorts': {}})
                conus_entry['cohorts'][cohort_key] = states_data['CONUS']
                # Per-state files DROP `dates` — same dates apply across
                # all states for the same (scenario, year, wy). JS reads
                # PAYLOAD.cohort_daily for the dates string array, saving
                # ~3 MB per state file.
                for st, daily_vals in states_data.items():
                    if st == 'CONUS':
                        continue
                    st_entry = (per_state.setdefault(st, {})
                                          .setdefault(scenario, {})
                                          .setdefault(year, {})
                                          .setdefault(wy, {'cohorts': {}}))
                    st_entry['cohorts'][cohort_key] = daily_vals
            if done == 1 or done == len(tasks) or done % 24 == 0:
                _log(f'    ({done:>3}/{len(tasks)}) {scenario} y{year} {cohort_key}')
    return conus_only, per_state


def panel_intermediate_annual(
    res_inter: dict[tuple[str, str, str, str, int], Path],
    com_inter: dict[tuple[str, str, str, str, int], Path],
) -> dict:
    """Parallel over every total-enduse intermediate file: annual GWh per wy."""
    out: dict[str, dict[int, dict[int, dict[str, dict[str, float]]]]] = {s: {} for s in SCENARIOS}
    for paths, sector in [(res_inter, 'residential'), (com_inter, 'commercial')]:
        relevant = [(k, p) for k, p in paths.items() if k[3] == 'total' and k[1] == sector]
        _log(f'  intermediate_annual — {sector}: {len(relevant)} files across {WORKERS} processes')
        done = 0
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for key, ann_by_wy in pool.map(_intermediate_annual_task, relevant):
                done += 1
                sc, sec, coh, _eu, y = key
                for wy, v in ann_by_wy.items():
                    (out.setdefault(sc, {})
                        .setdefault(y, {})
                        .setdefault(int(wy), {})
                        .setdefault(sec, {}))[coh] = v
                if done == 1 or done == len(relevant) or done % 20 == 0:
                    _log(f'    {sector} ({done:>3}/{len(relevant)}) {sc}/{coh}/y{y}')
    return out


def panel_stock_counts(res_run_dir: Path, com_run_dir: Path) -> dict:
    """Per-state, per-cohort stock-count trajectories on the ResStock-modeled
    basis (not the raw AEO total), at every stock year.

    The displayed counts reflect the stock that's actually represented in the
    projected energy data: the ResStock 2018 baseline (occupied households /
    modeled floor space) scaled by AEO cohort growth factors, NOT AEO totals
    directly. The two differ because the ResStock sampling frame's modeled
    units / sqft don't exactly equal AEO 2018 — residential matches within
    ~4 %, commercial undermodels by ~30 % (the gap sector).

    Cohort breakdown matches the energy cohorts in the intermediate files:
        residential: NC, SA, SNA
        commercial:  NC, SA, SNA, gap

    Calendar logic:
        Before ANCHOR_YEAR (2027), no adoption has begun and no new
        construction has accumulated — every modeled unit lives in the
        SNA cohort. The 2018 cell falls in this regime.
        From ANCHOR_YEAR onward, growth_factors' cohort_split functions
        define the per-cohort AEO fractions; we scale those by the
        ResStock-to-AEO ratio at the anchor year to land on the modeled
        basis (modeled_2018 / AEO_anchor).

    Per-state share comes from aux_coverage at 2018 (DC folded into MD,
    AK/HI dropped). The share is applied per cohort identically — this
    assumes state-level cohort proportions track the CONUS proportions.

    Output:
        {
          'residential_units': {year: {cohort: {state: M occupied HH}}},
          'commercial_sqft':   {year: {cohort: {state: B sqft}}},
        }
    """
    from projections.growth_factors import (
        residential_cohort_split, commercial_cohort_split,
        commercial_total_floorspace, ANCHOR_YEAR, GAP_FRACTION,
    )
    from projections.common import RES_OCCUPANCY_FRACTION

    aeo_dir = Path(__file__).resolve().parent.parent / 'AEO 2025'
    com_aeo18 = pd.read_csv(
        aeo_dir / 'Commercial_Sector_Key_Indicators_and_Consumption_2018.csv',
        skiprows=4)
    name_com = com_aeo18.columns[1]
    aeo_com_2018 = float(com_aeo18.loc[com_aeo18[name_com] ==
        'Commercial: Total Floorspace: Total: Reference case', '2018'].iloc[0])

    # Per-state share + 2018 modeled basis from aux_coverage.
    drop_states = {'AK', 'HI'}
    res_aux = pd.read_csv(res_run_dir / 'aux_coverage_upgradeAll-Baseline_reg_b2018.csv')
    com_aux = pd.read_csv(com_run_dir / 'aux_coverage_upgradeAll-Baseline_reg_b2018.csv')
    res_aux = res_aux[~res_aux['state'].isin(drop_states)].copy()
    com_aux = com_aux[~com_aux['state'].isin(drop_states)].copy()
    res_aux['state'] = res_aux['state'].replace({'DC': 'MD'})
    com_aux['state'] = com_aux['state'].replace({'DC': 'MD'})
    res_state_share = (res_aux.groupby('state')['units_count'].sum()
                       / res_aux['units_count'].sum()).to_dict()
    com_state_share = (com_aux.groupby('state')['sqft'].sum()
                       / com_aux['sqft'].sum()).to_dict()
    modeled_res_2018 = res_aux['units_count'].sum() * RES_OCCUPANCY_FRACTION / 1e6
    modeled_com_non_gap_2018 = com_aux['sqft'].sum() / 1e9
    gap_ratio_com_2018 = (aeo_com_2018 - modeled_com_non_gap_2018) / modeled_com_non_gap_2018

    # Scale factors: modeled / AEO at the anchor year (used to map AEO
    # cohort_split values onto the ResStock-modeled basis).
    anchor_res_total       = residential_cohort_split(ANCHOR_YEAR)['total_households']
    anchor_com_non_gap     = commercial_cohort_split(ANCHOR_YEAR)['total_floorspace'] * (1 - GAP_FRACTION)
    ratio_res              = modeled_res_2018 / anchor_res_total
    ratio_com_non_gap      = modeled_com_non_gap_2018 / anchor_com_non_gap

    def _per_state(value: float, share: dict[str, float]) -> dict[str, float]:
        out = {st: float(value * s) for st, s in share.items()}
        out['CONUS'] = float(value)
        return out

    residential_units: dict = {}
    commercial_sqft:   dict = {}
    for year in STOCK_YEARS:
        if year < ANCHOR_YEAR:
            # 2018 calibration year. No adoption, no new construction —
            # every modeled unit/sqft lives in SNA. Gap (commercial only) is
            # the unmodeled portion at 2018.
            res_NC, res_SA = 0.0, 0.0
            res_SNA = modeled_res_2018
            com_NC, com_SA = 0.0, 0.0
            com_SNA = modeled_com_non_gap_2018
            com_gap = modeled_com_non_gap_2018 * gap_ratio_com_2018
        else:
            res = residential_cohort_split(year)
            com = commercial_cohort_split(year)
            res_NC  = res['cumulative_new_households']                 * ratio_res
            res_SA  = res['adopted_existing_households']               * ratio_res
            res_SNA = (res['eligible_not_adopted_existing_households']
                       + res['ineligible_existing_households'])         * ratio_res
            # Commercial non-gap cohorts on the modeled basis. The cohort
            # values from growth_factors are already non-gap-scoped for the
            # adopted/eligible/ineligible terms; only cumulative_new needs
            # the (1 - GAP_FRACTION) factor (the dict mixes scopes).
            com_NC  = com['cumulative_new_floorspace'] * (1 - GAP_FRACTION) * ratio_com_non_gap
            com_SA  = com['adopted_existing_floorspace']                    * ratio_com_non_gap
            com_SNA = (com['eligible_not_adopted_existing_floorspace']
                       + com['ineligible_existing_floorspace'])             * ratio_com_non_gap
            # Commercial gap: scale 2018 gap by AEO total commercial growth.
            # This keeps the gap proportional to the non-gap on the modeled
            # basis as time advances, matching the energy panel's gap logic.
            com_gap = (modeled_com_non_gap_2018 * gap_ratio_com_2018
                       * commercial_total_floorspace(year) / aeo_com_2018)

        residential_units[year] = {
            'NC':  _per_state(res_NC,  res_state_share),
            'SA':  _per_state(res_SA,  res_state_share),
            'SNA': _per_state(res_SNA, res_state_share),
        }
        commercial_sqft[year] = {
            'NC':  _per_state(com_NC,  com_state_share),
            'SA':  _per_state(com_SA,  com_state_share),
            'SNA': _per_state(com_SNA, com_state_share),
            'gap': _per_state(com_gap, com_state_share),
        }

    _log(f'  stock_counts (modeled basis): '
         f'2018 res={modeled_res_2018:.1f}M HH all-SNA, '
         f'com_non_gap={modeled_com_non_gap_2018:.1f}B sqft + gap={modeled_com_non_gap_2018*gap_ratio_com_2018:.1f}B; '
         f'2050 res_total={sum(residential_units[2050][c]["CONUS"] for c in ("NC","SA","SNA")):.1f}M HH, '
         f'com_total={sum(commercial_sqft[2050][c]["CONUS"] for c in ("NC","SA","SNA","gap")):.1f}B sqft')
    return {
        'residential_units': residential_units,
        'commercial_sqft':   commercial_sqft,
    }


def panel_lbl(res_lbl_dir: Path, com_lbl_dir: Path) -> dict:
    """Per-file ProcessPool: each worker polars-scans one LBL CSV and returns
    its annual GWh scalar. Avoids the all-files-in-one-scan memory blow-up
    (the cross-file group_by was forcing materialization of the union)."""
    out: dict[str, dict[int, dict[int, dict[str, dict[str, float]]]]] = {s: {} for s in SCENARIOS}

    all_paths: list[str] = []
    for lbl_dir in (res_lbl_dir, com_lbl_dir):
        if not lbl_dir.exists():
            _log(f'  {lbl_dir} does not exist — skipping'); continue
        ts = [p for p in sorted(lbl_dir.iterdir())
              if _LBL_FILE_RE.match(p.name) and not p.name.startswith('aux_samples_')]
        if not ts:
            _log(f'  {lbl_dir.name}/ has no LBL timeseries CSVs — skipping'); continue
        _log(f'  {lbl_dir.parent.name}/{lbl_dir.name}: queueing {len(ts)} CSVs')
        all_paths.extend(str(p) for p in ts)

    if not all_paths:
        return out
    _log(f'  LBL — {len(all_paths)} CSVs across {WORKERS} processes (polars 1-thread per worker)')
    done = 0
    # Spawn (not fork) so workers don't inherit the parent's polars thread
    # pool. The 2018 baseline path runs polars in the parent before this
    # stage, and fork()-inherited polars threads deadlock when workers try
    # to spin up their own scan_csv reads.
    spawn_ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=spawn_ctx,
                              initializer=_lbl_worker_init) as pool:
        for result in pool.map(_lbl_one_file_task, all_paths):
            done += 1
            if result is None:
                continue
            sc, sec, coh, year, wy, gwh = result
            (out.setdefault(sc, {})
                .setdefault(year, {})
                .setdefault(wy, {})
                .setdefault(sec, {}))[coh] = gwh
            if done == 1 or done == len(all_paths) or done % 30 == 0:
                _log(f'    LBL ({done:>3}/{len(all_paths)}) {sc}/{sec}/{coh}/y{year}/amy{wy}'
                     f' = {gwh:,.0f} GWh')
    return out


# === Top-level orchestrator ===============================================
def build_payload(res_run_dir: Path, com_run_dir: Path) -> dict:
    res_reeds = _parse_reeds_files(res_run_dir / 'ReEDs')
    com_reeds = _parse_reeds_files(com_run_dir / 'ReEDs')
    reeds_keys = sorted(set(res_reeds) | set(com_reeds))
    _log(f'A. ReEDs streaming — res={len(res_reeds)} files, com={len(com_reeds)} files, merged={len(reeds_keys)}')

    # ReEDs only needs CONUS scalars now (panel1 CONUS trajectory). Per-state
    # annual + peak comes from intermediate/state files via state_by_sector —
    # that path has the sector dimension we need and produces the same per-state
    # numbers as ReEDs (verified to 0.000 % in the prior bake).
    series_annual: dict = {s: {} for s in SCENARIOS}
    series_peak:   dict = {s: {} for s in SCENARIOS}
    wy_seen: set[int] = set()

    for i, k in enumerate(reeds_keys, 1):
        scenario, year = k
        r = _read_with_index(res_reeds[k]) if k in res_reeds else None
        c = _read_with_index(com_reeds[k]) if k in com_reeds else None
        if r is None and c is None:
            continue
        df = c if r is None else (r if c is None else r.add(c, fill_value=0.0))
        conus = df.sum(axis=1)
        g = conus.groupby(conus.index.year)
        ann = (g.sum() / 1000.0).to_dict()
        pk  = (g.max() / 1000.0).to_dict()
        series_annual.setdefault(scenario, {})[year] = {int(w): float(v) for w, v in ann.items()}
        series_peak.setdefault(scenario,   {})[year] = {int(w): float(v) for w, v in pk.items()}
        wy_seen.update(ann.keys())
        _log(f'  ReEDs {i:>2}/{len(reeds_keys):<2} {scenario} y{year}  '
             f'(CONUS ann={max(ann.values())/1e6:.2f} MGWh, peak={max(pk.values()):.1f} GW)')

    panel1 = {'annual_gwh': series_annual, 'peak_gw': series_peak,
              'weather_years': sorted(int(w) for w in wy_seen)}

    res_inter = _parse_intermediate_files(res_run_dir / 'intermediate' / 'state')
    com_inter = _parse_intermediate_files(com_run_dir / 'intermediate' / 'state')
    _log(f'Intermediate index: res={len(res_inter)} com={len(com_inter)} files')

    _log('B. panel3_peak_week...')
    panel3_conus, panel3_per_state = panel3_peak_week(res_inter, com_inter)

    _log('C. panel_state_by_sector — per-state annual+peak per sector...')
    state_by_sector = panel_state_by_sector(res_inter, com_inter, res_run_dir, com_run_dir)

    # panel1 (CONUS trajectory) at the 2018 baseline: ReEDs has no 2018, so
    # derive it from state_by_sector.total.CONUS. The test suite verifies
    # this identity holds for the projection years — same source of truth,
    # one year earlier.
    for scen in SCENARIOS:
        ann_2018  = state_by_sector['annual_gwh'].get(scen, {}).get(BASELINE_YEAR, {})
        peak_2018 = state_by_sector['peak_gw']   .get(scen, {}).get(BASELINE_YEAR, {})
        if not ann_2018 or not peak_2018:
            continue
        series_annual.setdefault(scen, {})[BASELINE_YEAR] = {
            int(wy): float(by_sec['total']['CONUS'])
            for wy, by_sec in ann_2018.items()
        }
        series_peak.setdefault(scen, {})[BASELINE_YEAR] = {
            int(wy): float(by_sec['total']['CONUS']['annual'])
            for wy, by_sec in peak_2018.items()
        }
        wy_seen.update(int(w) for w in peak_2018.keys())
    panel1['weather_years'] = sorted(int(w) for w in wy_seen)

    _log('D. panel_cohort_daily_all_wys — daily cohort decomp, every wy...')
    cohort_daily_conus, cohort_daily_per_state = panel_cohort_daily_all_wys(res_inter, com_inter)

    _log('E. panel_intermediate_annual (kept for reconciliation)...')
    intermediate_annual = panel_intermediate_annual(res_inter, com_inter)

    _log('F. panel_lbl (per-file polars)...')
    lbl_annual = panel_lbl(res_run_dir / 'LBL', com_run_dir / 'LBL')
    _log(f'   LBL cells populated: {sum(1 for s in lbl_annual.values() for y in s.values() for w in y.values())}')

    _log('F.1 panel_stock_counts (per-state HH + sqft per stock year)...')
    stock_counts = panel_stock_counts(res_run_dir, com_run_dir)

    _log('G. axis pin maxes (per-state + per-sector)...')
    def _ceil_to_step(v: float, step: float) -> float:
        return float(np.ceil(v / step) * step)

    # CONUS axis pins (used by panel1-style summary; kept for back-compat).
    max_annual = max((v for by in panel1['annual_gwh'].values() for by_y in by.values() for v in by_y.values()),
                     default=1.0)
    max_peak = max((v for by in panel1['peak_gw'].values() for by_y in by.values() for v in by_y.values()),
                   default=1.0)

    # Per (state, sector, metric) max — for state-dropdown trajectory axes.
    # Walk state_by_sector and find max per (state, sector, metric) across all
    # (scenario, year, wy). Globally fixed → slider doesn't rescale.
    # peak_gw values are now dicts {annual, summer, winter}; for axis pinning
    # we use the annual scalar (the absolute peak across the whole year).
    traj_max: dict[str, dict[str, dict[str, float]]] = {}
    for metric_name, by_scen in (('annual_gwh', state_by_sector['annual_gwh']),
                                  ('peak_gw',    state_by_sector['peak_gw'])):
        for scen, by_year in by_scen.items():
            for year, by_wy in by_year.items():
                for wy, by_sec in by_wy.items():
                    for sector, by_state in by_sec.items():
                        for state, v in by_state.items():
                            scalar = v['annual'] if isinstance(v, dict) else v
                            d = traj_max.setdefault(state, {}).setdefault(sector, {})
                            if scalar > d.get(metric_name, 0.0):
                                d[metric_name] = float(scalar)

    # Per (sector, metric) max — for choropleth colorbar.
    chor_max: dict[str, dict[str, float]] = {}
    for metric_name in ('annual_gwh', 'peak_gw'):
        for state, by_sec in traj_max.items():
            if state == 'CONUS':
                continue
            for sector, by_metric in by_sec.items():
                v = by_metric.get(metric_name, 0.0)
                d = chor_max.setdefault(sector, {})
                if v > d.get(metric_name, 0.0):
                    d[metric_name] = float(v)

    # Main payload (CONUS-only for cohort daily + peak week; everything else
    # fully included). Per-state cohort + peak week data live in sidecar files.
    main_payload = {
        'scenarios':         SCENARIOS,
        'stock_years':       STOCK_YEARS,
        'res_cohorts':       RES_COHORTS,
        'com_cohorts':       COM_COHORTS,
        'lbl_cohorts':       LBL_COHORTS,
        'lbl_weather_years': LBL_WEATHER_YEARS,
        'colors':            _scenario_color(),
        'sectors':           ['residential', 'commercial', 'gap', 'total'],
        'states':            sorted(s for s in (traj_max.keys()) if s != 'CONUS'),
        'axis': {
            'annual_gwh_max': _ceil_to_step(max_annual * 1.10, 500_000.0),
            'peak_gw_max':    _ceil_to_step(max_peak * 1.10, 100.0),
            'trajectory_max': traj_max,
            'choropleth_max': chor_max,
        },
        'panel1':              panel1,
        'panel3':              panel3_conus,
        'state_by_sector':     state_by_sector,
        'cohort_daily':        cohort_daily_conus,
        'intermediate_annual': intermediate_annual,
        'lbl_annual':          lbl_annual,
        'stock_counts':        stock_counts,
        'meta': {
            'res_run_dir': str(res_run_dir),
            'com_run_dir': str(com_run_dir),
            'generated_iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }
    # Per-state sidecars: one entry per state. The dashboard lazy-loads these.
    state_keys = sorted(set(panel3_per_state.keys()) | set(cohort_daily_per_state.keys()))
    per_state_sidecars: dict[str, dict] = {
        st: {
            'cohort_daily': cohort_daily_per_state.get(st, {}),
            'peak_week':    panel3_per_state.get(st, {}),
        }
        for st in state_keys
    }
    return {'main': main_payload, 'per_state': per_state_sidecars}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--res-run-dir', type=Path, required=True,
                    help='ResStock run dir containing ReEDs/, intermediate/state/, LBL/.')
    ap.add_argument('--com-run-dir', type=Path, required=True,
                    help='ComStock run dir containing ReEDs/, intermediate/state/, LBL/.')
    ap.add_argument('--out-dir', type=Path, default=Path(__file__).parent / 'data',
                    help='Output dir for main.js + state_<postal>.js sidecars '
                         '(default: plots/data/).')
    args = ap.parse_args()

    _log(f'aggregate.py — WORKERS={WORKERS} (env BAKE_WORKERS={os.environ.get("BAKE_WORKERS","unset")}, cpu_count={os.cpu_count()})')
    _log(f'  res_run_dir: {args.res_run_dir}')
    _log(f'  com_run_dir: {args.com_run_dir}')

    bundle = build_payload(args.res_run_dir, args.com_run_dir)

    _log('Writing data/ sidecars...')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # data/main.js — sets window.PAYLOAD via an IIFE; fires onPayloadLoaded if
    # the dashboard wants to schedule something after main load.
    main_json = json.dumps(bundle['main'], separators=(',', ':'))
    main_js = (
        '(function(){\n'
        f'  window.PAYLOAD = {main_json};\n'
        '  if (window.onPayloadLoaded) window.onPayloadLoaded();\n'
        '})();\n'
    )
    main_path = args.out_dir / 'main.js'
    main_path.write_text(main_js)
    _log(f'  main.js  → {main_path.stat().st_size/1e6:.2f} MB')

    # data/state_<postal>.js — one per state. Registers into window.STATE_DATA
    # and fires onStateLoaded(postal) if a callback is set.
    per_state = bundle['per_state']
    _log(f'  per-state — {len(per_state)} sidecars')
    total_state_bytes = 0
    for st, st_data in per_state.items():
        st_json = json.dumps(st_data, separators=(',', ':'))
        st_js = (
            '(function(){\n'
            '  window.STATE_DATA = window.STATE_DATA || {};\n'
            f'  window.STATE_DATA[{json.dumps(st)}] = {st_json};\n'
            f'  if (window.onStateLoaded) window.onStateLoaded({json.dumps(st)});\n'
            '})();\n'
        )
        st_path = args.out_dir / f'state_{st}.js'
        st_path.write_text(st_js)
        total_state_bytes += st_path.stat().st_size
    _log(f'  state_*.js total → {total_state_bytes/1e6:.2f} MB across {len(per_state)} files'
         f' ({total_state_bytes/max(len(per_state),1)/1e6:.2f} MB avg)')
    _log('Done.')


if __name__ == '__main__':
    main()
