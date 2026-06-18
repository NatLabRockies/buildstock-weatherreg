"""Heavy aggregation step for the BuildStock projection dashboard.

Reads three handoff folders (ReEDs/, intermediate/state/, LBL/) from one
ResStock and one ComStock run_dir, pre-aggregates everything to the smallest
shapes the dashboard's four-tab plot set actually needs, and writes the
result as a single payload.json file to disk.

This file is the *intermediate* between heavy I/O and HTML emission, so you
can iterate on plot design (dashboard_template.html, build_dashboard.py)
without re-reading the source CSVs. Re-run aggregate.py only when the
source run_dirs change.

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
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


_T0: float = time.time()


def _log(msg: str) -> None:
    """Print a progress line prefixed with elapsed wall time, flushed immediately
    so the SLURM .out file shows progress in real time."""
    elapsed = time.time() - _T0
    mm, ss = divmod(int(elapsed), 60)
    print(f'[{mm:02d}:{ss:02d}] {msg}', flush=True)


# === Constants ============================================================
SCENARIOS: list[str] = ['Baseline', 'ASHP', 'GHP', 'GHP+Envelope']
STOCK_YEARS: list[int] = [2027, 2030, 2035, 2040, 2045, 2050]
RES_COHORTS: list[str] = ['NC', 'SA', 'SNA']
COM_COHORTS: list[str] = ['NC', 'SA', 'SNA', 'gap']
LBL_COHORTS: list[str] = ['NC', 'SA', 'SNA']  # LBL excludes gap by spec
LBL_WEATHER_YEARS: list[int] = [2012, 2018]   # LBL ships only these two AMYs

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
    for p in sorted(reeds_dir.iterdir()):
        m = _REEDS_FILE_RE.match(p.name)
        if not m:
            continue
        out[(m['scenario'], int(m['year']))] = p
    return out


def _parse_intermediate_files(intermediate_state_dir: Path) -> dict[tuple[str, str, str, str, int], Path]:
    out: dict[tuple[str, str, str, str, int], Path] = {}
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
    """One (scenario, year) → seasonal peak windows per weather year, with
    per-cohort hourly slices for the cohort decomposition panel.

    Loads each of the 7 cohort 'total' intermediate files individually (so
    per-cohort series are retained), sums them to res / com / total CONUS,
    finds summer + winter peak hours per weather year, and extracts the
    ±3-day window for both res+com (back-compat) and per-cohort traces.
    """
    res_inter, com_inter, scenario, year = args
    cohort_series_gw: dict[str, pd.Series] = {}

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
            df = _read_with_index(p)
            cohort_series_gw[prefix + cohort] = df.sum(axis=1)  # CONUS hourly GWh

    if not cohort_series_gw:
        return scenario, year, {}

    # Sector totals from the cohort series we have in memory. Use the
    # accumulator-pattern (`None → first, then .add`) — `sum(... , start=empty)`
    # gives NaN because pandas aligns indexes against the empty Series.
    res_gw: pd.Series | None = None
    com_gw: pd.Series | None = None
    for k, s in cohort_series_gw.items():
        if k.startswith('res_'):
            res_gw = s if res_gw is None else res_gw.add(s, fill_value=0.0)
        else:
            com_gw = s if com_gw is None else com_gw.add(s, fill_value=0.0)
    if res_gw is None or com_gw is None:
        return scenario, year, {}
    total_gw = res_gw.add(com_gw, fill_value=0.0)

    # Find peaks per (wy, season) by filtering the full series with masks,
    # but extract the ±3-day windows from the FULL series so windows that
    # straddle the Jan 1 boundary still get all 168 hours.
    full_year_idx = total_gw.index.year
    result: dict[int, dict[str, dict]] = {}
    for wy in sorted(set(int(y) for y in full_year_idx.unique())):
        wy_mask = (full_year_idx == wy)
        wy_total = total_gw[wy_mask]
        summer_mask = wy_total.index.month.isin([6, 7, 8, 9])
        winter_mask = wy_total.index.month.isin([12, 1, 2])
        summer_peak = wy_total[summer_mask].idxmax() if summer_mask.any() else None
        winter_peak = wy_total[winter_mask].idxmax() if winter_mask.any() else None

        seasons: dict[str, dict] = {}
        if summer_peak is not None:
            w = _extract_window(res_gw, com_gw, total_gw, summer_peak, cohort_series_gw)
            if w is not None:
                seasons['summer'] = w
        if winter_peak is not None:
            w = _extract_window(res_gw, com_gw, total_gw, winter_peak, cohort_series_gw)
            if w is not None:
                seasons['winter'] = w
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

    # Per-state per-wy: peak is intra-state coincident.
    ann = total.groupby(total.index.year).sum()
    peak = total.groupby(total.index.year).max()
    # CONUS: build joint hourly first, then aggregate. This is the *correct*
    # coincident peak — summing per-state peaks would over-count.
    conus_hourly = total.sum(axis=1)
    conus_ann = conus_hourly.groupby(conus_hourly.index.year).sum()
    conus_peak = conus_hourly.groupby(conus_hourly.index.year).max()

    ann_dict: dict = {}
    peak_dict: dict = {}
    for wy in ann.index:
        wy_int = int(wy)
        ann_dict[wy_int] = {st: float(v) for st, v in ann.loc[wy].items()}
        ann_dict[wy_int]['CONUS'] = float(conus_ann.loc[wy])
        peak_dict[wy_int] = {st: float(v) for st, v in peak.loc[wy].items()}
        peak_dict[wy_int]['CONUS'] = float(conus_peak.loc[wy])

    # Coincident decomposition: per (wy, state) → {res, com, gap} at the hour
    # of that state's total peak. CONUS uses the hour of CONUS total peak.
    # Summing the three sub-sectors at the chosen hour yields the total peak.
    coincident: dict | None = None
    if sector == 'total':
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
            # CONUS: at the hour of CONUS-summed total peak, sum each sub-sector
            # across states.
            t = wy_total.sum(axis=1).idxmax()
            coincident[wy_int]['CONUS'] = {
                'residential': float(wy_res.loc[t].sum()) if wy_res is not None else 0.0,
                'commercial':  float(wy_com.loc[t].sum()) if wy_com is not None else 0.0,
                'gap':         float(wy_gap.loc[t].sum()) if wy_gap is not None else 0.0,
            }

    return scenario, year, sector, ann_dict, peak_dict, coincident


def _cohort_daily_allwys_task(args: tuple) -> tuple[str, int, str, str, dict[int, list[float]], dict[int, list[str]]]:
    """One (scenario, year, sector_in_file, cohort) intermediate `total` file →
    daily-aggregated CONUS GWh for EVERY weather year. Returns:
      (scenario, year, file_sector, cohort, {wy: [daily_gwh]}, {wy: [iso_dates]})
    """
    paths, scenario, year, file_sector, cohort = args
    key = (scenario, file_sector, cohort, 'total', year)
    p = paths.get(key)
    if p is None:
        return scenario, year, file_sector, cohort, {}, {}
    df = _read_with_index(p)
    conus = df.sum(axis=1)  # GWh hourly CONUS
    daily_by_wy: dict[int, list[float]] = {}
    dates_by_wy: dict[int, list[str]] = {}
    for wy, group_idx in conus.groupby(conus.index.year).groups.items():
        s = conus.loc[group_idx]
        daily = s.resample('D').sum()
        daily_by_wy[int(wy)] = [float(v) for v in daily.values]
        dates_by_wy[int(wy)] = [d.isoformat() for d in daily.index]
    return scenario, year, file_sector, cohort, daily_by_wy, dates_by_wy


def _lbl_worker_init() -> None:
    """ProcessPool initializer: keep each worker's polars on a single thread.
    Without this, 15 workers × polars' default ~104-thread pool grossly
    over-subscribes the node and creates memory pressure from polars' chunk
    buffers."""
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
) -> dict:
    """Parallel over (scenario, year): summer + winter peak weeks per wy."""
    tasks = [(res_inter, com_inter, s, y) for s in SCENARIOS for y in STOCK_YEARS]
    weeks: dict[str, dict[int, dict[int, dict[str, dict]]]] = {s: {} for s in SCENARIOS}
    _log(f'  peak-week — {len(tasks)} (scenario, year) tasks across {WORKERS} processes')
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for scenario, year, result in pool.map(_peak_week_task, tasks):
            done += 1
            tag = 'skipped (no data)' if not result else f'{len(result)} wy'
            _log(f'    ({done:>2}/{len(tasks)}) {scenario} y{year}  {tag}')
            if result:
                weeks[scenario][year] = result
    return weeks


def panel_state_by_sector(
    res_inter: dict[tuple[str, str, str, str, int], Path],
    com_inter: dict[tuple[str, str, str, str, int], Path],
) -> dict:
    """Per-state annual + peak per (scenario, year, wy, sector).

    Tasks: 4 scenarios × 6 years × 4 sectors = 96. Each task computes its own
    'CONUS' pseudo-state from the joint hourly series (the coincident peak),
    so there's no parent-side combining that could over-count.
    """
    tasks = [(res_inter, com_inter, s, y, sec)
             for s in SCENARIOS for y in STOCK_YEARS
             for sec in ('residential', 'commercial', 'gap', 'total')]

    annual: dict = {s: {y: {} for y in STOCK_YEARS} for s in SCENARIOS}
    peak:   dict = {s: {y: {} for y in STOCK_YEARS} for s in SCENARIOS}
    coincident_decomp: dict = {s: {y: {} for y in STOCK_YEARS} for s in SCENARIOS}

    _log(f'  state-by-sector — {len(tasks)} (scenario, year, sector) tasks across {WORKERS} processes')
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for scenario, year, sector, ann_d, peak_d, coinc in pool.map(_state_sector_task, tasks):
            done += 1
            for wy, by_state in ann_d.items():
                annual[scenario][year].setdefault(wy, {})[sector] = by_state
            for wy, by_state in peak_d.items():
                peak[scenario][year].setdefault(wy, {})[sector] = by_state
            if coinc is not None:
                for wy, by_state in coinc.items():
                    coincident_decomp[scenario][year][wy] = by_state
            if done == 1 or done == len(tasks) or done % 16 == 0:
                _log(f'    ({done:>2}/{len(tasks)}) {scenario}/{year}/{sector}  '
                     f'states={len(next(iter(ann_d.values()), {}))}')

    return {
        'annual_gwh': annual,
        'peak_gw': peak,
        'peak_contributions': coincident_decomp,   # {scen}{year}{wy}{state} = {res, com, gap}
    }


def panel_cohort_daily_all_wys(
    res_inter: dict[tuple[str, str, str, str, int], Path],
    com_inter: dict[tuple[str, str, str, str, int], Path],
) -> dict:
    """Cohort daily decomposition for EVERY (scenario, year, weather_year).

    Tasks: 4 scenarios × 6 stock_years × 7 (sector,cohort) combos = 168.
    Each task returns daily GWh per wy and the matching ISO dates per wy.
    """
    sector_cohorts = ([('residential', c) for c in RES_COHORTS]
                      + [('commercial',  c) for c in COM_COHORTS])
    tasks = []
    for s in SCENARIOS:
        for y in STOCK_YEARS:
            for sec, coh in sector_cohorts:
                paths = res_inter if sec == 'residential' else com_inter
                tasks.append((paths, s, y, sec, coh))

    # Accumulator: {scenario: {year: {wy: {dates, cohorts: {cohort_key: [daily]}}}}}
    out: dict[str, dict[int, dict[int, dict]]] = {s: {y: {} for y in STOCK_YEARS} for s in SCENARIOS}
    _log(f'  cohort_daily_all_wys — {len(tasks)} (scenario, year, sector, cohort) tasks across {WORKERS} processes')
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for scenario, year, file_sector, cohort, daily_by_wy, dates_by_wy in pool.map(_cohort_daily_allwys_task, tasks):
            done += 1
            cohort_key = ('res_' if file_sector == 'residential' else 'com_') + cohort
            for wy, daily in daily_by_wy.items():
                entry = out[scenario][year].setdefault(wy, {'dates': dates_by_wy[wy], 'cohorts': {}})
                entry['cohorts'][cohort_key] = daily
            if done == 1 or done == len(tasks) or done % 24 == 0:
                _log(f'    ({done:>3}/{len(tasks)}) {scenario} y{year} {cohort_key}')
    return out


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
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_lbl_worker_init) as pool:
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
    panel3 = panel3_peak_week(res_inter, com_inter)

    _log('C. panel_state_by_sector — per-state annual+peak per sector...')
    state_by_sector = panel_state_by_sector(res_inter, com_inter)

    _log('D. panel_cohort_daily_all_wys — daily cohort decomp, every wy...')
    cohort_daily = panel_cohort_daily_all_wys(res_inter, com_inter)

    _log('E. panel_intermediate_annual (kept for reconciliation)...')
    intermediate_annual = panel_intermediate_annual(res_inter, com_inter)

    _log('F. panel_lbl (per-file polars)...')
    lbl_annual = panel_lbl(res_run_dir / 'LBL', com_run_dir / 'LBL')
    _log(f'   LBL cells populated: {sum(1 for s in lbl_annual.values() for y in s.values() for w in y.values())}')

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
    traj_max: dict[str, dict[str, dict[str, float]]] = {}
    for metric_name, by_scen in (('annual_gwh', state_by_sector['annual_gwh']),
                                  ('peak_gw',    state_by_sector['peak_gw'])):
        for scen, by_year in by_scen.items():
            for year, by_wy in by_year.items():
                for wy, by_sec in by_wy.items():
                    for sector, by_state in by_sec.items():
                        for state, v in by_state.items():
                            d = traj_max.setdefault(state, {}).setdefault(sector, {})
                            if v > d.get(metric_name, 0.0):
                                d[metric_name] = float(v)

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

    return {
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
            'trajectory_max': traj_max,    # [state][sector][metric] = max
            'choropleth_max': chor_max,    # [sector][metric] = max
        },
        'panel1':              panel1,
        'panel3':              panel3,
        'state_by_sector':     state_by_sector,    # {annual_gwh, peak_gw} → scen/year/wy/sector/state
        'cohort_daily':        cohort_daily,       # scen/year/wy → {dates, cohorts}
        'intermediate_annual': intermediate_annual,
        'lbl_annual':          lbl_annual,
        'meta': {
            'res_run_dir': str(res_run_dir),
            'com_run_dir': str(com_run_dir),
            'generated_iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--res-run-dir', type=Path, required=True,
                    help='ResStock run dir containing ReEDs/, intermediate/state/, LBL/.')
    ap.add_argument('--com-run-dir', type=Path, required=True,
                    help='ComStock run dir containing ReEDs/, intermediate/state/, LBL/.')
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'payload.json',
                    help='Output payload.json path (default: plots/payload.json).')
    args = ap.parse_args()

    _log(f'aggregate.py — WORKERS={WORKERS} (env BAKE_WORKERS={os.environ.get("BAKE_WORKERS","unset")}, cpu_count={os.cpu_count()})')
    _log(f'  res_run_dir: {args.res_run_dir}')
    _log(f'  com_run_dir: {args.com_run_dir}')

    payload = build_payload(args.res_run_dir, args.com_run_dir)
    _log('Serializing payload...')
    args.out.write_text(json.dumps(payload, separators=(',', ':')))
    sz = args.out.stat().st_size
    _log(f'Wrote {args.out}  ({sz/1e6:.2f} MB)')
    _log('Done.')


if __name__ == '__main__':
    main()
