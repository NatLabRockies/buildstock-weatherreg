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

A. ReEDs streaming  →  panel1 (trajectory) + panel2 (state map)
   For each of the 24 ReEDs CSVs per stock (scenario × stock_year):
     - read the file (timestamp_EST index, 48 lowercase-state-name columns, MWh)
     - add res + com row-wise if both run_dirs have the file
     - rename lowercase state names → 2-letter postals (Plotly choropleth)
     - sum across state columns → CONUS hourly MWh series
     - group by year-of-timestamp → 18 (annual GWh, peak GW) pairs per file
     - keep per-state annual GWh per weather year for the choropleth
   Output:
     panel1.annual_gwh[scenario][stock_year][weather_year] = GWh
     panel1.peak_gw  [scenario][stock_year][weather_year] = GW
     panel2.annual_gwh_by_state[scenario][stock_year][weather_year][postal] = GWh

B. panel3_peak_week  (parallel)
   For each (scenario, stock_year):
     - sum each cohort's `total` enduse intermediate file across cohorts:
         residential cohorts (NC + SA + SNA)        → res CONUS hourly GWh
         commercial  cohorts (NC + SA + SNA + gap)  → com CONUS hourly GWh
     - for each weather year:
         summer peak = max(total) in Jun-Sep
         winter peak = max(total) in Dec-Feb
         extract ±3-day windows (168 hours each)
   Output:
     panel3[scenario][stock_year][weather_year][summer|winter] =
       {timestamps, residential, commercial, peak_iso, peak_gw}

C. panel4_cohort_daily  (parallel)
   For each (scenario, stock_year):
     - pick the median CONUS-annual weather year (from panel1)
     - for each of the 7 cohort × sector channels (res NC/SA/SNA + com NC/SA/SNA/gap):
         read the cohort's `total` intermediate file
         sum across state cols → CONUS hourly GWh
         filter to the chosen weather year
         daily resample (sum) → 365 or 366 daily totals
   Output:
     panel4[scenario][stock_year][cohort_key] = [daily GWh / day]
       (+ 'dates': ISO dates, 'weather_year': the chosen median wy)

D. panel_intermediate_annual  (parallel)
   For every intermediate/state `total` file:
     - read, sum across state cols, group by year → annual GWh per weather year
   Used by Tabs 2 + 3 reconciliation.
   Output:
     intermediate_annual[scenario][stock_year][weather_year][sector][cohort] = GWh

E. panel_lbl  (polars one-pass)
   For every LBL/*.csv (long-format, ~9M rows each):
     - polars scan_csv with include_file_paths='source_file'
     - group_by('source_file') → sum(value_kwh) per file
     - kWh → GWh
   LBL ships only AMY 2012 and 2018 and excludes the commercial gap by spec.
   Output:
     lbl_annual[scenario][stock_year][weather_year][sector][cohort] = GWh

F. Axis pin maxes  (cheap)
   Compute max across all panel1 values for annual GWh and peak GW;
   add 10 % headroom; round to a tidy step (500 K GWh / 100 GW).
   Used to fix y-axis ranges so values across tabs stay visually comparable.

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
                    center_ts: pd.Timestamp) -> dict | None:
    """Build a payload dict for a 7-day window centered on center_ts."""
    win_start = center_ts - pd.Timedelta(days=3)
    win_end   = center_ts + pd.Timedelta(days=4) - pd.Timedelta(hours=1)
    res_win = res_wy.loc[win_start:win_end]
    com_win = com_wy.loc[win_start:win_end]
    tot_win = total_wy.loc[win_start:win_end]
    if len(res_win) < 24:
        return None
    return {
        'timestamps':  [t.isoformat() for t in res_win.index],
        'residential': [float(v) for v in res_win.values],
        'commercial':  [float(v) for v in com_win.values],
        'peak_iso':    center_ts.isoformat(),
        'peak_gw':     float(tot_win.max()),
    }


def _representative_weather_year_table(panel1: dict) -> dict[tuple[str, int], int]:
    """Median weather year per (scenario, stock_year) by CONUS annual GWh."""
    table: dict[tuple[str, int], int] = {}
    for scenario, by_year in panel1['annual_gwh'].items():
        for year, by_wy in by_year.items():
            if not by_wy:
                continue
            sorted_wy = sorted(by_wy.items(), key=lambda kv: kv[1])
            table[(scenario, year)] = sorted_wy[len(sorted_wy) // 2][0]
    return table


def _scenario_color() -> dict[str, str]:
    return {
        'Baseline':      '#7f8c8d',
        'ASHP':          '#2980b9',
        'GHP':           '#27ae60',
        'GHP+Envelope':  '#e67e22',
    }


# === Top-level worker functions (must be module-level for ProcessPool) =====
def _peak_week_task(args: tuple) -> tuple[str, int, dict]:
    """One (scenario, year) → seasonal peak windows per weather year."""
    res_inter, com_inter, scenario, year = args
    res_gw = _sum_intermediate_to_conus(res_inter, scenario, 'residential', year, 'total')
    com_gw = _sum_intermediate_to_conus(com_inter, scenario, 'commercial', year, 'total')
    if res_gw is None or com_gw is None:
        return scenario, year, {}
    total_gw = res_gw.add(com_gw, fill_value=0.0)

    result: dict[int, dict[str, dict]] = {}
    for wy, group_idx in total_gw.groupby(total_gw.index.year).groups.items():
        res_wy = res_gw.loc[group_idx]
        com_wy = com_gw.loc[group_idx]
        total_wy = total_gw.loc[group_idx]

        summer_mask = total_wy.index.month.isin([6, 7, 8, 9])
        winter_mask = total_wy.index.month.isin([12, 1, 2])

        summer_peak = total_wy[summer_mask].idxmax() if summer_mask.any() else None
        winter_peak = total_wy[winter_mask].idxmax() if winter_mask.any() else None

        seasons: dict[str, dict] = {}
        if summer_peak is not None:
            w = _extract_window(res_wy, com_wy, total_wy, summer_peak)
            if w is not None:
                seasons['summer'] = w
        if winter_peak is not None:
            w = _extract_window(res_wy, com_wy, total_wy, winter_peak)
            if w is not None:
                seasons['winter'] = w
        if seasons:
            result[int(wy)] = seasons
    return scenario, year, result


def _cohort_daily_task(args: tuple) -> tuple[str, int, dict]:
    """One (scenario, year) → per-cohort daily GWh at median wy."""
    res_inter, com_inter, scenario, year, wy = args
    entry: dict[str, list[float] | list[str] | int] = {}
    dates_set: list[str] | None = None

    def _series(paths, sector, cohort) -> pd.Series | None:
        key = (scenario, sector, cohort, 'total', year)
        p = paths.get(key)
        if p is None:
            return None
        df = _read_with_index(p)
        s = df.sum(axis=1)
        return s[s.index.year == wy]

    for cohort in RES_COHORTS:
        s = _series(res_inter, 'residential', cohort)
        if s is None:
            continue
        daily = s.resample('D').sum()
        entry[f'res_{cohort}'] = [float(v) for v in daily.values]
        if dates_set is None:
            dates_set = [d.isoformat() for d in daily.index]
    for cohort in COM_COHORTS:
        s = _series(com_inter, 'commercial', cohort)
        if s is None:
            continue
        daily = s.resample('D').sum()
        entry[f'com_{cohort}'] = [float(v) for v in daily.values]
        if dates_set is None:
            dates_set = [d.isoformat() for d in daily.index]

    if entry:
        entry['dates'] = dates_set or []
        entry['weather_year'] = wy
    return scenario, year, entry


def _intermediate_annual_task(args: tuple) -> tuple[tuple, dict[int, float]]:
    """One intermediate/state `total` file → annual CONUS GWh per weather year."""
    key, p = args
    df = _read_with_index(p)
    conus = df.sum(axis=1)
    ann_by_wy = conus.groupby(conus.index.year).sum()
    return key, {int(wy): float(v) for wy, v in ann_by_wy.items()}


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


def panel4_cohort_daily(
    res_inter: dict[tuple[str, str, str, str, int], Path],
    com_inter: dict[tuple[str, str, str, str, int], Path],
    representative_wy_by_scen_year: dict[tuple[str, int], int],
) -> dict:
    """Parallel over (scenario, year): per-cohort daily GWh at median wy."""
    tasks = [(res_inter, com_inter, s, y, wy)
             for (s, y), wy in representative_wy_by_scen_year.items()]
    cohorts: dict[str, dict[int, dict]] = {s: {} for s in SCENARIOS}
    _log(f'  cohort — {len(tasks)} (scenario, year) tasks across {WORKERS} processes')
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for scenario, year, entry in pool.map(_cohort_daily_task, tasks):
            done += 1
            n_coh = sum(1 for k in entry if k.startswith(('res_', 'com_')))
            _log(f'    ({done:>2}/{len(tasks)}) {scenario} y{year} @ amy{entry.get("weather_year","—")}  cohorts={n_coh}')
            if entry:
                cohorts[scenario][year] = entry
    return cohorts


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

    series_annual: dict = {s: {} for s in SCENARIOS}
    series_peak:   dict = {s: {} for s in SCENARIOS}
    by_scen_year_wy: dict = {s: {} for s in SCENARIOS}
    wy_seen: set[int] = set()
    states_seen: list[str] = []

    for i, k in enumerate(reeds_keys, 1):
        scenario, year = k
        r = _read_with_index(res_reeds[k]) if k in res_reeds else None
        c = _read_with_index(com_reeds[k]) if k in com_reeds else None
        if r is None and c is None:
            continue
        df = c if r is None else (r if c is None else r.add(c, fill_value=0.0))
        df = df.rename(columns=_LOWERNAME_TO_POSTAL)
        if not states_seen:
            states_seen = list(df.columns)

        conus = df.sum(axis=1)
        g = conus.groupby(conus.index.year)
        ann = (g.sum() / 1000.0).to_dict()
        pk  = (g.max() / 1000.0).to_dict()
        series_annual.setdefault(scenario, {})[year] = {int(w): float(v) for w, v in ann.items()}
        series_peak.setdefault(scenario,   {})[year] = {int(w): float(v) for w, v in pk.items()}
        wy_seen.update(ann.keys())

        ann_by_wy = df.groupby(df.index.year).sum() / 1000.0
        by_scen_year_wy.setdefault(scenario, {}).setdefault(year, {})
        for wy, row in ann_by_wy.iterrows():
            by_scen_year_wy[scenario][year][int(wy)] = {st: float(v) for st, v in row.items()}
        _log(f'  ReEDs {i:>2}/{len(reeds_keys):<2} {scenario} y{year}  '
             f'(CONUS ann={max(ann.values())/1e6:.2f} MGWh, peak={max(pk.values()):.1f} GW)')

    panel1 = {'annual_gwh': series_annual, 'peak_gw': series_peak,
              'weather_years': sorted(int(w) for w in wy_seen)}
    panel2 = {'annual_gwh_by_state': by_scen_year_wy, 'states': states_seen}

    res_inter = _parse_intermediate_files(res_run_dir / 'intermediate' / 'state')
    com_inter = _parse_intermediate_files(com_run_dir / 'intermediate' / 'state')
    _log(f'Intermediate index: res={len(res_inter)} com={len(com_inter)} files')

    _log('B. panel3_peak_week...')
    panel3 = panel3_peak_week(res_inter, com_inter)

    _log('C. panel4_cohort_daily...')
    panel4 = panel4_cohort_daily(res_inter, com_inter, _representative_weather_year_table(panel1))

    _log('D. panel_intermediate_annual...')
    intermediate_annual = panel_intermediate_annual(res_inter, com_inter)

    _log('E. panel_lbl (polars one-pass)...')
    lbl_annual = panel_lbl(res_run_dir / 'LBL', com_run_dir / 'LBL')
    _log(f'   LBL cells populated: {sum(1 for s in lbl_annual.values() for y in s.values() for w in y.values())}')

    _log('F. axis pin maxes...')
    def _ceil_to_step(v: float, step: float) -> float:
        return float(np.ceil(v / step) * step)
    max_annual = max((v for by in panel1['annual_gwh'].values() for by_y in by.values() for v in by_y.values()),
                     default=1.0)
    max_peak = max((v for by in panel1['peak_gw'].values() for by_y in by.values() for v in by_y.values()),
                   default=1.0)

    return {
        'scenarios':         SCENARIOS,
        'stock_years':       STOCK_YEARS,
        'res_cohorts':       RES_COHORTS,
        'com_cohorts':       COM_COHORTS,
        'lbl_cohorts':       LBL_COHORTS,
        'lbl_weather_years': LBL_WEATHER_YEARS,
        'colors':            _scenario_color(),
        'axis': {
            'annual_gwh_max': _ceil_to_step(max_annual * 1.10, 500_000.0),
            'peak_gw_max':    _ceil_to_step(max_peak * 1.10, 100.0),
        },
        'panel1':              panel1,
        'panel2':              panel2,
        'panel3':              panel3,
        'panel4':              panel4,
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
