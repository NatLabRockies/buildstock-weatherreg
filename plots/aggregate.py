"""Heavy aggregation step for the BuildStock projection dashboard.

Reads one folder (`intermediate/state/`) from a ResStock run_dir and a ComStock
run_dir, pre-aggregates everything to the smallest shapes the dashboard needs,
and writes:

  plots/data/main.js           ~55 MB   (CONUS payload — sets window.PAYLOAD)
  plots/data/state_<postal>.js ~10 MB × 49  (per-state — lazy-loaded via
                                            <script> injection on click)

The dashboard itself is `plots/dashboard.html` (tracked in git, edit
directly). It references `data/main.js` via <script src>. After this
script writes new data/, refresh the browser to pick it up.

Re-run aggregate.py only when the source run_dirs change. Edit
dashboard.html directly to iterate on plot design — no build step.

EXACTLY WHAT GETS COMPUTED
==========================

A. panel_state_by_sector  (parallel; ProcessPool, SCENARIOS × STOCK_YEARS tasks)
   Each (scenario, stock_year) task reads every per-(sector, cohort, enduse)
   leaf intermediate file and builds a per-state hourly matrix for all 28
   breakdown keys:
     - 4 rollups       : total, residential, commercial, gap
     - 6 cohort totals : <sector>_<cohort>          (sector × {NC, SA, SNA})
     - 18 leaves       : <sector>_<cohort>_<enduse> (× {cooling, heating, non_hvac})
   Per key: annual GWh = matrix.groupby(year).sum(); peak GW =
   {annual, summer, winter} max. CONUS sums hourly across states FIRST, then
   takes the max (correct coincident peak — summing per-state peaks over-counts).
   The 'total' key additionally yields peak_contributions: every leaf's (and
   gap's) value AT the system-total-peak hour. These sum EXACTLY to the total
   peak (coincident decomposition), so the trajectory's Peak GW Breakdown
   stack height equals the un-broken Total line.
   Output:
     state_by_sector.annual_gwh[scenario][year][wy][key][state] = GWh
     state_by_sector.peak_gw   [scenario][year][wy][key][state] = {annual,summer,winter} GW
     state_by_sector.peak_contributions[scenario][year][wy][state]
                              = {leaf_or_gap_key: GW at total-peak hour}   (19 keys)

   panel1 (headline CONUS trajectory) is derived inline in build_payload from
   the state_by_sector 'total' key at CONUS — one source of truth per value.

B. panel3_peak_week  (parallel; ProcessPool, SCENARIOS × PROJECTION_YEARS tasks)
   For each (scenario, stock_year):
     - load all 19 leaf files (res/com × NC/SA/SNA × {cooling, heating,
       non_hvac} + com gap) and sum to res / com / total CONUS hourly GW
     - for each weather year:
         summer peak = max(total) in Jun-Sep
         winter peak = max(total) in Dec-Feb
         extract ±3-day windows from the FULL series (not wy-sliced) so
         Jan-1 boundary peaks still get all 168 hours
   Output:
     panel3[scenario][stock_year][weather_year][summer|winter] =
       {timestamps, residential, commercial, peak_iso, peak_gw,
        cohorts: {<leaf_key>: [168 GW]}}   (the 19 stacked leaves)

C. panel_cohort_daily_all_wys  (parallel; ProcessPool, one task per
   (scenario, stock_year, series) over the 19 breakdown series)
   For each series:
     - read the series intermediate file
     - sum across state cols → CONUS hourly GWh
     - group by year-of-timestamp → per-wy daily resample (24h→1)
   Output:
     cohort_daily[scenario][year][wy] = {dates: [iso], cohorts: {series_key: [daily]}}
       (series_key ranges over the 19 breakdown series)

C.2 panel_cohort_hourly_maxmin  (parallel; SCENARIOS × PROJECTION_YEARS tasks)
   Per (scenario, stock_year), the daily MAX and MIN of the hourly CONUS
   series for Total and Com Total — the bottom-left panel's 'hourly*'
   granularity, showing the intra-day power envelope (4 lines). Populated
   for PROJECTION_YEARS only (historical years lack per-cohort files).
   Output:
     cohort_hourly_maxmin[scenario][year][wy] =
       {dates, series: {total_max, total_min, com_max, com_min}}   (GW)

D. panel_stock_counts
   Read per-cohort aux files from intermediate/state/. Output per-state
   sqft (com) / units_count + samples (res).

E. Axis pin maxes  (cheap, in build_payload)
   * annual_gwh_max / peak_gw_max (CONUS legacy pins, rounded to step)
   * trajectory_max[state][sector][metric] — per-state per-sector global max
     across all (scenario, year, wy); used to pin y-axes so sliding
     doesn't rescale.
   * choropleth_max[sector][metric] — per-sector global max across the
     non-CONUS states; used to pin the choropleth color scale.

CLI
===
  uv run python plots/aggregate.py \\
      --res-run-dir /projects/geohc/radhikar/outputs/resstock_cross_val_june8_2026 \\
      --com-run-dir /projects/geohc/radhikar/outputs/comstock_cross_val_may13_2026
  # output: plots/data/main.js  +  plots/data/state_<postal>.js (×49)

  # No build step: plots/dashboard.html loads data/main.js directly.
  # Refresh the browser (hard-refresh to bust cache) to pick up new data.
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
HISTORICAL_YEARS: list[int] = [2012, 2018, 2020]   # Baseline-only calibration years
PROJECTION_YEARS: list[int] = [2027, 2030, 2035, 2040, 2045, 2050]
STOCK_YEARS: list[int] = HISTORICAL_YEARS + PROJECTION_YEARS
RES_COHORTS: list[str] = ['NC', 'SA', 'SNA']
COM_COHORTS: list[str] = ['NC', 'SA', 'SNA', 'gap']

# 2-letter state postals for Plotly's USA-states choropleth. DC intentionally
# absent (merged into MD by the projection package's state-collapse convention).
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

_INTERMEDIATE_FILE_RE = re.compile(
    r'(?P<scenario>.+)_(?P<sector>residential|commercial)_(?P<cohort>NC|SA|SNA|gap)_'
    r'(?P<enduse>cooling|heating|non_hvac|total)_y(?P<year>\d{4})\.csv'
)

# Process worker cap. ProcessPoolExecutor for guaranteed parallelism — pandas'
# GIL release is inconsistent across read / groupby / dict-build, so threads
# can stall. On Linux fork() is COW so passing the path dicts per task is cheap.
WORKERS: int = max(1, min(16, int(os.environ.get('BAKE_WORKERS', os.cpu_count() or 4)) - 1))


# === File-system catalogers ===============================================
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
    # 19 leaves (18 residential/commercial cohort×enduse + gap), keyed by
    # the same identifiers as _STACK_ORDER so the dashboard can render in a
    # single pass without name translation.
    cohort_dfs: dict[str, pd.DataFrame] = {}

    for sector, paths in (('residential', res_inter), ('commercial', com_inter)):
        for cohort in ('NC', 'SA', 'SNA'):
            for enduse in ('cooling', 'heating', 'non_hvac'):
                p = paths.get((scenario, sector, cohort, enduse, year))
                if p is not None:
                    cohort_dfs[f'{sector}_{cohort}_{enduse}'] = _read_with_index(p)
    gap_p = com_inter.get((scenario, 'commercial', 'gap', 'total', year))
    if gap_p is not None:
        cohort_dfs['gap'] = _read_with_index(gap_p)

    if not cohort_dfs:
        return scenario, year, {}

    # Build sector-and-total state-x-time DataFrames from the leaves.
    res_df: pd.DataFrame | None = None
    com_df: pd.DataFrame | None = None
    for k, df in cohort_dfs.items():
        if k.startswith('residential_'):
            res_df = df if res_df is None else res_df.add(df, fill_value=0.0)
        elif k.startswith('commercial_') or k == 'gap':
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


ENDUSES: tuple[str, ...] = ('cooling', 'heating', 'non_hvac')

# Compound keys for the state_by_sector breakdown. Every key maps to a
# per-state annual + peak dict; the dashboard picks one for the choropleth
# and stacks the 19 leaves for the trajectory Breakdown view.
#
# Layer categories:
#   'total' / 'gap' / 'residential' / 'commercial'           — 4 rollup levels
#   '<sector>_<cohort>'                                      — 6 cohort totals
#   '<sector>_<cohort>_<enduse>'                             — 18 leaves
# = 28 keys per (scenario, year, wy).
#
# For per-key annual GWh, leaves sum to their parents. For per-key peak
# GW, the leaves' peaks generally DO NOT sum to the parent peak (max is not
# additive), so we store each level explicitly.

_LEAF_KEYS: tuple[str, ...] = tuple(
    f'{sector}_{cohort}_{enduse}'
    for sector in ('residential', 'commercial')
    for cohort in ('NC', 'SA', 'SNA')
    for enduse in ENDUSES
)  # 18 leaves

_COHORT_KEYS: tuple[str, ...] = tuple(
    f'{sector}_{cohort}'
    for sector in ('residential', 'commercial')
    for cohort in ('NC', 'SA', 'SNA')
)  # 6 cohort totals

_SECTOR_KEYS: tuple[str, ...] = ('residential', 'commercial', 'gap', 'total')  # 4 rollups

# Stacking order for the dashboard's Breakdown view (bottom → top of stack).
_STACK_ORDER: tuple[str, ...] = (
    'gap',
    'commercial_SNA_non_hvac', 'commercial_SNA_cooling', 'commercial_SNA_heating',
    'commercial_SA_non_hvac',  'commercial_SA_cooling',  'commercial_SA_heating',
    'commercial_NC_non_hvac',  'commercial_NC_cooling',  'commercial_NC_heating',
    'residential_SNA_non_hvac','residential_SNA_cooling','residential_SNA_heating',
    'residential_SA_non_hvac', 'residential_SA_cooling', 'residential_SA_heating',
    'residential_NC_non_hvac', 'residential_NC_cooling', 'residential_NC_heating',
)


def _state_sector_task(args: tuple) -> tuple[str, int, dict, dict, dict]:
    """One (scenario, year) → per-key per-state + CONUS annual+peak per wy.

    Reads every leaf cohort-enduse intermediate file for the (scenario, year)
    request, builds per-key hourly state-x-time matrices for all 28 slices
    (rollups + cohort totals + leaves + gap + total), and computes annual GWh
    + peak GW for each. For CONUS, the sum is joint-hourly first (correct
    coincident peak; summing per-state peaks over-counts).

    Returns three per-key dicts:
      * ann_by_key[key][wy][state]  = float GWh
      * peak_by_key[key][wy][state] = {annual, summer, winter} GW
      * coincident[wy][state]      = {leaf_or_gap_key: value at total-peak hour}

    `key` covers 28 slices: 'total', 4 rollups, 6 cohort totals, 18 leaves.
    """
    res_inter, com_inter, scenario, year = args

    # Load per-(sector, cohort, enduse) leaves. Missing files (empty scenario
    # at historical years, gap-in-residential, etc.) are silently skipped —
    # downstream state_by_sector slots stay empty for those keys.
    def _read(paths, sector, cohort, enduse):
        p = paths.get((scenario, sector, cohort, enduse, year))
        return _read_with_index(p) if p is not None else None

    leaves: dict[str, pd.DataFrame] = {}
    for sector, paths in (('residential', res_inter), ('commercial', com_inter)):
        for cohort in ('NC', 'SA', 'SNA'):
            for enduse in ENDUSES:
                df = _read(paths, sector, cohort, enduse)
                if df is not None:
                    leaves[f'{sector}_{cohort}_{enduse}'] = df

    gap_df = _read(com_inter, 'commercial', 'gap', 'total')

    if not leaves and gap_df is None:
        return scenario, year, {}, {}, {}

    def _sum(frames):
        acc = None
        for f in frames:
            if f is None:
                continue
            acc = f if acc is None else acc.add(f, fill_value=0.0)
        return acc

    # Roll up: cohort totals, sector totals, grand total.
    per_key: dict[str, pd.DataFrame] = dict(leaves)
    if gap_df is not None:
        per_key['gap'] = gap_df
    for sector in ('residential', 'commercial'):
        for cohort in ('NC', 'SA', 'SNA'):
            k = f'{sector}_{cohort}'
            children = [leaves.get(f'{k}_{eu}') for eu in ENDUSES]
            s = _sum(children)
            if s is not None:
                per_key[k] = s
    for sector in ('residential', 'commercial'):
        s = _sum([per_key.get(f'{sector}_{c}') for c in ('NC', 'SA', 'SNA')])
        if s is not None:
            per_key[sector] = s
    total = _sum([per_key.get('residential'), per_key.get('commercial'), per_key.get('gap')])
    if total is not None:
        per_key['total'] = total

    # Compute annual + peak stats for every key.
    ann_by_key: dict[str, dict] = {}
    peak_by_key: dict[str, dict] = {}
    for k, df in per_key.items():
        ann_by_key[k], peak_by_key[k] = _stats_from_state_hourly(df)

    # Coincident decomposition (only meaningful for the 'total' key). For each
    # (wy, state), find the hour that sets the total peak and report every
    # LEAF's value at that hour — trajectory Peak GW breakdown stacks these.
    coincident: dict = {}
    if 'total' in per_key:
        stack_frames = [(k, per_key.get(k)) for k in _STACK_ORDER if per_key.get(k) is not None]
        for wy in per_key['total'].index.year.unique():
            wy_int = int(wy)
            wy_total = per_key['total'][per_key['total'].index.year == wy]
            wy_frames = {k: df[df.index.year == wy] for k, df in stack_frames}
            coincident[wy_int] = {}
            for st in wy_total.columns:
                t = wy_total[st].idxmax()
                coincident[wy_int][st] = {k: float(f.loc[t, st]) for k, f in wy_frames.items()}
            t = wy_total.sum(axis=1).idxmax()
            coincident[wy_int]['CONUS'] = {k: float(f.loc[t].sum()) for k, f in wy_frames.items()}

    return scenario, year, ann_by_key, peak_by_key, coincident


def _stats_from_state_hourly(df: pd.DataFrame) -> tuple[dict, dict]:
    """Given a state-hourly GWh matrix, return per-wy per-state annual +
    peak dicts.  CONUS = joint hourly sum then aggregate (correct coincident
    peak; sum-of-state-peaks would over-count)."""
    summer_months = (6, 7, 8, 9)
    winter_months = (12, 1, 2)
    ann = df.groupby(df.index.year).sum()
    peak = df.groupby(df.index.year).max()
    conus_hourly = df.sum(axis=1)
    conus_ann = conus_hourly.groupby(conus_hourly.index.year).sum()
    conus_peak = conus_hourly.groupby(conus_hourly.index.year).max()
    ann_dict: dict = {}
    peak_dict: dict = {}
    for wy in ann.index:
        wy_int = int(wy)
        ann_dict[wy_int] = {st: float(v) for st, v in ann.loc[wy].items()}
        ann_dict[wy_int]['CONUS'] = float(conus_ann.loc[wy])
        wy_mask = df.index.year == wy
        wy_data = df[wy_mask]
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
    return ann_dict, peak_dict


def _cohort_hourly_maxmin_task(args: tuple) -> tuple[str, int, dict]:
    """One (scenario, year) → per-state per-wy per-day (max_hourly_gw,
    min_hourly_gw) for the TOTAL and Com-only composites.

    Reads all present cohort-enduse intermediate files (up to 18 leaves + gap),
    sums them hourly into `total_df` and `com_df` (gap + commercial leaves),
    then groups by day-of-year to compute max & min. Returned per-wy so the
    dashboard can pin daily-max / daily-min lines to the same date-axis
    used by the daily/monthly views.

    Returns: (scenario, year, by_wy) where
      by_wy[wy] = {
        'dates': [iso],
        'states': {state: {'total_max': [...], 'total_min': [...],
                           'com_max':   [...], 'com_min':   [...]},
                   'CONUS': {...}},
      }
    """
    res_inter, com_inter, scenario, year = args

    # Load every present leaf. This is memory-hungry (up to 19 × 8760 rows ×
    # 49 states of GWh floats ≈ 60 MB per (scenario, year)) but each worker
    # only holds one such set at a time.
    total_df: pd.DataFrame | None = None
    com_df:   pd.DataFrame | None = None
    for sector, paths in (('residential', res_inter), ('commercial', com_inter)):
        for cohort in ('NC', 'SA', 'SNA'):
            for enduse in ENDUSES:
                p = paths.get((scenario, sector, cohort, enduse, year))
                if p is None:
                    continue
                df = _read_with_index(p)
                total_df = df if total_df is None else total_df.add(df, fill_value=0.0)
                if sector == 'commercial':
                    com_df = df if com_df is None else com_df.add(df, fill_value=0.0)
    gap_p = com_inter.get((scenario, 'commercial', 'gap', 'total', year))
    if gap_p is not None:
        gap_df = _read_with_index(gap_p)
        total_df = gap_df if total_df is None else total_df.add(gap_df, fill_value=0.0)
        com_df   = gap_df if com_df   is None else com_df  .add(gap_df, fill_value=0.0)

    if total_df is None:
        return scenario, year, {}

    # Hourly GWh == hourly average GW (1-hour intervals). CONUS = state sum.
    total_conus = total_df.sum(axis=1)
    com_conus   = com_df.sum(axis=1) if com_df is not None else None

    by_wy: dict[int, dict] = {}
    for wy in total_df.index.year.unique():
        wy_int = int(wy)
        wy_mask = total_df.index.year == wy
        wy_total   = total_df[wy_mask]
        wy_com     = com_df[wy_mask] if com_df is not None else None
        wy_totalC  = total_conus[wy_mask]
        wy_comC    = com_conus[wy_mask] if com_conus is not None else None

        # Per-state per-day max / min. Group by day-of-year via .index.date.
        total_max = wy_total.groupby(wy_total.index.date).max()
        total_min = wy_total.groupby(wy_total.index.date).min()
        com_max   = wy_com.groupby(wy_com.index.date).max()   if wy_com is not None else None
        com_min   = wy_com.groupby(wy_com.index.date).min()   if wy_com is not None else None

        # CONUS: joint hourly first, then per-day agg (correct — daily max/min
        # of the CONUS-total series, not sum-of-state-daily-max).
        conus_total_max = wy_totalC.groupby(wy_totalC.index.date).max()
        conus_total_min = wy_totalC.groupby(wy_totalC.index.date).min()
        conus_com_max   = wy_comC  .groupby(wy_comC.index.date).max() if wy_comC is not None else None
        conus_com_min   = wy_comC  .groupby(wy_comC.index.date).min() if wy_comC is not None else None

        dates = [d.isoformat() for d in total_max.index]

        states_data: dict = {}
        for st in wy_total.columns:
            states_data[st] = {
                'total_max': [round(float(v), 3) for v in total_max[st].values],
                'total_min': [round(float(v), 3) for v in total_min[st].values],
                'com_max':   [round(float(v), 3) for v in com_max[st].values]   if com_max is not None else [],
                'com_min':   [round(float(v), 3) for v in com_min[st].values]   if com_min is not None else [],
            }
        states_data['CONUS'] = {
            'total_max': [float(v) for v in conus_total_max.values],
            'total_min': [float(v) for v in conus_total_min.values],
            'com_max':   [float(v) for v in conus_com_max.values] if conus_com_max is not None else [],
            'com_min':   [float(v) for v in conus_com_min.values] if conus_com_min is not None else [],
        }
        by_wy[wy_int] = {'dates': dates, 'states': states_data}
    return scenario, year, by_wy


def _cohort_daily_allwys_task(args: tuple) -> tuple[str, int, str, dict]:
    """One (scenario, year, series_key) intermediate file → daily-aggregated
    GWh per (state, wy) + CONUS sum. series_key is one of the 19 stacking
    keys (18 residential/commercial cohort_enduse leaves + 'gap').

    Returns: (scenario, year, series_key, by_wy) where
      by_wy[wy] = {'dates': [iso], 'states': {state: [daily], 'CONUS': [daily]}}
    """
    paths, scenario, year, series_key, sector, cohort, enduse = args
    p = paths.get((scenario, sector, cohort, enduse, year))
    if p is None:
        return scenario, year, series_key, {}
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
    return scenario, year, series_key, by_wy


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
    """Per-state annual + peak per (scenario, year, wy, key).

    `key` covers 28 breakdown slices — 4 rollups (total, residential,
    commercial, gap), 6 cohort totals (residential_NC/SA/SNA and
    commercial_NC/SA/SNA), and 18 leaves (each cohort × 3 enduses).

    Historical years emit only the Baseline scenario per projection package;
    upgrade-scenario tasks at those years produce empty dicts.
    """
    tasks = [(res_inter, com_inter, s, y) for s in SCENARIOS for y in STOCK_YEARS]

    annual: dict = {s: {y: {} for y in STOCK_YEARS} for s in SCENARIOS}
    peak:   dict = {s: {y: {} for y in STOCK_YEARS} for s in SCENARIOS}
    coincident_decomp: dict = {s: {y: {} for y in STOCK_YEARS} for s in SCENARIOS}

    _log(f'  state-by-sector — {len(tasks)} tasks across {WORKERS} processes')
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for scenario, year, ann_by_key, peak_by_key, coinc in pool.map(_state_sector_task, tasks):
            done += 1
            for key, by_wy in ann_by_key.items():
                for wy, by_state in by_wy.items():
                    annual[scenario][year].setdefault(wy, {})[key] = by_state
            for key, by_wy in peak_by_key.items():
                for wy, by_state in by_wy.items():
                    peak[scenario][year].setdefault(wy, {})[key] = by_state
            for wy, by_state in coinc.items():
                coincident_decomp[scenario][year][wy] = by_state
            if done == 1 or done == len(tasks) or done % 4 == 0:
                _log(f'    ({done:>2}/{len(tasks)}) {scenario}/{year} '
                     f'keys={len(ann_by_key)}')

    return {
        'annual_gwh': annual,
        'peak_gw': peak,
        'peak_contributions': coincident_decomp,   # {scen}{year}{wy}{state} = {leaf_key: v}
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

    Series keys are the 19 stacking keys (18 sector_cohort_enduse leaves + gap).
    """
    # Task tuples: (paths, scenario, year, series_key, sector, cohort, enduse).
    # 19 series × 4 scenarios × 6 years = 456 tasks. Historical years are
    # skipped (peak-week/cohort_daily panels are projection-year only).
    tasks: list = []
    for s in SCENARIOS:
        for y in PROJECTION_YEARS:
            for series_key in _STACK_ORDER:
                if series_key == 'gap':
                    tasks.append((com_inter, s, y, 'gap', 'commercial', 'gap', 'total'))
                    continue
                sector, cohort, enduse = series_key.split('_', 2)
                paths = res_inter if sector == 'residential' else com_inter
                tasks.append((paths, s, y, series_key, sector, cohort, enduse))

    conus_only: dict = {s: {y: {} for y in PROJECTION_YEARS} for s in SCENARIOS}
    per_state: dict = {}
    _log(f'  cohort_daily_all_wys — {len(tasks)} (scenario, year, series) tasks across {WORKERS} processes')
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for scenario, year, series_key, by_wy in pool.map(_cohort_daily_allwys_task, tasks):
            done += 1
            for wy, payload in by_wy.items():
                dates = payload['dates']
                states_data = payload['states']
                conus_entry = conus_only[scenario][year].setdefault(wy, {'dates': dates, 'cohorts': {}})
                conus_entry['cohorts'][series_key] = states_data['CONUS']
                # Per-state files DROP `dates` — same dates apply across
                # all states for the same (scenario, year, wy).
                for st, daily_vals in states_data.items():
                    if st == 'CONUS':
                        continue
                    st_entry = (per_state.setdefault(st, {})
                                          .setdefault(scenario, {})
                                          .setdefault(year, {})
                                          .setdefault(wy, {'cohorts': {}}))
                    st_entry['cohorts'][series_key] = daily_vals
            if done == 1 or done == len(tasks) or done % 40 == 0:
                _log(f'    ({done:>3}/{len(tasks)}) {scenario} y{year} {series_key}')
    return conus_only, per_state


def panel_cohort_hourly_maxmin(
    res_inter: dict[tuple[str, str, str, str, int], Path],
    com_inter: dict[tuple[str, str, str, str, int], Path],
) -> tuple[dict, dict]:
    """Per (scenario, year, wy, state) daily-max & daily-min of hourly GW for
    the Total (all cohorts + gap) and Com Total (com cohorts + gap) composites.
    Feeds the bottom-left panel's 'hourly*' granularity view — 4 series per
    state per wy: total_max / total_min / com_max / com_min. Populated only
    for PROJECTION_YEARS (historical years lack per-cohort intermediate
    files; the panel already shows 'No data' at those years).

    Returns:
      (conus_only, per_state)
      conus_only[scenario][year][wy] = {dates, series: {total_max, total_min,
                                                        com_max, com_min}}
      per_state[state][scenario][year][wy] = {series: {...}}    (dates dropped)
    """
    tasks = [(res_inter, com_inter, s, y) for s in SCENARIOS for y in PROJECTION_YEARS]
    conus_only: dict = {s: {} for s in SCENARIOS}
    per_state: dict = {}
    _log(f'  cohort_hourly_maxmin — {len(tasks)} (scenario, year) tasks across {WORKERS} processes')
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for scenario, year, by_wy in pool.map(_cohort_hourly_maxmin_task, tasks):
            done += 1
            if by_wy:
                conus_only[scenario][year] = {}
            for wy, payload in by_wy.items():
                dates = payload['dates']
                states_data = payload['states']
                conus_only[scenario][year][wy] = {
                    'dates': dates,
                    'series': states_data.get('CONUS', {}),
                }
                for st, st_data in states_data.items():
                    if st == 'CONUS':
                        continue
                    (per_state.setdefault(st, {})
                              .setdefault(scenario, {})
                              .setdefault(year, {}))[wy] = {'series': st_data}
            if done == 1 or done == len(tasks) or done % 4 == 0:
                _log(f'    ({done:>2}/{len(tasks)}) {scenario} y{year}')
    return conus_only, per_state


_AUX_FILE_RE = re.compile(
    r'^aux_(?P<scenario>.+)_(?P<sector>residential|commercial)_'
    r'(?P<cohort>NC|SA|SNA)_y(?P<year>\d{4})\.csv$'
)


def panel_stock_counts(res_inter_dir: Path, com_inter_dir: Path) -> dict:
    """Per-state, per-cohort stock counts read directly from the projection
    package's per-cohort aux files. Each aux file has 49 rows (one per state)
    with `sqft`, `units_count`, and `n_samples` columns.

    Output (each value is per-state with a CONUS key):
        {
          'residential_units':   {year: {cohort: {state: M HH}}},     # = units_count / 1e6
          'commercial_sqft':     {year: {cohort: {state: B sqft}}},   # = sqft / 1e9
          'residential_samples': {year: {cohort: {state: int}}},      # = n_samples
          'commercial_samples':  {year: {cohort: {state: int}}},
        }

    Cohort breakdown:
        residential: NC, SA, SNA
        commercial:  NC, SA, SNA  (no aux for the gap cohort by design;
                                   gap is unmodeled commercial floor space)
    """
    res_aux_files = _gather_aux_files(res_inter_dir, 'residential')
    com_aux_files = _gather_aux_files(com_inter_dir, 'commercial')

    residential_units:   dict = {}
    commercial_sqft:     dict = {}
    residential_samples: dict = {}
    commercial_samples:  dict = {}

    # Read from one *upgrade* scenario for projection years so we get the full
    # NC/SA/SNA cohort split (Baseline has no SA — adoption hasn't happened).
    # All upgrade scenarios share the same adoption ramp from growth_factors, so
    # which one we pick doesn't matter; first sorted name wins for determinism.
    # For HISTORICAL_YEARS we fall back to Baseline since only Baseline is
    # emitted at those years.
    def _pick_scenario(by_year_cohort: dict, year: int) -> str:
        scenarios = sorted({s for s, y, _ in by_year_cohort if y == year})
        upgrades = [s for s in scenarios if s != 'Baseline']
        return upgrades[0] if upgrades else 'Baseline'

    for (scenario, year, cohort), path in res_aux_files.items():
        if scenario != _pick_scenario(res_aux_files, year):
            continue
        df = pd.read_csv(path)
        per_state_units   = {row['state']: float(row['units_count']) / 1e6 for _, row in df.iterrows()}
        per_state_samples = {row['state']: int(row['n_samples'])           for _, row in df.iterrows()}
        per_state_units['CONUS']   = sum(per_state_units.values())
        per_state_samples['CONUS'] = sum(per_state_samples.values())
        residential_units  .setdefault(year, {})[cohort] = per_state_units
        residential_samples.setdefault(year, {})[cohort] = per_state_samples
    for (scenario, year, cohort), path in com_aux_files.items():
        if scenario != _pick_scenario(com_aux_files, year):
            continue
        df = pd.read_csv(path)
        per_state_sqft    = {row['state']: float(row['sqft']) / 1e9 for _, row in df.iterrows()}
        per_state_samples = {row['state']: int(row['n_samples'])    for _, row in df.iterrows()}
        per_state_sqft['CONUS']    = sum(per_state_sqft.values())
        per_state_samples['CONUS'] = sum(per_state_samples.values())
        commercial_sqft   .setdefault(year, {})[cohort] = per_state_sqft
        commercial_samples.setdefault(year, {})[cohort] = per_state_samples

    if 2050 in residential_units:
        res2050 = sum(residential_units[2050].get(c, {}).get('CONUS', 0) for c in ('NC','SA','SNA'))
        com2050 = sum(commercial_sqft.get(2050, {}).get(c, {}).get('CONUS', 0) for c in ('NC','SA','SNA'))
        res_n50 = sum(residential_samples[2050].get(c, {}).get('CONUS', 0) for c in ('NC','SA','SNA'))
        com_n50 = sum(commercial_samples.get(2050, {}).get(c, {}).get('CONUS', 0) for c in ('NC','SA','SNA'))
        _log(f'  stock_counts (aux): 2050 Baseline CONUS res={res2050:.1f}M HH '
             f'({res_n50} samples), com={com2050:.1f}B sqft ({com_n50} samples) '
             f'(years present: {sorted(residential_units.keys())})')

    return {
        'residential_units':   residential_units,
        'commercial_sqft':     commercial_sqft,
        'residential_samples': residential_samples,
        'commercial_samples':  commercial_samples,
    }


def _gather_aux_files(intermediate_state_dir: Path,
                      expect_sector: str) -> dict[tuple[str, int, str], Path]:
    """Return {(scenario, year, cohort): path} for every aux file at
    intermediate/state/ matching the given sector. Tolerates a missing dir
    (returns empty) so the bake doesn't fail when projection hasn't been
    run with aux output yet."""
    out: dict[tuple[str, int, str], Path] = {}
    if not intermediate_state_dir.is_dir():
        return out
    for p in sorted(intermediate_state_dir.iterdir()):
        m = _AUX_FILE_RE.match(p.name)
        if not m or m['sector'] != expect_sector:
            continue
        out[(m['scenario'], int(m['year']), m['cohort'])] = p
    return out


# === Top-level orchestrator ===============================================
def build_payload(res_run_dir: Path, com_run_dir: Path) -> dict:
    res_inter = _parse_intermediate_files(res_run_dir / 'intermediate' / 'state')
    com_inter = _parse_intermediate_files(com_run_dir / 'intermediate' / 'state')
    _log(f'Intermediate index: res={len(res_inter)} com={len(com_inter)} files')

    _log('A. panel_state_by_sector — per-state annual+peak per sector...')
    state_by_sector = panel_state_by_sector(res_inter, com_inter, res_run_dir, com_run_dir)

    # panel1 (CONUS trajectory) is the total-key CONUS scalar — derived
    # directly from state_by_sector so there is only one source of truth for
    # every value the dashboard displays. peak_gw values in state_by_sector
    # are seasonal dicts {annual, summer, winter}; panel1 collapses to the
    # `annual` scalar.
    series_annual: dict = {s: {} for s in SCENARIOS}
    series_peak:   dict = {s: {} for s in SCENARIOS}
    wy_seen: set[int] = set()
    for scen, by_year in state_by_sector['annual_gwh'].items():
        for year, by_wy in by_year.items():
            ann = {int(wy): float(by_key['total']['CONUS'])
                   for wy, by_key in by_wy.items() if 'total' in by_key}
            if ann:
                series_annual.setdefault(scen, {})[year] = ann
                wy_seen.update(ann.keys())
    for scen, by_year in state_by_sector['peak_gw'].items():
        for year, by_wy in by_year.items():
            pk = {int(wy): float(by_key['total']['CONUS']['annual'])
                  for wy, by_key in by_wy.items() if 'total' in by_key}
            if pk:
                series_peak.setdefault(scen, {})[year] = pk
    panel1 = {'annual_gwh': series_annual, 'peak_gw': series_peak,
              'weather_years': sorted(int(w) for w in wy_seen)}

    _log('B. panel3_peak_week...')
    panel3_conus, panel3_per_state = panel3_peak_week(res_inter, com_inter)

    _log('C. panel_cohort_daily_all_wys — daily cohort decomp, every wy...')
    cohort_daily_conus, cohort_daily_per_state = panel_cohort_daily_all_wys(res_inter, com_inter)

    _log('C.2 panel_cohort_hourly_maxmin — daily-max/min of hourly total & com...')
    hourly_mm_conus, hourly_mm_per_state = panel_cohort_hourly_maxmin(res_inter, com_inter)

    _log('D. panel_stock_counts (read per-cohort aux files from intermediate/)...')
    stock_counts = panel_stock_counts(res_run_dir / 'intermediate' / 'state',
                                       com_run_dir / 'intermediate' / 'state')

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
                for wy, by_key in by_wy.items():
                    for key, by_state in by_key.items():
                        for state, v in by_state.items():
                            scalar = v['annual'] if isinstance(v, dict) else v
                            d = traj_max.setdefault(state, {}).setdefault(key, {})
                            if scalar > d.get(metric_name, 0.0):
                                d[metric_name] = float(scalar)

    # Per (key, metric) max — for choropleth colorbar.
    chor_max: dict[str, dict[str, float]] = {}
    for metric_name in ('annual_gwh', 'peak_gw'):
        for state, by_key in traj_max.items():
            if state == 'CONUS':
                continue
            for key, by_metric in by_key.items():
                v = by_metric.get(metric_name, 0.0)
                d = chor_max.setdefault(key, {})
                if v > d.get(metric_name, 0.0):
                    d[metric_name] = float(v)

    # Main payload (CONUS-only for cohort daily + peak week; everything else
    # fully included). Per-state cohort + peak week data live in sidecar files.
    main_payload = {
        'scenarios':         SCENARIOS,
        'stock_years':       STOCK_YEARS,
        'res_cohorts':       RES_COHORTS,
        'com_cohorts':       COM_COHORTS,
        'stack_order':       list(_STACK_ORDER),   # bottom→top for Breakdown views
        'colors':            _scenario_color(),
        'sectors':           ['residential', 'commercial', 'gap', 'total'],
        'states':            sorted(s for s in (traj_max.keys()) if s != 'CONUS'),
        'axis': {
            'annual_gwh_max': _ceil_to_step(max_annual * 1.10, 500_000.0),
            'peak_gw_max':    _ceil_to_step(max_peak * 1.10, 100.0),
            'trajectory_max': traj_max,
            'choropleth_max': chor_max,
        },
        'panel1':          panel1,
        'panel3':          panel3_conus,
        'state_by_sector': state_by_sector,
        'cohort_daily':    cohort_daily_conus,
        'cohort_hourly_maxmin': hourly_mm_conus,
        'stock_counts':    stock_counts,
        'meta': {
            'res_run_dir': str(res_run_dir),
            'com_run_dir': str(com_run_dir),
            'generated_iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }
    # Per-state sidecars: one entry per state. The dashboard lazy-loads these.
    state_keys = sorted(set(panel3_per_state.keys())
                        | set(cohort_daily_per_state.keys())
                        | set(hourly_mm_per_state.keys()))
    per_state_sidecars: dict[str, dict] = {
        st: {
            'cohort_daily':         cohort_daily_per_state.get(st, {}),
            'peak_week':            panel3_per_state.get(st, {}),
            'cohort_hourly_maxmin': hourly_mm_per_state.get(st, {}),
        }
        for st in state_keys
    }
    return {'main': main_payload, 'per_state': per_state_sidecars}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--res-run-dir', type=Path, required=True,
                    help='ResStock run dir containing intermediate/state/.')
    ap.add_argument('--com-run-dir', type=Path, required=True,
                    help='ComStock run dir containing intermediate/state/.')
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
