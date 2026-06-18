"""Stage 1 of the bake pipeline.

Reads per-run handoff folders (ReEDs + intermediate/state) for one ResStock
run_dir and one ComStock run_dir, pre-aggregates everything to the smallest
shapes the four dashboard panels actually need, packs the result into one
JSON payload, and string-substitutes that payload into dashboard_template.html
to produce a single self-contained dashboard.html.

The browser side runs from PAYLOAD; no other data is fetched at runtime.
Plotly.js is loaded from a CDN.

CLI:
  uv run python plots/build_dashboard.py \
      --res-run-dir /projects/geohc/radhikar/outputs/resstock_cross_val_may13_2026 \
      --com-run-dir /projects/geohc/radhikar/outputs/comstock_cross_val_may13_2026
  # output: plots/dashboard.html (open in browser).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS: list[str] = ['Baseline', 'ASHP', 'GHP', 'GHP+Envelope']
STOCK_YEARS: list[int] = [2027, 2030, 2035, 2040, 2045, 2050]
RES_COHORTS: list[str] = ['NC', 'SA', 'SNA']
COM_COHORTS: list[str] = ['NC', 'SA', 'SNA', 'gap']

# ReEDs CSVs spell out state names in lowercase ("alabama, …") to match the
# ReEDs intake spec. Plotly's USA-states choropleth wants 2-letter postals,
# so rename on load. DC is intentionally absent — the ReEDs handoff merged DC
# into MD per spec, so the columns we read are 48 mainland states.
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
    df = pd.read_csv(path, index_col='timestamp_EST', parse_dates=['timestamp_EST'])
    return df


def panel1_trajectory(reeds_paths: dict[tuple[str, int], Path]) -> dict:
    """Annual GWh + peak GW per (scenario, stock_year, weather_year), CONUS sum.

    ReEDs CSVs already sum across cohorts and sectors and are in MWh. We sum
    across all state columns to get a single CONUS hourly series, then resample
    by weather year (year-of-timestamp) to get one (annual GWh, peak GW) point
    per weather year.
    """
    series_annual: dict[str, dict[int, dict[int, float]]] = {s: {} for s in SCENARIOS}
    series_peak: dict[str, dict[int, dict[int, float]]] = {s: {} for s in SCENARIOS}
    weather_years_seen: set[int] = set()

    for scenario in SCENARIOS:
        for year in STOCK_YEARS:
            p = reeds_paths.get((scenario, year))
            if p is None:
                continue
            df = _read_with_index(p)
            conus_mwh = df.sum(axis=1)
            grouped = conus_mwh.groupby(conus_mwh.index.year)
            annual_gwh = grouped.sum() / 1000.0
            peak_gw = grouped.max() / 1000.0
            series_annual[scenario][year] = {int(wy): float(v) for wy, v in annual_gwh.items()}
            series_peak[scenario][year] = {int(wy): float(v) for wy, v in peak_gw.items()}
            weather_years_seen.update(annual_gwh.index.tolist())

    return {
        'annual_gwh': series_annual,
        'peak_gw': series_peak,
        'weather_years': sorted(int(w) for w in weather_years_seen),
    }


def panel2_state_map(reeds_paths: dict[tuple[str, int], Path]) -> dict:
    """Annual GWh by state for each (scenario, stock_year, weather_year).

    We keep all weather years so the browser can switch — per state per scenario
    per year per weather year is 48 × 4 × 6 × 18 = 20,736 scalars; trivial.
    """
    by_scen_year_wy: dict[str, dict[int, dict[int, dict[str, float]]]] = {s: {} for s in SCENARIOS}
    states_seen: list[str] = []

    for scenario in SCENARIOS:
        for year in STOCK_YEARS:
            p = reeds_paths.get((scenario, year))
            if p is None:
                continue
            df = _read_with_index(p)
            if not states_seen:
                states_seen = list(df.columns)
            grouped = df.groupby(df.index.year)
            annual_by_wy = grouped.sum() / 1000.0
            by_scen_year_wy[scenario].setdefault(year, {})
            for wy, row in annual_by_wy.iterrows():
                by_scen_year_wy[scenario][year][int(wy)] = {
                    state: float(v) for state, v in row.items()
                }

    return {
        'annual_gwh_by_state': by_scen_year_wy,
        'states': states_seen,
    }


def _sum_intermediate_to_conus(
    paths: dict[tuple[str, str, str, str, int], Path],
    scenario: str, sector: str, year: int, enduse: str = 'total',
) -> pd.Series | None:
    """Sum every (cohort) file matching (scenario, sector, year, enduse) into a
    single CONUS hourly series in GWh (intermediate files are in GWh per the
    projection's filename convention). None if no files matched."""
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


def _extract_window(res_wy: pd.Series, com_wy: pd.Series, total_wy: pd.Series, center_ts: pd.Timestamp) -> dict | None:
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


def panel3_peak_week(
    res_inter: dict[tuple[str, str, str, str, int], Path],
    com_inter: dict[tuple[str, str, str, str, int], Path],
) -> dict:
    """Hourly residential + commercial CONUS load (GWh = GW for hourly bins)
    over the 7-day window centered on the SUMMER and WINTER annual peaks, for
    each (scenario, stock_year, weather_year). Summer = annual max in
    Jun–Sep; winter = annual max in Dec–Feb. The browser toggles between
    them — for electrification analyses winter is usually the more interesting
    story even though summer is bigger in absolute terms."""
    weeks: dict[str, dict[int, dict[int, dict[str, dict]]]] = {s: {} for s in SCENARIOS}

    for scenario in SCENARIOS:
        for year in STOCK_YEARS:
            res_gw = _sum_intermediate_to_conus(res_inter, scenario, 'residential', year, 'total')
            com_gw = _sum_intermediate_to_conus(com_inter, scenario, 'commercial', year, 'total')
            if res_gw is None or com_gw is None:
                continue
            total_gw = res_gw.add(com_gw, fill_value=0.0)

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
                    weeks[scenario].setdefault(year, {})[int(wy)] = seasons

    return weeks


def panel4_cohort_daily(
    res_inter: dict[tuple[str, str, str, str, int], Path],
    com_inter: dict[tuple[str, str, str, str, int], Path],
    representative_wy_by_scen_year: dict[tuple[str, int], int],
) -> dict:
    """Daily-aggregated CONUS load by cohort (GWh/day) over one representative
    weather year per (scenario, stock_year). Seven cohorts: residential NC,
    SA, SNA; commercial NC, SA, SNA, gap."""
    cohorts: dict[str, dict[int, dict]] = {s: {} for s in SCENARIOS}

    for scenario in SCENARIOS:
        for year in STOCK_YEARS:
            wy = representative_wy_by_scen_year.get((scenario, year))
            if wy is None:
                continue

            entry: dict[str, list[float] | list[str]] = {}
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
                daily_gwh = s.resample('D').sum()
                entry[f'res_{cohort}'] = [float(v) for v in daily_gwh.values]
                if dates_set is None:
                    dates_set = [d.isoformat() for d in daily_gwh.index]
            for cohort in COM_COHORTS:
                s = _series(com_inter, 'commercial', cohort)
                if s is None:
                    continue
                daily_gwh = s.resample('D').sum()
                entry[f'com_{cohort}'] = [float(v) for v in daily_gwh.values]
                if dates_set is None:
                    dates_set = [d.isoformat() for d in daily_gwh.index]

            if not entry:
                continue
            entry['dates'] = dates_set or []
            entry['weather_year'] = wy
            cohorts[scenario][year] = entry

    return cohorts


def _representative_weather_year_table(panel1: dict) -> dict[tuple[str, int], int]:
    """Median weather year per (scenario, stock_year) by CONUS annual GWh.
    Used by panel4 to pick one weather year for the cohort decomposition."""
    table: dict[tuple[str, int], int] = {}
    for scenario, by_year in panel1['annual_gwh'].items():
        for year, by_wy in by_year.items():
            if not by_wy:
                continue
            sorted_wy = sorted(by_wy.items(), key=lambda kv: kv[1])
            median_wy = sorted_wy[len(sorted_wy) // 2][0]
            table[(scenario, year)] = median_wy
    return table


def _summary_stats(panel1: dict) -> dict:
    """Headline numbers: median annual + peak across weather years per
    (scenario, last stock year)."""
    last = STOCK_YEARS[-1]
    out: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        ann = panel1['annual_gwh'].get(scenario, {}).get(last, {})
        pk = panel1['peak_gw'].get(scenario, {}).get(last, {})
        if not ann or not pk:
            continue
        out[scenario] = {
            'annual_gwh_median_2050': float(np.median(list(ann.values()))),
            'peak_gw_median_2050':    float(np.median(list(pk.values()))),
        }
    return out


def _scenario_color() -> dict[str, str]:
    return {
        'Baseline':      '#7f8c8d',
        'ASHP':          '#2980b9',
        'GHP':           '#27ae60',
        'GHP+Envelope':  '#e67e22',
    }


def build_payload(res_run_dir: Path, com_run_dir: Path) -> dict:
    res_reeds = _parse_reeds_files(res_run_dir / 'ReEDs')
    com_reeds = _parse_reeds_files(com_run_dir / 'ReEDs')

    # Stream the ReEDs files: for each (scenario, year), load res + com, sum,
    # extract panel-1 (CONUS annual + peak) and panel-2 (per-state annual)
    # contributions, then discard. Peak resident memory ~one DataFrame (~200 MB).
    print(f'Streaming ReEDs: res={len(res_reeds)} files, com={len(com_reeds)} files', file=sys.stderr)
    series_annual: dict = {s: {} for s in SCENARIOS}
    series_peak:   dict = {s: {} for s in SCENARIOS}
    by_scen_year_wy: dict = {s: {} for s in SCENARIOS}
    wy_seen: set[int] = set()
    states_seen: list[str] = []

    for k in sorted(set(res_reeds) | set(com_reeds)):
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
        g_conus = conus.groupby(conus.index.year)
        ann = (g_conus.sum() / 1000.0).to_dict()
        pk  = (g_conus.max() / 1000.0).to_dict()
        series_annual.setdefault(scenario, {})[year] = {int(w): float(v) for w, v in ann.items()}
        series_peak.setdefault(scenario,   {})[year] = {int(w): float(v) for w, v in pk.items()}
        wy_seen.update(ann.keys())

        ann_by_wy = df.groupby(df.index.year).sum() / 1000.0
        by_scen_year_wy.setdefault(scenario, {}).setdefault(year, {})
        for wy, row in ann_by_wy.iterrows():
            by_scen_year_wy[scenario][year][int(wy)] = {st: float(v) for st, v in row.items()}

    panel1 = {
        'annual_gwh': series_annual,
        'peak_gw':    series_peak,
        'weather_years': sorted(int(w) for w in wy_seen),
    }
    panel2 = {'annual_gwh_by_state': by_scen_year_wy, 'states': states_seen}

    # Panels 3 + 4 use the per-component intermediate/state files.
    res_inter = _parse_intermediate_files(res_run_dir / 'intermediate' / 'state')
    com_inter = _parse_intermediate_files(com_run_dir / 'intermediate' / 'state')
    print(f'Intermediate: res={len(res_inter)} com={len(com_inter)} files', file=sys.stderr)

    print('Building Panel 3 (peak week)...', file=sys.stderr)
    panel3 = panel3_peak_week(res_inter, com_inter)

    print('Building Panel 4 (cohort decomp)...', file=sys.stderr)
    rep_wy = _representative_weather_year_table(panel1)
    panel4 = panel4_cohort_daily(res_inter, com_inter, rep_wy)

    return {
        'scenarios':    SCENARIOS,
        'stock_years':  STOCK_YEARS,
        'res_cohorts':  RES_COHORTS,
        'com_cohorts':  COM_COHORTS,
        'colors':       _scenario_color(),
        'summary':      _summary_stats(panel1),
        'panel1':       panel1,
        'panel2':       panel2,
        'panel3':       panel3,
        'panel4':       panel4,
        'meta': {
            'res_run_dir': str(res_run_dir),
            'com_run_dir': str(com_run_dir),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--res-run-dir', type=Path, required=True,
                    help='ResStock run dir containing ReEDs/ and intermediate/state/.')
    ap.add_argument('--com-run-dir', type=Path, required=True,
                    help='ComStock run dir containing ReEDs/ and intermediate/state/.')
    here = Path(__file__).parent
    ap.add_argument('--template', type=Path, default=here / 'dashboard_template.html')
    ap.add_argument('--out',      type=Path, default=here / 'dashboard.html')
    args = ap.parse_args()

    payload = build_payload(args.res_run_dir, args.com_run_dir)
    payload_json = json.dumps(payload, separators=(',', ':'))
    payload_size_kb = len(payload_json) / 1024
    print(f'Payload: {payload_size_kb:.1f} KB', file=sys.stderr)

    tpl = args.template.read_text()
    if '__PAYLOAD__' not in tpl:
        sys.exit(f'ERROR: template {args.template} missing __PAYLOAD__ token')
    args.out.write_text(tpl.replace('__PAYLOAD__', payload_json))
    print(f'Wrote {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
