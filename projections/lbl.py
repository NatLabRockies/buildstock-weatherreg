"""LBL handoff: long-format county-group timeseries + per-cohort sample lists.

Reads from projections_county_group/ in one or more run_dirs (typically res and
com), filters to the LBL weather years (2012 and 2018), maps the package's
per-component groups to LBL's adoption cohorts (NC / SA / SNA), and writes:

  LBL/<scenario>_<sector>_<cohort>_<stock_year>_amy<weather_year>.csv
    Long-format timeseries. Columns: timestamp_EST, county_group, sector, cohort,
    enduse, value_kwh.

  LBL/aux_samples_<scenario>_y<stock_year>.csv
    Per-cohort sample IDs, per-building floor area, and cohort-scaled weights,
    combining all stocks/sectors.
    Columns: county_group, sector, cohort, bldg_id, sqft, weight.
    Consumers can reconstruct projected floor area for a cohort as
    sum(sqft * weight); the sqft column is per-building (unchanged from the
    aux_samples source), while weight = base_weight * cohort_factor.

Output directory defaults to `<first run_dir>/LBL/` (see `_resolve_out_dir`).

CLI: python -m projections.lbl <run_dir> [<run_dir> ...] [--out DIR]
                              [--only timeseries|samples|both]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from . import common, factors
from .common import (
    BASELINE_SPEC_NAME,
    COUNTY_TO_COUNTY_GROUP,
    ELIGIBLE_SPEC_NAME,
    INELIGIBLE_SPEC_NAME,
    PROJECTION_YEARS,
    Stock,
)
from .projection import _display_name
from .reeds import _parse_filename   # same filename regex


LBL_WEATHER_YEARS: tuple[int, ...] = (2012, 2018)

# The package's component-group name → the LBL cohort label it contributes to.
# Baseline's `surviving` is all non-adopting (baseline has no adoption);
# upgrade's `new_adoption` is the only new construction (all new construction
# adopts under an upgrade).
_GROUP_TO_COHORT: dict[str, str] = {
    'new_construction':       'NC',
    'surviving':              'SNA',
    'new_adoption':           'NC',
    'surviving_adoption':     'SA',
    'surviving_non_adoption': 'SNA',
}

_SECTOR_FROM_STOCK: dict[str, str] = {'res': 'residential', 'com': 'commercial'}


def _resolve_out_dir(run_dirs: list[str], out: str | None) -> str:
    out_dir = out if out is not None else os.path.join(run_dirs[0], 'LBL')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ============================================================================
# Timeseries.
# ============================================================================
def _list_component_files(run_dir: str) -> list[str]:
    cg_dir = os.path.join(run_dir, 'projections_county_group')
    if not os.path.isdir(cg_dir):
        raise FileNotFoundError(
            f'projections_county_group/ not found at {cg_dir} — '
            f'run `python -m projections {run_dir} --resolution county_group` first.')
    return sorted(glob.glob(os.path.join(cg_dir, 'proj_*_total_GWh_y*.csv')))


def _timeseries_out_path(out_dir: str, scenario: str, sector: str, cohort: str,
                         year: int, amy: int) -> str:
    return os.path.join(out_dir, f'{scenario}_{sector}_{cohort}_{year}_amy{amy}.csv')


def _write_one_timeseries(df_kwh: pd.DataFrame, amy: int, out_path: str,
                          sector: str, cohort: str) -> int | None:
    """Slice one weather year, melt to long, atomic-write. Returns row count or
    None if the slice is empty."""
    sliced = df_kwh.loc[df_kwh.index.year == amy]
    if sliced.empty:
        return None
    long_df = sliced.reset_index().melt(
        id_vars='timestamp_EST', var_name='county_group', value_name='value_kwh',
    )
    long_df['sector'] = sector
    long_df['cohort'] = cohort
    long_df['enduse'] = 'total'
    long_df = long_df[['timestamp_EST', 'county_group', 'sector', 'cohort',
                       'enduse', 'value_kwh']]
    tmp = f'{out_path}.tmp.{os.getpid()}'
    long_df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    return len(long_df)


def _process_one_source(args: tuple[str, str, tuple[int, ...]]) -> str:
    """Worker: read one source CSV once, emit long-format files for each weather
    year. Skips a weather year whose output already exists, so a re-run picks up
    where the previous one left off."""
    path, out_dir, weather_years = args
    info = _parse_filename(path)
    if info is None or info['group'] not in _GROUP_TO_COHORT:
        return f'  [skip] {os.path.basename(path)} (unrecognized)'

    cohort = _GROUP_TO_COHORT[info['group']]
    sector = _SECTOR_FROM_STOCK[info['stock']]
    scenario = info['scenario']
    year = int(info['year'])

    pending = [amy for amy in weather_years
               if not os.path.exists(_timeseries_out_path(out_dir, scenario, sector,
                                                          cohort, year, amy))]
    if not pending:
        return (f'  [skip] {scenario:20s} {sector:11s} {cohort:3s} y{year}: '
                f'all {len(weather_years)} weather years already written')

    t0 = pd.Timestamp.now()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = 'timestamp_EST'
    df_kwh = df * 1e6   # GWh → kWh

    written = []
    for amy in pending:
        rows = _write_one_timeseries(df_kwh, amy,
                                     _timeseries_out_path(out_dir, scenario, sector,
                                                          cohort, year, amy),
                                     sector, cohort)
        if rows is not None:
            written.append(f'amy{amy}={rows:,}')
    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    return (f'  [{elapsed:5.1f}s] {scenario:20s} {sector:11s} {cohort:3s} '
            f'y{year} → {", ".join(written) if written else "no slices"}')


def build_timeseries(run_dirs: list[str], out_dir: str,
                     weather_years: tuple[int, ...] = LBL_WEATHER_YEARS,
                     n_workers: int | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if n_workers is None:
        n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK') or os.cpu_count() or 8)
    print(f'lbl timeseries: writing to {out_dir} '
          f'(weather years {weather_years}, {n_workers} workers)')

    tasks = [(p, out_dir, weather_years)
             for rd in run_dirs for p in _list_component_files(rd)]
    print(f'lbl timeseries: dispatching {len(tasks)} source files across {n_workers} workers')

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for fut in as_completed([pool.submit(_process_one_source, t) for t in tasks]):
            print(fut.result(), flush=True)


# ============================================================================
# Per-cohort sample lists.
# ============================================================================
def _aux_samples_path(run_dir: str, spec_tag: str) -> str:
    return os.path.join(run_dir, f'aux_samples_upgrade{spec_tag}.csv')


def _load_aux_samples(run_dir: str, spec_tag: str) -> pd.DataFrame:
    """Load aux_samples for one spec and project to (county_group, bldg_id,
    sqft, base_weight). The county column is mostly GISJOIN ('G0100010') but
    the aux file also carries 'State, County Name' rows for AK/HI; we keep
    only GISJOIN-format rows and drop anything outside the county-group
    mapping.

    Schema differs by stock: res uses 'county', com uses 'nhgis_county_gisjoin'
    (see com_bsq_cols / res_bsq_cols in the switches snapshot).

    The raw aux_samples file stores `sqft` and `weight` on a TOTAL
    contribution basis (sum(sqft) equals aux_coverage sqft; sum(weight)
    equals aux_coverage units_count). We divide sqft by weight here to
    recover per-building floor area, so downstream `sum(sqft * weight)`
    reproduces the total floor area for the cohort — matching how a reader
    would naturally consume the LBL handoff."""
    path = _aux_samples_path(run_dir, spec_tag)
    header = pd.read_csv(path, nrows=0).columns
    county_col = 'county' if 'county' in header else 'nhgis_county_gisjoin'
    df = pd.read_csv(path, usecols=[county_col, 'bldg_id', 'sqft', 'weight'])
    df = df.rename(columns={county_col: 'county'})
    df = df[df['county'].str.match(r'^G\d{7}$', na=False)]
    df['county_fips'] = [str(int(g[1:3]) * 1000 + int(g[4:7])) for g in df['county']]
    df['county_group'] = df['county_fips'].map(COUNTY_TO_COUNTY_GROUP)
    df = df.dropna(subset=['county_group'])
    # Convert row-level totals to per-building floor area. `weight` here is
    # buildings-per-sample (equivalent to units_count contribution). Divide
    # by weight to recover the per-building sqft that stays constant when
    # weights get scaled by a cohort factor downstream.
    df['sqft_per_unit'] = df['sqft'] / df['weight']
    return (df[['county_group', 'bldg_id', 'sqft_per_unit', 'weight']]
            .rename(columns={'sqft_per_unit': 'sqft', 'weight': 'base_weight'}))


def _discover_run_specs(run_dir: str) -> tuple[list[dict], dict[str, str]]:
    """(run_specs list, scenario_names map) read from the run_dir's switches snapshot."""
    with open(os.path.join(run_dir, 'inputs', 'switches_agg.json')) as f:
        sw = json.load(f)
    return sw['run_specs'], sw.get('scenario_names') or {}


def _scenario_kinds(specs: list[dict], scenario_names: dict[str, str]) -> dict[str, str]:
    """Map display scenario name → 'baseline' or 'upgrade'. The eligible /
    ineligible cohort specs are skipped (they're sources, not scenarios)."""
    out: dict[str, str] = {}
    for s in specs:
        name = s['name']
        if name == BASELINE_SPEC_NAME:
            out[_display_name(name, scenario_names)] = 'baseline'
        elif name not in (ELIGIBLE_SPEC_NAME, INELIGIBLE_SPEC_NAME):
            out[_display_name(name, scenario_names)] = 'upgrade'
    return out


def _cohort_weights_for_scenario(run_dir: str, stock: Stock, year: int,
                                 kind: str) -> dict[str, pd.DataFrame]:
    """Per-cohort sample frames for one (scenario, stock, year).

    Returns {cohort_label: DataFrame(county_group, bldg_id, sqft, weight)}.
    Weights are base_weight × cohort_factor. `sqft` is the per-building floor
    area, carried through unchanged from the aux_samples file. A consumer can
    reconstruct the projected total floor area for a cohort as
    sum(sqft × weight).

    For baseline: NC + SNA. For upgrade: NC + SA + SNA. Upgrade NC and SA
    reuse the eligible-cohort sample list (Upgraded-Baseline is the same
    building set as Upgraded-Upgrade*; E_aux_query writes aux_samples only
    once per cohort).

    Sets the common.X_TAG module attrs from this run_dir's snapshot before
    delegating to factors.* — those functions read common.ELIGIBLE_TAG etc."""
    common.set_baseline_tags(run_dir)
    cols = ['county_group', 'bldg_id', 'sqft', 'weight']
    out: dict[str, pd.DataFrame] = {}
    if kind == 'baseline':
        fac = factors.baseline_scenario_factors(run_dir, stock, year)
        # Baseline NC sources from the eligible cohort (Upgraded-Baseline) —
        # same as the energy projection now does. SNA stays on All-Baseline.
        nc_base = _load_aux_samples(run_dir, common.ELIGIBLE_TAG)
        nc_df = nc_base.copy()
        nc_df['weight'] = nc_df['base_weight'] * fac['new_construction']
        out['NC'] = nc_df[cols]
        sna_base = _load_aux_samples(run_dir, common.ALL_BASELINE_TAG)
        sna_df = sna_base.copy()
        sna_df['weight'] = sna_df['base_weight'] * fac['surviving']
        out['SNA'] = sna_df[cols]
        return out

    fac = factors.upgrade_factors(run_dir, stock, year)
    eligible = _load_aux_samples(run_dir, common.ELIGIBLE_TAG)
    for cohort, key in (('NC', 'new_adoption'), ('SA', 'surviving_adoption')):
        df = eligible.copy()
        df['weight'] = df['base_weight'] * fac[key]
        out[cohort] = df[cols]

    eligible_sna   = eligible.copy()
    ineligible_sna = _load_aux_samples(run_dir, common.INELIGIBLE_TAG)
    eligible_sna['weight']   = eligible_sna['base_weight']   * fac['surviving_not_adopted_eligible']
    ineligible_sna['weight'] = ineligible_sna['base_weight'] * fac['surviving_not_adopted_ineligible']
    out['SNA'] = pd.concat([eligible_sna[cols], ineligible_sna[cols]],
                           ignore_index=True)
    return out


def _infer_stock(run_dir: str) -> Stock:
    """Run-dir is single-stock; read the snapshot's `comstock` flag."""
    with open(os.path.join(run_dir, 'inputs', 'switches_agg.json')) as f:
        return 'com' if json.load(f).get('comstock') else 'res'


def build_samples(run_dirs: list[str], out_dir: str,
                  projection_years: tuple[int, ...] = PROJECTION_YEARS) -> None:
    """One CSV per (scenario, stock_year), combining cohort sample weights from
    every input run_dir (typically res + com)."""
    os.makedirs(out_dir, exist_ok=True)
    print(f'lbl samples: writing to {out_dir}')

    # Per-run_dir context: (stock, {scenario_display → 'baseline'|'upgrade'}).
    contexts = []
    for rd in run_dirs:
        stock = _infer_stock(rd)
        specs, scenario_names = _discover_run_specs(rd)
        contexts.append((rd, stock, _scenario_kinds(specs, scenario_names)))

    all_scenarios = sorted({s for _, _, kinds in contexts for s in kinds})

    for scenario in all_scenarios:
        for year in projection_years:
            frames: list[pd.DataFrame] = []
            for rd, stock, kinds in contexts:
                kind = kinds.get(scenario)
                if kind is None:
                    continue
                sector = _SECTOR_FROM_STOCK[stock]
                for cohort, df in _cohort_weights_for_scenario(rd, stock, year, kind).items():
                    block = df.copy()
                    block['sector'] = sector
                    block['cohort'] = cohort
                    frames.append(block[['county_group', 'sector', 'cohort',
                                         'bldg_id', 'sqft', 'weight']])
            if not frames:
                continue
            combined = pd.concat(frames, ignore_index=True)
            out_path = os.path.join(out_dir, f'aux_samples_{scenario}_y{year}.csv')
            combined.to_csv(out_path, index=False)
            print(f'  {scenario:20s} y{year}: {len(combined):,} rows '
                  f'across {combined["cohort"].nunique()} cohorts')


# ============================================================================
# CLI.
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(prog='python -m projections.lbl', description=__doc__)
    ap.add_argument('run_dirs', nargs='+',
                    help='One or more run dirs with projections_county_group/ '
                         '(and aux_samples_*.csv) generated.')
    ap.add_argument('--out', default=None,
                    help='Output directory (default: <first run_dir>/LBL/).')
    ap.add_argument('--only', choices=('timeseries', 'samples', 'both'), default='both',
                    help='Which artifacts to generate (default: both).')
    args = ap.parse_args()

    out_dir = _resolve_out_dir(args.run_dirs, args.out)
    if args.only in ('timeseries', 'both'):
        build_timeseries(args.run_dirs, out_dir)
    if args.only in ('samples', 'both'):
        build_samples(args.run_dirs, out_dir)


if __name__ == '__main__':
    main()
