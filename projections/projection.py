"""The projection itself: one function per consumption component, plus the
parallel driver that writes them.

A run is split into components, each a function returning one GWh frame per
enduse (cooling, heating, non_hvac, total):

  Upgrade scenario (per Upgraded-Upgrade* spec; gap is commercial-only):
    get_new_adoption_upgrade            newly-built stock that adopts the upgrade
    get_surviving_adoption_upgrade      existing stock that survives and adopts
    get_surviving_non_adoption_upgrade  existing stock that survives without adopting
    get_gap                             ComStock's unsimulated commercial floorspace

  Baseline scenario (the All-Baseline spec; gap is commercial-only):
    get_new_construction_baseline       newly-built stock, no adoption
    get_surviving_baseline              existing stock surviving, no adoption
    get_gap

The two new-construction components share get_new_construction, which applies
new-construction efficiency to each HVAC source and sums them into `total`; this
makes new-construction `total` smaller than the source `total` scaled directly,
because the efficiency gain is weighted by each enduse's share. Every other
component keeps existing efficiency, so its `total` is the plain
cooling+heating+non_hvac sum (equal to the agg `total` file scaled by the same
factor). The full projected load for a scenario is the sum of its component
files per enduse; nothing writes a pre-summed total.

Output: one CSV per (stock, scenario, component, enduse, year):
    proj_<stock>_<scenario>_<component>_<enduse>_GWh_y<YYYY>.csv
where <scenario> is the configured display name from `scenario_names` in
inputs/switches_agg.json (e.g. 'GHP' for Upgraded-Upgrade8), falling back to
the spec's short identifier when no mapping is configured.

CLI:  python -m projections <run_dir> [--stock res|com] [--resolution state|county|county_group]
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import cast

import pandas as pd

from . import common, factors, gap
from .common import (
    ALL_BASELINE_TAG,
    BASELINE_SPEC_NAME,
    COUNTY_GROUP_TO_STATE_POSTAL,
    COUNTY_TO_COUNTY_GROUP,
    ELIGIBLE_SPEC_NAME,
    ELIGIBLE_TAG,
    ENDUSES,
    INELIGIBLE_SPEC_NAME,
    INELIGIBLE_TAG,
    HISTORICAL_YEARS,
    PROJECTION_YEARS,
    STATE_FIPS_TO_POSTAL,
    EnduseFrames,
    Enduse,
    GroupName,
    GroupTask,
    Resolution,
    Scenario,
    SpecName,
    SpecTag,
    Stock,
    collapse_counties_to_county_groups,
    collapse_counties_to_states,
    load_agg_gwh,
    state_fips_from_county,
)

# Output subfolder per resolution. Used by both _projection_path and the
# pre-flight makedirs in project_run_dir.
_SUBFOLDER: dict[Resolution, str] = {
    'state':        'projections_state',
    'county':       'projections_county',
    'county_group': 'projections_county_group',
}

HVAC_ENDUSES: tuple[Enduse, ...] = ('cooling_elec', 'heating_elec', 'non_hvac_elec')

BASELINE_GROUPS: tuple[GroupName, ...] = ('new_construction', 'surviving')
UPGRADE_GROUPS:  tuple[GroupName, ...] = (
    'new_adoption', 'surviving_adoption', 'surviving_non_adoption',
)
GAP_GROUP: GroupName = 'gap_consumption'


def scenario_groups(scenario: Scenario, stock: Stock) -> tuple[GroupName, ...]:
    """Groups a scenario emits. Gap is commercial-only. At county_group
    resolution, gap is built by collapsing the cached county-level S3 gap data
    via the county_group mapping (see gap.load_gap)."""
    groups = BASELINE_GROUPS if scenario == 'baseline' else UPGRADE_GROUPS
    if stock == 'com':
        groups = (*groups, GAP_GROUP)
    return groups


def group_enduses(group: GroupName) -> tuple[Enduse, ...]:
    # Gap is total-electricity only; everything else writes all four enduses.
    return ('total',) if group == GAP_GROUP else ENDUSES


def _collapse(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate county-FIPS columns to the configured resolution."""
    if common.RESOLUTION == 'state':
        return collapse_counties_to_states(df)
    if common.RESOLUTION == 'county_group':
        return collapse_counties_to_county_groups(df)
    return df   # county: keep county-FIPS columns


def _load_hvac_sources(run_dir: str, stock: Stock, spec_tag: SpecTag) -> EnduseFrames:
    """The cooling/heating/non_hvac agg frames for one spec, at the configured
    resolution. `total` is synthesized by callers, not loaded."""
    return {enduse: _collapse(load_agg_gwh(run_dir, stock, spec_tag, enduse))
            for enduse in HVAC_ENDUSES}


def _add_total(per_enduse: EnduseFrames) -> EnduseFrames:
    """Append total = cooling + heating + non_hvac (how the agg total is defined)."""
    out = dict(per_enduse)
    out['total'] = out['cooling_elec'] + out['heating_elec'] + out['non_hvac_elec']
    return out


def _column_to_state_postal(col: str) -> str:
    """Resolve a column label to its state postal — needed for efficiency
    lookup. Counties resolve via FIPS; county-groups via the mapping table;
    state columns are already state postals."""
    if common.RESOLUTION == 'state':
        return col
    if common.RESOLUTION == 'county_group':
        return COUNTY_GROUP_TO_STATE_POSTAL[col]
    return STATE_FIPS_TO_POSTAL[state_fips_from_county(col)]


def _efficiency_per_column(stock: Stock, enduse: Enduse, year: int,
                           cols: Iterable[str]) -> pd.Series:
    """New-construction efficiency per column (resolved to a state)."""
    return pd.Series(
        {c: factors.new_construction_efficiency_factor(
                stock, enduse, year, _column_to_state_postal(c))
         for c in cols},
        index=cols,
    )


def _reference_axes(run_dir: str, stock: Stock) -> tuple[pd.DatetimeIndex, pd.Index]:
    """The timestamp index and column labels of the All-Baseline agg, read
    without its payload, so gap data can be aligned to the same axes."""
    path = common.agg_path(run_dir, stock, ALL_BASELINE_TAG, 'total')
    index = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path, usecols=[0]).iloc[:, 0]))
    index.name = 'timestamp_EST'
    county_cols = pd.read_csv(path, nrows=0, index_col=0).columns
    if common.RESOLUTION == 'state':
        columns = pd.Index(
            sorted({STATE_FIPS_TO_POSTAL[state_fips_from_county(c)] for c in county_cols}),
            name='state',
        )
    elif common.RESOLUTION == 'county_group':
        columns = pd.Index(
            sorted({COUNTY_TO_COUNTY_GROUP[c] for c in county_cols}),
            name='county_group',
        )
    else:
        columns = county_cols
    return index, columns


def get_new_construction(sources: EnduseFrames, growth_factor: float,
                         stock: Stock, year: int) -> EnduseFrames:
    """Newly-built stock: each HVAC source scaled by its new-construction
    efficiency, summed into total, then everything scaled by `growth_factor`.
    Shared by the baseline and upgrade new-construction components."""
    efficient: EnduseFrames = {}
    for enduse in HVAC_ENDUSES:
        src = sources[enduse]
        efficient[enduse] = src.mul(_efficiency_per_column(stock, enduse, year, src.columns), axis=1)
    efficient['total'] = efficient['cooling_elec'] + efficient['heating_elec'] + efficient['non_hvac_elec']
    return {enduse: growth_factor * frame for enduse, frame in efficient.items()}


def get_new_adoption_upgrade(upgrade_scenario: SpecTag, run_dir: str, stock: Stock,
                             year: int) -> EnduseFrames:
    """Newly-built stock that adopts the upgrade."""
    fac = factors.upgrade_factors(run_dir, stock, year)
    sources = _load_hvac_sources(run_dir, stock, upgrade_scenario)
    return get_new_construction(sources, fac['new_adoption'], stock, year)


def get_surviving_adoption_upgrade(upgrade_scenario: SpecTag, run_dir: str,
                                   stock: Stock, year: int) -> EnduseFrames:
    """Existing stock that survives and adopts the upgrade (existing efficiency)."""
    fac = factors.upgrade_factors(run_dir, stock, year)
    sources = _add_total(_load_hvac_sources(run_dir, stock, upgrade_scenario))
    factor = fac['surviving_adoption']
    return {enduse: factor * frame for enduse, frame in sources.items()}


def get_surviving_non_adoption_upgrade(upgrade_scenario: SpecTag, run_dir: str,
                                       stock: Stock, year: int) -> EnduseFrames:
    """Existing stock that survives without adopting, summed across the eligible
    and ineligible cohorts.

    Depends only on (stock, year), so it's identical across upgrade specs;
    `upgrade_scenario` is accepted only to keep the upgrade components' signatures
    uniform for the driver, and the result is written once per spec.
    """
    del upgrade_scenario
    fac = factors.upgrade_factors(run_dir, stock, year)
    eligible   = _add_total(_load_hvac_sources(run_dir, stock, ELIGIBLE_TAG))
    ineligible = _add_total(_load_hvac_sources(run_dir, stock, INELIGIBLE_TAG))
    eligible_factor   = fac['surviving_not_adopted_eligible']
    ineligible_factor = fac['surviving_not_adopted_ineligible']
    return {enduse: eligible_factor * eligible[enduse] + ineligible_factor * ineligible[enduse]
            for enduse in eligible}


def get_gap(run_dir: str, stock: Stock, year: int,
            target_years: Sequence[int]) -> EnduseFrames:
    """ComStock's unsimulated commercial floorspace (total electricity only).
    Empty for residential. Scaled by floorspace growth and the fixed T&D
    derating factor (see factors.GAP_DERATING_FACTOR)."""
    if stock != 'com':
        return {}
    index, columns = _reference_axes(run_dir, stock)
    g = gap.load_gap(target_years, columns)
    g = g.reindex(index).reindex(columns=columns, fill_value=0)
    return {'total': factors.GAP_DERATING_FACTOR * factors.gap_growth_factor(year) * g}


def get_new_construction_baseline(run_dir: str, stock: Stock, year: int) -> EnduseFrames:
    """Newly-built stock with no adoption.  New construction is always sourced
    from the Upgraded-Baseline (eligible) cohort — new buildings have modern
    construction supporting the upgrade even when the Baseline scenario
    doesn't install it."""
    fac = factors.baseline_scenario_factors(run_dir, stock, year)
    sources = _load_hvac_sources(run_dir, stock, ELIGIBLE_TAG)
    return get_new_construction(sources, fac['new_construction'], stock, year)


def get_surviving_baseline(run_dir: str, stock: Stock, year: int) -> EnduseFrames:
    """Existing stock surviving with no adoption (existing efficiency)."""
    fac = factors.baseline_scenario_factors(run_dir, stock, year)
    sources = _add_total(_load_hvac_sources(run_dir, stock, ALL_BASELINE_TAG))
    factor = fac['surviving']
    return {enduse: factor * frame for enduse, frame in sources.items()}


# =========================================================================
# Aux file projection. For every (scenario, year, cohort) we emit a per-
# building aux file alongside the per-enduse energy files. Same factors —
# from upgrade_factors / baseline_scenario_factors — applied to the source
# aux_coverage rows' sqft and units_count columns. The dashboard sums these
# across cohorts and aggregates to state; LBL reconstructs aux_samples by
# summing weights across cohorts.
# =========================================================================

def _aux_samples_path(run_dir: str, spec_tag: SpecTag) -> str:
    """Sample-level aux file for one spec — one row per (county, bldg_id)
    with characteristic sqft and sampling weight."""
    return os.path.join(run_dir, f'aux_samples_upgrade{spec_tag}.csv')


def _load_aux_for_spec(run_dir: str, stock: Stock, spec_tag: SpecTag) -> pd.DataFrame:
    """Load aux for one spec — per-county aux from BSQ, with an n_samples
    column tacked on (distinct bldg_id count from aux_samples_<spec>.csv).
    n_samples is a structural count: it doesn't scale with the projection
    factor (a simulation sample stays a simulation sample regardless of
    how many real-world units it represents at year Y)."""
    coverage = pd.read_csv(common.aux_path(run_dir, spec_tag))
    samples = pd.read_csv(_aux_samples_path(run_dir, spec_tag))
    county_col = coverage.columns[0]
    n_samples = (samples.groupby(county_col)['bldg_id']
                        .nunique()
                        .reset_index(name='n_samples'))
    return coverage.merge(n_samples, on=county_col, how='left').fillna({'n_samples': 0})


def _scale_aux(aux: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Scale the projected-quantity columns (sqft, units_count) by `factor`.
    n_samples is left alone — sample count is structural, not projected."""
    out = aux.copy()
    out['sqft'] = out['sqft'] * factor
    out['units_count'] = out['units_count'] * factor
    return out


def _sum_aux(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Sum per-county aux frames row-wise. Frames must share row identity
    (same county FIPS / nhgis_county_gisjoin in column 0). Non-numeric
    columns (county_name, state) come from the first frame. n_samples sums
    because eligible/ineligible cohorts are disjoint sets of bldg_ids."""
    key = dfs[0].columns[0]
    base = dfs[0].set_index(key)
    cols = ['sqft', 'units_count', 'n_samples']
    summed = base[cols].copy()
    for d in dfs[1:]:
        summed = summed.add(d.set_index(key)[cols], fill_value=0.0)
    meta_cols = [c for c in base.columns if c not in cols]
    return base[meta_cols].join(summed).reset_index()


def _aggregate_aux(aux: pd.DataFrame, resolution: Resolution) -> pd.DataFrame:
    """Roll per-county aux up to the target resolution. The energy code
    aggregates by hourly column; aux aggregates by row identity:
        state         → 49 rows keyed by state postal
        county_group  → ~1,038 rows keyed by BuildStock county_group
        county        → original ~3,100 rows unchanged
    """
    cols = ['sqft', 'units_count', 'n_samples']
    if resolution == 'state':
        return aux.groupby('state', as_index=False)[cols].sum()
    if resolution == 'county_group':
        county_col = aux.columns[0]
        a = aux.copy()
        a['county_group'] = a[county_col].map(COUNTY_TO_COUNTY_GROUP)
        a = a.dropna(subset=['county_group'])
        return a.groupby(['county_group', 'state'], as_index=False)[cols].sum()
    return aux  # county resolution: as-is


def _aux_for_group(group: GroupName, spec_tag: SpecTag, run_dir: str,
                   stock: Stock, year: int) -> pd.DataFrame | None:
    """Project the aux file for one (group, spec, year). Returns None for
    the gap cohort (commercial only — no aux representation).

    The energy projection sources `_load_hvac_sources` from the upgrade
    spec_tag for new_adoption / surviving_adoption — those rows represent
    *upgrade-applied load* per building. The AUX representation, however,
    counts *buildings*. Buildings being "upgraded" are still the same
    bldg_ids as in the Upgraded-Baseline (eligible) cohort, so all upgrade
    aux paths source from ELIGIBLE_TAG. Only the energy path needs the
    upgrade spec's load profile; the aux is invariant to which upgrade is
    installed.
    """
    if group == GAP_GROUP:
        return None
    if group == 'new_construction':
        # Baseline NC sources from Upgraded-Baseline (eligible).
        fac = factors.baseline_scenario_factors(run_dir, stock, year)['new_construction']
        return _scale_aux(_load_aux_for_spec(run_dir, stock, ELIGIBLE_TAG), fac)
    if group == 'surviving':
        fac = factors.baseline_scenario_factors(run_dir, stock, year)['surviving']
        return _scale_aux(_load_aux_for_spec(run_dir, stock, ALL_BASELINE_TAG), fac)
    if group == 'new_adoption':
        fac = factors.upgrade_factors(run_dir, stock, year)['new_adoption']
        return _scale_aux(_load_aux_for_spec(run_dir, stock, ELIGIBLE_TAG), fac)
    if group == 'surviving_adoption':
        fac = factors.upgrade_factors(run_dir, stock, year)['surviving_adoption']
        return _scale_aux(_load_aux_for_spec(run_dir, stock, ELIGIBLE_TAG), fac)
    if group == 'surviving_non_adoption':
        fac = factors.upgrade_factors(run_dir, stock, year)
        elig   = _scale_aux(_load_aux_for_spec(run_dir, stock, ELIGIBLE_TAG),
                            fac['surviving_not_adopted_eligible'])
        inelig = _scale_aux(_load_aux_for_spec(run_dir, stock, INELIGIBLE_TAG),
                            fac['surviving_not_adopted_ineligible'])
        return _sum_aux(elig, inelig)
    raise ValueError(f'unknown group {group!r}')


def _aux_output_path(run_dir: str, stock: Stock, display_name: str,
                     group: GroupName, year: int) -> str:
    return os.path.join(run_dir, _SUBFOLDER[common.RESOLUTION],
                        f'aux_{stock}_{display_name}_{group}_y{year}.csv')


def _compute_group(group: GroupName, spec_tag: SpecTag, run_dir: str, stock: Stock,
                   year: int, target_years: Sequence[int]) -> EnduseFrames:
    if group == 'new_construction':
        return get_new_construction_baseline(run_dir, stock, year)
    if group == 'surviving':
        return get_surviving_baseline(run_dir, stock, year)
    if group == 'new_adoption':
        return get_new_adoption_upgrade(spec_tag, run_dir, stock, year)
    if group == 'surviving_adoption':
        return get_surviving_adoption_upgrade(spec_tag, run_dir, stock, year)
    if group == 'surviving_non_adoption':
        return get_surviving_non_adoption_upgrade(spec_tag, run_dir, stock, year)
    if group == GAP_GROUP:
        return get_gap(run_dir, stock, year, target_years)
    raise ValueError(f'unknown group {group!r}')


def _discover_target_years(run_dir: str) -> list[int]:
    """The weather years to project, parsed from the switches snapshot. The
    target_year field is a list of ints and 'lo-hi' range strings."""
    snap = os.path.join(run_dir, 'inputs', 'switches_agg.json')
    with open(snap) as f:
        sw = json.load(f)
    years: list[int] = []
    for entry in sw['run_specs'][0]['target_year']:
        if isinstance(entry, int):
            years.append(entry)
        elif '-' in str(entry):
            lo, hi = (int(p) for p in str(entry).split('-'))
            years.extend(range(lo, hi + 1))
        else:
            years.append(int(entry))
    return sorted(set(years))


def _discover_specs(run_dir: str) -> Iterator[tuple[SpecName, SpecTag, Scenario]]:
    """Yield (spec_name, spec_tag, scenario) per run spec. All-Baseline is the
    baseline scenario; the two cohort anchors are skipped (they're sources, not
    scenarios); everything else is an upgrade scenario."""
    snap = os.path.join(run_dir, 'inputs', 'switches_agg.json')
    with open(snap) as f:
        sw = json.load(f)
    for s in sw['run_specs']:
        name = cast(SpecName, s['name'])
        tag = cast(SpecTag, f"{name}_{'reg' if s['apply_regression'] else 'ref'}_b{s['base_year']}")
        if name == BASELINE_SPEC_NAME:
            yield (name, tag, 'baseline')
        elif name in (ELIGIBLE_SPEC_NAME, INELIGIBLE_SPEC_NAME):
            continue
        else:
            yield (name, tag, 'upgrade')


def _load_scenario_names(run_dir: str) -> dict[str, str]:
    """Mapping from a spec's short identifier (e.g. 'Upgrade8') to the display
    name (e.g. 'GHP') used in output filenames. Returns {} if the snapshot has
    no scenario_names key."""
    with open(os.path.join(run_dir, 'inputs', 'switches_agg.json')) as f:
        return json.load(f).get('scenario_names') or {}


def _display_name(spec_name: SpecName, scenario_names: dict[str, str]) -> str:
    """Map a spec name to its filename-friendly display name. Strips the
    'Upgraded-' prefix and looks the result up in scenario_names; falls back to
    the stripped form if no mapping is configured."""
    short = spec_name.removeprefix('Upgraded-')
    return scenario_names.get(short, short)


def _projection_path(run_dir: str, stock: Stock, display_name: str,
                     group: GroupName, enduse: Enduse, year: int) -> str:
    return os.path.join(run_dir, _SUBFOLDER[common.RESOLUTION],
                        f'proj_{stock}_{display_name}_{group}_{enduse}_GWh_y{year}.csv')


def _project_one_group(args: GroupTask) -> str:
    """Compute one component for one (spec, year) and write its per-enduse CSV
    files. Idempotent: returns early if every file already exists, so a
    re-submitted job skips finished work. Runs in a worker process; its frames
    are freed when it returns."""
    (run_dir, stock, display_name, spec_tag, scenario, group,
     year, target_years) = args

    enduses = group_enduses(group)
    out_paths = {e: _projection_path(run_dir, stock, display_name, group, e, year)
                 for e in enduses}
    aux_path = _aux_output_path(run_dir, stock, display_name, group, year)
    has_aux = group != GAP_GROUP  # gap cohort has no per-building aux

    energy_done = all(os.path.exists(p) for p in out_paths.values())
    aux_done    = (not has_aux) or os.path.exists(aux_path)
    if energy_done and aux_done:
        return (f'  [skip] {scenario:8s} {display_name:22s} {group:22s} '
                f'y{year} → all {len(enduses)} enduses + aux exist')

    t0 = pd.Timestamp.now()
    shape: tuple[int, int] = (0, 0)

    if not energy_done:
        frames = _compute_group(group, spec_tag, run_dir, stock, year, target_years)
        for enduse, df in frames.items():
            path = out_paths[enduse]
            tmp = f'{path}.tmp.{os.getpid()}'
            df.to_csv(tmp)
            os.replace(tmp, path)
            shape = df.shape

    if has_aux and not aux_done:
        aux = _aux_for_group(group, spec_tag, run_dir, stock, year)
        if aux is not None:
            aux = _aggregate_aux(aux, common.RESOLUTION)
            tmp = f'{aux_path}.tmp.{os.getpid()}'
            aux.to_csv(tmp, index=False)
            os.replace(tmp, aux_path)

    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    return (f'  [{elapsed:5.1f}s] {scenario:8s} {display_name:22s} {group:22s} '
            f'y{year} → {len(enduses)} enduses + aux'
            f' ({shape[0]} rows × {shape[1]} cols each)')


def project_run_dir(run_dir: str, stock: Stock, n_workers: int | None = None) -> None:
    """Project every (spec, year, component) for `stock`, one worker process per
    component. Peak memory per worker scales with resolution: tiny at state
    (~49 cols), large at county (~3,100 cols × up to four enduse frames), so
    county runs want fewer workers."""
    print(f'\n=== {stock} | {run_dir} ===')
    target_years = _discover_target_years(run_dir)
    print(f'  target weather years: {target_years[0]}..{target_years[-1]} '
          f'({len(target_years)} years)')

    out_dir = os.path.join(run_dir, _SUBFOLDER[common.RESOLUTION])
    os.makedirs(out_dir, exist_ok=True)
    print(f'  resolution={common.RESOLUTION!r}; writing to {out_dir}')

    # Build the per-(spec, group, year) task list. Historical years (2012,
    # 2018, 2020) only get the Baseline scenario — adoption hasn't begun
    # before ANCHOR_YEAR, so upgrade scenarios at those years would be
    # identical-by-construction to Baseline.
    scenario_names = _load_scenario_names(run_dir)
    tasks: list[GroupTask] = []
    for spec_name, spec_tag, scenario in _discover_specs(run_dir):
        years = list(PROJECTION_YEARS)
        if scenario == 'baseline':
            years += list(HISTORICAL_YEARS)
        for group in scenario_groups(scenario, stock):
            for year in years:
                tasks.append((run_dir, stock,
                              _display_name(spec_name, scenario_names),
                              spec_tag, scenario, group, year, target_years))

    if n_workers is None:
        n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK') or os.cpu_count() or 8)

    # SLURM job array support — partition tasks across array indices so a
    # county or county_group run can fan out across many standard-partition
    # nodes (each node = one array task with `n_workers` ProcessPool slots).
    # No SLURM_ARRAY_TASK_COUNT → behaves as a single non-array job.
    array_id    = int(os.environ.get('SLURM_ARRAY_TASK_ID')    or 0)
    array_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT') or 1)
    if array_count > 1:
        my_tasks = tasks[array_id::array_count]
        print(f'  array task {array_id}/{array_count}: handling '
              f'{len(my_tasks)}/{len(tasks)} tasks across {n_workers} workers')
    else:
        my_tasks = tasks
        print(f'  dispatching {len(my_tasks)} projection tasks across {n_workers} workers')

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_project_one_group, t) for t in my_tasks]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog='python -m projections',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('run_dir', help='Run output directory (agg + aux + '
                                    'inputs/switches_agg.json).')
    ap.add_argument('--stock', choices=('res', 'com'), default=None,
                    help='Override stock (default: inferred from switches `comstock`).')
    ap.add_argument('--workers', type=int, default=None,
                    help='Worker processes (default: SLURM_CPUS_PER_TASK or cpu_count or 8).')
    ap.add_argument('--resolution', choices=('state', 'county', 'county_group'),
                    default='state',
                    help='state (default): 49 state cols, local gap CSV. '
                         'county: county-FIPS cols, per-county gap from S3 (cached). '
                         'county_group: 1,038 BuildStock county-group cols, gap excluded.')
    args = ap.parse_args()

    if args.stock is None:
        with open(os.path.join(args.run_dir, 'inputs', 'switches_agg.json')) as f:
            args.stock = 'com' if json.load(f).get('comstock') else 'res'

    # Set before the pool forks so workers inherit it (see common.RESOLUTION).
    common.RESOLUTION = cast(Resolution, args.resolution)
    project_run_dir(args.run_dir, cast(Stock, args.stock), n_workers=args.workers)


if __name__ == '__main__':
    main()
