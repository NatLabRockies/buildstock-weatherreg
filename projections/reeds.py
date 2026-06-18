"""ReEDS handoff: state-aggregated total electricity, wide format in MWh.

Reads the per-component CSVs from projections_state/, sums res + com + gap
across every group/cohort and both stocks, merges DC into MD (ReEDS treats
them as one state per spec), and writes one wide CSV per (scenario, stock_year):
    ReEDs/<scenario>_y<stock_year>.csv
with the timestamp_EST index and one column per state — state names spelled
out in lowercase (alabama, arizona, …, wyoming) — values in MWh.

CLI: python -m projections.reeds <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

from .common import STATE_POSTAL_TO_NAME


_GROUPS = ('new_construction', 'surviving',
           'new_adoption', 'surviving_adoption', 'surviving_non_adoption',
           'gap_consumption')
_ENDUSES = ('cooling_elec', 'heating_elec', 'non_hvac_elec', 'total')
_FNAME_RE = re.compile(
    r'^proj_(?P<stock>res|com)_(?P<scenario>.+?)'
    r'_(?P<group>' + '|'.join(_GROUPS) + r')'
    r'_(?P<enduse>' + '|'.join(_ENDUSES) + r')'
    r'_GWh_y(?P<year>\d+)\.csv$'
)

# Postal → lowercase full state name, used to rename the output columns.
# DC is intentionally absent here — it gets summed into MD before this lookup
# fires, and MD then renames to 'maryland' which carries the combined value.
_POSTAL_TO_LOWER_NAME: dict[str, str] = {
    postal: name.lower() for postal, name in STATE_POSTAL_TO_NAME.items()
    if postal != 'DC'
}


def _parse_filename(path: str) -> dict[str, str] | None:
    m = _FNAME_RE.match(os.path.basename(path))
    return m.groupdict() if m else None


def _group_by_scenario_year(state_dir: str) -> dict[tuple[str, int], list[str]]:
    """{(scenario, stock_year): [paths]} for every total-enduse file."""
    grouped: dict[tuple[str, int], list[str]] = {}
    for p in sorted(glob.glob(os.path.join(state_dir, 'proj_*_total_GWh_y*.csv'))):
        info = _parse_filename(p)
        if info is None:
            continue
        grouped.setdefault((info['scenario'], int(info['year'])), []).append(p)
    return grouped


def _merge_dc_into_md(df: pd.DataFrame) -> pd.DataFrame:
    """ReEDS groups DC with MD as one state (spec: 'n=48; DC grouped with MD')."""
    if 'DC' in df.columns and 'MD' in df.columns:
        df = df.copy()
        df['MD'] = df['MD'] + df['DC']
        df = df.drop(columns=['DC'])
    elif 'DC' in df.columns:
        df = df.rename(columns={'DC': 'MD'})
    return df


def _build_one(paths: list[str], out_path: str) -> tuple[int, int]:
    """Sum all per-component files for one (scenario, year), keep wide, rename
    state-postal columns to lowercase full names, convert GWh → MWh, write CSV."""
    total: pd.DataFrame | None = None
    for p in paths:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        total = df if total is None else total.add(df, fill_value=0)
    assert total is not None
    total.index.name = 'timestamp_EST'
    total = _merge_dc_into_md(total)
    total = (total * 1e3).rename(columns=_POSTAL_TO_LOWER_NAME)   # GWh → MWh
    total = total.reindex(columns=sorted(total.columns))           # stable column order
    total.to_csv(out_path)
    return total.shape


def build(run_dirs: list[str], out_dir: str | None = None) -> None:
    """Aggregate projections_state/ across one or more run_dirs (typically the
    res and com run_dirs) and emit one CSV per (scenario, stock_year)."""
    grouped: dict[tuple[str, int], list[str]] = {}
    for rd in run_dirs:
        state_dir = os.path.join(rd, 'projections_state')
        if not os.path.isdir(state_dir):
            raise FileNotFoundError(
                f'projections_state/ not found at {state_dir} — '
                f'run `python -m projections {rd}` first.')
        for key, paths in _group_by_scenario_year(state_dir).items():
            grouped.setdefault(key, []).extend(paths)

    out_dir = out_dir if out_dir is not None else os.path.join(run_dirs[0], 'ReEDs')
    os.makedirs(out_dir, exist_ok=True)
    print(f'reeds handoff: writing to {out_dir} '
          f'(aggregating {len(run_dirs)} run_dir(s))')

    for (scenario, year), paths in sorted(grouped.items()):
        out_path = os.path.join(out_dir, f'{scenario}_y{year}.csv')
        shape = _build_one(paths, out_path)
        print(f'  {scenario:20s} y{year}: {len(paths):2d} components → '
              f'{shape[0]:,} rows × {shape[1]} cols (MWh)')


def main() -> None:
    ap = argparse.ArgumentParser(prog='python -m projections.reeds', description=__doc__)
    ap.add_argument('run_dirs', nargs='+',
                    help='One or more run dirs with projections_state/ generated. '
                         'Typically the res and com run_dirs; output sums across them.')
    ap.add_argument('--out', default=None,
                    help='Output directory (default: <first run_dir>/ReEDs/).')
    args = ap.parse_args()
    build(args.run_dirs, args.out)


if __name__ == '__main__':
    main()
