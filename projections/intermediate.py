"""Intermediate handoff: per-component projection profiles relabeled for
publishing/debugging.

Combines both resolutions (state + county_group), both stocks (res + com),
every cohort (NC/SA/SNA + gap), every enduse (cooling, heating, non_hvac,
total), and every weather year. The source data already exists as per-component
CSVs in projections_state/ and projections_county_group/; this module relabels
them with sector/cohort/enduse names from the LBL spec and writes the result
into intermediate/{state,county_group}/. By default each entry is a relative
symlink (instant, zero data duplication; works with pandas/cat/grep/etc.);
pass --copy for standalone copies (the folder becomes self-contained for
shipping but takes ~equal disk space to the source).

Filename: <scenario>_<sector>_<cohort>_<enduse>_y<stock_year>.csv

CLI: python -m projections.intermediate <run_dir> [<run_dir>...] [--out DIR] [--copy]
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

from .reeds import _parse_filename


_GROUP_TO_COHORT: dict[str, str] = {
    'new_construction':       'NC',
    'surviving':              'SNA',
    'new_adoption':           'NC',
    'surviving_adoption':     'SA',
    'surviving_non_adoption': 'SNA',
    'gap_consumption':        'gap',
}
_SECTOR_FROM_STOCK: dict[str, str] = {'res': 'residential', 'com': 'commercial'}
_ENDUSE_LABEL: dict[str, str] = {
    'cooling_elec':  'cooling',
    'heating_elec':  'heating',
    'non_hvac_elec': 'non_hvac',
    'total':         'total',
}
_SOURCE_TO_TARGET: dict[str, str] = {
    'projections_state':        'state',
    'projections_county_group': 'county_group',
}


def _relabel(info: dict[str, str]) -> str:
    sector = _SECTOR_FROM_STOCK[info['stock']]
    cohort = _GROUP_TO_COHORT[info['group']]
    enduse = _ENDUSE_LABEL[info['enduse']]
    return f"{info['scenario']}_{sector}_{cohort}_{enduse}_y{info['year']}.csv"


def _emit(src: str, dst: str, copy: bool) -> None:
    """Replace dst with a relative symlink (or copy) of src."""
    if os.path.lexists(dst):
        os.remove(dst)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def build(run_dirs: list[str], out_dir: str, copy: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f'intermediate: writing to {out_dir} ({"copying" if copy else "symlinking"})')

    counts = {'state': 0, 'county_group': 0, 'skipped': 0}
    for rd in run_dirs:
        for src_subdir, dst_subdir in _SOURCE_TO_TARGET.items():
            src_root = os.path.join(rd, src_subdir)
            if not os.path.isdir(src_root):
                continue
            target = os.path.join(out_dir, dst_subdir)
            os.makedirs(target, exist_ok=True)
            for src in sorted(glob.glob(os.path.join(src_root, 'proj_*.csv'))):
                info = _parse_filename(src)
                if info is None or info['group'] not in _GROUP_TO_COHORT:
                    counts['skipped'] += 1
                    continue
                _emit(src, os.path.join(target, _relabel(info)), copy)
                counts[dst_subdir] += 1

    print(f'intermediate: {counts["state"]} state + {counts["county_group"]} '
          f'county_group files ({counts["skipped"]} skipped)')


def main() -> None:
    ap = argparse.ArgumentParser(prog='python -m projections.intermediate',
                                 description=__doc__)
    ap.add_argument('run_dirs', nargs='+',
                    help='Run dirs with projections_state/ and/or '
                         'projections_county_group/ generated.')
    ap.add_argument('--out', default=None,
                    help='Output directory (default: <first run_dir>/intermediate/).')
    ap.add_argument('--copy', action='store_true',
                    help='Copy files instead of symlinking (default: symlink). '
                         'Copy makes the folder shippable but ~equal to the '
                         'source size on disk (~1.7 TB for both stocks).')
    args = ap.parse_args()
    out = args.out if args.out is not None else os.path.join(args.run_dirs[0], 'intermediate')
    build(args.run_dirs, out, copy=args.copy)


if __name__ == '__main__':
    main()
