#!/usr/bin/env python
'''
This script develops county-level building cooling.elec / heating.elec /
non_hvac.elec load profiles for ReEDS (in GWh) based on resstock/comstock
EULP outputs (MWh). One GWh file per enduse is produced per run, plus a
`total` aggregate (sum of the three enduses), all dropped into the run
output directory alongside the chunk MWh files.

The `upgrade_tag` is the per-spec identifier from `switches_agg.json`'s
`run_specs`, of the form `<name>_<reg|ref>_b<base_year>` — e.g.,
`All-Baseline_ref_b2018`. (Older runs used `<upgrade_id>_<reg|ref>_b<base_year>`
before the name-based tagging change; old tags still parse since the leading
segment is treated as an opaque label.)

Usage:
    # Run aggregation for one spec (writes 4 outputs: cool, heat, non_hvac, total).
    python agg_buildings.py --bldg-path <run_dir> --bldg-type res --upgrade-tag All-Baseline_ref_b2018

    # Backfill missing total aggs across every spec in a run dir.
    python agg_buildings.py --bldg-path <run_dir> --bldg-type res --backfill-totals

Or import the functions:
    from agg_buildings import aggregate, backfill_totals
    aggregate(run_dir, "res", "All-Baseline_ref_b2018")  # returns list of output paths
    backfill_totals(run_dir, "res")                       # returns list of newly-written paths
'''

import argparse
import glob
import os
import shutil
import pandas as pd


def _chunks_eulp_dir(bldg_path: str, upgrade_tag: str) -> str:
    '''Path to the per-spec EULP chunks subfolder.

    upgrade_tag is `<upgrade_id>_<reg|ref>_b<base_year>`. The folder
    convention is `chunks_<reg|ref>_b<base_year>` — i.e., the upgrade_tag
    minus its leading `<upgrade_id>_` segment.
    '''
    suffix = upgrade_tag.split('_', 1)[1]  # e.g. "reg_b2018"
    return os.path.join(bldg_path, f'chunks_{suffix}')


# Enduse tokens written by D into chunk filenames. Each token produces one
# stitched GWh CSV. Order is irrelevant.
# KEEP IN SYNC with the two resume-check tuples in B_building_stock_parallel_agg.py
# (per-spec agg-presence check and per-chunk presence check). If this drifts
# from those, you'll either re-run already-done chunks forever or skip
# specs that are silently missing one enduse output.
ENDUSES = ('cooling_elec', 'heating_elec', 'non_hvac_elec')


def _agg_path(bldg_path: str, bldg_type: str, upgrade_tag: str,
              enduse: str) -> str:
    '''Canonical agg GWh file path for one (bldg_type, upgrade_tag, enduse).'''
    return os.path.join(
        bldg_path,
        f'agg_{bldg_type}_eulp_{enduse}_GWh_upgrade{upgrade_tag}.csv',
    )


def _aggregate_one(bldg_path: str, bldg_type: str, upgrade_tag: str,
                   enduse: str):
    '''Stitch chunk MWh CSVs for one (bldg_type, upgrade_tag, enduse) into a
    single GWh CSV. Returns (out_file_path, df_sum) — the caller may use the
    in-memory frame to compute derived aggregates (e.g. the cross-enduse
    total) without re-reading the CSV from disk.

    Raises FileNotFoundError if the chunks subfolder is missing or contains
    no matching files.
    '''
    bldg_tech = f'upgrade{upgrade_tag}'
    chunks_dir = _chunks_eulp_dir(bldg_path, upgrade_tag)

    if not os.path.isdir(chunks_dir):
        raise FileNotFoundError(
            f'Chunks subfolder not found: {chunks_dir}'
        )

    # Trailing underscore on the prefix is required: without it,
    # `upgrade0_reg_*.csv` would also match a `upgrade0_*` query.
    chunk_prefix = f'{bldg_type}_eulp_{enduse}_MWh_{bldg_tech}_'
    eulp_files = [
        os.path.join(chunks_dir, f)
        for f in os.listdir(chunks_dir)
        if f.startswith(chunk_prefix)
    ]
    if not eulp_files:
        raise FileNotFoundError(
            f'No chunk files matching {chunk_prefix}* in {chunks_dir}'
        )

    ls_df = []
    for f in eulp_files:
        print(f)
        df = pd.read_csv(f, index_col=0)
        counties = [c.replace('"', "'").replace("('", "").replace("')", "").split("', '")[0] for c in df.columns]
        cnty_fips = [int(c[1:3] + c[4:7]) for c in counties]
        df.columns = cnty_fips
        df_sum = df.groupby(df.columns, axis=1).sum()
        ls_df.append(df_sum)

    df_sum = pd.concat(ls_df, axis=1)
    df_sum = df_sum.groupby(df_sum.columns, axis=1).sum()
    df_sum = df_sum / 1000  # Convert to GWh

    out_file_path = _agg_path(bldg_path, bldg_type, upgrade_tag, enduse)
    print(f'Outputting csv: {out_file_path}')
    df_sum.to_csv(out_file_path)
    return out_file_path, df_sum


def _write_total_agg(bldg_path: str, bldg_type: str, upgrade_tag: str,
                     total_df) -> str:
    '''Write the total-enduse GWh aggregate to disk and return its path.'''
    out_path = _agg_path(bldg_path, bldg_type, upgrade_tag, 'total')
    print(f'Outputting csv: {out_path}')
    total_df.to_csv(out_path)
    return out_path


def aggregate(bldg_path: str, bldg_type: str, upgrade_tag: str) -> list:
    '''Stitch chunk MWh CSVs for all enduses of one (bldg_type, upgrade_tag),
    plus a `total` aggregate that sums the per-enduse GWh frames.

    Streams the total: keeps only the running sum and the current enduse's
    frame in memory at any one time, never all three at once. Returns the
    full list of output paths (one per enduse plus the total).
    '''
    paths = []
    total = None
    for enduse in ENDUSES:
        path, df = _aggregate_one(bldg_path, bldg_type, upgrade_tag, enduse)
        paths.append(path)
        # Defensive add with fill_value=0 — by construction every enduse's
        # frame should have the same row index (hourly timestamps) and the
        # same column set (county FIPS) since they're derived from the same
        # bldg_id roster and D's county-collapse step. fill_value=0 makes
        # any inadvertent index/column mismatch silently merge rather than
        # leaking NaN into the total.
        total = df if total is None else total.add(df, fill_value=0)
    paths.append(_write_total_agg(bldg_path, bldg_type, upgrade_tag, total))
    return paths


def _list_upgrade_tags(bldg_path: str, bldg_type: str):
    '''Return every upgrade_tag in `bldg_path` for which a cooling_elec agg
    exists. Used by backfill_totals to enumerate completed specs.
    '''
    pattern = _agg_path(bldg_path, bldg_type, '*', 'cooling_elec')
    tags = []
    for p in sorted(glob.glob(pattern)):
        # Parse the tag out of '.../agg_<bt>_eulp_cooling_elec_GWh_upgrade<TAG>.csv'.
        # Splitting on the literal 'GWh_upgrade' delimiter is robust to
        # underscores inside the tag's `reg`/`ref` segment.
        fname = os.path.basename(p)
        tag = fname.split('GWh_upgrade', 1)[1].rsplit('.csv', 1)[0]
        tags.append(tag)
    return tags


def backfill_totals(bldg_path: str, bldg_type: str) -> list:
    '''For each upgrade_tag in `bldg_path` that has all three per-enduse aggs
    but is missing its `total` agg, read the three from disk, sum them, and
    write the total. Idempotent — already-present totals are skipped.

    Reads one frame at a time and accumulates into a running sum so peak
    memory stays at ~2 frames regardless of the per-enduse file size
    (reg-spec frames are ~3-4 GB each in pandas, so this matters).
    '''
    written = []
    for tag in _list_upgrade_tags(bldg_path, bldg_type):
        total_path = _agg_path(bldg_path, bldg_type, tag, 'total')
        if os.path.exists(total_path):
            print(f'[skip] total already exists: {total_path}')
            continue
        per_enduse_paths = {
            e: _agg_path(bldg_path, bldg_type, tag, e) for e in ENDUSES
        }
        missing = [p for p in per_enduse_paths.values() if not os.path.exists(p)]
        if missing:
            print(f'[skip] {tag}: incomplete enduse set; missing {missing}')
            continue
        print(f'[backfill] {tag}: summing enduses into total')
        total = None
        for enduse, p in per_enduse_paths.items():
            print(f'  read {os.path.basename(p)}')
            df = pd.read_csv(p, index_col=0)
            total = df if total is None else total.add(df, fill_value=0)
        written.append(_write_total_agg(bldg_path, bldg_type, tag, total))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description='Aggregate chunk MWh CSVs into GWh CSVs.')
    parser.add_argument('--bldg-path', required=True,
                        help='Run output directory containing chunk CSVs')
    parser.add_argument('--bldg-type', required=True, choices=['res', 'com'],
                        help='Building stock type (res or com)')
    parser.add_argument('--upgrade-tag', required=False, default=None,
                        help='Upgrade tag (e.g. All-Baseline_ref_b2018). '
                             'Required unless --backfill-totals is set.')
    parser.add_argument('--backfill-totals', action='store_true',
                        help='Scan the run dir for completed (cool/heat/non_hvac) '
                             'agg trios and write any missing total aggs from them. '
                             'When set, --upgrade-tag is ignored and the per-spec '
                             'chunk aggregation is skipped.')
    args = parser.parse_args(argv)

    try:
        shutil.copy2(os.path.realpath(__file__), args.bldg_path)
    except Exception as exc:
        print(f'Warning: failed to copy script into run dir: {exc}')

    if args.backfill_totals:
        backfill_totals(args.bldg_path, args.bldg_type)
        return

    if not args.upgrade_tag:
        parser.error('--upgrade-tag is required unless --backfill-totals is set.')
    aggregate(args.bldg_path, args.bldg_type, args.upgrade_tag)


if __name__ == '__main__':
    main()
