#!/usr/bin/env python
'''
This script develops county-level building cooling.elec and heating.elec load
profiles for ReEDS (in GWh) based on resstock/comstock EULP outputs (MWh).
Two GWh files are produced per run — one per enduse — and dropped into the
run output directory alongside the chunk MWh files.

The `upgrade_tag` is the per-spec identifier from `switches_agg.json`'s
`upgrade_specs` of the form `<upgrade_id>_<reg|ref>` — e.g., `0_reg` for
the regressed run of upgrade 0, `0_ref` for the direct-query reference.

Usage:
    python agg_buildings.py --bldg-path <run_dir> --bldg-type res --upgrade-tag 0_reg

Or import the function:
    from agg_buildings import aggregate
    aggregate(run_dir, "res", "0_reg")  # returns list of output paths
'''

import argparse
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
ENDUSES = ('cooling_elec', 'heating_elec')


def _aggregate_one(bldg_path: str, bldg_type: str, upgrade_tag: str,
                   enduse: str) -> str:
    '''Stitch chunk MWh CSVs for one (bldg_type, upgrade_tag, enduse) into a
    single GWh CSV. Returns the output file path. Raises FileNotFoundError
    if the chunks subfolder is missing or contains no matching files.
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

    out_file_path = f'{bldg_path}/agg_{bldg_type}_eulp_{enduse}_GWh_{bldg_tech}.csv'
    print(f'Outputting csv: {out_file_path}')
    df_sum.to_csv(out_file_path)
    return out_file_path


def aggregate(bldg_path: str, bldg_type: str, upgrade_tag: str) -> list:
    '''Stitch chunk MWh CSVs for all enduses of one (bldg_type, upgrade_tag).
    Returns list of output file paths (one per enduse).
    '''
    return [_aggregate_one(bldg_path, bldg_type, upgrade_tag, enduse)
            for enduse in ENDUSES]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Aggregate chunk MWh CSVs into a single GWh CSV.')
    parser.add_argument('--bldg-path', required=True,
                        help='Run output directory containing chunk CSVs')
    parser.add_argument('--bldg-type', required=True, choices=['res', 'com'],
                        help='Building stock type (res or com)')
    parser.add_argument('--upgrade-tag', required=True,
                        help='Upgrade tag of the form <id>_<reg|ref>, e.g. 0_reg')
    args = parser.parse_args(argv)

    try:
        shutil.copy2(os.path.realpath(__file__), args.bldg_path)
    except Exception as exc:
        print(f'Warning: failed to copy script into run dir: {exc}')

    aggregate(args.bldg_path, args.bldg_type, args.upgrade_tag)


if __name__ == '__main__':
    main()
