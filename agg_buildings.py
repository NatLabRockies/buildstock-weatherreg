#!/usr/bin/env python
'''
This script develops county-level building hvac load profiles for ReEDS (in GWh) based on resstock/comstock EULP outputs (MWh). The resulting GWh file is dropped into the run output directory alongside the chunk MWh files.

The `upgrade_tag` is the per-spec identifier from `switches_agg.json`'s
`upgrade_specs` of the form `<upgrade_id>_<reg|ref>` — e.g., `0_reg` for
the regressed run of upgrade 0, `0_ref` for the direct-query reference.

Usage:
    python agg_buildings.py --bldg-path <run_dir> --bldg-type res --upgrade-tag 0_reg

Or import the function:
    from agg_buildings import aggregate
    aggregate(run_dir, "res", "0_reg")
'''

import argparse
import os
import shutil
import pandas as pd


def aggregate(bldg_path: str, bldg_type: str, upgrade_tag: str) -> str:
    '''Stitch chunk MWh CSVs for one (bldg_type, upgrade_tag) into a single GWh CSV.

    Returns the output file path. Raises FileNotFoundError if no chunk files
    are found.
    '''
    bldg_tech = f'upgrade{upgrade_tag}'

    # Trailing underscore on the prefix is required: without it,
    # `upgrade0_reg_*.csv` would also match a `upgrade0_*` query.
    chunk_prefix = f'{bldg_type}_eulp_hvac_elec_MWh_{bldg_tech}_'
    eulp_files = [
        f'{bldg_path}/{f}'
        for f in os.listdir(bldg_path)
        if f.startswith(chunk_prefix)
    ]
    if not eulp_files:
        raise FileNotFoundError(
            f'No chunk files found matching {chunk_prefix}* in {bldg_path}'
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

    out_file_path = f'{bldg_path}/agg_{bldg_type}_eulp_hvac_elec_GWh_{bldg_tech}.csv'
    print(f'Outputting csv: {out_file_path}')
    df_sum.to_csv(out_file_path)
    return out_file_path


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
