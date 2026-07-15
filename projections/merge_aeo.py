"""Merge AEO Reference Case CSVs across vintages into one CSV per sector.

Reads every `<Residential|Commercial>_Sector_Key_Indicators_and_Consumption_<vintage>.csv`
in `AEO 2025/`, plus the un-vintaged `Table_{4,5}._*.csv` (which are the
AEO 2025 vintage exports), and produces:

  projections/data/aeo_merged_residential.csv
  projections/data/aeo_merged_commercial.csv

The merge rule is **most-recent-vintage-wins per (row, year)**:
  AEO 2025 covers 2023-2050 → wins those years.
  AEO 2018 covers 2016-2050 → wins 2016-2022 (newer vintages don't have).
  AEO 2012 covers 2009-2035 → wins 2009-2015.
  AEO 2010 covers 2007-2035 → wins 2007-2008.

Some vintage files are **partial single-indicator exports** (e.g. AEO 2020
carries only Commercial Surviving Floorspace; AEO 2019/2021 only Residential
Households) rather than full-sector dumps. Because the rule is per (row, year),
a partial file only overrides the specific rows it contains — the intermediate
vintages (2015/2019/2020/2021) refine individual recent-year cells that the
older full-sector vintages would otherwise supply, leaving unrelated rows
untouched.

Output covers years 2007-2050 inclusive. Same 4-line preamble + header
+ data shape as the source AEO CSVs, so growth_factors._load_aeo_csv
reads them with `skiprows=4` without any code change.

This script is idempotent — re-run it any time the source AEO files
change. Output goes under `projections/data/` (git-tracked), which is
the projection package's preferred location for derived inputs.

CLI:
  python -m projections.merge_aeo
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


_REPO_DIR: Path = Path(__file__).resolve().parent.parent
_AEO_DIR:  Path = _REPO_DIR / 'AEO 2025'
_OUT_DIR:  Path = Path(__file__).resolve().parent / 'data'

_TARGET_YEARS: list[str] = [str(y) for y in range(2007, 2051)]


def _vintage_year(path: Path) -> int:
    """Extract vintage year. Table_*.csv → 2025 (the current AEO);
    *_<YYYY>.csv → that 4-digit suffix."""
    if path.name.startswith('Table_'):
        return 2025
    m = re.search(r'_(\d{4})\.csv$', path.name)
    return int(m.group(1)) if m else -1


def _list_vintages(sector: str) -> list[Path]:
    """Sorted newest-first."""
    sec_cap = sector.title()
    paths = list(_AEO_DIR.glob(
        f'{sec_cap}_Sector_Key_Indicators_and_Consumption_*.csv'))
    table_path = _AEO_DIR / (
        f'Table_4._{sec_cap}_Sector_Key_Indicators_and_Consumption.csv'
        if sector == 'residential'
        else f'Table_5._{sec_cap}_Sector_Key_Indicators_and_Consumption.csv'
    )
    if table_path.exists():
        paths.append(table_path)
    return sorted(paths, key=_vintage_year, reverse=True)


def _read_aeo(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    """Return (df, name_col, year_cols). df is indexed by row position; year
    columns are detected by 4-digit string headers."""
    df = pd.read_csv(path, skiprows=4)
    name_col = df.columns[1]
    df = df.dropna(subset=[name_col])
    year_cols = [c for c in df.columns if isinstance(c, str) and c.isdigit()]
    return df, name_col, year_cols


def _merge_sector(sector: str) -> pd.DataFrame:
    """Merge all vintages for one sector. Most-recent-wins per (row, year)
    via pandas `combine_first`: the newer frame's values stay, NaNs are
    filled from the older frame."""
    vintages = _list_vintages(sector)
    if not vintages:
        raise RuntimeError(f'no AEO vintages found for {sector!r}')

    # Take metadata (non-year columns) from the newest vintage. Newer vintages
    # have richer / corrected api_keys, so we want their version. Rows that
    # only exist in older vintages get appended below.
    newest_df, name_col, newest_years = _read_aeo(vintages[0])
    non_year_cols = [
        c for c in newest_df.columns
        if not (isinstance(c, str) and c.isdigit()) and c != name_col
    ]

    # Year matrix: each vintage contributes its year columns, indexed by
    # full-name. combine_first does the right thing: newer wins, older fills.
    year_matrices: list[pd.DataFrame] = []
    for vp in vintages:
        df, vname_col, vyears = _read_aeo(vp)
        keep = [vname_col] + [y for y in vyears if y in _TARGET_YEARS]
        m = df[keep].rename(columns={vname_col: '_name'}).set_index('_name')
        year_matrices.append(m)

    merged = year_matrices[0]
    for older in year_matrices[1:]:
        merged = merged.combine_first(older)

    # Ensure every target year is present (some may be missing in all
    # vintages — keep them as NaN columns so downstream readers see a
    # consistent shape).
    for y in _TARGET_YEARS:
        if y not in merged.columns:
            merged[y] = pd.NA
    merged = merged[_TARGET_YEARS]

    # Re-attach metadata. Rows that exist in older-only get NaN metadata —
    # rare; if it matters we could enrich from older vintages.
    metadata = newest_df.set_index(name_col)[non_year_cols]
    metadata.index.name = '_name'
    out = metadata.join(merged, how='outer')
    out = out.reset_index().rename(columns={'_name': name_col})
    # Reorder so name_col sits where it does in source files (position 1).
    cols_ordered = ([out.columns[0] if out.columns[0] != name_col
                     else non_year_cols[0]]
                    + [name_col]
                    + [c for c in non_year_cols if c != out.columns[0]]
                    + _TARGET_YEARS)
    # Simpler: just put name_col second + all metadata + years.
    other_meta = [c for c in non_year_cols if c not in (name_col,)]
    section_col = newest_df.columns[0]  # the "section" / first column
    out_cols = [section_col, name_col] + other_meta + _TARGET_YEARS
    # Some out_cols may not be present after the join; filter.
    out_cols = [c for c in out_cols if c in out.columns]
    return out[out_cols]


def write_merged(sector: str) -> Path:
    """Write `aeo_merged_<sector>.csv` under projections/data/."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = _merge_sector(sector)
    out = _OUT_DIR / f'aeo_merged_{sector}.csv'
    # 4-line preamble matching the AEO source format so growth_factors
    # can read with skiprows=4 unchanged.
    sec_cap = sector.title()
    preamble = (
        f'"{sec_cap} Sector Key Indicators and Consumption (merged AEO vintages)"\n'
        f'"https://github.com/NatLabRockies/buildstock-weatherreg/tree/main/projections/merge_aeo.py"\n'
        f'"Most-recent-vintage-wins per (row, year); years 2007-2050"\n'
        f'"Source: U.S. Energy Information Administration (AEO 2010, 2012, 2018, 2025)"\n'
    )
    with open(out, 'w') as f:
        f.write(preamble)
    merged.to_csv(out, mode='a', index=False)
    return out


def main() -> None:
    for sector in ('residential', 'commercial'):
        out = write_merged(sector)
        df = pd.read_csv(out, skiprows=4)
        n_years = sum(1 for c in df.columns if isinstance(c, str) and c.isdigit())
        print(f'wrote {out.relative_to(_REPO_DIR)}  '
              f'({len(df)} rows × {n_years} year columns)')


if __name__ == '__main__':
    main()
