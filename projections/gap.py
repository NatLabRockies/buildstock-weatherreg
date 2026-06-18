"""The ComStock gap-model loader: the commercial floorspace ComStock doesn't
simulate. Total-electricity only, from a fixed 2018 base replicated across the
target weather years. `load_gap` picks the source by common.RESOLUTION:
state → one local CSV (49 cols); county → one cached CSV per county from S3.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import common
from .common import REPO_DIR, Gisjoin, GwhFrame, GwhSeries

_GAP_BY_STATE_CSV: str = os.path.join(REPO_DIR, 'gap_by_state.csv')

_GAP_S3_URL_BASE: str = (
    's3://oedi-data-lake/nrel-pds-building-stock/'
    'end-use-load-profiles-for-us-building-stock/2025/'
    'comstock_amy2018_release_2/commercial_gap_model/by_county/upgrade=0/'
)
_GAP_CACHE_DIR: str = '/projects/geohc/radhikar/outputs/gap_model_cache'
_GAP_S3_STORAGE_OPTIONS: dict[str, object] = {'anon': True, 'client_kwargs': {'verify': False}}


def _replicate_2018_across_years(base: GwhFrame, target_years: Sequence[int]) -> GwhFrame:
    """Re-stamp an 8760-row 2018 profile onto each target year, day-of-week aligned."""
    base_jan1_dow = base.index[0].dayofweek
    year_frames = []
    for yr in target_years:
        yr_idx = pd.date_range(start=dt.datetime(yr, 1, 1, 1, 0, 0),
                               periods=len(base), freq='h')
        shift_hours = (base_jan1_dow - yr_idx[0].dayofweek) * 24
        rolled = np.roll(base.values, shift_hours, axis=0)
        year_frames.append(pd.DataFrame(rolled, index=yr_idx, columns=base.columns))
    out = pd.concat(year_frames)
    out.index.name = 'timestamp_EST'
    return out


def load_gap_gwh(target_years: Sequence[int]) -> GwhFrame:
    """State-level gap (49 state-postal columns), replicated to target years."""
    raw = pd.read_csv(_GAP_BY_STATE_CSV)
    raw.columns = ['timestamp', 'kwh', 'state']
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    base = raw.pivot(index='timestamp', columns='state', values='kwh') / 1e6
    base = base.iloc[:8760]   # drop leap-day rows beyond EULP's 8760 hrs/year
    return _replicate_2018_across_years(base, target_years)


def _gap_county_cache_path(gisjoin: Gisjoin, url_base: str) -> str:
    # Hash the URL base into the name so a release bump auto-invalidates the cache.
    digest = hashlib.sha1(url_base.encode('utf-8')).hexdigest()[:12]
    return os.path.join(_GAP_CACHE_DIR, f'{gisjoin}_{digest}.csv')


def _gap_county_s3_key(gisjoin: Gisjoin) -> str:
    return f'{_GAP_S3_URL_BASE}county={gisjoin}/up0-{gisjoin}-gap.csv'


def _load_one_gap_county(gisjoin: Gisjoin) -> GwhSeries | None:
    """One county's 8760-row GWh gap profile from cache or S3; None if absent on S3."""
    cache_path = _gap_county_cache_path(gisjoin, _GAP_S3_URL_BASE)
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        try:
            df = pd.read_csv(_gap_county_s3_key(gisjoin),
                             storage_options=_GAP_S3_STORAGE_OPTIONS)
        except FileNotFoundError:
            return None
        os.makedirs(_GAP_CACHE_DIR, exist_ok=True)
        tmp = cache_path + f'.tmp.{os.getpid()}'   # atomic write: tmp + rename
        df.to_csv(tmp, index=False)
        os.replace(tmp, cache_path)

    kwh_col = 'out.electricity.total.energy_consumption..kwh'
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    series = df.set_index('timestamp')[kwh_col].iloc[:8760] / 1e6   # kWh → GWh
    series.name = gisjoin
    return series


def load_gap_gwh_county(target_years: Sequence[int],
                        gisjoins: Sequence[Gisjoin]) -> GwhFrame:
    """County-level gap (county-FIPS columns) for `gisjoins`, replicated to target
    years. Threaded S3 pulls, hash-cached on disk; counties missing on S3 are
    dropped."""
    print(f'  fetching gap data for {len(gisjoins)} counties (cache: {_GAP_CACHE_DIR})')
    t0 = pd.Timestamp.now()
    with ThreadPoolExecutor(max_workers=32) as pool:
        series_list = list(pool.map(_load_one_gap_county, gisjoins))
    n_ok = sum(s is not None for s in series_list)
    print(f'  gap fetch: {n_ok}/{len(gisjoins)} counties ok '
          f'in {(pd.Timestamp.now() - t0).total_seconds():.1f}s')
    series_ok = [s for s in series_list if s is not None]
    if not series_ok:
        return pd.DataFrame()
    base = pd.concat(series_ok, axis=1).sort_index().iloc[:8760]
    # GISJOIN 'G0100010' → state 01 + county 001 → FIPS '1001' (agg convention).
    base.columns = [str(int(g[1:3]) * 1000 + int(g[4:7])) for g in base.columns]
    return _replicate_2018_across_years(base, target_years)


def load_gap(target_years: Sequence[int], columns: Iterable[str]) -> GwhFrame:
    """Gap GWh at the configured resolution. At state, returns the local 49-state
    CSV. At county, fetches per-county from S3 (cached) for the FIPS columns
    given. At county_group, fetches all CONUS counties and collapses via the
    county_group mapping (state-level gap data isn't apportionable to groups)."""
    if common.RESOLUTION == 'state':
        return load_gap_gwh(target_years)

    if common.RESOLUTION == 'county_group':
        fips_list = list(common.COUNTY_TO_COUNTY_GROUP.keys())
        gisjoins = [f'G{int(c) // 1000:02d}0{int(c) % 1000:03d}0' for c in fips_list]
        county_gap = load_gap_gwh_county(target_years, gisjoins)
        if county_gap.empty:
            return county_gap
        cg_labels = pd.Index(
            [common.COUNTY_TO_COUNTY_GROUP[c] for c in county_gap.columns],
            name='county_group',
        )
        return county_gap.T.groupby(cg_labels).sum().T

    gisjoins = [f'G{int(c) // 1000:02d}0{int(c) % 1000:03d}0' for c in columns]
    return load_gap_gwh_county(target_years, gisjoins)
