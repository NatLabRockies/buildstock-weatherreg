"""Shared vocabulary for the projection package: type aliases, configuration,
the state geography, and the agg/aux input loaders.

Every other module sits on top of this one; nothing here imports them.

RESOLUTION and the three baseline tags (ALL_BASELINE_TAG / ELIGIBLE_TAG /
INELIGIBLE_TAG) are mutated at startup — RESOLUTION by projection.main(),
the tags by set_baseline_tags() called from projection.main() or per-run_dir
in lbl.main(). Read all four as `common.X` (attribute access on this module)
rather than `from .common import X`, so workers see the value set at fork
time instead of a stale import-time copy.
"""

from __future__ import annotations

import json
import os
from typing import Literal

import pandas as pd


type Stock      = Literal['res', 'com']
type Enduse     = Literal['cooling_elec', 'heating_elec', 'non_hvac_elec', 'total']
type Resolution = Literal['state', 'county', 'county_group']
type Scenario   = Literal['baseline', 'upgrade']
type GroupName  = Literal[
    'new_construction', 'surviving',
    'new_adoption', 'surviving_adoption', 'surviving_non_adoption',
    'gap_consumption',
]

# Spec names enumerated from switches_agg_{res,com}stock.json. A spec's on-disk
# tag is `{name}_{'reg' if apply_regression else 'ref'}_b{base_year}`; the
# suffix is fixed within one run_dir but varies across runs, so SpecTag is a
# bare str instead of a Literal.
type SpecName = Literal[
    'All-Baseline', 'Upgraded-Baseline', 'Non-Upgraded-Baseline',
    'Upgraded-Upgrade4', 'Upgraded-Upgrade8', 'Upgraded-Upgrade32',
    'Upgraded-Upgrade1-14', 'Upgraded-Upgrade55', 'Upgraded-Upgrade59',
]
type SpecTag = str

type StatePostal = str   # 'CO'                — agg/gap columns at state resolution
type StateName   = str   # 'Colorado'          — shell-factor table key
type CountyFips  = str   # '8013'              — agg/gap columns at county resolution
type Gisjoin     = str   # 'G0800130'          — S3 county-partition key
type CountyGroup = str   # 'county_group_29'   — BuildStock county-group key (1,038 CONUS)

type GwhFrame  = pd.DataFrame   # DatetimeIndex('timestamp_EST') × geo-keyed GWh cols
type GwhSeries = pd.Series

type ShellFactorKey   = tuple[Stock, Enduse, int, StateName]
type ShellFactorTable = dict[ShellFactorKey, float]
type FactorTable      = dict[str, float]
type EnduseFrames     = dict[Enduse, GwhFrame]

type GroupTask = tuple[
    str,        # run_dir
    Stock,
    str,        # display name used in the output filename (from scenario_names)
    SpecTag,
    Scenario,
    GroupName,
    int,        # projection year
    list[int],  # target weather years
]


PROJECTION_YEARS: tuple[int, ...] = (2027, 2030, 2035, 2040, 2045, 2050)

# Historical / calibration-anchor years. Only Baseline scenario is meaningful
# here — pre-anchor years carry no adoption (adoption_rate = 0 for
# year < ANCHOR_YEAR in growth_factors). The dashboard renders these as a
# single SNA cohort (existing-not-adopted) since the cohort_split functions
# in growth_factors special-case year < ANCHOR_YEAR to all-SNA.
HISTORICAL_YEARS: tuple[int, ...] = (2012, 2018, 2020)

# Combined: every stock year the projection emits outputs for. Past-year
# entries appear only under the Baseline spec; projection-year entries
# appear under every spec the run_specs configuration enumerates.
STOCK_YEARS: tuple[int, ...] = HISTORICAL_YEARS + PROJECTION_YEARS

# state  → county-FIPS agg cols summed to 49 state cols; gap from gap_by_state.csv.
# county → county-FIPS cols kept; gap fetched per-county from S3 (cached).
RESOLUTION: Resolution = 'state'

BASELINE_SPEC_NAME:   SpecName = 'All-Baseline'
ELIGIBLE_SPEC_NAME:   SpecName = 'Upgraded-Baseline'
INELIGIBLE_SPEC_NAME: SpecName = 'Non-Upgraded-Baseline'

# Mutated by set_baseline_tags() before the work starts. The placeholder
# suffix matches the cross_val convention; it gets overwritten as soon as a
# run_dir is loaded.
ALL_BASELINE_TAG: SpecTag = f'{BASELINE_SPEC_NAME}_reg_b2018'
ELIGIBLE_TAG:     SpecTag = f'{ELIGIBLE_SPEC_NAME}_reg_b2018'
INELIGIBLE_TAG:   SpecTag = f'{INELIGIBLE_SPEC_NAME}_reg_b2018'


def set_baseline_tags(run_dir: str) -> None:
    """Set the three module-level tag constants from a run_dir's switches
    snapshot. The suffix is `_reg_b<year>` or `_ref_b<year>` depending on the
    All-Baseline spec's apply_regression flag and base_year. Call this once
    per run_dir before reading any agg/aux file."""
    with open(os.path.join(run_dir, 'inputs', 'switches_agg.json')) as f:
        specs = json.load(f)['run_specs']
    baseline = next(s for s in specs if s['name'] == BASELINE_SPEC_NAME)
    suffix = f"_{'reg' if baseline['apply_regression'] else 'ref'}_b{baseline['base_year']}"
    global ALL_BASELINE_TAG, ELIGIBLE_TAG, INELIGIBLE_TAG
    ALL_BASELINE_TAG = f'{BASELINE_SPEC_NAME}{suffix}'
    ELIGIBLE_TAG     = f'{ELIGIBLE_SPEC_NAME}{suffix}'
    INELIGIBLE_TAG   = f'{INELIGIBLE_SPEC_NAME}{suffix}'

ENDUSES: tuple[Enduse, ...] = ('cooling_elec', 'heating_elec', 'non_hvac_elec', 'total')

# Repo root — one level up from this package, where the data CSVs live.
REPO_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Fraction of ResStock-modeled housing units that are occupied. AEO residential
# numbers report only OCCUPIED households; aux['units_count'] sums every unit in
# the ResStock sampling frame (occupied + vacant). When a cohort amount taken
# from AEO is divided by units_count, the resulting per-building factor is too
# small by this fraction. Multiplying the units_count denominator by 0.878
# brings the basis onto AEO's "occupied" footing — see load_aux_cohort_size.
#
# Source: ResStock options saturation for Vacancy.
RES_OCCUPANCY_FRACTION: float = 0.878


# CONUS+DC only (AK/HI are filtered out upstream).
STATE_FIPS_TO_POSTAL: dict[int, StatePostal] = {
    1: 'AL',   4: 'AZ',   5: 'AR',   6: 'CA',   8: 'CO',   9: 'CT',
    10: 'DE',  11: 'DC',  12: 'FL',  13: 'GA',  16: 'ID',  17: 'IL',
    18: 'IN',  19: 'IA',  20: 'KS',  21: 'KY',  22: 'LA',  23: 'ME',
    24: 'MD',  25: 'MA',  26: 'MI',  27: 'MN',  28: 'MS',  29: 'MO',
    30: 'MT',  31: 'NE',  32: 'NV',  33: 'NH',  34: 'NJ',  35: 'NM',
    36: 'NY',  37: 'NC',  38: 'ND',  39: 'OH',  40: 'OK',  41: 'OR',
    42: 'PA',  44: 'RI',  45: 'SC',  46: 'SD',  47: 'TN',  48: 'TX',
    49: 'UT',  50: 'VT',  51: 'VA',  53: 'WA',  54: 'WV',  55: 'WI',
    56: 'WY',
}
STATE_POSTAL_TO_NAME: dict[StatePostal, StateName] = {
    'AL': 'Alabama',           'AZ': 'Arizona',         'AR': 'Arkansas',
    'CA': 'California',        'CO': 'Colorado',        'CT': 'Connecticut',
    'DE': 'Delaware',          'DC': 'District of Columbia',
    'FL': 'Florida',           'GA': 'Georgia',         'ID': 'Idaho',
    'IL': 'Illinois',          'IN': 'Indiana',         'IA': 'Iowa',
    'KS': 'Kansas',            'KY': 'Kentucky',        'LA': 'Louisiana',
    'ME': 'Maine',             'MD': 'Maryland',        'MA': 'Massachusetts',
    'MI': 'Michigan',          'MN': 'Minnesota',       'MS': 'Mississippi',
    'MO': 'Missouri',          'MT': 'Montana',         'NE': 'Nebraska',
    'NV': 'Nevada',            'NH': 'New Hampshire',   'NJ': 'New Jersey',
    'NM': 'New Mexico',        'NY': 'New York',        'NC': 'North Carolina',
    'ND': 'North Dakota',      'OH': 'Ohio',            'OK': 'Oklahoma',
    'OR': 'Oregon',            'PA': 'Pennsylvania',    'RI': 'Rhode Island',
    'SC': 'South Carolina',    'SD': 'South Dakota',    'TN': 'Tennessee',
    'TX': 'Texas',             'UT': 'Utah',            'VT': 'Vermont',
    'VA': 'Virginia',          'WA': 'Washington',      'WV': 'West Virginia',
    'WI': 'Wisconsin',         'WY': 'Wyoming',
}


def state_fips_from_county(county_fips: CountyFips | int) -> int:
    return int(county_fips) // 1000


def collapse_counties_to_states(df: GwhFrame) -> GwhFrame:
    """Sum county-FIPS columns into state-postal columns."""
    postal_labels = pd.Index(
        [STATE_FIPS_TO_POSTAL[state_fips_from_county(c)] for c in df.columns],
        name='state',
    )
    return df.T.groupby(postal_labels).sum().T


def _load_county_group_mapping() -> tuple[dict[CountyFips, CountyGroup],
                                          dict[CountyGroup, StatePostal]]:
    """Read county_group_mapping.csv (CONUS only; county groups are state-bounded).
    Returns (county_fips → county_group, county_group → state_postal).
    """
    df = pd.read_csv(os.path.join(REPO_DIR, 'county_group_mapping.csv'),
                     usecols=['state', 'county_fips5', 'county_groups'])
    df = df[~df['state'].isin(('AK', 'HI'))]
    fips_to_group = {str(int(r.county_fips5)): r.county_groups
                     for r in df.itertuples(index=False)}
    group_to_state = {r.county_groups: r.state for r in df.itertuples(index=False)}
    return fips_to_group, group_to_state


COUNTY_TO_COUNTY_GROUP, COUNTY_GROUP_TO_STATE_POSTAL = _load_county_group_mapping()


def collapse_counties_to_county_groups(df: GwhFrame) -> GwhFrame:
    """Sum county-FIPS columns into county-group columns."""
    group_labels = pd.Index(
        [COUNTY_TO_COUNTY_GROUP[c] for c in df.columns],
        name='county_group',
    )
    return df.T.groupby(group_labels).sum().T


def agg_path(run_dir: str, stock: Stock, spec_tag: SpecTag, enduse: Enduse) -> str:
    return os.path.join(run_dir, f'agg_{stock}_eulp_{enduse}_GWh_upgrade{spec_tag}.csv')


def aux_path(run_dir: str, spec_tag: SpecTag) -> str:
    return os.path.join(run_dir, f'aux_coverage_upgrade{spec_tag}.csv')


def load_agg_gwh(run_dir: str, stock: Stock, spec_tag: SpecTag, enduse: Enduse) -> GwhFrame:
    """Hourly GWh, timestamp index, county-FIPS columns."""
    df = pd.read_csv(agg_path(run_dir, stock, spec_tag, enduse), index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'timestamp_EST'
    return df


def load_aux_cohort_size(run_dir: str, stock: Stock, spec_tag: SpecTag) -> float:
    """Cohort size on the AEO-comparable basis: billion sqft (com) or million
    OCCUPIED households (res).

    For residential we multiply units_count by RES_OCCUPANCY_FRACTION so the
    denominator matches AEO's occupied-household convention. Without this
    correction, every AEO-cohort / units_count ratio (in factors.py's
    upgrade_factors / baseline_scenario_factors) silently under-counts by
    (1 - occupancy) ≈ 12 %, which propagates into 14 % under-projection of
    residential load.

    Commercial sqft is already on the same basis as AEO total floorspace, so
    no occupancy correction applies there.
    """
    aux = pd.read_csv(aux_path(run_dir, spec_tag))
    if stock == 'com':
        return float(aux['sqft'].sum()) / 1e9
    return float(aux['units_count'].sum()) * RES_OCCUPANCY_FRACTION / 1e6
