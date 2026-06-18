"""The multipliers the projection applies.

new_construction_efficiency_factor varies per (stock, enduse, year, state) and
comes from shell_factors_combined.csv. upgrade_factors / baseline_scenario_factors
return scalar cohort-growth ratios per (stock, year), derived from
growth_factors' AEO cohort splits. gap_growth_factor scales the commercial gap
profile by AEO floorspace growth. GAP_DERATING_FACTOR is a year-/scenario-
independent constant applied to the ComStock gap profile (T&D losses + other
consumption-side adjustments between the gap-model output and what shows up at
the consumption meter).
"""

from __future__ import annotations

import os
from typing import Literal

import pandas as pd

from . import common
from .common import (
    ELIGIBLE_TAG,
    INELIGIBLE_TAG,
    STATE_POSTAL_TO_NAME,
    Enduse,
    FactorTable,
    ShellFactorTable,
    StatePostal,
    Stock,
)
from .growth_factors import (
    ANCHOR_YEAR,
    commercial_cohort_split,
    commercial_total_floorspace,
    residential_cohort_split,
)

_SHELL_FACTORS_CSV: str = os.path.join(common.REPO_DIR, 'shell_factors_combined.csv')


def _load_shell_factors(path: str) -> ShellFactorTable:
    """Parse shell_factors_combined.csv → {(stock, enduse, year, state_name): factor}."""
    df = pd.read_csv(path)
    df.columns = ['shell_factor', 'table_name', 'source', 'year', 'type', 'state']
    df['stock']  = df['table_name'].map({'Commercial': 'com', 'Residential': 'res'})
    df['enduse'] = df['type'].map({'Cooling': 'cooling_elec', 'Heating': 'heating_elec'})
    df['year']   = df['year'].astype(int)
    return {
        (r.stock, r.enduse, r.year, r.state): float(r.shell_factor)
        for r in df.itertuples(index=False)
    }


_SHELL_FACTORS: ShellFactorTable = _load_shell_factors(_SHELL_FACTORS_CSV)


def new_construction_efficiency_factor(stock: Stock, enduse: Enduse, year: int,
                                       state_postal: StatePostal) -> float:
    """Efficiency of newly-built stock relative to the base year, per state.

    1.0 for non_hvac_elec and total (the table covers only HVAC enduses; new
    construction is assumed unchanged for non-HVAC). Raises KeyError on a
    missing (stock, enduse, year, state) rather than silently scaling by 1.
    """
    if enduse in ('non_hvac_elec', 'total'):
        return 1.0
    return _SHELL_FACTORS[(stock, enduse, year, STATE_POSTAL_TO_NAME[state_postal])]


def gap_growth_factor(year: int) -> float:
    """Commercial gap floorspace growth relative to the anchor year."""
    return (commercial_cohort_split(year)['total_floorspace']
            / commercial_total_floorspace(ANCHOR_YEAR))


# Year- and scenario-independent derating applied to the ComStock gap profile
# in get_gap. Accounts for T&D losses and other downstream adjustments between
# the gap-model's raw output and what shows up at the consumption side of the
# meter ReEDS/LBL ultimately care about.
GAP_DERATING_FACTOR: float = 0.5


def upgrade_factors(run_dir: str, stock: Stock, projection_year: int) -> FactorTable:
    """Cohort-growth ratios for the upgrade scenario at one (stock, year).

    Keys: new_adoption, surviving_adoption, surviving_not_adopted_eligible,
    surviving_not_adopted_ineligible. Each cohort amount from the AEO split is
    divided by the simulated cohort size so it scales the simulated load.

    The eligible/ineligible cohort sizes come from the aux files. AEO amounts
    are in billion sqft / million households, so load_aux_cohort_size already
    rescales the aux denominators to match.
    """
    eligible_size   = common.load_aux_cohort_size(run_dir, stock, ELIGIBLE_TAG)
    ineligible_size = common.load_aux_cohort_size(run_dir, stock, INELIGIBLE_TAG)

    cohort: dict[str, float]
    unit: Literal['floorspace', 'households']
    if stock == 'com':
        cohort = commercial_cohort_split(projection_year)
        unit = 'floorspace'
    else:
        cohort = residential_cohort_split(projection_year)
        unit = 'households'

    return {
        'new_adoption':                     cohort[f'adopted_new_{unit}']                   / eligible_size,
        'surviving_adoption':               cohort[f'adopted_existing_{unit}']              / eligible_size,
        'surviving_not_adopted_eligible':   cohort[f'eligible_not_adopted_existing_{unit}'] / eligible_size,
        'surviving_not_adopted_ineligible': (cohort[f'ineligible_existing_{unit}'] / ineligible_size
                                             if ineligible_size > 0 else 0.0),
    }


def baseline_scenario_factors(stock: Stock, year: int) -> FactorTable:
    """Stock-growth ratios for the no-adoption baseline at one (stock, year).

    Keys: new_construction, surviving. These multiply the All-Baseline load
    directly (All-Baseline is the cohort, so no aux denominator).
    """
    if stock == 'com':
        cohort = commercial_cohort_split(year)
        anchor = commercial_total_floorspace(ANCHOR_YEAR)
        return {
            'new_construction': cohort['cumulative_new_floorspace'] / anchor,
            'surviving':        cohort['surviving_floorspace']      / anchor,
        }
    cohort = residential_cohort_split(year)
    anchor = residential_cohort_split(ANCHOR_YEAR)['total_households']
    return {
        'new_construction': cohort['cumulative_new_households'] / anchor,
        'surviving':        cohort['surviving_households']      / anchor,
    }
