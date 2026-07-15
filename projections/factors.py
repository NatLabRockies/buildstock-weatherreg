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
    STATE_POSTAL_TO_NAME,
    Enduse,
    FactorTable,
    ShellFactorTable,
    StatePostal,
    Stock,
)
from .growth_factors import (
    ANCHOR_YEAR,
    GAP_FRACTION,
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
    construction is assumed unchanged for non-HVAC). 1.0 for year < ANCHOR_YEAR
    too — pre-anchor years carry no NC cohort (baseline_scenario_factors
    returns new_construction = 0 for historical years), so the eventual
    multiplication zeros out anyway and we'd rather avoid the shell_factors
    KeyError. Raises KeyError on a missing entry at projection years.
    """
    if enduse in ('non_hvac_elec', 'total'):
        return 1.0
    if year < ANCHOR_YEAR:
        return 1.0
    return _SHELL_FACTORS[(stock, enduse, year, STATE_POSTAL_TO_NAME[state_postal])]


def gap_growth_factor(year: int) -> float:
    """Commercial gap floorspace growth relative to the anchor year.
    Uses `commercial_total_floorspace` directly (full AEO, gap + non-gap)
    rather than `commercial_cohort_split['total_floorspace']` because the
    cohort split now returns non-gap only — but the gap's growth ratio
    inherently depends on full AEO floorspace."""
    return (commercial_total_floorspace(year)
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
    eligible_size   = common.load_aux_cohort_size(run_dir, stock, common.ELIGIBLE_TAG)
    ineligible_size = common.load_aux_cohort_size(run_dir, stock, common.INELIGIBLE_TAG)

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


def baseline_scenario_factors(run_dir: str, stock: Stock, year: int) -> FactorTable:
    """Stock-growth ratios for the no-adoption baseline at one (stock, year).

    Keys: new_construction, surviving. Both denominators come from
    `load_aux_cohort_size`, which puts the ResStock raw `units_count` on the
    AEO occupied-households basis (× RES_OCCUPANCY_FRACTION; for commercial
    sqft is already on AEO basis). The factor shape is uniform with
    `upgrade_factors`: AEO_cohort_amount(year) / load_aux_cohort_size(source).

    * `new_construction` is applied to the *Upgraded-Baseline* (eligible)
      load source, not All-Baseline. The reasoning: new buildings will be
      modern-construction tracked to ComStock/ResStock coverage and will
      have characteristics that *support* the upgrade — whether the upgrade
      is installed is the scenario question, not the stock question. Factor
      = `cumulative_new_<unit>` divided by the *eligible* aux cohort size.
      The commercial cohort split now reports `cumulative_new_floorspace`
      on a non-gap basis (see `commercial_cohort_split`), so no
      `(1 - GAP_FRACTION)` correction is needed here — it's already baked
      into the cohort.
    * `surviving` is applied to the All-Baseline source. Factor =
      `surviving_<unit>` / load_aux_cohort_size(ALL_BASELINE) — the
      occupancy-aware total stock denominator. Previously this used
      AEO(ANCHOR_YEAR) directly, which silently dropped the
      ResStock raw→occupied correction and under-projected baseline load by
      ~12 % (the missing 1 / RES_OCCUPANCY_FRACTION ≈ 1.139 factor). The
      commercial surviving cohort is now non-gap to match the All-Baseline
      source's non-gap basis (previously mixed gap-incl numerator with
      non-gap denominator, which over-projected by 1/0.6 and double-counted
      the gap that `get_gap` separately handles).
    """
    eligible_size     = common.load_aux_cohort_size(run_dir, stock, common.ELIGIBLE_TAG)
    all_baseline_size = common.load_aux_cohort_size(run_dir, stock, common.ALL_BASELINE_TAG)
    if stock == 'com':
        cohort = commercial_cohort_split(year)
        return {
            'new_construction': cohort['cumulative_new_floorspace'] / eligible_size,
            'surviving':        cohort['surviving_floorspace']      / all_baseline_size,
        }
    cohort = residential_cohort_split(year)
    return {
        'new_construction': cohort['cumulative_new_households']   / eligible_size,
        'surviving':        cohort['surviving_households']        / all_baseline_size,
    }
