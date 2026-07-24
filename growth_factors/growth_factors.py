"""
growth_factors.py — Year-by-year cohort splits for ReEDS load projection.

Replaces the calculations in `growth_factors.xlsx` with a transparent Python
implementation. Raw EIA AEO 2025 Reference Case totals are loaded from the
`AEO 2025/` CSVs at module import. Everything else is a constant at the top
or a closed-form formula in the body.

Defaults reproduce the Excel sheet `adoption_2027 (start in 2027)`. Change a
constant below to model a different scenario.

CLI:
  python growth_factors.py     # writes growth_factors.csv (2024..2050, all cohort columns)

API:
  from growth_factors import (
      commercial_cohort_split,
      residential_cohort_split,
  )
"""

import os

import pandas as pd


# ============================================================================
# Scenario constants — edit to change the projection scenario.
# Defaults reproduce the Excel sheet `adoption_2027 (start in 2027)`.
# ============================================================================
ANCHOR_YEAR      = 2027   # Year the adoption ramp begins.
RAMP_END_YEAR    = 2046   # Year adoption hits its maximum (linear ramp between).
GAP_FRACTION     = 0.40   # Commercial only: fraction of stock not in ComStock.

# Maximum adoption rate / eligibility share. Empirical values derived from
# the BSQ aux_coverage_* files in our run directories:
#   MAX_ADOPTION_RES = Upgraded-Baseline.units_count
#                    / (Upgraded-Baseline + Non-Upgraded-Baseline).units_count
#   MAX_ADOPTION_COM = Upgraded-Baseline.sqft
#                    / (Upgraded-Baseline + Non-Upgraded-Baseline).sqft
# Residential is measured in households (units_count); commercial in
# floorspace (sqft). Re-run E_aux_query.py and recompute these whenever
# the restrict/avoid filter in run_specs changes.
# Excel used 0.80 and 0.55 — our BSQ-derived values differ because the
# spec's restrict predicate now selects a different cohort.
MAX_ADOPTION_RES = 0.69   # Maximum residential adoption rate (= eligibility share).
MAX_ADOPTION_COM = 0.56   # Maximum non-gap commercial adoption rate (= eligibility share).

# Per-type residential demolition rates (annual fraction demolished).
# from literature, Berrill et al. 2021
# These are assumed CONSTANTS across years;
RESIDENTIAL_DEMOLITION_RATE = {
    'sf': 0.007053400357414416,   # Single-Family
    'mf': 0.011758511221970355,   # Multifamily
    'mh': 0.03700468124019561,    # Mobile Homes
}


# ============================================================================
# AEO 2025 Reference Case data — loaded once at module import.
# ============================================================================
_AEO_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AEO 2025')
_AEO_COMMERCIAL_CSV = os.path.join(_AEO_DIRECTORY,
    'Table_5._Commercial_Sector_Key_Indicators_and_Consumption.csv')
_AEO_RESIDENTIAL_CSV = os.path.join(_AEO_DIRECTORY,
    'Table_4._Residential_Sector_Key_Indicators_and_Consumption.csv')

# "Full name" rows we read from each AEO CSV.
COMMERCIAL_TOTAL_ROW         = 'Commercial: Total Floorspace: Total: Reference case'
COMMERCIAL_NEW_ADDITIONS_ROW = 'Commercial: Total Floorspace: New Additions: Reference case'
RESIDENTIAL_TOTAL_ROW        = 'Residential: Key Indicators: Households: Total: Reference case'
RESIDENTIAL_SF_ROW           = 'Residential: Key Indicators: Households: Single-Family: Reference case'
RESIDENTIAL_MF_ROW           = 'Residential: Key Indicators: Households: Multifamily: Reference case'
RESIDENTIAL_MH_ROW           = 'Residential: Key Indicators: Households: Mobile Homes: Reference case'


def _load_aeo_csv(path):
    """Parse an AEO 2025 CSV into {full_name: {year: value}}.

    File format: 4 preamble lines, then a header row with year columns, then
    rows of (section, full_name, api_key, units, 2023..2050, growth).
    """
    df = pd.read_csv(path, skiprows=4, header=0)
    df = df.dropna(subset=[df.columns[1]])
    name_column = df.columns[1]
    year_columns = [c for c in df.columns if isinstance(c, str) and c.isdigit()]
    return {
        row[name_column]: {int(year): float(row[year])
                           for year in year_columns
                           if pd.notna(row[year])}
        for _, row in df.iterrows()
    }


_AEO_COMMERCIAL  = _load_aeo_csv(_AEO_COMMERCIAL_CSV)
_AEO_RESIDENTIAL = _load_aeo_csv(_AEO_RESIDENTIAL_CSV)


def commercial_total_floorspace(year):
    """Total commercial floor space (billion sq ft) at `year`."""
    return _AEO_COMMERCIAL[COMMERCIAL_TOTAL_ROW][year]


def commercial_new_construction_floorspace_at(year):
    """Commercial floor space added during `year` (billion sq ft)."""
    return _AEO_COMMERCIAL[COMMERCIAL_NEW_ADDITIONS_ROW][year]


def residential_total_households(year):
    """Total residential households (millions) at `year`."""
    return _AEO_RESIDENTIAL[RESIDENTIAL_TOTAL_ROW][year]


def residential_total_households_by_type(year):
    """Per-type residential households (millions) at `year`."""
    return {
        'sf': _AEO_RESIDENTIAL[RESIDENTIAL_SF_ROW][year],
        'mf': _AEO_RESIDENTIAL[RESIDENTIAL_MF_ROW][year],
        'mh': _AEO_RESIDENTIAL[RESIDENTIAL_MH_ROW][year],
    }


def adoption_rate_existing(year, stock):
    """Cumulative adoption rate for existing buildings of `stock` at `year`.

    Linear ramp over (RAMP_END_YEAR - ANCHOR_YEAR + 1) years inclusive:
        rate(ANCHOR_YEAR)   = max / N    (first year of adoption)
        rate(RAMP_END_YEAR) = max        (last year of ramp; flat after)
    where N = RAMP_END_YEAR - ANCHOR_YEAR + 1.

    This matches the Excel where, e.g., the residential rate at the anchor
    year is 0.04 = 0.80/20, not zero.
    """
    if year < ANCHOR_YEAR:
        return 0.0
    maximum = MAX_ADOPTION_RES if stock == 'res' else MAX_ADOPTION_COM
    if year >= RAMP_END_YEAR:
        return maximum
    ramp_duration_years = RAMP_END_YEAR - ANCHOR_YEAR + 1
    years_into_ramp = year - ANCHOR_YEAR + 1
    return maximum * years_into_ramp / ramp_duration_years


def adoption_rate_new(year):
    """Adoption rate for new construction: 1.0 from ANCHOR_YEAR onward."""
    return 1.0 if year >= ANCHOR_YEAR else 0.0


def commercial_new_cumulative_floorspace(year):
    """Commercial floor space added between ANCHOR_YEAR and `year`, inclusive."""
    if year < ANCHOR_YEAR:
        return 0.0
    return sum(commercial_new_construction_floorspace_at(y)
               for y in range(ANCHOR_YEAR, year + 1))


def commercial_surviving_floorspace(year):
    """Commercial anchor-year stock still standing at `year`.

    No explicit commercial demolition is modeled; surviving stock equals total
    floor space at `year` minus cumulative new construction since the anchor.
    """
    if year < ANCHOR_YEAR:
        return 0.0
    return (commercial_total_floorspace(year)
            - commercial_new_cumulative_floorspace(year))


def residential_surviving_households(year):
    """Residential anchor-year households still standing at `year`.

    Per-type compound demolition:
        surviving = sum over type (
            households_at_anchor[type] * (1 - demolition_rate[type]) ** years_elapsed
        )
    """
    if year < ANCHOR_YEAR:
        return 0.0
    years_elapsed = year - ANCHOR_YEAR
    households_at_anchor = residential_total_households_by_type(ANCHOR_YEAR)
    return sum(
        households_at_anchor[building_type]
            * (1 - RESIDENTIAL_DEMOLITION_RATE[building_type]) ** years_elapsed
        for building_type in ('sf', 'mf', 'mh')
    )


def residential_new_construction_households(year):
    """Residential new construction since anchor; implicit from demolition."""
    if year < ANCHOR_YEAR:
        return 0.0
    return residential_total_households(year) - residential_surviving_households(year)


# ============================================================================
# Cohort splits — the main payload.
# ============================================================================
def commercial_cohort_split(year):
    """Commercial cohort quantities at `year` (all in billion sq ft).

    Returns a dict whose keys describe the cohort. The `_floorspace` suffix
    on every quantity makes the unit explicit at every call site:
        total_floorspace                          - total (gap + non-gap, new + surviving)
        cumulative_new_floorspace                 - new since anchor (gap + non-gap)
        surviving_floorspace                      - anchor stock surviving to year (gap + non-gap)
        gap_total_floorspace                      - total in gap segment (gap_fraction * total)
        adopted_existing_floorspace               - non-gap existing that has adopted by year
        adopted_new_floorspace                    - non-gap new that adopted (all new from anchor onward)
        eligible_not_adopted_existing_floorspace  - non-gap existing eligible to adopt, not yet
        ineligible_existing_floorspace            - non-gap existing that never qualifies
        not_adopted_new_floorspace                - non-gap new that has not adopted (0 from anchor onward)

    The gap/non-gap intermediates are local-only — downstream consumers only
    need the gap_total (for the gap-growth factor) and the adopted/not-adopted
    cohorts (already non-gap-scoped). The gap/non-gap of cumulative_new and
    surviving_floorspace can be reconstructed as `value * GAP_FRACTION` /
    `value * (1 - GAP_FRACTION)` if ever needed.
    """
    total_floorspace                = commercial_total_floorspace(year)
    cumulative_new_floorspace       = commercial_new_cumulative_floorspace(year)
    surviving_floorspace            = commercial_surviving_floorspace(year)
    cumulative_new_nongap_floorspace = cumulative_new_floorspace * (1 - GAP_FRACTION)
    surviving_nongap_floorspace     = surviving_floorspace       * (1 - GAP_FRACTION)
    existing_adoption_rate          = adoption_rate_existing(year, 'com')
    new_adoption_rate               = adoption_rate_new(year)

    return {
        'total_floorspace':                          total_floorspace,
        'cumulative_new_floorspace':                 cumulative_new_floorspace,
        'surviving_floorspace':                      surviving_floorspace,
        'gap_total_floorspace':                      total_floorspace * GAP_FRACTION,
        'adopted_existing_floorspace':               existing_adoption_rate * surviving_nongap_floorspace,
        'adopted_new_floorspace':                    new_adoption_rate      * cumulative_new_nongap_floorspace,
        'eligible_not_adopted_existing_floorspace':  (MAX_ADOPTION_COM - existing_adoption_rate) * surviving_nongap_floorspace,
        'ineligible_existing_floorspace':            (1 - MAX_ADOPTION_COM) * surviving_nongap_floorspace,
        'not_adopted_new_floorspace':                (1 - new_adoption_rate) * cumulative_new_nongap_floorspace,
    }


def residential_cohort_split(year):
    """Residential cohort quantities at `year` (all in million households).

    Residential has no gap concept (all households are in the ResStock
    simulation), so the cohort structure is simpler than commercial. The
    `_households` suffix on every quantity makes the unit explicit:
        total_households                          - total households
        cumulative_new_households                 - new since anchor
        surviving_households                      - anchor stock surviving to year
        adopted_existing_households               - existing that has adopted by year
        adopted_new_households                    - new that adopted (all new from anchor onward)
        eligible_not_adopted_existing_households  - existing eligible to adopt, not yet
        ineligible_existing_households            - existing that never qualifies
        not_adopted_new_households                - new that has not adopted (0 from anchor onward)
    """
    total_households            = residential_total_households(year)
    surviving_households        = residential_surviving_households(year)
    cumulative_new_households   = total_households - surviving_households
    existing_adoption_rate      = adoption_rate_existing(year, 'res')
    new_adoption_rate           = adoption_rate_new(year)

    return {
        'total_households':                          total_households,
        'cumulative_new_households':                 cumulative_new_households,
        'surviving_households':                      surviving_households,
        'adopted_existing_households':               existing_adoption_rate * surviving_households,
        'adopted_new_households':                    new_adoption_rate      * cumulative_new_households,
        'eligible_not_adopted_existing_households':  (MAX_ADOPTION_RES - existing_adoption_rate) * surviving_households,
        'ineligible_existing_households':            (1 - MAX_ADOPTION_RES) * surviving_households,
        'not_adopted_new_households':                (1 - new_adoption_rate) * cumulative_new_households,
    }


# ============================================================================
# CLI: prints a CSV with one row per year for spot-checking against Excel.
# ============================================================================
def _full_table(years):
    rows = []
    for year in years:
        commercial = commercial_cohort_split(year)
        residential = residential_cohort_split(year)
        row = {'Year': year}
        for key, value in commercial.items():
            row[f'com_{key}'] = value
        for key, value in residential.items():
            row[f'res_{key}'] = value
        row['adoption_rate_existing_com'] = adoption_rate_existing(year, 'com')
        row['adoption_rate_existing_res'] = adoption_rate_existing(year, 'res')
        row['adoption_rate_new']          = adoption_rate_new(year)
        rows.append(row)
    return pd.DataFrame(rows)


_OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'growth_factors.csv'
)


def main():
    df = _full_table(range(2024, 2051))
    df.to_csv(_OUTPUT_CSV, index=False, float_format='%.6f')
    print(f'wrote {_OUTPUT_CSV} ({len(df)} rows, {len(df.columns)} columns)')


if __name__ == '__main__':
    main()
