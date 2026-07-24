"""
growth_factors.py — Year-by-year cohort splits for ReEDS load projection.

Replaces the calculations in `growth_factors.xlsx` with a transparent Python
implementation. Raw EIA AEO 2025 Reference Case totals are loaded from the
`AEO 2025/` CSVs at module import. Everything else is a constant at the top
or a closed-form formula in the body.

Defaults reproduce the Excel sheet `adoption_2027 (start in 2027)`. Change a
constant below to model a different scenario.

CLI:
  # AEO cohort amounts + adoption rates only (no run_dir needed):
  python -m projections.growth_factors
  # ...plus the factor multipliers the projection actually applies:
  python -m projections.growth_factors --run-dir-res <res_run_dir> --run-dir-com <com_run_dir>
  # Writes growth_factors.csv over HISTORICAL_YEARS + 2021..2050, with a
  # plain-English description row under the header. The two-run-dir form also
  # inlines the cohort-size denominators and writes growth_factors_denominators.csv.
  # Override the output path with --out.

API:
  from projections.growth_factors import (
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
# This module lives in projections/, but the data CSVs sit at the repo root
# (one level up), so resolve paths relative to the parent directory.
# Single merged CSV per sector, covering 2007-2050. Produced by
# `python -m projections.merge_aeo`. Most-recent-vintage-wins per (row, year)
# so we can pull historical AEO data (2007-2022 from older vintages) and
# future projections (2023-2050 from AEO 2025) through one loader call —
# without any year-vs-vintage logic in this module.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AEO_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_AEO_RESIDENTIAL_CSV = os.path.join(_AEO_DATA_DIR, 'aeo_merged_residential.csv')
_AEO_COMMERCIAL_CSV  = os.path.join(_AEO_DATA_DIR, 'aeo_merged_commercial.csv')

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
def _all_existing_cohort_split_commercial(year):
    """For year < ANCHOR_YEAR: every existing non-gap sqft sits in the
    `ineligible_existing` cohort (which the dashboard maps to SNA), with
    NC = SA = 0.

    Every `*_floorspace` quantity here is on a NON-GAP basis (matches what
    ComStock actually simulates and what the dashboard displays). The full
    AEO trajectory including the unsimulated gap is reconstructable as
    `total_floorspace + gap_total_floorspace`."""
    total = commercial_total_floorspace(year)
    nongap = total * (1 - GAP_FRACTION)
    return {
        'total_floorspace':                          nongap,
        'cumulative_new_floorspace':                 0.0,
        'surviving_floorspace':                      nongap,
        'gap_total_floorspace':                      total * GAP_FRACTION,
        'adopted_existing_floorspace':               0.0,
        'adopted_new_floorspace':                    0.0,
        'eligible_not_adopted_existing_floorspace':  0.0,
        'ineligible_existing_floorspace':            nongap,
        'not_adopted_new_floorspace':                0.0,
    }


def _all_existing_cohort_split_residential(year):
    """Pre-anchor residential split: everything in `ineligible_existing`
    (= dashboard SNA). NC = SA = 0; eligible_not_adopted = 0 because no
    adoption ramp has started."""
    total = residential_total_households(year)
    return {
        'total_households':                          total,
        'cumulative_new_households':                 0.0,
        'surviving_households':                      total,
        'adopted_existing_households':               0.0,
        'adopted_new_households':                    0.0,
        'eligible_not_adopted_existing_households':  0.0,
        'ineligible_existing_households':            total,
        'not_adopted_new_households':                0.0,
    }


def commercial_cohort_split(year):
    """Commercial cohort quantities at `year` (all in billion sq ft).

    Every `_floorspace` key here is on a NON-GAP basis (the portion ComStock
    actually simulates, ~60% of AEO total). The gap is reported as a separate
    `gap_total_floorspace` key for informational use — it's NOT included in
    `total_floorspace` / `surviving_floorspace` / `cumulative_new_floorspace`.
    To reconstruct the full AEO total (gap + non-gap): `total_floorspace +
    gap_total_floorspace`.

    Keys:
        total_floorspace                          - non-gap total (= ComStock simulated frame at year)
        cumulative_new_floorspace                 - non-gap new since anchor
        surviving_floorspace                      - non-gap anchor stock surviving to year
        gap_total_floorspace                      - gap segment (gap_fraction × AEO total, informational)
        adopted_existing_floorspace               - existing that has adopted by year
        adopted_new_floorspace                    - new that adopted (all new from anchor onward)
        eligible_not_adopted_existing_floorspace  - existing eligible to adopt, not yet
        ineligible_existing_floorspace            - existing that never qualifies
        not_adopted_new_floorspace                - new that has not adopted (0 from anchor onward)

    For year < ANCHOR_YEAR the split collapses to all-SNA-equivalent — that's
    the historical / calibration regime where neither adoption nor new
    construction (since anchor) has accumulated yet.

    The non-gap basis matches the residential cohort split (which is on
    ResStock raw / total-dwelling-units basis), so cross-stock comparisons
    line up. The dashboard's commercial trajectory is now continuous across
    the ANCHOR_YEAR boundary (no longer drops 30 B sqft from 2020 → 2027).
    """
    if year < ANCHOR_YEAR:
        return _all_existing_cohort_split_commercial(year)
    total_floorspace_aeo            = commercial_total_floorspace(year)
    cumulative_new_floorspace_aeo   = commercial_new_cumulative_floorspace(year)
    surviving_floorspace_aeo        = commercial_surviving_floorspace(year)
    cumulative_new_nongap_floorspace = cumulative_new_floorspace_aeo * (1 - GAP_FRACTION)
    surviving_nongap_floorspace     = surviving_floorspace_aeo       * (1 - GAP_FRACTION)
    total_nongap_floorspace         = total_floorspace_aeo           * (1 - GAP_FRACTION)
    existing_adoption_rate          = adoption_rate_existing(year, 'com')
    new_adoption_rate               = adoption_rate_new(year)

    return {
        'total_floorspace':                          total_nongap_floorspace,
        'cumulative_new_floorspace':                 cumulative_new_nongap_floorspace,
        'surviving_floorspace':                      surviving_nongap_floorspace,
        'gap_total_floorspace':                      total_floorspace_aeo * GAP_FRACTION,
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

    For year < ANCHOR_YEAR the split collapses to all-SNA-equivalent: the
    historical / calibration regime where adoption hasn't begun and no
    new construction has been tracked from the anchor's perspective.
    """
    if year < ANCHOR_YEAR:
        return _all_existing_cohort_split_residential(year)
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
# With --run-dir-res / --run-dir-com, also includes the actual factor
# multipliers (AEO_cohort_amount / load_aux_cohort_size(source_spec)) the
# projection applies. Without those args, the CSV only has AEO cohort
# amounts.
# ============================================================================
def _full_table(years):
    """Per-year AEO direct inputs + cohort amounts + adoption rates.

    Residential cohort amounts are converted from AEO's occupied-household
    basis to ResStock's total dwelling-unit basis (includes occupied +
    vacant) by dividing by RES_OCCUPANCY_FRACTION = 0.878. Column names use
    `_dwelling_units` instead of `_households` to make the basis unambiguous.
    `res_aeo_total_households` keeps the raw AEO value (occupied only) as a
    transparency column at the top.

    Commercial cohort amounts are on a ComStock non-gap basis (excludes the
    ~40 percent unsimulated gap). `com_aeo_total_floorspace` keeps the raw
    AEO value (incl gap) at the top; `com_gap_total_floorspace` reports the
    gap portion separately.
    """
    # Import here to avoid a top-level cycle with common.py at module load.
    from .common import RES_OCCUPANCY_FRACTION
    rows = []
    for year in years:
        commercial = commercial_cohort_split(year)
        residential = residential_cohort_split(year)
        # AEO direct inputs — copied verbatim from the merged AEO data, plus
        # two AEO intermediates (cumulative new since 2027 and surviving stock)
        # so the downstream cohort columns can reference left-only columns.
        res_by_type = residential_total_households_by_type(year)
        # res_aeo_surviving: for year < 2027 there's no anchor concept yet,
        # so "surviving" is just the full AEO total (everything is existing).
        # For year >= 2027 it's the 2027 anchor stock decayed by per-type
        # demolition rates (handled by residential_surviving_households).
        res_aeo_surviving = (residential_total_households(year)
                             if year < ANCHOR_YEAR
                             else residential_surviving_households(year))
        row = {
            'Year': year,
            'com_aeo_total_floorspace':            commercial_total_floorspace(year),
            'com_aeo_new_construction_floorspace': commercial_new_construction_floorspace_at(year),
            'com_aeo_cumulative_new_floorspace':   commercial_new_cumulative_floorspace(year),
            'res_aeo_total_households':            residential_total_households(year),
            'res_aeo_sf_households':               res_by_type['sf'],
            'res_aeo_mf_households':               res_by_type['mf'],
            'res_aeo_mh_households':               res_by_type['mh'],
            'res_aeo_surviving_households':        res_aeo_surviving,
        }
        for key, value in commercial.items():
            row[f'com_{key}'] = value
        for key, value in residential.items():
            # AEO occupied households -> total dwelling units (incl vacant).
            new_key = key.replace('_households', '_dwelling_units')
            row[f'res_{new_key}'] = value / RES_OCCUPANCY_FRACTION
        row['adoption_rate_existing_com'] = adoption_rate_existing(year, 'com')
        row['adoption_rate_existing_res'] = adoption_rate_existing(year, 'res')
        row['adoption_rate_new']          = adoption_rate_new(year)
        rows.append(row)
    return pd.DataFrame(rows)


def _factor_table(years, run_dir_res, run_dir_com):
    """Per-year factor multipliers actually applied by the projection.

    All factors are AEO_cohort_amount(year) / load_aux_cohort_size(source_spec),
    where the source_spec depends on the cohort:
      * NC + adoption-related cohorts → ELIGIBLE (Upgraded-Baseline) aux
      * Baseline surviving             → ALL_BASELINE aux
      * Ineligible cohort              → INELIGIBLE aux

    Requires `run_dir_res` and `run_dir_com` because `load_aux_cohort_size`
    reads the aux_coverage_*.csv files for the specific BSQ run.
    """
    from . import factors
    rows = []
    for year in years:
        row = {'Year': year}
        bres = factors.baseline_scenario_factors(run_dir_res, 'res', year)
        bcom = factors.baseline_scenario_factors(run_dir_com, 'com', year)
        for k, v in bres.items(): row[f'factor_res_baseline_{k}'] = v
        for k, v in bcom.items(): row[f'factor_com_baseline_{k}'] = v
        # Upgrade factors are only meaningful at year >= ANCHOR_YEAR — pre-anchor
        # the cohort split degenerates to all-SNA, which puts every household /
        # sqft into ineligible_existing and produces a misleading factor when
        # divided by the (smaller) INELIGIBLE-only denominator.
        if year >= ANCHOR_YEAR:
            ures = factors.upgrade_factors(run_dir_res, 'res', year)
            ucom = factors.upgrade_factors(run_dir_com, 'com', year)
            for k, v in ures.items(): row[f'factor_res_upgrade_{k}']  = v
            for k, v in ucom.items(): row[f'factor_com_upgrade_{k}']  = v
        rows.append(row)
    return pd.DataFrame(rows)


def _denominator_summary(run_dir_res, run_dir_com):
    """One-row dataframe with the cohort-size denominators every factor in
    the file divides by — expressed on the *same basis* as the cohort-amount
    columns:
      * residential → total dwelling units (includes vacant) = aux units_count
        sum / 1e6, no occupancy correction. This is the ResStock raw count
        the dashboard sees.
      * commercial  → billion sqft, as in AEO and ComStock aux.

    Reproduction: factor = res_<cohort>_dwelling_units(year) / res_<SPEC>_M_dwelling_units.
    (The occupancy correction cancels because it applies to both numerator
    and denominator equally in the residential case.)
    """
    from . import common
    # load_aux_cohort_size returns residential on AEO-occupied basis (× 0.878);
    # divide by 0.878 to express on ResStock raw / total-dwelling-units basis
    # so the denominator basis matches the cohort-amount columns above.
    rd = lambda spec: common.load_aux_cohort_size(run_dir_res, 'res', spec) / common.RES_OCCUPANCY_FRACTION
    cd = lambda spec: common.load_aux_cohort_size(run_dir_com, 'com', spec)
    return pd.DataFrame([{
        # Order: ELIGIBLE, INELIGIBLE, ALL_BASELINE so the "total" denominator
        # comes after its two parts (it equals their sum).
        'com_ELIGIBLE_B_sqft':                 cd(common.ELIGIBLE_TAG),
        'com_INELIGIBLE_B_sqft':               cd(common.INELIGIBLE_TAG),
        'com_ALL_BASELINE_B_sqft':             cd(common.ALL_BASELINE_TAG),
        'res_ELIGIBLE_M_dwelling_units':       rd(common.ELIGIBLE_TAG),
        'res_INELIGIBLE_M_dwelling_units':     rd(common.INELIGIBLE_TAG),
        'res_ALL_BASELINE_M_dwelling_units':   rd(common.ALL_BASELINE_TAG),
    }])


_OUTPUT_CSV = os.path.join(_REPO_DIR, 'growth_factors.csv')


# Column descriptions used to write a human-readable second row in the CSV.
# Written in plain English (no unicode, no internal code-constant names) so a
# reader without access to the codebase can understand every column. Returns
# empty string for any unrecognized column so row alignment never breaks.
_COLUMN_DESCRIPTIONS: dict[str, str] = {
    'Year': 'Calendar year (the integer year for this row).',

    # ---- AEO direct inputs (raw values from the merged AEO data; root of every derivation) ----
    'com_aeo_total_floorspace':
        "Total commercial floorspace at year, B sqft (includes the unsimulated gap). Read directly from the merged AEO data, row 'Commercial: Total Floorspace: Total: Reference case', column for this row's Year.",
    'com_aeo_new_construction_floorspace':
        "New commercial floorspace added during year, B sqft (annual increment, includes gap-share). Read directly from the merged AEO data, row 'Commercial: Total Floorspace: New Additions: Reference case', column for this row's Year.",
    'com_aeo_cumulative_new_floorspace':
        "Cumulative new commercial floorspace built between 2027 and year (B sqft, AEO basis including gap-share). For Year < 2027: = 0 (no construction relative to a future anchor). For Year >= 2027: = sum of com_aeo_new_construction_floorspace from Year=2027 through this row's Year inclusive (cross-row running sum). Used downstream to derive com_surviving_floorspace and com_cumulative_new_floorspace.",
    'res_aeo_total_households':
        "Total residential occupied households at year, M HH. Read directly from the merged AEO data, row 'Residential: Key Indicators: Households: Total: Reference case', column for this row's Year.",
    'res_aeo_sf_households':
        "Single-family occupied households at year, M HH. Read directly from the merged AEO data, row 'Residential: Key Indicators: Households: Single-Family: Reference case', column for this row's Year.",
    'res_aeo_mf_households':
        "Multifamily occupied households at year, M HH. Read directly from the merged AEO data, row 'Residential: Key Indicators: Households: Multifamily: Reference case', column for this row's Year.",
    'res_aeo_mh_households':
        "Mobile-home occupied households at year, M HH. Read directly from the merged AEO data, row 'Residential: Key Indicators: Households: Mobile Homes: Reference case', column for this row's Year.",
    'res_aeo_surviving_households':
        "AEO occupied households from the 2027 anchor stock still standing at year, M HH. For Year < 2027: = res_aeo_total_households (no anchor yet, all existing households are 'surviving'). For Year >= 2027: = sum over type in {sf, mf, mh} of (res_aeo_<type>_households read from the Year=2027 row) * (1 - demolition_rate_<type>) ** (this row's Year - 2027), where demolition rates are sf=0.007053, mf=0.011759, mh=0.037005 per year (from Berrill et al. 2021). Cross-row dependency: requires looking up the Year=2027 row's per-type values.",

    # ---- Cohort-size denominator constants (same value every row; from BSQ aux files at run time) ----
    'com_ELIGIBLE_B_sqft':
        "Total commercial floorspace in the upgrade-eligible cohort, B sqft (excludes gap). Same value every row. Calculated once per run: sum of the 'sqft' column in aux_coverage_upgradeUpgraded-Baseline_reg_b2018.csv, divided by 1e9.",
    'com_INELIGIBLE_B_sqft':
        "Total commercial floorspace in the ineligible cohort, B sqft (excludes gap). Same value every row. Calculated once per run: sum of the 'sqft' column in aux_coverage_upgradeNon-Upgraded-Baseline_reg_b2018.csv, divided by 1e9.",
    'com_ALL_BASELINE_B_sqft':
        'Total commercial floorspace across the full ComStock simulation frame, B sqft (excludes gap). Same value every row. = com_ELIGIBLE_B_sqft + com_INELIGIBLE_B_sqft (the All-Baseline spec is the union of eligible and ineligible).',
    'res_ELIGIBLE_M_dwelling_units':
        "Total residential dwelling units in the upgrade-eligible cohort, M units (raw, includes vacant). Same value every row. Calculated once per run: sum of the 'units_count' column in aux_coverage_upgradeUpgraded-Baseline_reg_b2018.csv, divided by 1e6.",
    'res_INELIGIBLE_M_dwelling_units':
        "Total residential dwelling units in the ineligible cohort, M units (raw, includes vacant). Same value every row. Calculated once per run: sum of the 'units_count' column in aux_coverage_upgradeNon-Upgraded-Baseline_reg_b2018.csv, divided by 1e6.",
    'res_ALL_BASELINE_M_dwelling_units':
        'Total residential dwelling units across the full ResStock simulation frame, M units (raw, includes vacant). Same value every row. = res_ELIGIBLE_M_dwelling_units + res_INELIGIBLE_M_dwelling_units (the All-Baseline spec is the union of eligible and ineligible).',

    # ---- Adoption rates ----
    'adoption_rate_existing_com':
        'Cumulative fraction of existing commercial that has adopted by year. = 0 for Year < 2027; = 0.56 * (Year - 2026) / 20 for 2027 <= Year <= 2046 (linear ramp from 0 to 0.56 over 20 years); = 0.56 for Year > 2046 (flat after the ramp). The 0.56 maximum is the empirical eligibility share derived from the BSQ aux files.',
    'adoption_rate_existing_res':
        'Cumulative fraction of existing residential that has adopted by year. = 0 for Year < 2027; = 0.69 * (Year - 2026) / 20 for 2027 <= Year <= 2046 (linear ramp from 0 to 0.69 over 20 years); = 0.69 for Year > 2046 (flat after the ramp). The 0.69 maximum is the empirical eligibility share derived from the BSQ aux files.',
    'adoption_rate_new':
        'Fraction of new (post-2027) construction that adopts the upgrade. = 1.0 for Year >= 2027; = 0 for Year < 2027 (no new construction tracked from before the anchor).',

    # ---- Commercial cohort amounts (B sqft, ComStock non-gap basis unless noted) ----
    # Layout: surviving (input from AEO intermediates) -> its sub-cohorts ->
    # cumulative_new (input from AEO intermediate) -> its sub-cohorts ->
    # total (sum of the two parents) -> gap (informational).
    'com_surviving_floorspace':
        'Commercial floorspace from 2027 still standing at year, B sqft (excludes gap). = (com_aeo_total_floorspace - com_aeo_cumulative_new_floorspace) * 0.6 (subtract cumulative new since 2027 from the full AEO total, then take the 60 percent that ComStock simulates).',
    'com_adopted_existing_floorspace':
        'Existing commercial that has adopted the upgrade by year, B sqft (excludes gap). = adoption_rate_existing_com * com_surviving_floorspace.',
    'com_eligible_not_adopted_existing_floorspace':
        'Existing commercial eligible to adopt but has not yet, B sqft (excludes gap). = (0.56 - adoption_rate_existing_com) * com_surviving_floorspace; equals 0 for Year < 2027 (adoption has not started). The 0.56 is the max eligibility share for commercial.',
    'com_ineligible_existing_floorspace':
        'Existing commercial that never qualifies for the upgrade, B sqft (excludes gap). = (1 - 0.56) * com_surviving_floorspace = 0.44 * com_surviving_floorspace for Year >= 2027; = com_surviving_floorspace for Year < 2027 (the entire surviving stock sits here pre-anchor since adoption has not started).',
    'com_cumulative_new_floorspace':
        'Cumulative new commercial built since 2027 at year, B sqft (excludes gap). = com_aeo_cumulative_new_floorspace * 0.6 (the 60 percent of AEO cumulative new that ComStock simulates).',
    'com_adopted_new_floorspace':
        'New commercial built since 2027 that adopted the upgrade, B sqft (excludes gap). = adoption_rate_new * com_cumulative_new_floorspace.',
    'com_not_adopted_new_floorspace':
        'New commercial built since 2027 that has not adopted, B sqft (excludes gap). = (1 - adoption_rate_new) * com_cumulative_new_floorspace; always 0 because new construction always adopts post-2027.',
    'com_total_floorspace':
        'Total commercial floorspace simulated by ComStock at year, B sqft (excludes gap). = com_surviving_floorspace + com_cumulative_new_floorspace; equivalently com_aeo_total_floorspace * 0.6.',
    'com_gap_total_floorspace':
        'Unsimulated gap portion of AEO commercial total at year, B sqft. = com_aeo_total_floorspace * 0.4 (the 40 percent of AEO total that ComStock does not simulate; supplied to the projection only as an hourly load profile, separate from the cohort columns above and not summed into com_total_floorspace).',

    # ---- Residential cohort amounts (M dwelling units, ResStock raw basis incl vacant) ----
    'res_surviving_dwelling_units':
        'Residential dwelling units from 2027 still standing at year, M units (raw basis, includes vacant). = res_aeo_surviving_households / 0.878 (convert AEO occupied basis to ResStock raw basis by dividing by the 0.878 occupancy fraction).',
    'res_adopted_existing_dwelling_units':
        'Existing residential that has adopted the upgrade by year, M units (raw basis). = adoption_rate_existing_res * res_surviving_dwelling_units.',
    'res_eligible_not_adopted_existing_dwelling_units':
        'Existing residential eligible to adopt but has not yet, M units (raw basis). = (0.69 - adoption_rate_existing_res) * res_surviving_dwelling_units; equals 0 for Year < 2027 (adoption has not started). The 0.69 is the max eligibility share for residential.',
    'res_ineligible_existing_dwelling_units':
        'Existing residential that never qualifies for the upgrade, M units (raw basis). = (1 - 0.69) * res_surviving_dwelling_units = 0.31 * res_surviving_dwelling_units for Year >= 2027; = res_surviving_dwelling_units for Year < 2027 (the entire surviving stock sits here pre-anchor).',
    'res_cumulative_new_dwelling_units':
        'Cumulative new residential built since 2027 at year, M units (raw basis). = (res_aeo_total_households - res_aeo_surviving_households) / 0.878.',
    'res_adopted_new_dwelling_units':
        'New residential built since 2027 that adopted the upgrade, M units (raw basis). = adoption_rate_new * res_cumulative_new_dwelling_units.',
    'res_not_adopted_new_dwelling_units':
        'New residential built since 2027 that has not adopted, M units (raw basis). = (1 - adoption_rate_new) * res_cumulative_new_dwelling_units; always 0 because new construction always adopts post-2027.',
    'res_total_dwelling_units':
        'Total residential dwelling units at year, M units (raw basis, includes vacant). = res_surviving_dwelling_units + res_cumulative_new_dwelling_units; equivalently res_aeo_total_households / 0.878.',

    # ---- Baseline-scenario factors (multipliers) ----
    'factor_res_baseline_new_construction':
        'Multiplier applied to the upgrade-eligible residential energy source to produce the baseline new-construction cohort load. = res_cumulative_new_dwelling_units / res_ELIGIBLE_M_dwelling_units.',
    'factor_res_baseline_surviving':
        'Multiplier applied to the full-baseline residential energy source to produce the baseline surviving cohort load. = res_surviving_dwelling_units / res_ALL_BASELINE_M_dwelling_units.',
    'factor_com_baseline_new_construction':
        'Multiplier applied to the upgrade-eligible commercial energy source to produce the baseline new-construction cohort load (non-gap basis). = com_cumulative_new_floorspace / com_ELIGIBLE_B_sqft.',
    'factor_com_baseline_surviving':
        'Multiplier applied to the full-baseline commercial energy source to produce the baseline surviving cohort load (non-gap basis). = com_surviving_floorspace / com_ALL_BASELINE_B_sqft.',

    # ---- Upgrade-scenario factors ----
    'factor_res_upgrade_new_adoption':
        'Multiplier applied to the upgrade-eligible residential energy source to produce the new-adopted (NC) cohort load. = res_adopted_new_dwelling_units / res_ELIGIBLE_M_dwelling_units.',
    'factor_res_upgrade_surviving_adoption':
        'Multiplier applied to the upgrade-eligible residential energy source to produce the surviving-adopted (SA) cohort load. = res_adopted_existing_dwelling_units / res_ELIGIBLE_M_dwelling_units.',
    'factor_res_upgrade_surviving_not_adopted_eligible':
        'Multiplier applied to the upgrade-eligible residential energy source to produce the eligible-not-adopted portion of SNA. = res_eligible_not_adopted_existing_dwelling_units / res_ELIGIBLE_M_dwelling_units.',
    'factor_res_upgrade_surviving_not_adopted_ineligible':
        'Multiplier applied to the ineligible residential energy source to produce the ineligible portion of SNA. = res_ineligible_existing_dwelling_units / res_INELIGIBLE_M_dwelling_units.',
    'factor_com_upgrade_new_adoption':
        'Multiplier applied to the upgrade-eligible commercial energy source to produce the new-adopted (NC) cohort load (non-gap basis). = com_adopted_new_floorspace / com_ELIGIBLE_B_sqft.',
    'factor_com_upgrade_surviving_adoption':
        'Multiplier applied to the upgrade-eligible commercial energy source to produce the surviving-adopted (SA) cohort load (non-gap basis). = com_adopted_existing_floorspace / com_ELIGIBLE_B_sqft.',
    'factor_com_upgrade_surviving_not_adopted_eligible':
        'Multiplier applied to the upgrade-eligible commercial energy source to produce the eligible-not-adopted portion of SNA (non-gap basis). = com_eligible_not_adopted_existing_floorspace / com_ELIGIBLE_B_sqft.',
    'factor_com_upgrade_surviving_not_adopted_ineligible':
        'Multiplier applied to the ineligible commercial energy source to produce the ineligible portion of SNA (non-gap basis). = com_ineligible_existing_floorspace / com_INELIGIBLE_B_sqft.',
}


# Canonical column order for the main CSV. Each column on the right
# can be computed from columns to its left. Section flow:
#   identifier
#   -> AEO direct inputs (raw values copied from AEO data)
#   -> cohort-size denominators (constants; ALL_BASELINE = ELIGIBLE + INELIGIBLE)
#   -> adoption rates
#   -> commercial cohort amounts (atomic pieces, then summary sums, total last)
#   -> residential cohort amounts (same shape)
#   -> factors (cohort_amount / denominator, both visible to the left)
_MAIN_CSV_ORDER: tuple[str, ...] = (
    'Year',

    # ---- AEO direct inputs + AEO intermediates (running sum / decay) ----
    'com_aeo_total_floorspace',
    'com_aeo_new_construction_floorspace',
    'com_aeo_cumulative_new_floorspace',
    'res_aeo_total_households',
    'res_aeo_sf_households',
    'res_aeo_mf_households',
    'res_aeo_mh_households',
    'res_aeo_surviving_households',

    # ---- Cohort-size denominator constants (ALL_BASELINE = ELIGIBLE + INELIGIBLE) ----
    'com_ELIGIBLE_B_sqft',
    'com_INELIGIBLE_B_sqft',
    'com_ALL_BASELINE_B_sqft',
    'res_ELIGIBLE_M_dwelling_units',
    'res_INELIGIBLE_M_dwelling_units',
    'res_ALL_BASELINE_M_dwelling_units',

    # ---- Adoption rates ----
    'adoption_rate_existing_com',
    'adoption_rate_existing_res',
    'adoption_rate_new',

    # ---- Commercial cohort amounts ----
    # surviving (parent, derived from AEO intermediates) -> its sub-cohorts ->
    # cumulative_new (parent) -> its sub-cohorts -> total -> gap (informational).
    'com_surviving_floorspace',
    'com_adopted_existing_floorspace',
    'com_eligible_not_adopted_existing_floorspace',
    'com_ineligible_existing_floorspace',
    'com_cumulative_new_floorspace',
    'com_adopted_new_floorspace',
    'com_not_adopted_new_floorspace',
    'com_total_floorspace',
    'com_gap_total_floorspace',

    # ---- Residential cohort amounts (same shape as commercial) ----
    'res_surviving_dwelling_units',
    'res_adopted_existing_dwelling_units',
    'res_eligible_not_adopted_existing_dwelling_units',
    'res_ineligible_existing_dwelling_units',
    'res_cumulative_new_dwelling_units',
    'res_adopted_new_dwelling_units',
    'res_not_adopted_new_dwelling_units',
    'res_total_dwelling_units',

    # ---- Baseline factors ----
    'factor_res_baseline_new_construction',
    'factor_res_baseline_surviving',
    'factor_com_baseline_new_construction',
    'factor_com_baseline_surviving',

    # ---- Upgrade factors ----
    'factor_res_upgrade_new_adoption',
    'factor_res_upgrade_surviving_adoption',
    'factor_res_upgrade_surviving_not_adopted_eligible',
    'factor_res_upgrade_surviving_not_adopted_ineligible',
    'factor_com_upgrade_new_adoption',
    'factor_com_upgrade_surviving_adoption',
    'factor_com_upgrade_surviving_not_adopted_eligible',
    'factor_com_upgrade_surviving_not_adopted_ineligible',
)


def _write_csv_with_descriptions(df: pd.DataFrame, path: str,
                                 float_format: str = '%.6f') -> None:
    """Write `df` to `path` with a description row inserted between the header
    and the data, while preserving `float_format` for the numeric data rows.
    Output schema: [column names, descriptions, data...]. We can't just concat
    because mixing the str description row with float data would coerce
    everything to object dtype and drop `float_format`."""
    desc_row = pd.DataFrame([{c: _COLUMN_DESCRIPTIONS.get(c, '') for c in df.columns}])
    desc_row.to_csv(path, index=False)                     # writes header + 1 desc row
    df.to_csv(path, index=False, header=False, mode='a',
              float_format=float_format)                   # appends data rows


def main():
    import argparse
    from .common import HISTORICAL_YEARS
    ap = argparse.ArgumentParser(prog='python -m projections.growth_factors',
                                 description=__doc__)
    ap.add_argument('--run-dir-res', default=None,
                    help='Residential run_dir — if given alongside --run-dir-com, '
                         'includes the actual factor multipliers applied by the '
                         'projection in the output.')
    ap.add_argument('--run-dir-com', default=None,
                    help='Commercial run_dir — pair with --run-dir-res.')
    ap.add_argument('--out', default=_OUTPUT_CSV,
                    help=f'Output CSV path (default: {_OUTPUT_CSV})')
    args = ap.parse_args()

    # Cover historical anchor years + the full annual range used in projection.
    years = sorted(set(HISTORICAL_YEARS) | set(range(2021, 2051)))
    df = _full_table(years)

    if args.run_dir_res and args.run_dir_com:
        factor_df = _factor_table(years, args.run_dir_res, args.run_dir_com)
        df = df.merge(factor_df, on='Year')
        # Inline the cohort-size denominators as constant columns so the
        # main CSV is self-contained — every factor's `cohort_amount /
        # denominator` formula references columns visible in the same file.
        denom = _denominator_summary(args.run_dir_res, args.run_dir_com)
        for col in denom.columns:
            df[col] = denom[col].iloc[0]
        # Also emit a standalone denominators CSV for quick reference.
        denom_path = args.out.replace('.csv', '_denominators.csv')
        _write_csv_with_descriptions(denom, denom_path)
        print(f'wrote {denom_path} (denominators only, also inlined into main CSV)')

    # Reorder to the canonical section layout. Drop unknown columns to nothing
    # silently (none expected; this is a defensive against typos in upstream).
    ordered_cols = [c for c in _MAIN_CSV_ORDER if c in df.columns]
    extra_cols   = [c for c in df.columns if c not in _MAIN_CSV_ORDER]
    df = df[ordered_cols + extra_cols]

    _write_csv_with_descriptions(df, args.out)
    print(f'wrote {args.out} ({len(df)} data rows + 1 description row, '
          f'{len(df.columns)} columns)')


if __name__ == '__main__':
    main()
