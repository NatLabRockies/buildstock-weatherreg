#!/usr/bin/env python
"""
Combine 2018 non-HVAC load timeseries with HVAC load timeseries from various weather years.

This script:
1. Loads 2018 non-HVAC profiles (ResStock and ComStock) by state
2. Loads HVAC profiles from different weather years (e.g., 2007-2013, 2016-2023) 
3. Aggregates county-level HVAC data to state-level
4. Matches day-of-week patterns between HVAC weather years and 2018
5. Combines the two load profiles by state and upgrade level

By default the output is a single CSV per upgrade or scenario (e.g. ``Baseline.csv``
or ``upgrade_3.csv``) with an hourly index matching the HVAC input data and one
column per state.  The file sums loads across both residential and commercial
building types and across any number of HVAC source directories (specified by
``RES_HVAC_DIRCTORY`` and ``COM_HVAC_DIRECTORY``). 
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import re
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - use relative paths from script location
SCRIPT_DIR = Path(__file__).parent
NON_HVAC_BASE_PATH = Path("/projects/geohc/data")
HVAC_OUTPUTS_BASE = Path("/projects/geohc/data/regression/regression_run1")
OUTPUT_DIR = SCRIPT_DIR / "outputs"
COMSTOCK_GAP_FILE = NON_HVAC_BASE_PATH / "ComStock Gap Model" / "gap_by_state.csv"

# Static state FIPS -> USPS state abbreviation map. The county GISJOIN format
# encodes the state FIPS in characters [1:3] (e.g. "G0100010" -> "01" -> "AL").
GIS_MAP_FILE = SCRIPT_DIR / "gis_map.json"
_STATE_FIPS_TO_ABBREV = json.loads(GIS_MAP_FILE.read_text())

# USER CONFIGURATION: Specify HVAC directories to process (one per upgrade).
# Lists of directories are supported so multiple upgrades per building type can be
# combined into a single run.
RES_HVAC_DIRECTORIES = [
    HVAC_OUTPUTS_BASE / "resstock_reference_u0_base2018_to2007_2023",
    HVAC_OUTPUTS_BASE / "resstock_ashp_mass_adoption_u4_base2018_to2007_2023",
    HVAC_OUTPUTS_BASE / "resstock_ghp_mass_adoption_u8_base2018_to2007_2023",
    HVAC_OUTPUTS_BASE / "resstock_ghp_mass_adoption_envelope_u32_base2018_to2007_2023",
]
COM_HVAC_DIRECTORIES = [
    HVAC_OUTPUTS_BASE / "comstock_reference_u0_base2018_to2007_2023",
    HVAC_OUTPUTS_BASE / "comstock_ashp_mass_adoption_u1_u14_base2018_to2007_2023",
    HVAC_OUTPUTS_BASE / "comstock_ghp_mass_adoption_u55_base2018_to2007_2023",
    HVAC_OUTPUTS_BASE / "comstock_ghp_mass_adoption_envelope_u59_base2018_to2007_2023",
]

# SCENARIO MAPPING
# Map a combination of ResStock and ComStock upgrade levels to a scenario name.
# Each scenario can be defined by the pair of upgrades used for res/com data.  If
# only one building type applies the other may be set to None.
SCENARIO_MAPPING = {
    "Baseline": {"res": "0", "com": "0"},
    "ASHP":     {"res": "4", "com": ["1", "14"]},
    "GHP":      {"res": "8", "com": "55"},
    "GHP + Envelope": {"res": "32", "com": "59"},
}


def load_comstock_gap_profiles(downscale_factor=0.5035):
    """
    Load additional ComStock gap profiles by state.

    This data should be added to every ComStock upgrade level.
    
    Returns a dictionary of {state: Series} with hourly gap profiles in MWh (converted from kWh).
    """
    if not COMSTOCK_GAP_FILE.exists():
        logger.warning(f"ComStock gap file not found: {COMSTOCK_GAP_FILE}")
        return {}

    df = pd.read_csv(COMSTOCK_GAP_FILE, parse_dates=["timestamp"], index_col="timestamp")
    elec_col = next((col for col in df.columns if "electricity" in col.lower()), df.columns[0])

    profiles = {}
    for state in df['State'].unique():
        profiles[state] = df[df['State'] == state][elec_col] * downscale_factor
    logger.info(f"Loaded ComStock gap profiles for {len(profiles)} states")
    return profiles


def scenario_for(building_type, upgrade):
    """Return the scenario name for a given building type and upgrade.

    List-valued mappings (mutually-exclusive upgrade bundles) match against
    the synthetic combined ID ``"_".join(upgrades)`` produced by
    ``inject_mutex_combined_upgrades``.
    """
    for scenario, mapping in SCENARIO_MAPPING.items():
        val = mapping.get(building_type)
        key = "_".join(val) if isinstance(val, list) else val
        if key == str(upgrade):
            return scenario
    return None


def inject_mutex_combined_upgrades(all_combined, baseline="0"):
    """For each list-valued upgrade in SCENARIO_MAPPING, inject a synthetic
    combined entry. Mutually-exclusive upgrades each apply to a disjoint
    slice of the stock, so the combined load is:

        combined = baseline + sum(u_i - baseline)

    ``combine_profiles`` is linear, so this identity holds for the
    post-combined (HVAC + non-HVAC) profiles stored in ``all_combined``.
    """
    for mapping in SCENARIO_MAPPING.values():
        for bldg, ups in mapping.items():
            if not isinstance(ups, list):
                continue
            combined_id = "_".join(ups)
            # Collect profiles by state, ignoring source (baseline and
            # constituent upgrades typically come from different directories).
            by_state = {}
            needed = set(ups) | {baseline}
            for (source, b, state, u), profile in all_combined.items():
                if b == bldg and str(u) in needed:
                    by_state.setdefault(state, {})[str(u)] = profile
            for state, ups_dict in by_state.items():
                if baseline not in ups_dict or not all(u in ups_dict for u in ups):
                    continue
                base = ups_dict[baseline]
                combined = base + sum(ups_dict[u] - base for u in ups)
                all_combined[(f"synthetic_{combined_id}", bldg, state, combined_id)] = combined

def load_non_hvac_profiles(building_type='res'):
    """
    Load 2018 non-HVAC profiles from ResStock or ComStock.
    
    Args:
        building_type: 'res' for ResStock or 'com' for ComStock
    
    Returns:
        Dictionary of {(bldg, state, upgrade): Series} with hourly non-HVAC loads in MWh (converted from kWh).
    """
    logger.info(f"Loading {building_type.upper()} non-HVAC profiles...")
    
    # Determine the directory
    if building_type.lower() == 'res':
        search_dir = NON_HVAC_BASE_PATH / "ResStock 2025 Release 1 (AMY 2018)"
        file_pattern = "resstock_amy2018_r1_2025_upgrade_*.csv"
    else:
        search_dir = NON_HVAC_BASE_PATH / "ComStock 2025 Release 2 (AMY2018)"
        file_pattern = "manual_query_comstock_upgrade*_*.csv"
    
    if not search_dir.exists():
        logger.error(f"Directory not found: {search_dir}")
        return {}
    
    profiles_by_state = {}
    gap_profiles = {}
    if building_type.lower() == 'com':
        gap_profiles = load_comstock_gap_profiles()
    
    # Find all upgrade files
    files = sorted(search_dir.glob(file_pattern))
    logger.info(f"Found {len(files)} non-HVAC files")
    
    for filepath in files:
        logger.info(f"  Loading {filepath.name}...")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Extract upgrade level from filename
        # Handles both "..._upgrade_0_2026mar11" and "..._upgrade0_mar17"
        m = re.search(r'upgrade_?(\d+)', filepath.stem)
        upgrade = m.group(1) if m else filepath.stem.split('_')[-1]
        
        # Get the electricity consumption column (last one)
        elec_col = df.columns[-1]
        
        # Group by state to get state-level timeseries
        for state in df['state'].unique():
            state_data = df[df['state'] == state][elec_col]
            if state in gap_profiles:
                # roll gap profiles to match state data timestamp if needed
                if not gap_profiles[state].index.equals(state_data.index): 
                    index_diff = int((gap_profiles[state].index[0] - state_data.index[0]).total_seconds() / 3600)
                    shifted_gap_profiles = np.roll(gap_profiles[state].values, index_diff)
                else:
                    shifted_gap_profiles = gap_profiles[state]
                state_data = state_data.add(shifted_gap_profiles, fill_value=0.0)
            elif building_type.lower() == 'com':
                logger.debug(f"  No gap profile for state {state}, using non-HVAC data as-is")
            key = (building_type, state, upgrade)
            profiles_by_state[key] = state_data / 1e3  # Convert kWh to MWh
            logger.debug(f"  Total non-HVAC demand for {key}: {round(sum(state_data)/1e9,1)} TWh")

    logger.info(f"  -> Loaded {len(profiles_by_state)} state-upgrade combinations")
    return profiles_by_state


def load_hvac_profiles(hvac_output_path, building_type='res'):
    """
    Load HVAC profiles from a specific output directory and aggregate to state level.
    
    Args:
        hvac_output_path: Path to the output directory containing HVAC files
        building_type: 'res' for ResStock or 'com' for ComStock
    
    Returns:
        Dictionary of {(upgrade, state): Series} with hourly HVAC loads in MWh
    """
    hvac_path = Path(hvac_output_path)
    if not hvac_path.exists():
        logger.warning(f"HVAC path does not exist: {hvac_path}")
        return {}
    
    logger.debug(f"  Loading HVAC profiles for {building_type}...")
    
    # Find all HVAC files for this building type
    hvac_files = sorted(hvac_path.glob(f"{building_type}_eulp_hvac_elec_MWh_*.csv"))
    
    if not hvac_files:
        logger.debug(f"  No HVAC files found in {hvac_path}")
        return {}
    
    profiles_by_state = {}
    
    for filepath in hvac_files:
        logger.debug(f"    Loading {filepath.name}...")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Extract upgrade level from filename
        parts = filepath.stem.split('upgrade')
        upgrade = parts[1].split('_')[0]
        
        # Process each county column and aggregate to state.
        # GISJOIN format: G + 2-digit state FIPS + 0 + 3-digit county FIPS + 0
        for county_col in df.columns:
            state = _STATE_FIPS_TO_ABBREV.get(county_col[1:3])
            if not state:
                logger.debug(f"    skipping column {county_col!r}: could not determine state")
                continue
            key = (upgrade, state)
            if key not in profiles_by_state:
                profiles_by_state[key] = pd.Series(0.0, index=df.index, dtype=float)
            profiles_by_state[key] += df[county_col]
    
    logger.debug(f"  -> Loaded {len(profiles_by_state)} state-upgrade combinations")
    return profiles_by_state


def match_day_patterns(non_hvac, hvac_index, hvac_year=None):
    """
    Shift 2018 non-HVAC data to match HVAC day-of-week patterns.

    Each year in ``hvac_index`` is handled independently: the 2018 hourly
    profile is rolled by the difference in Jan-1 day-of-week between 2018
    and that year, then assigned to the year's slice of the output. Leap
    years are padded by repeating the last 24 hours of the rolled profile.

    Args:
        non_hvac: 2018 non-HVAC timeseries (hourly or coarser).
        hvac_index: Target DatetimeIndex (may span multiple years).
        hvac_year: Unused; retained for backwards compatibility.

    Returns:
        Series of 2018 non-HVAC data day-of-week-aligned to ``hvac_index``.
    """
    hvac_index = pd.to_datetime(hvac_index)
    non_hvac_hourly = non_hvac.resample('h').interpolate(method='linear')
    non_hvac_values = non_hvac_hourly.values
    non_hvac_jan1_dow = non_hvac_hourly.index.dayofweek[0]

    out = pd.Series(0.0, index=hvac_index, dtype=float)
    for year in hvac_index.year.unique():
        year_idx = hvac_index[hvac_index.year == year]
        year_jan1_dow = year_idx.dayofweek[0]
        shift_hours = (non_hvac_jan1_dow - year_jan1_dow) * 24
        rolled = np.roll(non_hvac_values, shift_hours)

        n = len(year_idx)
        if n > len(rolled):
            # Leap year: pad by repeating the last 24 hours of the rolled profile.
            pad = np.tile(rolled[-24:], int(np.ceil((n - len(rolled)) / 24)))
            year_values = np.concatenate([rolled, pad])[:n]
        else:
            year_values = rolled[:n]
        out.loc[year_idx] = year_values

    return out


def combine_profiles(non_hvac_profile, hvac_profile, hvac_year=2018):
    """
    Combine non-HVAC and HVAC profiles.
    
    When HVAC data is from a different year than 2018, this function shifts the 
    non-HVAC 2018 data to match the HVAC day-of-week patterns, enabling one-to-many
    mapping (one year of 2018 non-HVAC combined with multiple weather years of HVAC).
    
    Args:
        non_hvac_profile: Series with 2018 non-HVAC loads (in MWh units)
        hvac_profile: Series with HVAC loads (in MWh) - may be from different year
        hvac_year: Year of HVAC data (used to detect if alignment is needed)
    
    Returns:
        Combined Series with total loads, indexed to HVAC year timestamps
    """
    # If indices already align, do a direct sum.
    if (len(hvac_profile) == len(non_hvac_profile)
            and hvac_profile.index.equals(non_hvac_profile.index)):
        return non_hvac_profile + hvac_profile

    # Otherwise shift the 2018 non-HVAC data to match HVAC day-of-week patterns
    # for every year present in the HVAC index (handles multi-year HVAC spans).
    logger.debug(f"Aligning 2018 non-HVAC data to HVAC index using day-of-week matching...")
    non_hvac_shifted = match_day_patterns(non_hvac_profile, hvac_profile.index, hvac_year)
    return non_hvac_shifted + hvac_profile


def process_hvac_directory(hvac_dir_path, non_hvac_data_res, non_hvac_data_com):
    """
    Process a single HVAC output directory and combine with non-HVAC data.
    
    Args:
        hvac_dir_path: Path to HVAC output directory
        non_hvac_data_res: Dictionary of ResStock non-HVAC profiles
        non_hvac_data_com: Dictionary of ComStock non-HVAC profiles
    
    Returns:
        Dictionary of combined profiles {(source, bldg, state, upgrade): Series}
    """
    hvac_dir = Path(hvac_dir_path)
    source_name = hvac_dir.name
    
    combined_data = {}
    
    # Get HVAC year from the data
    hvac_year = None
    for building_type in ['res', 'com']:
        hvac_files = list(hvac_dir.glob(f"{building_type}_eulp_hvac_elec_MWh_*.csv"))
        if hvac_files:
            try:
                df_sample = pd.read_csv(hvac_files[0], nrows=1, index_col=0, parse_dates=True)
                if len(df_sample.index) > 0:
                    hvac_year = df_sample.index[0].year
                    break
            except Exception as e:
                logger.debug(f"Could not read year from {hvac_files[0].name}: {e}")
    
    logger.info(f"Processing {source_name} (weather year: {hvac_year})")
    
    # Process ResStock
    logger.debug(f"  Processing ResStock...")
    res_hvac = load_hvac_profiles(hvac_dir, 'res')
    for (hvac_upgrade, state), hvac_data in res_hvac.items():
        non_hvac_key = ('res', state, hvac_upgrade)
        if non_hvac_key in non_hvac_data_res:
            combined_key = (source_name, 'res', state, hvac_upgrade)
            combined_data[combined_key] = combine_profiles(
                non_hvac_data_res[non_hvac_key], hvac_data, hvac_year=hvac_year
            )
    
    # Process ComStock
    logger.debug(f"  Processing ComStock...")
    com_hvac = load_hvac_profiles(hvac_dir, 'com')
    for (hvac_upgrade, state), hvac_data in com_hvac.items():
        non_hvac_key = ('com', state, hvac_upgrade)
        if non_hvac_key in non_hvac_data_com:
            combined_key = (source_name, 'com', state, hvac_upgrade)
            logger.debug(f"Processing combined key: {combined_key}")
            combined_data[combined_key] = combine_profiles(
                non_hvac_data_com[non_hvac_key], hvac_data, hvac_year=hvac_year
            )
    
    logger.info(f"  -> Created {len(combined_data)} combined profiles")
    return combined_data


def save_combined_profiles(combined_data, output_dir):
    """
    Save combined profiles to CSV files organized by scenario with one column per state.

    This function only outputs scenarios that are explicitly defined in SCENARIO_MAPPING.
    Any profiles with upgrades that do not map to a scenario are excluded from output.

    Args:
        combined_data: Dictionary of combined profiles keyed by
            (source, bldg, state, upgrade)
        output_dir: Output directory for saving files
    """
    output_base_dir = Path(output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if not combined_data:
        logger.info(f"  No data to save")
        return

    # Filter combined_data to only include profiles that map to a defined scenario.
    # Exclude any upgrades that don't match SCENARIO_MAPPING.
    filtered_data = {}
    for key, profile in combined_data.items():
        source, bldg, state, upgrade = key
        scen_name = scenario_for(bldg, upgrade)
        if scen_name:
            filtered_data[key] = profile

    if not filtered_data:
        logger.info("  No profiles matching defined scenarios. No output files created.")
        return

    # Group profiles by scenario then state.
    # Keep track of which building types contribute to each scenario
    # to enforce the requirement that both ResStock and ComStock data be present before writing.
    scenarios = {}
    scenario_buildings = {}

    for (source, bldg, state, upgrade), profile in filtered_data.items():
        scen_name = scenario_for(bldg, upgrade)
        scenario_buildings.setdefault(scen_name, set()).add(bldg)

        state_dict = scenarios.setdefault(scen_name, {})
        if state not in state_dict:
            state_dict[state] = pd.Series(0.0, index=profile.index, dtype=float)
        existing = state_dict[state]
        union_idx = existing.index.union(profile.index)
        existing = existing.reindex(union_idx, fill_value=0.0)
        to_add = profile.reindex(union_idx, fill_value=0.0)
        state_dict[state] = existing + to_add

    # now write one file per scenario
    for scen, state_dict in scenarios.items():
        # Ensure both res and com building types are present for this scenario
        mapping = SCENARIO_MAPPING[scen]
        buildings = scenario_buildings.get(scen, set())
        
        if mapping.get('res') and mapping.get('com'):
            if not ({'res', 'com'} <= buildings):
                logger.info(
                    f"Skipping output for scenario '{scen}' - ``res`` and ``com`` "
                    f"data not both present (found: {buildings})"
                )
                continue

        df = pd.concat([state_dict[state] for state in sorted(state_dict.keys())], axis=1)
        df.columns = sorted(state_dict.keys())

        # fill any remaining NaNs
        df = df.fillna(0)

        # shift index back an hour to match ReEDS hour-beginning convention
        df.index = df.index - pd.Timedelta(hours=1)

        # name using scenario label rather than raw upgrade
        filename = f"{scen}.csv" if scen.isidentifier() else f"scenario_{scen}.csv"
        output_file = output_base_dir / filename
        logger.debug(f"  Saving aggregated scenario '{scen}': {output_file}")
        df.to_csv(output_file)


def aggregate_national_sector_totals(combined_data):
    """Aggregate national totals by scenario and building sector."""
    sector_totals = {}
    for (_, bldg, _, upgrade), profile in combined_data.items():
        scen_name = scenario_for(bldg, upgrade)
        if not scen_name:
            continue
        key = (scen_name, bldg)
        sector_totals.setdefault(key, []).append(profile)

    aggregated = {}
    for (scen_name, bldg), profiles in sector_totals.items():
        aggregated_series = pd.concat(profiles, axis=1).sum(axis=1)
        aggregated[(scen_name, bldg)] = aggregated_series.sort_index()
    return aggregated


def save_national_sector_totals(combined_data, output_dir):
    """Save national residential and commercial totals for each scenario."""
    output_base_dir = Path(output_dir) / "sector_totals"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if not combined_data:
        logger.info("  No combined data available to save sector totals")
        return

    sector_totals = aggregate_national_sector_totals(combined_data)
    scenario_names = sorted({scen for scen, _ in sector_totals.keys()})

    for scen_name in scenario_names:
        # Check if both residential and commercial data are present
        has_res = (scen_name, 'res') in sector_totals
        has_com = (scen_name, 'com') in sector_totals
        
        if not (has_res and has_com):
            logger.warning(
                f"Skipping sector totals output for scenario '{scen_name}': "
                f"both residential and commercial data must be present. "
                f"Found: residential={has_res}, commercial={has_com}"
            )
            continue
        
        res_series = sector_totals[(scen_name, 'res')]
        com_series = sector_totals[(scen_name, 'com')]

        series_index = res_series.index.union(com_series.index).sort_values()
        res_series = res_series.reindex(series_index, fill_value=0.0)
        com_series = com_series.reindex(series_index, fill_value=0.0)

        df = pd.DataFrame({
            'residential_MWh': res_series,
            'commercial_MWh': com_series,
            'total_MWh': res_series + com_series,
        }, index=series_index)

        # shift index back an hour to match ReEDS hour-beginning convention
        df.index = df.index - pd.Timedelta(hours=1)
        df.index.name = 'timestamp'

        filename = f"{scen_name}_national_sector_totals.csv"
        output_file = output_base_dir / filename
        logger.debug(f"  Saving national sector totals for scenario '{scen_name}': {output_file}")
        df.to_csv(output_file)


def get_hvac_directories(base_path, min_date=None):
    """
    Get list of HVAC output directories, optionally filtered by date.
    
    Args:
        base_path: Base path to HVAC outputs
        min_date: Optional datetime to filter directories after this date
    
    Returns:
        Sorted list of Path objects
    """
    base = Path(base_path)
    hvac_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith('outputs_')]
    hvac_dirs.sort()
    
    if min_date:
        hvac_dirs = [d for d in hvac_dirs if datetime.strptime(d.name.split('_', 1)[1], '%Y-%m-%d-%H-%M-%S') >= min_date]
    
    return hvac_dirs


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("Starting combine_profiles.py")
    logger.info("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load non-HVAC profiles
    logger.info("\nLoading non-HVAC profiles...")
    logger.info("-" * 70)
    res_non_hvac = load_non_hvac_profiles('res')
    com_non_hvac = load_non_hvac_profiles('com')
    
    logger.info(f"Total ResStock non-HVAC: {len(res_non_hvac)} state-upgrade combinations")
    logger.info(f"Total ComStock non-HVAC: {len(com_non_hvac)} state-upgrade combinations")
    
    # Get HVAC directory (or directories) to process
    logger.info("\nPreparing HVAC data...")
    logger.info("-" * 70)

    hvac_dirs_to_process = []
    # if specific paths are configured, add them
    for HVAC_DIRECTORY in list(RES_HVAC_DIRECTORIES) + list(COM_HVAC_DIRECTORIES):
        if HVAC_DIRECTORY:
            hvac_path = Path(HVAC_DIRECTORY)
            if not hvac_path.exists():
                logger.error(f"Specified HVAC directory does not exist: {hvac_path}")
                return
            hvac_dirs_to_process.append(hvac_path)
            logger.info(f"Processing specified HVAC directory: {hvac_path.name}")

    # if no explicit directories were provided, scan the base directory
    if not hvac_dirs_to_process:
        logger.info("Scanning HVAC output directories...")
        all_hvac_dirs = get_hvac_directories(HVAC_OUTPUTS_BASE)
        logger.info(f"Found {len(all_hvac_dirs)} total HVAC output directories")
        if not all_hvac_dirs:
            logger.error("No HVAC directories found. Exiting.")
            return
        hvac_dirs_to_process = all_hvac_dirs
    
    # Process HVAC directories
    logger.info("\nProcessing HVAC directories...")
    logger.info("-" * 70)
    all_combined = {}
    
    for i, hvac_dir in enumerate(hvac_dirs_to_process, 1):
        logger.info(f"\n[{i}/{len(hvac_dirs_to_process)}] {hvac_dir.name}")
        combined = process_hvac_directory(hvac_dir, res_non_hvac, com_non_hvac)
        all_combined.update(combined)
    
    # Synthesize combined entries for mutually-exclusive upgrade bundles
    # (e.g. ComStock ASHP = u1 + u14).
    inject_mutex_combined_upgrades(all_combined)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Total combined profiles created: {len(all_combined)}")
    logger.info(f"{'=' * 70}")

    # Save results
    logger.info("\nSaving combined profiles...")
    logger.info("-" * 70)
    
    # write aggregated files to the main output directory
    save_combined_profiles(all_combined, OUTPUT_DIR)
    save_national_sector_totals(all_combined, OUTPUT_DIR)
    
    logger.info("\n" + "=" * 70)
    logger.info("Script complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
