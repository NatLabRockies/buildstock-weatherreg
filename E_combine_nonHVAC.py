#!/usr/bin/env python
"""
Combine 2018 non-HVAC load timeseries with HVAC load timeseries from various weather years.

This script:
1. Loads 2018 non-HVAC profiles (ResStock and ComStock) by state
2. Loads HVAC profiles from different weather years (e.g., 2007-2013, 2016-2023) 
3. Aggregates county-level HVAC data to state-level
4. Matches day-of-week patterns between HVAC weather years and 2018
5. Combines the two load profiles by state and upgrade level

By default the output is a single CSV per upgrade (e.g. ``upgrade_0.csv``) with
an hourly index matching the HVAC input data and one column per state.  The
file sums loads across both residential and commercial building types and
across any number of HVAC source directories (specified by ``RES_HVAC_DIRCTORY``
and ``COM_HVAC_DIRECTORY``). 
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - use relative paths from script location
SCRIPT_DIR = Path(__file__).parent
NON_HVAC_BASE_PATH = SCRIPT_DIR / "inputs" / "non-hvac"
HVAC_OUTPUTS_BASE = Path("/projects/geohc/geo_predict/outputs")
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# USER CONFIGURATION: Specify HVAC directory to process
RES_HVAC_DIRCTORY = HVAC_OUTPUTS_BASE / "outputs_2025-12-15-14-51-32"
COM_HVAC_DIRECTORY = HVAC_OUTPUTS_BASE / "outputs_2025-12-10-15-53-30"

def extract_state_from_county_id(county_str):
    """
    Extract the two-letter state abbreviation from a county identifier column
    in the HVAC output files.  The string can take one of two forms:

    * ComStock:  ("G0101010", "AL, Montgomery County", "AL", "G0100010")
    * ResStock:  ("G0100010", "Autauga County", "AL")

    1. Search for any two‑letter uppercase code between single quotes.  This
       picks up the state entry in both ResStock and ComStock tuples and
       ignores FIPS codes containing digits.
    2. If that fails, fall back to the old splitting logic and pick the first
       two‑letter element encountered.
    3. If nothing sane is found, return ``None`` and the caller will skip that
       column.
    """
    import re

    # look for two-letter state codes inside single quotes
    m = re.findall(r"'([A-Z]{2})'", county_str)
    if m:
        # return the first match; normally there will be two identical
        # occurrences in a ComStock tuple and one in a ResStock tuple
        return m[0]

    # fallback: try naive parsing but be careful about extra
    # commas inside the county name
    try:
        parts = county_str.replace("'", "").replace("(", "").replace(")", "").split(", ")
        for part in parts:
            if part.isalpha() and len(part) == 2:
                return part
    except Exception as e:
        logger.warning(f"Could not parse county string: {county_str}, error: {e}")

    return None


def load_non_hvac_profiles(building_type='res'):
    """
    Load 2018 non-HVAC profiles from ResStock or ComStock.
    
    Args:
        building_type: 'res' for ResStock or 'com' for ComStock
    
    Returns:
        Dictionary of {(bldg, state, upgrade): Series} with hourly non-HVAC loads
    """
    logger.info(f"Loading {building_type.upper()} non-HVAC profiles...")
    
    # Determine the directory
    if building_type.lower() == 'res':
        search_dir = NON_HVAC_BASE_PATH / "ResStock 2025 Release 1 (AMY 2018)"
        file_pattern = "resstock_amy2018_r1_2025_upgrade_*.csv"
    else:
        search_dir = NON_HVAC_BASE_PATH / "ComStock 2025 Release 2 (AMY2018)"
        file_pattern = "comstock_amy2018_r2_2025_ts_aggregates_upgrade_*.csv"
    
    if not search_dir.exists():
        logger.error(f"Directory not found: {search_dir}")
        return {}
    
    profiles_by_state = {}
    
    # Find all upgrade files
    files = sorted(search_dir.glob(file_pattern))
    logger.info(f"Found {len(files)} non-HVAC files")
    
    for filepath in files:
        logger.info(f"  Loading {filepath.name}...")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Extract upgrade level from filename
        parts = filepath.stem.split('_')
        upgrade = parts[-1]
        
        # Get the electricity consumption column (last one)
        elec_col = df.columns[-1]
        
        # Group by state to get state-level timeseries
        for state in df['state'].unique():
            state_data = df[df['state'] == state][elec_col]
            key = (building_type, state, upgrade)
            profiles_by_state[key] = state_data
    
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
        
        # Process each county column and aggregate to state
        for county_col in df.columns:
            state = extract_state_from_county_id(county_col)
            if not state:
                logger.debug(f"    skipping column {county_col!r}: could not determine state")
                continue
            key = (upgrade, state)
            if key not in profiles_by_state:
                profiles_by_state[key] = pd.Series(0.0, index=df.index, dtype=float)
            profiles_by_state[key] += df[county_col]
    
    logger.debug(f"  -> Loaded {len(profiles_by_state)} state-upgrade combinations")
    return profiles_by_state


def get_day_of_week(df_index):
    """Get day of week (0=Monday, 6=Sunday) for each timestamp."""
    return df_index.dayofweek


def match_day_patterns(non_hvac, hvac_index, hvac_year):
    """
    Shift non-HVAC 2018 data to match HVAC day-of-week patterns.
    
    This function creates a one-to-many mapping by shifting 2018 non-HVAC data
    to align with the day-of-week patterns in the HVAC weather year while 
    preserving seasonal trends from 2018.
    
    Algorithm:
    1. Calculate the day of week differences between HVAC year and 2018
    2. Roll that 2018 hourly profile to match HVAC day of week
    3. Return non-HVAC data indexed to HVAC timestamps
    
    Args:
        non_hvac: Non-HVAC timeseries from 2018 (source data)
        hvac_index: DatetimeIndex from HVAC data (target index/shape)
        hvac_year: Integer year of HVAC data (for context)
    
    Returns:
        Series with 2018 non-HVAC data shifted to match HVAC day-of-week patterns,
        indexed to hvac_index timestamps
    """
    # Ensure hvac index is DateTimeIndex
    hvac_index = pd.to_datetime(hvac_index)
        
    # Ensure hourly resolution
    non_hvac_hourly = non_hvac.resample('H').interpolate(method='linear')

    hvac_jan1_dow = hvac_index.dayofweek[0]  # Day of week for HVAC data
    non_hvac_jan1_dow = non_hvac_hourly.index.dayofweek[0]  # Day of week for 2018 non-HVAC data

    # Calculate the difference in days of week (shift amount in hours)
    day_diff = non_hvac_jan1_dow - hvac_jan1_dow
    if hvac_index.is_leap_year[0] and not non_hvac_hourly.index.is_leap_year[0]:
        # If HVAC year is leap year but non-HVAC is not, roll the non-HVAC data 
        # back by one more day to account for the extra day in HVAC year
        day_diff += -1
    shift_hours = day_diff * 24  # Convert to hours
    
    # Shift non-HVAC data by the calculated number of hours
    shifted_non_hvac = np.roll(non_hvac_hourly.values, shift_hours)
    
    # Create a new Series with the shifted data and HVAC index
    shifted_non_hvac = pd.Series(shifted_non_hvac, index=hvac_index, dtype=float)
 
    return shifted_non_hvac


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
    # If HVAC is from a different year, shift non-HVAC to match HVAC day-of-week
    if hvac_year is not None and hvac_year != 2018:
        logger.debug(f"Shifting 2018 non-HVAC data to HVAC year {hvac_year} using day-of-week matching...")
        non_hvac_shifted = match_day_patterns(non_hvac_profile, hvac_profile.index, hvac_year)
        combined = non_hvac_shifted + hvac_profile
    else:
        # Same year - direct alignment
        if len(hvac_profile) == len(non_hvac_profile) and hvac_profile.index.equals(non_hvac_profile.index):
            # Same length and aligned index
            combined = non_hvac_profile + hvac_profile
        else:
            # Reindex/interpolate to match
            non_hvac_reindexed = non_hvac_profile.reindex(hvac_profile.index, method='ffill')
            combined = non_hvac_reindexed.fillna(0) + hvac_profile
    
    return combined


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
    Save combined profiles to CSV files organized by upgrade with one column per state.

    This function aggregates across both ResStock and ComStock results as well as
    across any HVAC source directories that were processed. The resulting files
    have an hourly time index matching the HVAC data and columns for each state.

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

    # Group profiles by upgrade, then by state.  Sum together any entries that
    # belong to the same state/upgrade regardless of building type or source.
    upgrades = {}
    for (source, bldg, state, upgrade), profile in combined_data.items():
        # ensure we have a dict for this upgrade
        state_dict = upgrades.setdefault(upgrade, {})
        if state not in state_dict:
            # start with a zero series using the profile's index
            state_dict[state] = pd.Series(0.0, index=profile.index, dtype=float)
        # align indices in case they differ and add
        existing = state_dict[state]
        # reindex both series to the union of their indices to avoid dropping
        union_idx = existing.index.union(profile.index)
        existing = existing.reindex(union_idx, fill_value=0.0)
        to_add = profile.reindex(union_idx, fill_value=0.0)
        state_dict[state] = existing + to_add

    # now write one file per upgrade
    for upgrade, state_dict in upgrades.items():
        # create a DataFrame sorted by state name
        df = pd.concat([state_dict[state] for state in sorted(state_dict.keys())],
                       axis=1)
        df.columns = sorted(state_dict.keys())

        # fill any remaining NaNs and convert units from kWh to MWh
        df = df.fillna(0) / 1000.0

        output_file = output_base_dir / f"upgrade_{upgrade}.csv"
        logger.debug(f"  Saving aggregated Upgrade {upgrade}: {output_file}")
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
    for HVAC_DIRECTORY in [RES_HVAC_DIRCTORY, COM_HVAC_DIRECTORY]:
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
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Total combined profiles created: {len(all_combined)}")
    logger.info(f"{'=' * 70}")
    
    # Save results
    logger.info("\nSaving combined profiles...")
    logger.info("-" * 70)
    
    # write aggregated files to the main output directory
    save_combined_profiles(all_combined, OUTPUT_DIR)
    
    logger.info("\n" + "=" * 70)
    logger.info("Script complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
