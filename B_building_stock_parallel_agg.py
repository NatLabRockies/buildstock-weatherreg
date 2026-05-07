import os
import certifi
_CA = certifi.where()
os.environ.setdefault("AWS_CA_BUNDLE", _CA)
os.environ.setdefault("CURL_CA_BUNDLE", _CA)
os.environ.setdefault("SSL_CERT_FILE", _CA)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _CA)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import fnmatch
import sys
import shutil
import datetime as dt
import subprocess
import time as pytime
import logging
import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path

COUNTY_PARQUET_CACHE_DIR = Path("/projects/geohc/radhikar/outputs/county_parquet_cache")

# Bypass SSL certificate verification for all HTTPS requests in this script.


LOG_LEVEL = os.environ.get("BUILDSTOCK_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | pid=%(process)d | %(message)s",
)
logger = logging.getLogger(__name__)

# S3 options for this environment: anonymous access with certificate checks disabled.
S3_STORAGE_OPTIONS = {
    "anon": True,
    "client_kwargs": {"verify": False},
}

# Helper function to load a single county's parquet in parallel
def _county_cache_path(state, county, upgrade, url_bldg, parquet_cols):
    """Build a deterministic cache path for a county parquet load.

    The hash covers `url_bldg` and the sorted `parquet_cols` so that any change
    to dataset source or requested columns invalidates the cache.
    """
    cols_key = "|".join(sorted(parquet_cols)) if parquet_cols else ""
    digest = hashlib.sha1(f"{url_bldg}||{cols_key}".encode("utf-8")).hexdigest()[:12]
    fname = f"{state}_{county}_upgrade{upgrade}_{digest}.parquet"
    return COUNTY_PARQUET_CACHE_DIR / fname


def _load_county_parquet(state, county, upgrade, url_bldg, parquet_cols, max_retries=5, initial_backoff=1.0):
    """Load a single county's parquet file with exponential backoff retry logic for S3 rate limits."""
    url = (
        f"{url_bldg}metadata_and_annual_results_aggregates/by_state_and_county/full/parquet/"
        f"state={state}/county={county}/{state}_{county}_upgrade{upgrade}_agg.parquet"
    )
    key = url.replace("s3://", "")
    t0 = pytime.perf_counter()

    cache_path = _county_cache_path(state, county, upgrade, url_bldg, parquet_cols)
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            elapsed = pytime.perf_counter() - t0
            logger.info(
                "County read cache hit | state=%s county=%s upgrade=%s rows=%s elapsed=%.2fs path=%s",
                state, county, upgrade, len(df), elapsed, cache_path,
            )
            return df
        except Exception as e:
            logger.warning(
                "County cache read failed, falling back to S3 | path=%s error_type=%s error=%s",
                cache_path, type(e).__name__, e,
            )

    logger.debug(
        "County read start | state=%s county=%s upgrade=%s key=%s",
        state,
        county,
        upgrade,
        key,
    )
    
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            logger.debug(
                "County read attempt | state=%s county=%s upgrade=%s attempt=%s/%s",
                state,
                county,
                upgrade,
                attempt + 1,
                max_retries,
            )
            df = pd.read_parquet(
                url,
                columns=parquet_cols,
                storage_options=S3_STORAGE_OPTIONS,
            )
            df["in.state"] = state
            df["in.nhgis_county_gisjoin"] = county
            elapsed = pytime.perf_counter() - t0
            logger.info(
                "County read success | state=%s county=%s upgrade=%s rows=%s elapsed=%.2fs",
                state,
                county,
                upgrade,
                len(df),
                elapsed,
            )
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
                df.to_parquet(tmp_path, index=False)
                os.replace(tmp_path, cache_path)
                logger.debug("County cache write | path=%s", cache_path)
            except Exception as e:
                logger.warning(
                    "County cache write failed | path=%s error_type=%s error=%s",
                    cache_path, type(e).__name__, e,
                )
            return df
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                elapsed = pytime.perf_counter() - t0
                logger.warning(
                    f"{url} | County parquet missing | state=%s county=%s upgrade=%s elapsed=%.2fs key=%s",
                    state,
                    county,
                    upgrade,
                    elapsed,
                    key,
                )
                return None
            if attempt < max_retries - 1:
                # Check if it's a rate limit or connection error worth retrying
                error_str = str(e).lower()
                retry_tokens = [
                    'throttling',
                    'slowdown',
                    'too many requests',
                    'timed out',
                    'timeout',
                    'temporarily unavailable',
                    'connection reset',
                    '503',
                    '429',
                ]
                if any(token in error_str for token in retry_tokens):
                    logger.warning(
                        "County read retry | state=%s county=%s upgrade=%s attempt=%s/%s backoff=%.1fs error_type=%s error=%s",
                        state,
                        county,
                        upgrade,
                        attempt + 1,
                        max_retries,
                        backoff,
                        type(e).__name__,
                        e,
                    )
                    pytime.sleep(backoff)
                    backoff = min(backoff * 2, 60)  # Cap backoff at 60 seconds
                    continue
                else:
                    # Other errors, don't retry
                    elapsed = pytime.perf_counter() - t0
                    logger.error(
                        "County read failed (non-retryable) | state=%s county=%s upgrade=%s elapsed=%.2fs error_type=%s error=%s key=%s",
                        state,
                        county,
                        upgrade,
                        elapsed,
                        type(e).__name__,
                        e,
                        key,
                    )
                    return None
            else:
                elapsed = pytime.perf_counter() - t0
                logger.error(
                    "County read failed (retries exhausted) | state=%s county=%s upgrade=%s retries=%s elapsed=%.2fs error_type=%s error=%s key=%s",
                    state,
                    county,
                    upgrade,
                    max_retries,
                    elapsed,
                    type(e).__name__,
                    e,
                    key,
                )
                return None

# Helper function for running parallelized tasks via multiprocessing
def run_task(cmd):
    print(f'Running command: {" ".join(cmd)} at {dt.datetime.now()}\n')
    result = subprocess.run(cmd, env=os.environ.copy(), check=True,
                            stdout=sys.stdout.buffer, stderr=sys.stderr.buffer)
    return result.returncode

if __name__ == "__main__":
    # Detect if running on HPC
    # hpc = bool(int(os.environ.get('REEDS_USE_SLURM', 0)))
    hpc = 'SLURM_JOB_ID' in os.environ
    logger.info("Running on HPC: %s", hpc)

    if not hpc:
        # Detect available CPU cores dynamically
        num_cores = max(2, os.cpu_count() - 2)  # Leave 2 cores free
        logger.info("Using %s parallel processes", num_cores)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Optional CLI arg: path to a switches JSON. Lets a user maintain separate
    # ResStock/ComStock switch files and submit them back-to-back via:
    #   sbatch A_start_building_stock_parallel_agg.sh switches_resstock.json
    # Defaults to script_dir/switches_agg.json for backwards compatibility.
    if len(sys.argv) > 1:
        switches_path = os.path.abspath(sys.argv[1])
    else:
        switches_path = os.path.join(script_dir, 'switches_agg.json')
    if not os.path.isfile(switches_path):
        raise FileNotFoundError(f"Switches file not found: {switches_path}")
    logger.info("Using switches file: %s", switches_path)

    # Pre-read switches to resolve output_dir before touching the filesystem.
    # The full canonical load happens later from the snapshot copy.
    with open(switches_path, 'r') as _f:
        _pre_switch = json.load(_f)
    output_dir = _pre_switch['output_dir']
    if os.path.exists(output_dir):
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Refusing to "
            f"clobber prior results. Edit `output_dir` in {switches_path} "
            f"or remove/rename the existing directory."
        )
    os.makedirs(output_dir, exist_ok=False)
    logger.info("Created output directory at %s", output_dir)

    ## Function to exclude directories and files that shouldn't be copied into outputs
    EXCLUDE_DIRS = {'outputs', '__pycache__', 'aggregates'}
    EXCLUDE_PATTERNS = ['slurm-*.out']
    def exclude_dir(_dirname, filenames):
        return [
            name for name in filenames
            if name.startswith('.')
            or name in EXCLUDE_DIRS
            or any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_PATTERNS)
        ]

    ## Copy script_dir into output_dir/inputs (output_dir already exists; copytree creates the inputs/ leaf).
    shutil.copytree(script_dir, f'{output_dir}/inputs', ignore=exclude_dir)
    logger.info("Copied input files into %s/inputs", output_dir)
    ## Canonicalize the chosen switches file at inputs/switches_agg.json so D
    ## (and any other consumer) finds it at one well-known path. Overwrites
    ## whatever the copytree default placed there.
    snapshot_switches_path = f'{output_dir}/inputs/switches_agg.json'
    shutil.copy2(switches_path, snapshot_switches_path)
    ## Write the commit hash to a text file in the output folder
    with open(f'{output_dir}/inputs/commit_hash.txt', 'w') as f:
        f.write(subprocess.check_output(['git', 'rev-parse', 'HEAD'])
                        .strip()
                        .decode('utf-8'))

    logger.info("Loading switches from snapshot at %s", snapshot_switches_path)
    # SWITCHES #TODO: Only import necessary for this script & reorder
    # Read from the snapshot (not the source) so B and D are guaranteed to see
    # identical bytes regardless of edits to the source between submissions.
    with open(snapshot_switches_path, 'r') as f:
        switch = json.load(f)
    sw_testmode = switch['testmode']
    ## Switch that designates comstock or resstock data
    sw_comstock = switch['comstock'] # if `False`, then resstock
    ## Each spec is {"upgrade_id": int, "apply_regression": bool,
    ## "base_year": int, "target_year": list}. The same upgrade_id may appear
    ## with different base_year/regression combinations to produce ref+reg
    ## across multiple training years in one run; outputs are disambiguated
    ## by an `<id>_<reg|ref>_b<base_year>` tag.
    run_specs = switch['run_specs']

    ## Validate that all specs produce unique upgrade_tags. Two specs that
    ## differ only in target_year would collide and overwrite each other.
    seen_tags = set()
    for s in run_specs:
        tag = (
            f"{s['upgrade_id']}_"
            f"{'reg' if s['apply_regression'] else 'ref'}_"
            f"b{s['base_year']}"
        )
        if tag in seen_tags:
            raise ValueError(
                f"Duplicate upgrade_tag {tag!r} across run_specs. "
                f"Each (upgrade_id, apply_regression, base_year) triple "
                f"must be unique within a single switches file."
            )
        seen_tags.add(tag)

    n_bldngs = switch['n_bldngs'] # 'all' for all buildings, 'assign' for assigned building id list from csv
    comstock_year, comstock_release = switch['version_comstock'][0], switch['version_comstock'][1]
    resstock_year, resstock_release = switch['version_resstock'][0], switch['version_resstock'][1]
    chunk_size = switch['chunk_size'] # number of combinations to pull at a time
    bsq_cols = switch['com_bsq_cols'] if sw_comstock else switch['res_bsq_cols'] # columns to group by
    applied_only = switch['applied_only'] # if `True`, only buildings with upgrade applied

    # set chunk_size to 500 for resStock and 50 for ComStock if passed -1
    if chunk_size == -1:
        chunk_size = 500 if not sw_comstock else 50
    # Define columns to load from parquet files to speed up reads
    parquet_cols = [
        'upgrade',
        'in.state',
        'in.nhgis_county_gisjoin' if sw_comstock else 'in.county',
        'applicability',
        'in.sqft..ft2',
        'weight',
    ]
    # Add all groupby columns
    parquet_cols.extend([f'in.{col}' for col in bsq_cols])
    
    # Add energy enduse columns
    if sw_comstock:
        elec_cols = [
            'out.electricity.heating.energy_consumption',
            'out.electricity.cooling.energy_consumption',
            'out.electricity.fans.energy_consumption',
            'out.electricity.heat_recovery.energy_consumption',
            'out.electricity.heat_rejection.energy_consumption',
            'out.electricity.pumps.energy_consumption'
        ]
    else:
        elec_cols = [
            'out.electricity.heating.energy_consumption',
            'out.electricity.heating_fans_pumps.energy_consumption',
            'out.electricity.heating_hp_bkup.energy_consumption',
            'out.electricity.heating_hp_bkup_fa.energy_consumption',
            'out.electricity.cooling.energy_consumption',
            'out.electricity.cooling_fans_pumps.energy_consumption'
        ]
    
    parquet_cols.extend([col + '..kwh' for col in elec_cols])
    parquet_cols.append('out.natural_gas.heating.energy_consumption..kwh')
    parquet_cols = list(set(parquet_cols))  # Remove any duplicates

    county_parquet_cols = parquet_cols
    if sw_comstock:
        county_parquet_cols = [col for col in parquet_cols if col != 'in.county_name']

    # URLs (the base/target year vary per spec, so url_bldg is built inside
    # the loop below).
    url_base = switch['url_base']

    # Load the state_county_map outside the loop
    state_county_map = pd.read_csv(
        f"{url_base}2025/comstock_amy2018_release_2/geographic_information/"
        "spatial_tract_lookup_table_publish_v8 1.csv",
        storage_options=S3_STORAGE_OPTIONS,
    )
    logger.info("Loaded state_county_map with shape: %s", state_county_map.shape)
    # Subset the DataFrame to include only the specified columns
    state_county_map = state_county_map[
        ["nhgis_county_gisjoin", "resstock_county_id", "state_abbreviation"]
    ]

    # Save the subsetted DataFrame as a CSV to the output directory
    state_county_map.to_csv(
        os.path.join(output_dir, "inputs", "spatial_tract_lookup_table.csv"),
        index=False
    )
    logger.info("Saved spatial_tract_lookup_table.csv to output directory.")

    # MAIN
    for spec_index, spec in enumerate(run_specs):
        upgrade = spec['upgrade_id']
        apply_regression = spec['apply_regression']
        base_year = spec['base_year']
        target_year = spec['target_year']  # noqa: F841 (D consumes via spec_index)
        regression_tag = 'reg' if apply_regression else 'ref'
        upgrade_tag = f'{upgrade}_{regression_tag}_b{base_year}'

        # Per-spec dataset URLs (base_year now varies per spec).
        url_comstock = f'{url_base}{comstock_year}/comstock_amy{base_year}_release_{comstock_release}/'
        url_resstock = f'{url_base}{resstock_year}/resstock_amy{base_year}_release_{resstock_release}/'
        url_bldg = url_comstock if sw_comstock else url_resstock

        if sw_comstock and comstock_year == "2025" and comstock_release == "2":
            logger.info("Using custom metadata load logic for ComStock 2025 Release 2")

            if sw_testmode:
                state_meta = ['VT']
            else:
                state_meta = state_county_map["state_abbreviation"].unique().tolist()

            # Collect all (state, county) pairs to load in parallel
            county_pairs = []
            for state in state_meta:
                state_county_map_iter = state_county_map[state_county_map["state_abbreviation"].isin([state])]
                county_meta = state_county_map_iter["nhgis_county_gisjoin"].unique().tolist()
                logger.info(
                    "Processing state %s with %s counties for upgrade %s",
                    state,
                    len(county_meta),
                    upgrade,
                )
                county_pairs.extend([(state, county, upgrade) for county in county_meta])

            # Load all counties in parallel.
            # ThreadPoolExecutor is used (not ProcessPoolExecutor) because S3 reads are
            # I/O-bound and threads share the event loop, avoiding aiobotocore asyncio
            # conflicts that occur when forking processes with an active async S3 client.
            all_meta = []
            num_workers = max(2, min(num_cores, 16)) if not hpc else 8  # Threads are lighter; use more
            logger.info(
                "Loading %s counties in parallel with %s threads for upgrade %s",
                len(county_pairs),
                num_workers,
                upgrade,
            )

            loaded_count = 0
            missing_or_failed_count = 0
            crashed_future_count = 0
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_load_county_parquet, state, county, upgrade, url_bldg, county_parquet_cols)
                    for state, county, upgrade in county_pairs
                }
                for future in as_completed(futures):
                    try:
                        df = future.result()
                        if df is not None:
                            loaded_count += 1
                            all_meta.append(df)
                        else:
                            missing_or_failed_count += 1
                    except Exception as e:
                        crashed_future_count += 1
                        logger.exception(
                            "County worker crashed for upgrade %s with error_type=%s error=%s",
                            upgrade,
                            type(e).__name__,
                            e,
                        )

            logger.info(
                "County read summary | upgrade=%s requested=%s loaded=%s missing_or_failed=%s worker_crashes=%s",
                upgrade,
                len(county_pairs),
                loaded_count,
                missing_or_failed_count,
                crashed_future_count,
            )

            if not all_meta:
                raise RuntimeError(f"No metadata loaded for upgrade {upgrade}.")

            df_meta = pd.concat(all_meta, ignore_index=True)
            # TODO: It doesn't make sense to filter by upgrade here - should be above `for spec in run_specs`
            df_meta = df_meta[df_meta["upgrade"] == upgrade]

            # Merge state_county_map w/ df_meta to bring in resstock_county_id
            df_meta = df_meta.merge(
                state_county_map[
                    ["nhgis_county_gisjoin", "resstock_county_id"]
                ].drop_duplicates(),
                how="left",
                left_on="in.nhgis_county_gisjoin",
                right_on="nhgis_county_gisjoin"
            )

            # Assign resstock_county_id to in.county_name
            df_meta["in.county_name"] = df_meta["resstock_county_id"]

        else:
            # Reformat metadata filepath based on upgrade number
            url_meta = f'metadata_and_annual_results/national/full/parquet/upgrade{upgrade}.parquet'

            # Read Parquet file into a DataFrame
            df_meta = pd.read_parquet(
                url_bldg + url_meta,
                columns=parquet_cols,
                storage_options=S3_STORAGE_OPTIONS,
            )

            if sw_testmode:
                df_meta = df_meta[df_meta['in.state'] == 'VT']

        # Remove Alaska and Hawaii
        df_meta = df_meta[~df_meta['in.state'].isin(['AK', 'HI'])]

        # Set `county` based on `sw_comstock` value
        county = 'in.nhgis_county_gisjoin' if sw_comstock else 'in.county'

        # TODO: Alter testing code blocks to incorporate ComStock 2025.2
        # TESTING - DELETE for production or comment out
        # Testing Subset 1
        ## For ResStock testing purposes, subset to 'Single-Family Detached'
        # df_meta = df_meta[df_meta['in.geometry_building_type_recs']
        #                           == 'Single-Family Detached']
        
        # For ComStock testing purposes, subset to 'LargeOffice'
        # df_meta = df_meta[df_meta['in.comstock_building_type'] == 'LargeOffice']

        # counties = df_meta[county].unique()[:1] # TODO: counties (or restrict) should be a switch
        
        ## For testing purposes, subset counties
        # Note: G5000030 is the NHGIS code for Bennington County, VT
        # G1901630 = Scott County, IA; G1901530 = Polk County, IA
        # df_meta = df_meta[df_meta[county].isin(['G5000030'])] # or next line
        # df_meta = df_meta[df_meta[county].isin(counties)]
        # Testing Subset 1 end

        # # Testing Subset 2
        # states = ['WY', 'VT', 'AK', 'ND', 'SD']

        # # Filter df_meta to only include the specified states
        # df_meta = df_meta[df_meta['in.state'].isin(states)]

        # # Group by state and select the first three counties from each state
        # df_meta_subset = df_meta.groupby('in.state').apply(lambda x: x[county].unique()[:3]).reset_index()

        # # Flatten the list of counties for each state
        # counties = [county for sublist in df_meta_subset[0].tolist() for county in sublist]

        # # Filter df_meta to only include the selected counties
        # df_meta = df_meta[df_meta[county].isin(counties)]
        # # Testing Subset 2 end

        # Restrict to buildings upgraded in the current upgrade iteration
        if applied_only:
            df_meta = df_meta[df_meta['applicability']]

        # Apply weight to sqft and natural gas energy consumption
        df_meta['in.sqft'] = df_meta['in.sqft..ft2'] * df_meta['weight']

        # Define elec_enduses based on whether it's ComStock or ResStock
        if sw_comstock:
            elec_enduses = [
                'out.electricity.heating.energy_consumption',
                'out.electricity.cooling.energy_consumption',
                'out.electricity.fans.energy_consumption',
                'out.electricity.heat_recovery.energy_consumption',
                'out.electricity.heat_rejection.energy_consumption',
                'out.electricity.pumps.energy_consumption'
            ]

        else:
            elec_enduses = [
                'out.electricity.heating.energy_consumption',
                'out.electricity.heating_fans_pumps.energy_consumption',
                'out.electricity.heating_hp_bkup.energy_consumption',
                'out.electricity.heating_hp_bkup_fa.energy_consumption',
                'out.electricity.cooling.energy_consumption',
                'out.electricity.cooling_fans_pumps.energy_consumption'
            ]

        elec_enduses = [item + '..kwh' for item in elec_enduses]

        gas_enduses = ['out.natural_gas.heating.energy_consumption']

        gas_enduses = [enduse + '..kwh' for enduse in gas_enduses]

        # Apply weight to energy consumption columns
        for enduse in elec_enduses + gas_enduses:
            df_meta[enduse] = df_meta[enduse] * df_meta['weight']

        # Create a new column for total electricity consumption
        df_meta['meta_HVAC.elec'] = df_meta[elec_enduses].sum(axis=1)

        # Create a new column for total natural gas consumption
        df_meta['meta_natural_gas.heating.energy_consumption'] = (
            df_meta[gas_enduses].sum(axis=1)
        )
        # Format groupby columns based on BuildStockQuery groups
        group_cols = [f'in.{col}' for col in bsq_cols]

        grouped = df_meta.groupby(group_cols, observed=True)
        df_meta = grouped.agg({
            'in.sqft': 'sum',
            'meta_HVAC.elec': 'sum',
            'meta_natural_gas.heating.energy_consumption': 'sum'
        }).reset_index()

        # Error check: Convert energy consumption columns from kWh to MWh & round
        df_meta['meta_HVAC.elec'] = (df_meta['meta_HVAC.elec'] / 1000).round(6)
        df_meta['meta_natural_gas.heating.energy_consumption'] = (
            df_meta['meta_natural_gas.heating.energy_consumption'] / 1000).round(6)

        df_meta['bldg_id'] = df_meta[group_cols].apply(tuple, axis=1).astype(str)
        df_meta.set_index('bldg_id', inplace=True)
        df_meta = df_meta.sort_values(by=[
            'in.state',
            'in.nhgis_county_gisjoin' if sw_comstock else 'in.county',
        ])

        # Save single upgrade DataFrame to CSV file (one per spec; the
        # `<id>_<reg|ref>` tag keeps regressed and direct runs separate).
        prefix = 'com_' if sw_comstock else 'res_'
        meta_path = os.path.join(
            output_dir, f'{prefix}meta_master_upgrade{upgrade_tag}.csv')
        df_meta.to_csv(meta_path)

        unique_counties = df_meta[county].unique()

        # Build the chunks list (used by both HPC and local paths). Each
        # entry is (start_index, end_index, counties_str).
        chunks = []
        for i in range(0, len(unique_counties), chunk_size):
            start_index = i
            end_index = i + chunk_size
            county_chunk = unique_counties[i:i + chunk_size]
            counties_str = '_'.join(county_chunk)
            chunks.append((start_index, end_index, counties_str))

        # Initialize cross-iteration accumulators on first spec.
        try:
            tasks  # Check if it exists
        except NameError:
            tasks = []
        try:
            local_specs  # Check if it exists
        except NameError:
            local_specs = []

        if hpc:
            if not chunks:
                logger.warning("No counties to chunk for upgrade_tag %s; skipping array submission.", upgrade_tag)
                continue

            # Write the per-spec manifest. C reads its row by SLURM_ARRAY_TASK_ID.
            manifest_path = f'{output_dir}/inputs/manifest_upgrade{upgrade_tag}.csv'
            with open(manifest_path, 'w') as mf:
                mf.write('chunk_idx,start_index,end_index,counties_str\n')
                for idx, (s_idx, e_idx, c_str) in enumerate(chunks):
                    mf.write(f'{idx},{s_idx},{e_idx},{c_str}\n')
            logger.info("Wrote manifest with %d chunks at %s", len(chunks), manifest_path)

            # Submit one array job per spec, capped at 100 concurrent tasks.
            n_chunks = len(chunks)
            result = subprocess.run([
                'sbatch',
                f'--job-name={prefix}chunk_{upgrade_tag}',
                f'--array=0-{n_chunks - 1}%100',
                './C_run_bldg_chunk_agg.sh',
                manifest_path, meta_path, str(upgrade), prefix,
                output_dir, script_dir, str(spec_index),
            ], check=True, capture_output=True, text=True)
            array_job_id = result.stdout.strip().split()[-1]
            logger.info(
                "Submitted array job %s with %d tasks (%%100 concurrency) for upgrade_tag %s",
                array_job_id, n_chunks, upgrade_tag,
            )

            # Aggregation depends on the entire array completing successfully.
            # afterok on an array is satisfied only when every task exits 0;
            # --kill-on-invalid-dep=yes cancels the agg job if any task fails.
            bldg_type = prefix.rstrip('_')  # 'res' or 'com'
            agg_result = subprocess.run([
                'sbatch',
                f'--job-name={prefix}agg_{upgrade_tag}',
                f'--dependency=afterok:{array_job_id}',
                '--kill-on-invalid-dep=yes',
                './F_aggregate_chunks.sh',
                output_dir, bldg_type, upgrade_tag,
            ], check=True, capture_output=True, text=True)
            agg_job_id = agg_result.stdout.strip().split()[-1]
            logger.info(
                "Queued aggregation job %s for upgrade_tag %s (depends on array %s)",
                agg_job_id, upgrade_tag, array_job_id,
            )

        else:
            # Local: queue each chunk for the multiprocessing pool below.
            for start_index, end_index, counties_str in chunks:
                cmd = [
                    sys.executable,
                    f'{output_dir}/inputs/D_process_chunk_agg.py',
                    str(start_index), str(end_index), meta_path,
                    str(upgrade), prefix, output_dir, script_dir, counties_str,
                    str(spec_index),
                ]
                tasks.append(cmd)
            if upgrade_tag not in {s['upgrade_tag'] for s in local_specs}:
                local_specs.append({'upgrade_tag': upgrade_tag, 'prefix': prefix})

    # Local: run all chunks in parallel, then aggregate per spec in series.
    if not hpc and tasks:
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(run_task, tasks)
        for s in local_specs:
            tag = s['upgrade_tag']
            bldg_type = s['prefix'].rstrip('_')
            logger.info("Aggregating chunks for upgrade_tag %s", tag)
            subprocess.run([
                sys.executable,
                f'{output_dir}/inputs/agg_buildings.py',
                '--bldg-path', output_dir,
                '--bldg-type', bldg_type,
                '--upgrade-tag', tag,
            ], check=True)

    print("All chunks processed successfully!")
