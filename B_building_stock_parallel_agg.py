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
import re
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


def _compress_array_indices(indices):
    """Format an iterable of ints as a SLURM --array index spec with ranges.

    Examples
    --------
    >>> _compress_array_indices([0, 1, 2, 5, 7, 8, 9])
    '0-2,5,7-9'
    >>> _compress_array_indices([3])
    '3'
    """
    indices = sorted(set(indices))
    if not indices:
        return ''
    parts = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f'{start}-{prev}')
        start = prev = i
    parts.append(str(start) if start == prev else f'{start}-{prev}')
    return ','.join(parts)

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
    is_resume = os.path.isdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if is_resume:
        logger.info("RESUMING existing run at %s", output_dir)
    else:
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

    ## Always refresh the inputs/ snapshot from the current source tree.
    ## Resume scenarios often involve hot-fixes to D/agg_buildings.py between
    ## submissions; chunks read these files from the snapshot at runtime, so
    ## a stale snapshot means new chunks would silently run old code. The
    ## in-flight chunks have their D loaded in memory and aren't affected by
    ## the rewrite. Per-chunk provenance lives in slurm-out/ (preserved).
    inputs_dir = f'{output_dir}/inputs'
    if os.path.isdir(inputs_dir):
        shutil.rmtree(inputs_dir)
        logger.info("Refreshing inputs/ snapshot at %s (resume)", inputs_dir)
    shutil.copytree(script_dir, inputs_dir, ignore=exclude_dir)
    logger.info("Copied input files into %s", inputs_dir)

    ## Co-locate per-job slurm-out files with the run. Chunk and agg sbatches
    ## write directly here via --output=<slurm_out_dir>/slurm-%x_%j.out. The
    ## launcher's own slurm-out stays at script_dir (where the user submitted).
    slurm_out_dir = f'{output_dir}/slurm-out'
    os.makedirs(slurm_out_dir, exist_ok=True)
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

    ## Names propagate into upgrade_tag, which is the leading segment of every
    ## chunk/manifest/meta/agg filename and of SLURM job names. The agg-side
    ## parser splits the tag on the FIRST underscore to recover the
    ## `<reg|ref>_b<year>` suffix (see agg_buildings._chunks_eulp_dir), so an
    ## underscore inside `name` would steal that split. Enforce ASCII letters,
    ## digits, dots, and hyphens — fail loudly otherwise. Keep this regex and
    ## the matching one in D_process_chunk_agg.py in sync.
    _VALID_NAME_RE = re.compile(r'^[A-Za-z0-9.\-]+$')

    def _validate_spec_name(name):
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"run_specs entry is missing required 'name' field (got {name!r})"
            )
        if not _VALID_NAME_RE.match(name):
            raise ValueError(
                f"run_spec name {name!r} contains disallowed characters. "
                "Allowed: letters, digits, '.', '-' (no '_', no spaces, no '/')."
            )
        return name

    ## Validate that all specs have unique `name` values. With name-based
    ## upgrade_tag, two specs sharing a name (even if they differ in
    ## upgrade_id, base_year, etc.) would collide on filenames. We also
    ## validate that each spec's derived upgrade_tag is unique (defense in
    ## depth — name uniqueness already implies this, but it documents intent).
    seen_names = set()
    seen_tags = set()
    for s in run_specs:
        name = _validate_spec_name(s['name'])
        if name in seen_names:
            raise ValueError(
                f"Duplicate spec name {name!r} in run_specs. "
                f"Each run_specs entry must have a unique 'name'."
            )
        seen_names.add(name)
        tag = (
            f"{name}_"
            f"{'reg' if s['apply_regression'] else 'ref'}_"
            f"b{s['base_year']}"
        )
        if tag in seen_tags:
            raise ValueError(
                f"Duplicate upgrade_tag {tag!r} across run_specs."
            )
        seen_tags.add(tag)

    n_bldngs = switch['n_bldngs'] # 'all' for all buildings, 'assign' for assigned building id list from csv
    comstock_year, comstock_release = switch['version_comstock'][0], switch['version_comstock'][1]
    resstock_year, resstock_release = switch['version_resstock'][0], switch['version_resstock'][1]
    # Top-level chunk_size acts as the default for any spec that doesn't set
    # its own. Each run_spec may override via `chunk_size` to pick a value
    # appropriate for that spec's compute profile (e.g., bigger for direct
    # ref pulls to reduce Athena query count, smaller for reg specs to keep
    # per-chunk wall time under the SLURM time limit).
    chunk_size = switch['chunk_size'] # default number of combinations to pull at a time
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
    prefix = 'com_' if sw_comstock else 'res_'
    bldg_type = prefix.rstrip('_')
    # Cross-spec accumulators. Hoisted out of the loop so they're defined even
    # when every spec early-skips (resume with all aggs already produced).
    tasks = []          # local-mode multiprocessing commands
    local_specs = []    # local-mode aggregation roster (per spec needing agg)
    agg_job_ids = []    # HPC-mode F job IDs, collected for the Z post-pipeline dependency
    for spec_index, spec in enumerate(run_specs):
        spec_name = _validate_spec_name(spec['name'])
        # upgrade_id may be a scalar or a list. List form means "iterate
        # over these upgrades and sum the resulting timeseries" (handled
        # in D); on the B side we load each upgrade's parquet metadata
        # and concat them so per-county energy totals sum across the set.
        _raw_upgrade = spec['upgrade_id']
        upgrade_ids = (
            list(_raw_upgrade)
            if isinstance(_raw_upgrade, (list, tuple))
            else [_raw_upgrade]
        )
        n_upgrades = len(upgrade_ids)
        # Single representative upgrade for paths that still take a scalar
        # (ComStock 2025.2 county-parquet loader). The multi-upgrade list
        # path is currently only supported by the else-branch parquet loader.
        upgrade = upgrade_ids[0]
        # argv-safe token (no spaces) carrying the full upgrade list to D.
        # D reads the authoritative value from spec_index, so this is purely
        # informational for rerun-command logs.
        upgrade_argv_token = ','.join(str(u) for u in upgrade_ids)
        apply_regression = spec['apply_regression']
        base_year = spec['base_year']
        target_year = spec['target_year']  # noqa: F841 (D consumes via spec_index)
        regression_tag = 'reg' if apply_regression else 'ref'
        upgrade_tag = f'{spec_name}_{regression_tag}_b{base_year}'

        # Multi-upgrade specs combined with applied_only=True would over- or
        # under-count static features (sqft) in unpredictable ways depending
        # on each building's applicability pattern across the upgrade set.
        # Disallow the combination until we add a proper per-bldg_id dedup.
        if n_upgrades > 1 and applied_only:
            raise NotImplementedError(
                f"Spec {spec_name!r}: applied_only=True is not supported "
                f"with a list-valued upgrade_id ({upgrade_ids}). Set "
                f"applied_only=false in switches or pass a scalar upgrade_id."
            )

        # ===== RESUME CHECK 1 (per-spec) =====
        # If BOTH agg GWh CSVs are already on disk, this spec is fully done.
        # Skip the entire S3 metadata pull + chunking + array submission.
        # Saves hours of compute when re-running after a partial-failure
        # recovery. (D writes one chunk file per enduse; agg_buildings.py
        # writes one agg file per enduse.)
        # KEEP IN SYNC with agg_buildings.ENDUSES and the per-chunk resume
        # tuple lower in this function. All three lists must enumerate the
        # same enduses or resume will mis-skip work.
        expected_aggs = [
            os.path.join(
                output_dir,
                f'agg_{bldg_type}_eulp_{enduse}_GWh_upgrade{upgrade_tag}.csv',
            )
            for enduse in ('cooling_elec', 'heating_elec', 'non_hvac_elec')
        ]
        if all(os.path.exists(p) for p in expected_aggs):
            logger.info("[skip] %s already complete (all agg files present)", upgrade_tag)
            continue

        # Resolve effective chunk_size: per-spec override > top-level default.
        # A spec value of -1 falls back to the stock-type default like the
        # top-level switch does.
        spec_chunk_size = spec.get('chunk_size', chunk_size)
        if spec_chunk_size == -1:
            spec_chunk_size = 500 if not sw_comstock else 50

        # Per-spec dataset URLs (base_year now varies per spec).
        url_comstock = f'{url_base}{comstock_year}/comstock_amy{base_year}_release_{comstock_release}/'
        url_resstock = f'{url_base}{resstock_year}/resstock_amy{base_year}_release_{resstock_release}/'
        url_bldg = url_comstock if sw_comstock else url_resstock

        if sw_comstock and comstock_year == "2025" and comstock_release == "2":
            logger.info(
                "Using custom metadata load logic for ComStock 2025 Release 2 "
                "(upgrades=%s)", upgrade_ids,
            )

            if sw_testmode:
                state_meta = ['VT']
            else:
                state_meta = state_county_map["state_abbreviation"].unique().tolist()

            # Build (state, county, upgrade) triples covering EVERY upgrade
            # in upgrade_ids. The county-parquet loader is per-upgrade and
            # caches per (state, county, upgrade), so multi-upgrade specs
            # just become a longer task list with no special handling in
            # the loader itself. After concat, the same parquet-path
            # post-processing kicks in: applied_only filter, weight, energy
            # aggregation, and the post-groupby sqft /= n_upgrades correction
            # (further down) that undoes the per-upgrade row duplication.
            county_pairs = []
            for state in state_meta:
                state_county_map_iter = state_county_map[state_county_map["state_abbreviation"].isin([state])]
                county_meta = state_county_map_iter["nhgis_county_gisjoin"].unique().tolist()
                logger.info(
                    "Processing state %s with %s counties for upgrades %s",
                    state,
                    len(county_meta),
                    upgrade_ids,
                )
                for upg in upgrade_ids:
                    county_pairs.extend([(state, county, upg) for county in county_meta])

            # Load all counties in parallel.
            # ThreadPoolExecutor is used (not ProcessPoolExecutor) because S3 reads are
            # I/O-bound and threads share the event loop, avoiding aiobotocore asyncio
            # conflicts that occur when forking processes with an active async S3 client.
            all_meta = []
            num_workers = max(2, min(num_cores, 16)) if not hpc else 8  # Threads are lighter; use more
            logger.info(
                "Loading %s county-upgrade tasks in parallel with %s threads (upgrades=%s)",
                len(county_pairs),
                num_workers,
                upgrade_ids,
            )

            loaded_count = 0
            missing_or_failed_count = 0
            crashed_future_count = 0

            # Map future -> (state, county, upg) so the exception logger can
            # attribute crashes to the specific task that failed. With
            # multi-upgrade list specs this matters even more — the outer
            # `upgrade` variable doesn't identify which upgrade's task
            # crashed.
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _load_county_parquet, state, county, upg,
                        url_bldg, county_parquet_cols,
                    ): (state, county, upg)
                    for state, county, upg in county_pairs
                }
                for future in as_completed(futures):
                    f_state, f_county, f_upg = futures[future]
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
                            "County worker crashed | state=%s county=%s upgrade=%s "
                            "error_type=%s error=%s",
                            f_state, f_county, f_upg,
                            type(e).__name__,
                            e,
                        )

            logger.info(
                "County read summary | upgrades=%s requested=%s loaded=%s "
                "missing_or_failed=%s worker_crashes=%s",
                upgrade_ids,
                len(county_pairs),
                loaded_count,
                missing_or_failed_count,
                crashed_future_count,
            )

            if not all_meta:
                raise RuntimeError(f"No metadata loaded for upgrades {upgrade_ids}.")

            df_meta = pd.concat(all_meta, ignore_index=True)
            # Defensive filter: keep only the rows for the upgrades we asked
            # for. Loader returns single-upgrade frames so this is normally
            # a no-op, but it guards against any cross-upgrade leakage.
            df_meta = df_meta[df_meta["upgrade"].isin(upgrade_ids)]

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
            # Multi-upgrade specs: load each upgrade's parquet and concat,
            # so the later county groupby sums energy across (buildings ×
            # upgrades) — which gives "total energy across the upgrade
            # set" per county, matching the way D sums ts_agg across the
            # same upgrades. We correct sqft over-counting after the
            # groupby; see comment near the agg call below.
            df_meta_list = []
            for upg in upgrade_ids:
                url_meta = f'metadata_and_annual_results/national/full/parquet/upgrade{upg}.parquet'
                df_one = pd.read_parquet(
                    url_bldg + url_meta,
                    columns=parquet_cols,
                    storage_options=S3_STORAGE_OPTIONS,
                )
                if sw_testmode:
                    df_one = df_one[df_one['in.state'] == 'VT']
                df_meta_list.append(df_one)
            df_meta = pd.concat(df_meta_list, ignore_index=True)

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

        # Multi-upgrade list specs concat the same bldg_id roster `n_upgrades`
        # times before this groupby, so sqft (a static feature) gets counted
        # `n_upgrades` times per building. Energy is supposed to sum across
        # the upgrade set, so we leave it alone. This correction is only valid
        # for applied_only=False (every bldg_id appears in every upgrade's
        # parquet); applied_only=True is rejected above for this reason.
        if n_upgrades > 1:
            df_meta['in.sqft'] = df_meta['in.sqft'] / n_upgrades

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
        # Idempotent: overwrites any prior copy (deterministic from buildstock data).
        meta_path = os.path.join(
            output_dir, f'{prefix}meta_master_upgrade{upgrade_tag}.csv')
        df_meta.to_csv(meta_path)

        # Weather-location column. This is the key change vs. the previous
        # source-county chunking: we group regression work by the column that
        # determines the weather file (one EPW per value), so each chunk-task
        # in D loads weather ONCE per location and trains ONE RF per location
        # — instead of one per (source_county × as_sim) pair as before.
        #
        # For ResStock and pre-2025.2 ComStock, weather_col equals `county`
        # (county owns its EPW), so chunking-by-weather_col reduces to
        # chunking-by-county — i.e., the same physical chunks the old code
        # produced. Only ComStock 2025.2 sees a real change (many source
        # counties collapse onto one as_sim).
        weather_col = (
            'in.as_simulated_nhgis_county_gisjoin'
            if (sw_comstock and comstock_year == "2025" and comstock_release == "2")
            else county
        )
        unique_weather_locs = df_meta[weather_col].unique()

        # Bldg_id count per weather-location (informational; not the bin-pack
        # metric). For ResStock this is always 1; for ComStock 2025.2 it
        # varies — an as_sim with 50 source counties contributes 50 bldg_ids
        # that all roll up into one trained model.
        bldg_per_loc = df_meta.groupby(weather_col).size().reindex(unique_weather_locs)

        # Bin-pack by NUMBER OF LOCATIONS per chunk. Each location is one
        # regression (one weather load + one RF training per energy type),
        # so loc count is the right balance metric — bldg_id count would
        # over-budget chunks where as_sims serve many source counties (the
        # share-out work is cheap; the training is what costs minutes).
        # Locations stay contiguous (filename indices reference the order of
        # `unique_weather_locs` so resume's existence check is stable).
        chunks = []
        chunk_loc_counts = []
        chunk_bldg_counts = []
        cur_start = 0
        cur_end = 0
        cur_locs = []
        cur_loc_count = 0
        cur_bldg_count = 0
        for loc in unique_weather_locs:
            if cur_loc_count + 1 > spec_chunk_size and cur_locs:
                chunks.append((cur_start, cur_end, '_'.join(cur_locs)))
                chunk_loc_counts.append(cur_loc_count)
                chunk_bldg_counts.append(cur_bldg_count)
                cur_start = cur_end
                cur_locs = []
                cur_loc_count = 0
                cur_bldg_count = 0
            cur_locs.append(loc)
            cur_end += 1
            cur_loc_count += 1
            cur_bldg_count += int(bldg_per_loc.loc[loc])
        if cur_locs:
            chunks.append((cur_start, cur_end, '_'.join(cur_locs)))
            chunk_loc_counts.append(cur_loc_count)
            chunk_bldg_counts.append(cur_bldg_count)

        if chunks:
            logger.info(
                "Bin-packed %d weather locations / %d bldg_ids into %d chunks "
                "(target=%d locs/chunk; actual locs: min=%d, max=%d, mean=%.0f; "
                "actual bldg_ids: min=%d, max=%d, mean=%.0f)",
                len(unique_weather_locs), int(bldg_per_loc.sum()), len(chunks),
                spec_chunk_size,
                min(chunk_loc_counts), max(chunk_loc_counts),
                sum(chunk_loc_counts) / len(chunk_loc_counts),
                min(chunk_bldg_counts), max(chunk_bldg_counts),
                sum(chunk_bldg_counts) / len(chunk_bldg_counts),
            )

        # ===== RESUME CHECK 2 (per-chunk) =====
        # Detect which chunks have already produced ALL per-enduse EULP MWh
        # CSVs; we only resubmit the missing ones. A chunk is "done" only
        # when every enduse file exists on disk.
        # KEEP IN SYNC with agg_buildings.ENDUSES and the per-spec resume
        # tuple higher in this function.
        chunks_eulp_dir = os.path.join(
            output_dir, f'chunks_{regression_tag}_b{base_year}'
        )
        existing_indices = set()
        for idx, (s_idx, e_idx, _) in enumerate(chunks):
            chunk_files = [
                os.path.join(
                    chunks_eulp_dir,
                    f'{prefix}eulp_{enduse}_MWh_upgrade{upgrade_tag}_'
                    f'{s_idx:04d}-{e_idx:04d}.csv',
                )
                for enduse in ('cooling_elec', 'heating_elec', 'non_hvac_elec')
            ]
            if all(os.path.exists(p) for p in chunk_files):
                existing_indices.add(idx)
        missing_indices = [i for i in range(len(chunks)) if i not in existing_indices]
        n_done, n_total, n_missing = len(existing_indices), len(chunks), len(missing_indices)
        if n_done > 0:
            logger.info(
                "[resume] %s: %d/%d chunks already done; %d missing",
                upgrade_tag, n_done, n_total, n_missing,
            )
        else:
            logger.info("[fresh] %s: %d chunks to compute", upgrade_tag, n_total)

        if hpc:
            if not chunks:
                logger.warning("No counties to chunk for upgrade_tag %s; skipping array submission.", upgrade_tag)
                continue

            # Always (re)write the manifest. Same content if chunking is unchanged.
            # `weather_locs_str` is an underscore-joined list of values from
            # `weather_col` (as_sim GISJOINs for ComStock 2025.2; county codes
            # for ResStock and older ComStock). D filters df_meta and the BSQ
            # ts_agg query by these to scope the chunk's training set.
            manifest_path = f'{output_dir}/inputs/manifest_upgrade{upgrade_tag}.csv'
            with open(manifest_path, 'w') as mf:
                mf.write('chunk_idx,start_index,end_index,weather_locs_str\n')
                for idx, (s_idx, e_idx, c_str) in enumerate(chunks):
                    mf.write(f'{idx},{s_idx},{e_idx},{c_str}\n')
            logger.info("Wrote manifest with %d chunks at %s", n_total, manifest_path)

            # Chunk-worker profile, sized for the standard partition
            # (48 cores / 246 GB / node). If the standard pool is ever
            # unusable, edit these four values + A's `#SBATCH --partition=`
            # together; nothing else needs to change.
            chunk_partition = 'standard'
            chunk_cpus = '48'
            chunk_mem = '246064'
            array_cap = '200'

            if n_missing == 0:
                # All chunks present but agg missing — submit agg directly.
                # No array dependency needed; chunks already exist on disk.
                # --time passed explicitly so a stray SBATCH_TIMELIMIT in
                # the env can't shrink it (SLURM precedence: CLI > env > #SBATCH).
                agg_result = subprocess.run([
                    'sbatch',
                    f'--job-name={prefix}agg_{upgrade_tag}',
                    f'--partition={chunk_partition}',
                    f'--mem={chunk_mem}',
                    '--time=02:00:00',
                    f'--output={slurm_out_dir}/slurm-%x_%j.out',
                    './F_aggregate_chunks.sh',
                    output_dir, bldg_type, upgrade_tag,
                ], check=True, capture_output=True, text=True)
                agg_job_id = agg_result.stdout.strip().split()[-1]
                agg_job_ids.append(agg_job_id)
                logger.info(
                    "Queued aggregation job %s for upgrade_tag %s (no array; all chunks present)",
                    agg_job_id, upgrade_tag,
                )
            else:
                # Submit array for missing chunks only; agg depends on it.
                # _compress_array_indices yields "0-2,5,7-9" style; SLURM
                # accepts these as a sparse array spec.
                array_spec = _compress_array_indices(missing_indices)
                # --time passed explicitly so a stray SBATCH_TIMELIMIT in
                # the env can't shrink it (SLURM precedence: CLI > env > #SBATCH).
                result = subprocess.run([
                    'sbatch',
                    f'--job-name={prefix}chunk_{upgrade_tag}',
                    f'--partition={chunk_partition}',
                    f'--cpus-per-task={chunk_cpus}',
                    f'--mem={chunk_mem}',
                    '--time=03:00:00',
                    f'--array={array_spec}%{array_cap}',
                    f'--output={slurm_out_dir}/slurm-%x_%A_%a.out',
                    './C_run_bldg_chunk_agg.sh',
                    manifest_path, meta_path, upgrade_argv_token, prefix,
                    output_dir, script_dir, str(spec_index),
                ], check=True, capture_output=True, text=True)
                array_job_id = result.stdout.strip().split()[-1]
                logger.info(
                    "Submitted array job %s for upgrade_tag %s with %d tasks "
                    "(array=%s, %%%s concurrency, partition=%s, cpus=%s, mem=%sMB)",
                    array_job_id, upgrade_tag, n_missing, array_spec,
                    array_cap, chunk_partition, chunk_cpus, chunk_mem,
                )

                # Aggregation depends on the array completing successfully.
                # afterok on an array is satisfied only when every task exits 0;
                # --kill-on-invalid-dep=yes cancels the agg job if any task fails.
                agg_result = subprocess.run([
                    'sbatch',
                    f'--job-name={prefix}agg_{upgrade_tag}',
                    f'--partition={chunk_partition}',
                    f'--mem={chunk_mem}',
                    '--time=02:00:00',
                    f'--dependency=afterok:{array_job_id}',
                    '--kill-on-invalid-dep=yes',
                    f'--output={slurm_out_dir}/slurm-%x_%j.out',
                    './F_aggregate_chunks.sh',
                    output_dir, bldg_type, upgrade_tag,
                ], check=True, capture_output=True, text=True)
                agg_job_id = agg_result.stdout.strip().split()[-1]
                agg_job_ids.append(agg_job_id)
                logger.info(
                    "Queued aggregation job %s for upgrade_tag %s (depends on array %s)",
                    agg_job_id, upgrade_tag, array_job_id,
                )

                # Stagger between array submissions so multiple specs don't
                # all start hammering Athena in the same window. Belt-and-
                # suspenders alongside D's `sleep_seconds` randomized startup.
                logger.info("Sleeping 60s before submitting next spec's array...")
                pytime.sleep(60)

        else:
            # Local: queue each missing chunk for the multiprocessing pool below.
            for idx, (start_index, end_index, weather_locs_str) in enumerate(chunks):
                if idx in existing_indices:
                    continue  # already done from a prior run
                cmd = [
                    sys.executable,
                    f'{output_dir}/inputs/D_process_chunk_agg.py',
                    str(start_index), str(end_index), meta_path,
                    upgrade_argv_token, prefix, output_dir, script_dir, weather_locs_str,
                    str(spec_index),
                ]
                tasks.append(cmd)
            if upgrade_tag not in {s['upgrade_tag'] for s in local_specs}:
                local_specs.append({'upgrade_tag': upgrade_tag, 'prefix': prefix})

    # Local: run any remaining chunks in parallel, then aggregate any specs
    # that need it. The two are gated separately because resume can yield
    # local_specs without tasks (all chunks already on disk, agg still pending).
    if not hpc:
        if tasks:
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

    # HPC end-to-end: submit the aux-query step and chain the projection +
    # handoff stages on top. Z dispatches state and county_group projections
    # and queues ReEDs/intermediate/LBL handoffs with `--dependency=afterok`.
    # Skipped in local mode (no SLURM to chain) and skipped when no F was
    # submitted (resume run where every agg CSV was already on disk).
    if hpc and agg_job_ids:
        stock = 'com' if sw_comstock else 'res'

        # E (aux query) only needs the switches snapshot, not the chunks/aggs,
        # so it runs in parallel with C/F. Z waits on F's AND E so projection
        # has both the calibrated agg CSVs and the aux_coverage_*.csv it reads.
        e_result = subprocess.run([
            'sbatch',
            f'--output={slurm_out_dir}/slurm-%x_%j.out',
            './E_run_aux.sh', output_dir,
        ], check=True, capture_output=True, text=True)
        e_job_id = e_result.stdout.strip().split()[-1]
        logger.info("Queued aux-query job %s", e_job_id)

        dep = ':'.join(agg_job_ids + [e_job_id])
        z_result = subprocess.run([
            'sbatch',
            f'--dependency=afterok:{dep}',
            '--kill-on-invalid-dep=yes',
            f'--output={slurm_out_dir}/slurm-%x_%j.out',
            './Z_post_pipeline.sh', output_dir, stock,
        ], check=True, capture_output=True, text=True)
        z_job_id = z_result.stdout.strip().split()[-1]
        logger.info(
            "Queued Z post-pipeline job %s (depends on %d agg jobs + aux %s)",
            z_job_id, len(agg_job_ids), e_job_id,
        )
    elif hpc:
        logger.info(
            "Skipping Z post-pipeline: no aggregation jobs were submitted "
            "(everything already on disk). Run Z manually if handoffs need "
            "refreshing: sbatch Z_post_pipeline.sh %s %s",
            output_dir, 'com' if sw_comstock else 'res',
        )

    print("All chunks processed successfully!")
