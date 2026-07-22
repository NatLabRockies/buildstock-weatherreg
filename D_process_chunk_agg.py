"""Per-chunk BSQ pull + (optional) calibration + regression + write.

Run by C as a SLURM array task. For one chunk (a counties-by-weather-loc bin):

  1. Athena pulls of meta + ts_agg from the spec's base_run and target_run.
  2. ResStock calibration (only when the run_type has `adjustment_factor` set):
     `_apply_state_adjustment` multiplies every electricity column of ts_agg
     uniformly by `factor[state, hour]` so downstream gap / net consumption
     math sees a calibrated trajectory. PV scales in lockstep with total, so
     `net = total - pv` calibrates exactly.
  3. Builds the regression feature matrix (weather + temperature lags +
     building metadata), fits a RandomForest / hybrid model, predicts each
     target year, and writes one chunk CSV per (spec, weather year) to
     <output_dir>/chunks_*_b<base_year>/.

When `apply_regression: false`, step 3 is skipped and the base-year load is
written directly. Calibration (step 2) runs in both modes — it sits *before*
the regression branch.
"""

# IMPORTS
# Import libraries
import os
import certifi
_CA = certifi.where()
os.environ.setdefault("AWS_CA_BUNDLE", _CA)
os.environ.setdefault("CURL_CA_BUNDLE", _CA)
os.environ.setdefault("SSL_CERT_FILE", _CA)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _CA)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime as dt
import json
import shutil
import subprocess
from buildstock_query import BuildStockQuery
import sqlglot
import time
import re
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set environment variable to disable OneDNN prior to importing tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Defer TensorFlow import to avoid forking a TF-initialized runtime.
# Import and seed only when/if the hybrid model is used.
_TF = None
def _ensure_tf():
    global _TF
    if _TF is None:
        import tensorflow as tf
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        _TF = tf
    return _TF

def _parse_year_entry(entry):
    """
    Parse one target year entry.
    Supported forms:
      - 2018
      - "2018"
      - "2007-2013"
    """
    if isinstance(entry, int):
        return [entry]
    if isinstance(entry, str):
        token = entry.strip()
        if not token:
            return []
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid year range: {entry}")
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            if end < start:
                raise ValueError(f"Year range end < start: {entry}")
            return list(range(start, end + 1))
        return [int(token)]
    raise TypeError(f"Unsupported target_year entry type: {type(entry)}")

def parse_target_years(year_spec):
    """
    Parse target_year setting into a sorted unique list of years.
    Supported forms:
      - 2018
      - "2018"
      - ["2007-2013", 2016, "2018"]
    """
    if isinstance(year_spec, (int, str)):
        years = _parse_year_entry(year_spec)
    elif isinstance(year_spec, list):
        years = []
        for entry in year_spec:
            years.extend(_parse_year_entry(entry))
    else:
        raise TypeError(f"Unsupported target_year type: {type(year_spec)}")

    years = sorted(set(years))
    if not years:
        raise ValueError("target_year parsed to an empty year list")
    return years

script_start_time = dt.datetime.now()
print('Script start time:', script_start_time)

# Import command line arguments
start_index = int(sys.argv[1])
end_index = int(sys.argv[2])
meta_path = sys.argv[3]
# Passed by C_run_bldg_chunk_agg.sh as `$upgrade`. With list-valued upgrade
# specs this is a stringified token (e.g. "[1, 14, 58]") used only in the
# "how to re-run this exact chunk" log message below; the authoritative
# upgrade_id is `spec['upgrade_id']`, read from the switches JSON.
upgrade_token_display = sys.argv[4]
prefix = sys.argv[5]
output_dir = sys.argv[6]
script_dir = sys.argv[7]
# Underscore-joined list of weather-location values (as_sim GISJOINs for
# ComStock 2025.2; county codes for ResStock and older ComStock).
weather_locs_str = sys.argv[8]
spec_index = int(sys.argv[9])

# Import switches first so we can resolve per-spec values from spec_index.
with open(os.path.join(output_dir, 'inputs', 'switches_agg.json'), 'r') as f:
    switch = json.load(f)


# Names propagate into upgrade_tag, which is the leading segment of every
# chunk/manifest/meta/agg filename and of SLURM job names. The agg-side
# parser splits the tag on the FIRST underscore to recover the
# `<reg|ref>_b<year>` suffix (see agg_buildings._chunks_eulp_dir), so an
# underscore inside `name` would steal that split. Enforce ASCII letters,
# digits, dots, and hyphens — fail loudly otherwise.
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


# Per-spec values: each run_specs entry carries its own
# apply_regression / base_year / target_year / name / restrict / avoid,
# so two specs with the same upgrade_id can produce different outputs in
# one run. The supplied `name` field disambiguates them.
spec = switch['run_specs'][spec_index]
spec_name = _validate_spec_name(spec['name'])
sw_apply_regression = bool(spec['apply_regression'])
base_year = spec['base_year']
target_years = parse_target_years(spec['target_year'])
comparison_year = (
    base_year if base_year in target_years else target_years[0]
) # Year used for df_meta annual regression comparison columns
regression_tag = 'reg' if sw_apply_regression else 'ref'
upgrade_tag = f'{spec_name}_{regression_tag}_b{base_year}'

# `upgrade_id` may be a scalar OR a list. For a list, we call process_chunk_agg
# once per upgrade and sum the resulting timeseries — the spec's restrict/avoid
# applies identically to each upgrade pull. `upgrade_ids` is always a list
# internally; `upgrade_display` is just for log lines.
_raw_upgrade = spec['upgrade_id']
upgrade_ids = list(_raw_upgrade) if isinstance(_raw_upgrade, (list, tuple)) else [_raw_upgrade]
upgrade_display = (
    str(upgrade_ids[0]) if len(upgrade_ids) == 1 else f'{upgrade_ids!r}'
)

# Spec-level applied filters. Each is an optional dict of the shape
#   { "all_of": [<upgrade_id>, ...], "any_of": [<upgrade_id>, ...] }
# Both keys are optional; missing/None means "no extra predicate." When
# present, the dict gets translated into a BSQ `get_applied_buildings_filter`
# RestrictTuple inside process_chunk_agg and appended to the BSQ
# restrict= or avoid= list.
spec_restrict_filter = spec.get('restrict') or None
spec_avoid_filter = spec.get('avoid') or None

# Per-spec subfolders inside the run output dir. D writes its three kinds of
# chunk artifacts directly into these (rather than the run-dir root) so the
# top-level layout stays browsable across many concurrent specs. Master
# metadata and aggregated GWh outputs continue to live at the run-dir root.
# Concurrent makedirs from many array tasks is safe with exist_ok=True.
chunks_eulp_dir = os.path.join(output_dir, f'chunks_{regression_tag}_b{base_year}')
chunks_meta_dir = os.path.join(output_dir, f'chunks_meta_b{base_year}')
chunks_sql_dir = os.path.join(output_dir, f'chunks_sql_{regression_tag}_{base_year}')
for _d in (chunks_eulp_dir, chunks_meta_dir, chunks_sql_dir):
    os.makedirs(_d, exist_ok=True)

print('start_index:', start_index)
print('end_index:', end_index)
print('meta_path:', meta_path)
print('upgrade_ids:', upgrade_ids)
print('spec_name:', spec_name)
print('spec_restrict_filter:', spec_restrict_filter)
print('spec_avoid_filter:', spec_avoid_filter)
print('prefix:', prefix)
print('output_dir:', output_dir)
print('script_dir:', script_dir)
print('weather_locs_str:', weather_locs_str)
print('spec_index:', spec_index)
print('apply_regression:', sw_apply_regression)
print('base_year:', base_year)
print('target_years:', target_years)
print('upgrade_tag:', upgrade_tag)

print('To rerun just this chunk as a single-task array:')
_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '<task_idx_from_manifest>')
_manifest_path = f'{output_dir}/inputs/manifest_upgrade{upgrade_tag}.csv'
print(f'sbatch --job-name={prefix}chunk_{upgrade_tag} '
      f'--array={_task_id} ./C_run_bldg_chunk_agg.sh '
      f'{_manifest_path} {meta_path} {upgrade_token_display} {prefix} '
      f'{output_dir} {script_dir} {spec_index}')

## Switch that designates comstock or resstock data
sw_comstock = switch['comstock'] # if `False`, then resstock
sw_savings_shape = switch['savings_shape'] # if `False`, aggregate_timeseries
applied_only = switch['applied_only'] # if `True`, only buildings with upgrade applied
sleep_seconds = switch['sleep_seconds'] # Number of seconds to sleep at the start of the script to prevent AWS token errors when multiple jobs are run simultaneously
## Columns to group by; Note: if this changes, language in the
## `process_chunk_agg` function will need to be updated following the AWS call
bsq_cols =  switch['com_bsq_cols'] if sw_comstock else switch['res_bsq_cols']
## Number of buildings to pull per upgrade
n_bldngs = switch['n_bldngs'] # 'all' for all buildings, 'assign' for assigned building id list from csv
sw_test_base = switch['test_base']
sw_save_metrics = switch['save_metrics']
sw_show_fit = switch['show_fit']
sw_save_fit = switch['save_fit']
sw_test_target = switch['test_target']
sw_cross_val = switch['cross_val']  # If True, perform cross-validation, fitting accuracy will be improved and runtime will be longer
sw_hybrid_model = switch['hybrid_model']  # If True, use a hybrid model of random forest and polynomial regression to overcome the limitations of random forest extrapolation, and runtime will be longer. If False, use only random forest.
sw_mode = switch['mode'] # Choose HVAC electricity usage, "heat_and_cool" for all HVAC (default), "heat" for heating only, "cool" for cooling only
comstock_year, comstock_release = switch['version_comstock'][0], switch['version_comstock'][1]
resstock_year, resstock_release = switch['version_resstock'][0], switch['version_resstock'][1]
lag_hours = switch['lag_hours_temperature']   # Lagged features for the dry bulb temperature to include the load inertia
base_run = spec['base_run']     # Per-spec BuildStockQuery base run type
target_run = spec['target_run'] # Per-spec BuildStockQuery target run type
run_types = switch['run_types'] # Run types for the BuildStockQuery object
weather_data_base = switch['weather_data_base']

if sw_test_target and len(target_years) != 1:
    raise ValueError(
        f"sw_test_target=True requires exactly one target_year per spec; "
        f"run_specs[{spec_index}] has {len(target_years)}: {target_years}."
    )

# Force program to sleep for a random amount of seconds between 0 and sleep_seconds
# Prevents AWS token errors when multiple jobs are run simultaneously
time.sleep(random.uniform(0, sleep_seconds))

# FUNCTIONS
# Detect HPC by presence of a SLURM_JOB_ID (only set under sbatch/srun).
def _is_hpc() -> bool:
    return 'SLURM_JOB_ID' in os.environ

# Process one weather-location worker function — for HPC multiprocessing.
# Each weather location (an as_sim GISJOIN for ComStock 2025.2; a county code
# for ResStock and older ComStock) trains THREE RFs on the per-loc hourly
# aggregates — one each for cooling.elec, heating.elec, and natural-gas
# heating — and returns predicted hourly profiles plus annual sums for the
# comparison_year. Per-bldg_id share-out happens after all locations finish.
def _process_one_location(args):
    """
    Worker: train RF for each energy type, return:
      - location identifier
      - cooling.elec hourly DataFrame
      - heating.elec hourly DataFrame
      - NG hourly DataFrame
      - cooling/heating/NG annual sums at this location for comparison_year
    Notes:
      - Metrics and plots are disabled in workers to avoid file contention.
      - The 9th positional arg of `prediction()` is named bldg_id but is
        only used to (a) tag log lines, (b) index df_eulp_targ when
        sw_test_target=True. We pass `loc` here; df_eulp_targ_local is
        pre-aggregated per loc by the caller and has both cooling.elec and
        heating.elec columns.
    """
    (loc,
     df_eulp_pred,
     df_eulp_targ_local,
     base_year,
     target_years,
     sw_test_base,
     sw_test_target) = args

    # Ensure globals used in prediction/test_fit exist but do nothing noisy
    global i, sw_save_metrics, sw_show_fit, sw_save_fit
    i = 0
    sw_save_metrics = False
    sw_show_fit = False
    sw_save_fit = False

    # Load weather once for this location.
    weather_base_df = weather_data(weather_data_base, base_year, loc)
    target_weather_frames = []
    target_year_by_row = []
    for yr in target_years:
        year_df = weather_data(weather_data_base, yr, loc)
        target_weather_frames.append(year_df)
        target_year_by_row.extend([yr] * len(year_df))
    weather_target_df = pd.concat(target_weather_frames, ignore_index=True)
    target_year_by_row = np.asarray(target_year_by_row)

    def _predict(energy_type):
        df_out = prediction(
            base_year, df_eulp_pred, sw_test_base, target_years, sw_test_target,
            energy_type, weather_base_df, weather_target_df, loc,
            df_eulp_targ_local, target_year_by_row,
        )
        annual_sum = (
            df_out.loc[
                (df_out['timestamp'] - pd.Timedelta(hours=1)).dt.year
                == comparison_year,
                energy_type,
            ].sum().round(6)
        )
        return df_out, annual_sum

    df_cool, cool_sum = _predict('cooling.elec')
    df_heat, heat_sum = _predict('heating.elec')
    df_ng, ng_sum = _predict('natural_gas.heating.energy_consumption')

    return loc, df_cool, df_heat, df_ng, cool_sum, heat_sum, ng_sum

# Substrings that indicate an error is transient (worth retrying). Matched
# case-insensitively against the exception message — covers Athena/S3
# throttling, transient TLS/socket failures, and AWS slow-down responses.
_TRANSIENT_ERROR_MARKERS = (
    'HIVE_S3_THROTTLING',
    'Status Code: 503',
    'Status Code: 429',
    'SlowDown',
    'TooManyRequests',
    'Throttling',
    'ThrottlingException',
    'RequestLimitExceeded',
    'timed out',
    'TimeoutError',
    'Connection reset',
    'BrokenPipe',
    'ServiceUnavailable',
)


def query_execution(query, my_run, max_attempts=6, base_delay=30, max_delay=600):
    """Run a BSQ query with exponential backoff + jitter on transient failures.

    Athena/S3 returns throttling 503s ("HIVE_S3_THROTTLING") when many parallel
    tasks query at once. Total retry budget is ~25 min worst case (well within
    a 4hr chunk wall) — generous enough to wait out a real throttle window.
    Per-retry jitter (±50%) desyncs concurrent retries across array tasks so we
    don't re-stampede AWS in lockstep.

    Non-transient errors (e.g., SQL syntax errors) fail fast — no point eating
    the retry budget on a permanent failure.

    Parameters:
        query: The query string to execute.
        my_run: The BuildStockQuery object.
        max_attempts: Total attempts including the first (default 6).
        base_delay: Seconds for the first backoff (default 30; doubled each retry).
        max_delay: Cap on per-retry sleep regardless of exponent (default 600).

    Returns:
        DataFrame: the query results.

    Raises:
        Exception: re-raises the last exception when retries are exhausted, or
        immediately on non-transient errors.
    """
    for attempt in range(max_attempts):
        try:
            print(f"Executing query (attempt {attempt + 1}/{max_attempts}): {query}\n")
            return my_run.execute(query)
        except Exception as e:
            err_str = str(e).lower()
            is_last = attempt == max_attempts - 1
            is_transient = any(m.lower() in err_str for m in _TRANSIENT_ERROR_MARKERS)

            print(f"Query attempt {attempt + 1}/{max_attempts} failed: {e}")

            if is_last:
                print(f"Out of retries after {max_attempts} attempts. Re-raising.")
                raise
            if not is_transient:
                # Likely permanent error (e.g., SQL syntax) — failing fast keeps
                # the developer feedback loop tight.
                print("Error does not look transient; failing fast (no retry).")
                raise

            # Exponential backoff: base * 2^attempt, capped at max_delay.
            base = min(base_delay * (2 ** attempt), max_delay)
            # Full ±50% jitter to spread concurrent retries across array tasks.
            jitter = base * 0.5 * (2 * random.random() - 1)
            sleep_for = max(1.0, base + jitter)
            print(f"Transient error; sleeping {sleep_for:.1f}s before retry "
                  f"{attempt + 2}/{max_attempts}...")
            time.sleep(sleep_for)

def write_pretty_sql(sql, path):
    """Write `sql` to `path`, pretty-printed via sqlglot when possible.

    Falls back to writing the raw SQL on any sqlglot failure — capturing
    the query is the primary goal; formatting is best-effort.
    """
    raw = str(sql)
    try:
        formatted = sqlglot.transpile(
            raw, read='athena', write='athena', pretty=True
        )[0]
    except Exception as e:
        print(f'sqlglot pretty-format failed ({type(e).__name__}: {e}); writing raw SQL.')
        formatted = raw
    with open(path, 'w') as f:
        f.write(formatted)


def _apply_state_adjustment(ts_agg, df_meta, adjustment_factor_path):
    """Multiply every electricity column in `ts_agg` by per-(timestamp,
    bldg-state) calibration factors from a (hours × state-postal) parquet.

    Scaling every electric component uniformly preserves the net-calibration
    target because:
        new_total − new_pv = factor × total − factor × pv
                           = factor × (total − pv)
                           = factor × old_net
    The non_hvac.elec residual computed by the caller (total − cooling −
    heating − ev) automatically scales by the same factor since all four
    inputs do. Natural gas is intentionally NOT scaled — the parquet
    calibrates net electric load, not gas.

    `adjustment_factor_path` is resolved against `script_dir` if relative.
    Rows whose (timestamp, state) isn't in the parquet fall through with
    factor 1.0 (no adjustment).
    """
    path = (adjustment_factor_path if os.path.isabs(adjustment_factor_path)
            else os.path.join(script_dir, adjustment_factor_path))
    adj = pd.read_parquet(path)
    adj.index.name = 'timestamp'
    adj_long = adj.stack()          # MultiIndex (timestamp, state) -> factor
    adj_long.index.names = ['timestamp', 'state']

    # df_meta uses BSQ-prefixed column names ('in.state', not 'state')
    row_states = ts_agg.index.map(df_meta['in.state'])
    keys = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(ts_agg['timestamp']), row_states],
        names=['timestamp', 'state'],
    )
    factors = adj_long.reindex(keys).fillna(1.0).to_numpy()
    matched = int((factors != 1.0).sum())
    print(f'  ResStock calibration: {matched:,} / {len(factors):,} rows '
          f'scaled via {os.path.basename(path)}')

    for col in ('cooling.elec', 'heating.elec', 'ev', 'pv', 'total'):
        if col in ts_agg.columns:
            ts_agg[col] = ts_agg[col].to_numpy() * factors


def process_chunk_agg(run_type, upgrade, weather_locs, weather_col, bsq_cols,
                      sw_comstock, chunk_states, sw_savings_shape, df_meta,
                      applied_only, query_label='base',
                      spec_restrict=None, spec_avoid=None):
    """
    This function aggregates timeseries data for a specific run type, upgrade,
    enduse, and set of weather locations.
    It then processes the aggregated data to calculate the 'HVAC.elec' column
    and returns the DataFrame.

    Parameters:
    run_type (str): The type of run to process.
    upgrade (int): The upgrade ID to process.
    weather_locs (list): The weather-location values to filter on (as_sim
        GISJOINs for ComStock 2025.2; county codes for other stocks).
    weather_col (str): The column in the BSQ schema to filter on (e.g.
        'in.as_simulated_nhgis_county_gisjoin' or 'in.county').
    bsq_cols (list): The columns to group by when aggregating.
    sw_comstock (bool): Whether the data is from ComStock (True) or ResStock.
    chunk_states (list): The states to process.
    sw_savings_shape (bool): Method - savings_shape or aggregate_timeseries.
    df_meta (DataFrame): The metadata DataFrame.
    applied_only (bool): If True, only buildings with upgrade applied are used.
    query_label (str): Role of this call ('base' or 'target'). Controls the
        filename of the saved Athena SQL — 'base' uses no suffix so it sits
        next to the chunk's meta CSV; other labels are appended.
    spec_restrict (dict | None): Optional applied-upgrade predicate to AND
        into BSQ's restrict list. Shape:
        ``{"all_of": [<upgrade_id>...], "any_of": [<upgrade_id>...]}``;
        either key may be omitted. The pair is translated to a
        `get_applied_buildings_filter` RestrictTuple and appended to
        restrict so only bldg_ids satisfying the predicate are pulled.
    spec_avoid (dict | None): Same shape as `spec_restrict`, but the
        resulting RestrictTuple is appended to BSQ's avoid list — i.e.
        bldg_ids satisfying the predicate are EXCLUDED.

    Returns:
    ts_agg (DataFrame): Aggregated timeseries HVAC electricity.
    """
    aws_cols = [c for c in bsq_cols]
    aws_weather_locs = weather_locs.copy()
    aws_run_type = run_types[run_type].copy()
    # Optional ResStock calibration. The (8760 × 49-state) parquet, if set on
    # the run_type, multiplies the BSQ-pulled electric load columns *before*
    # the non_hvac residual is computed (so the residual stays consistent)
    # and *before* any regression. Pop it out so BSQ doesn't see the unknown
    # kwarg. None means "no calibration" — leaves ts_agg unchanged.
    adjustment_factor_path = aws_run_type.pop('adjustment_factor', None)
    natural_gas = ['out.natural_gas.heating.energy_consumption']
    # Split HVAC enduses into pure-heating, pure-cooling, and ambiguous
    # (heating-or-cooling depending on which mode the air handler / hydronic
    # loop is in that day). ResStock breaks fans/pumps out by mode already,
    # so its ambiguous list is empty. ComStock's fans/pumps/heat_recovery
    # serve whichever mode is running and we allocate them by daily share
    # of heating.energy vs cooling.energy (50/50 fallback if both zero).
    if sw_comstock:
        # `elec_enduse` order must match the historical pipeline so BSQ
        # caches the SAME SQL string hash and serves prior chunk queries
        # from cache.
        elec_enduse = [
            'out.electricity.heating.energy_consumption',
            'out.electricity.cooling.energy_consumption',
            'out.electricity.fans.energy_consumption',
            'out.electricity.heat_recovery.energy_consumption',
            'out.electricity.heat_rejection.energy_consumption',
            'out.electricity.pumps.energy_consumption',
        ]
        heating_pure = ['out.electricity.heating.energy_consumption']
        cooling_pure = [
            'out.electricity.cooling.energy_consumption',
            'out.electricity.heat_rejection.energy_consumption',
        ]
        heating_and_cooling = [
            'out.electricity.fans.energy_consumption',
            'out.electricity.heat_recovery.energy_consumption',
            'out.electricity.pumps.energy_consumption',
        ]
        total = ['out.electricity.total.energy_consumption']
        pv = ['out.electricity.pv.energy_consumption']
        ev = []
        avoid = [('in.hvac_system_type', ('PTAC with gas boiler',))]
    else:
        # ResStock's natural ordering already happens to equal
        # `heating_pure + cooling_pure` so cache stays warm without
        # special handling.
        heating_pure = [
            'out.electricity.heating.energy_consumption..kwh',
            'out.electricity.heating_fans_pumps.energy_consumption..kwh',
            'out.electricity.heating_hp_bkup.energy_consumption..kwh',
            'out.electricity.heating_hp_bkup_fa.energy_consumption..kwh',
        ]
        cooling_pure = [
            'out.electricity.cooling.energy_consumption..kwh',
            'out.electricity.cooling_fans_pumps.energy_consumption..kwh',
        ]
        heating_and_cooling = []
        ev = ['out.electricity.ev_charging.energy_consumption..kwh']
        total = ['out.electricity.total.energy_consumption..kwh']
        pv = ['out.electricity.pv.energy_consumption..kwh']
        avoid = []

        elec_enduse = heating_pure + cooling_pure + heating_and_cooling
        natural_gas = ['out.natural_gas.heating.energy_consumption..kwh']
    restrict = [('state', chunk_states),
                  (weather_col, aws_weather_locs),
                 ]
    my_run = BuildStockQuery(**aws_run_type)

    # Spec-level applied filters. `get_applied_buildings_filter` returns a
    # `(cols, subquery)` RestrictTuple ready to drop into restrict= or avoid=.
    # It returns None when both predicate lists are empty/None — we guard
    # against that explicitly. A spec can carry both `restrict` and `avoid`
    # (e.g., "upgrade 32 applied AND upgrade 4 NOT applied"), so the two
    # predicates are built and slotted independently.
    if spec_restrict:
        applied_restrict_filter = my_run.get_applied_buildings_filter(
            any_of=spec_restrict.get('any_of'),
            all_of=spec_restrict.get('all_of'),
        )
        if applied_restrict_filter is not None:
            restrict = restrict + [applied_restrict_filter]
    if spec_avoid:
        applied_avoid_filter = my_run.get_applied_buildings_filter(
            any_of=spec_avoid.get('any_of'),
            all_of=spec_avoid.get('all_of'),
        )
        if applied_avoid_filter is not None:
            avoid = avoid + [applied_avoid_filter]

    ts_agg_query = my_run.query(
        upgrade_id=upgrade,
        applied_only=False,
        enduses=elec_enduse + natural_gas + ev + total + pv,
        # Filter by weather_col directly. For ComStock 2025.2 this pulls
        # every source-county served by these as_sims; for ResStock and
        # older ComStock weather_col == county so behavior is unchanged.
        restrict=restrict,
        avoid=avoid,
        timestamp_grouping_func="hour",
        group_by=aws_cols,
        get_query_only=True,
        annual_only=False
    )

    # Include the upgrade id in the SQL filename so multi-upgrade specs
    # (upgrade_id given as a list) write one SQL file per call rather than
    # overwriting a single shared path.
    sql_suffix = '' if query_label == 'base' else f'_{query_label}'
    sql_path = os.path.join(
        chunks_sql_dir,
        f'{prefix}meta_upgrade{upgrade_tag}_u{upgrade}_'
        f'{start_index:04}-{end_index:04}{sql_suffix}.sql'
    )
    write_pretty_sql(ts_agg_query, sql_path)

    ts_agg = query_execution(ts_agg_query, my_run)
    ts_agg = ts_agg.sort_values(aws_cols + ['timestamp']).reset_index(drop=True)
    ts_agg['timestamp'] = pd.to_datetime(ts_agg['timestamp']) + pd.Timedelta(hours=1)
    # Normalize column names: drop 'out.' prefix and ResStock's '..kwh' suffix
    # so downstream lookups use canonical names.
    def _norm(c):
        return c.replace('out.', '').replace('..kwh', '')
    ts_agg.columns = [_norm(c) for c in ts_agg.columns]
    heating_pure = [_norm(c) for c in heating_pure]
    cooling_pure = [_norm(c) for c in cooling_pure]
    heating_and_cooling = [_norm(c) for c in heating_and_cooling]
    ev_cols = [_norm(c) for c in ev]
    total_cols = [_norm(c) for c in total]
    pv_cols = [_norm(c) for c in pv]

    if sw_comstock:
        state_county_map = pd.read_csv(
            os.path.join(
                output_dir, "inputs", "spatial_tract_lookup_table.csv")
        )

        # Merge state_county_map w/ df_meta to bring in resstock_county_id
        ts_agg = ts_agg.merge(
            state_county_map[
                ["nhgis_county_gisjoin", "resstock_county_id"]
            ].drop_duplicates(),
            how="left",
            on="nhgis_county_gisjoin"
        )

        # Assign resstock_county_id to in.county_name
        ts_agg["county_name"] = ts_agg["resstock_county_id"]

    # Add building ID column from groupby columns and set as index
    ts_agg['bldg_id'] = ts_agg[aws_cols].apply(tuple, axis=1).astype(str)
    ts_agg.set_index('bldg_id', inplace=True)

    # Pure-mode contributions sum cleanly per row. For ResStock this is the
    # whole calculation (ambiguous is empty). For ComStock the ambiguous
    # enduses (fans/pumps/heat_recovery) get allocated by daily heating-share
    # vs cooling-share computed from the pure signals.
    ts_agg['heating.elec'] = ts_agg[heating_pure].sum(axis=1)
    ts_agg['cooling.elec'] = ts_agg[cooling_pure].sum(axis=1)

    if heating_and_cooling:
        # Compute heating/cooling shares per (bldg_id, day). Day boundaries
        # use the model-year convention (timestamp is hour-ending, so the
        # local day is `(ts - 1h).date()`). Both sums zero → 50/50 (covers
        # off-days where fans/pumps run for ventilation only).
        ts_tmp = ts_agg.reset_index()
        ts_tmp['_day'] = (ts_tmp['timestamp'] - pd.Timedelta(hours=1)).dt.date
        grp = ts_tmp.groupby(['bldg_id', '_day'])
        daily_heat = grp['heating.elec'].transform('sum').to_numpy()
        daily_cool = grp['cooling.elec'].transform('sum').to_numpy()
        daily_total = daily_heat + daily_cool
        # `np.divide(..., where=..., out=...)` skips the divide at zero-total
        # rows entirely (no RuntimeWarning) and leaves them at the 0.5
        # initializer — the user-spec'd off-day fallback.
        heating_share = np.divide(
            daily_heat, daily_total,
            out=np.full_like(daily_total, 0.5, dtype=float),
            where=daily_total > 0,
        )
        cooling_share = 1.0 - heating_share
        amb_sum = ts_tmp[heating_and_cooling].sum(axis=1).to_numpy()
        ts_tmp['heating.elec'] = ts_tmp['heating.elec'].to_numpy() + amb_sum * heating_share
        ts_tmp['cooling.elec'] = ts_tmp['cooling.elec'].to_numpy() + amb_sum * cooling_share
        ts_tmp = ts_tmp.drop(columns=['_day'])
        ts_agg = ts_tmp.set_index('bldg_id')

    # EV / PV / total carry through from the Athena pull. `non_hvac.elec`
    # is the residual after removing HVAC (cooling+heating, already
    # ambiguous-allocated above) and EV charging from the site meter
    # total. PV is NOT subtracted: `electricity.total.energy_consumption`
    # in EULP is gross site consumption and does not net out onsite
    # generation. For ComStock, ev_cols is empty so the EV term is zero.
    ts_agg['ev'] = ts_agg[ev_cols].sum(axis=1) if ev_cols else 0.0
    ts_agg['pv'] = ts_agg[pv_cols[0]] if pv_cols else 0.0
    ts_agg['total'] = ts_agg[total_cols[0]]

    # ResStock electric-load calibration: per-(state, hour) factor applied
    # uniformly to every electricity column (cooling, heating, ev, pv, total).
    # Scaling pv along with total means net = total − pv also scales by the
    # factor — same calibration target. The non_hvac.elec residual below
    # inherits the same scaling since all of its inputs do. Natural gas is
    # left untouched (factor calibrates ELECTRICITY).
    if adjustment_factor_path:
        _apply_state_adjustment(ts_agg, df_meta, adjustment_factor_path)

    ts_agg['non_hvac.elec'] = (
        ts_agg['total']
        - ts_agg['cooling.elec']
        - ts_agg['heating.elec']
        - ts_agg['ev']
    )

    ts_agg = ts_agg[['timestamp', 'cooling.elec', 'heating.elec',
                     'natural_gas.heating.energy_consumption',
                     'ev', 'pv', 'total', 'non_hvac.elec']]

    # Convert all energy columns from kWh to MWh & round.
    for c in ['cooling.elec', 'heating.elec',
              'natural_gas.heating.energy_consumption',
              'ev', 'pv', 'total', 'non_hvac.elec']:
        ts_agg[c] = (ts_agg[c] / 1000).round(6)

    return ts_agg

# Constant: EST offset from UTC (no DST). BuildStockQuery returns EULP hourly
# timestamps in EST, so weather data — typically stored in the location's
# local standard time per EPW spec — must be rolled to align with EST before
# being fed into the regression. Mainland US (excluding AK/HI, which we
# filter earlier) spans UTC offsets -5..-8, giving shifts of 0..3 hours.
_EST_UTC_OFFSET = -5


def _read_epw_timezone(epw_path):
    """Return the timezone offset (hours from UTC) from an EPW file's header.

    The LOCATION line is always the first line of an EPW file, comma-separated:
        LOCATION,<city>,<state>,<country>,<src>,<wmo>,<lat>,<lon>,<TZ>,<elev>
    Field 8 is the UTC offset (e.g., -6 for CST). Returns float.
    """
    with open(epw_path) as f:
        line1 = f.readline()
    parts = line1.strip().split(',')
    if len(parts) < 9 or parts[0].upper() != 'LOCATION':
        raise ValueError(f"Unexpected EPW header line in {epw_path}: {line1!r}")
    try:
        return float(parts[8])
    except ValueError as e:
        raise ValueError(
            f"Could not parse TZ offset from EPW header field 8 in {epw_path}: {parts[8]!r}"
        ) from e


def weather_data(url_base, year, county_id):
    """
    Retrieves weather data from a URL and performs data preprocessing.

    The EPW file stores hourly weather in the location's local standard time.
    BSQ-derived EULP energy timestamps are in EST. We roll the EPW columns
    by `EST_offset - local_offset` so each weather row corresponds to the
    same EST hour as the matching EULP row.

    Parameters:
    url_base (str): The base URL for the weather data.
    year (int): The year for which the weather data is retrieved.
    county_id (str): The county ID for which the weather data is retrieved.

    Returns:
    df_weather (DataFrame): The preprocessed weather data as a DataFrame.
    """
    print(f'Retrieving weather data for {year} county {county_id}.')

    # Local EPW path support (e.g., /projects/geohc/EPW/epw_symlinks)
    epw_path = os.path.join(url_base, f'FIPS_{year}', f'{county_id}_{year}.epw')
    if not os.path.isfile(epw_path):
        raise FileNotFoundError(f'Local EPW file not found: {epw_path}')

    # Compute the EST shift from the EPW's LOCATION header. Standard time
    # → standard time, integer-hour shift; no DST math needed.
    tz_offset = _read_epw_timezone(epw_path)
    shift_hours = int(round(_EST_UTC_OFFSET - tz_offset))

    # EPW has 8 metadata lines, then hourly data with no column header.
    df_epw = pd.read_csv(epw_path, skiprows=8, header=None)
    if df_epw.empty:
        raise ValueError(f'Empty EPW file: {epw_path}')

    df_weather = pd.DataFrame({
        'Dry Bulb Temperature [°C]': pd.to_numeric(df_epw.iloc[:, 6], errors='coerce'),
        'Relative Humidity [%]': pd.to_numeric(df_epw.iloc[:, 8], errors='coerce'),
        'Wind Speed [m/s]': pd.to_numeric(df_epw.iloc[:, 21], errors='coerce'),
        'Wind Direction [Deg]': pd.to_numeric(df_epw.iloc[:, 20], errors='coerce'),
        'Global Horizontal Radiation [W/m2]': pd.to_numeric(df_epw.iloc[:, 13], errors='coerce'),
        'Direct Normal Radiation [W/m2]': pd.to_numeric(df_epw.iloc[:, 14], errors='coerce'),
        'Diffuse Horizontal Radiation [W/m2]': pd.to_numeric(df_epw.iloc[:, 15], errors='coerce')
    })

    # Roll all weather columns forward by `shift_hours` so each row aligns
    # with the corresponding EST hour. np.roll wraps around: the first
    # `shift_hours` slots take values from the year's last `shift_hours`
    # rows (e.g., for PST→EST shift=+3, Jan 1 01:00–03:00 EST receives
    # Dec 31 22:00–24:00 PST data, which IS the physically-correct
    # late-Dec-evening data for those EST hours).
    if shift_hours != 0:
        for col in df_weather.columns:
            df_weather[col] = np.roll(df_weather[col].values, shift_hours)

    # Build a time index by row count so downstream features stay consistent.
    # After the roll, this index represents EST hours.
    df_weather.index = pd.date_range(
        start=f'{year}-01-01 01:00:00',
        periods=len(df_weather),
        freq='h'
    )

    # Time-of-day feature is now EST hour-of-day (the regression's target Y
    # is in EST, so X must use the same time-of-day labels).
    df_weather['Time of Day'] = df_weather.index.hour
    df_weather['Weekend'] = df_weather.index.weekday.isin([5, 6]).astype(int)

    # Lagged features built AFTER the EST roll so they reference the
    # EST-aligned temperature series.
    for lag in lag_hours:
        df_weather[f'Dry Bulb Temperature Lag {lag}h'] = (
            df_weather['Dry Bulb Temperature [°C]'].shift(lag)
        )

    # Fill NaN values caused by the lagging operation
    df_weather.bfill(inplace=True)
    df_weather.ffill(inplace=True)

    # Reset the index for further operations
    df_weather.reset_index(drop=True, inplace=True)

    # Match EULP convention: keep only the first 8760 hourly rows per year.
    df_weather = df_weather.iloc[:8760].copy()

    return df_weather

def test_fit(yr_type, year, prefix, upgrade_tag, bldg_id, model, Y_test, Y_pred,
             X_train, sw_save_metrics, output_dir, sw_save_fit, sw_show_fit, i,
             df_meta, Y, start_index, end_index, energy_type):
    """
    Perform testing and evaluation of the random forest model.

    Args:
        yr_type (str): The type of year for which the model is being tested.
        year (int): The year for which the model is being tested.
        prefix (str): Prefix string for identification.
        upgrade (int): Upgrade identifier.
        bldg_id (int): Building identifier.
        model (RandomForestRegressor): Random forest model.
        Y_test (pd.Series): Test data target.
        Y_pred (np.array): Predicted target values.
        X_train (pd.DataFrame): Training data features.
        sw_save_metrics (bool): Flag to save metrics to a file.
        output_dir (str): The output directory path.
        sw_save_fit (bool): Flag to save the fit plot to a file.
        sw_show_fit (bool): Flag to show the fit plot.
        i (int): Index of the loop.
        df_meta (pd.DataFrame): Metadata DataFrame.
        Y (pd.Series): Actual values.
        start_index (int): The starting index of the chunk.
        end_index (int): The ending index of the chunk.
        energy_type (str): 'HVAC.elec' or 'natural_gas.heating.energy_consumption'

    Returns:
        None
    """
    energy_out = 'HVAC.elec' if energy_type == 'HVAC.elec' else 'natural_gas'
    # Print the building ID/upgrade/year combination
    print(f'{yr_type}{year}_{prefix}up{upgrade:02}_{str(bldg_id)}_{energy_out}')
    # Calculate the metrics
    mae = format(mean_absolute_error(Y_test, Y_pred), '.3g')
    mse = format(mean_squared_error(Y_test, Y_pred), '.3g')
    r2 = round(r2_score(Y_test, Y_pred), 3)
    feature_importances = [round(importance, 3) 
                           for importance in model.feature_importances_]

    # Print metrics to terminal
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R2 Score: {r2}')
    print('Feature Importances:')
    for feature, importance in zip(X_train.columns, feature_importances):
        print(f'    {feature}: {importance}')
    print('\n')

    fig_dir = f'{output_dir}/{yr_type}{year}'
    metrics = f'{prefix}metrics_upgrade{upgrade_tag}_{start_index:04}-{end_index:04}'
    if sw_save_metrics or sw_save_fit:
        os.makedirs(fig_dir, exist_ok=True)
    if sw_save_metrics:
        # Save the metrics and feature importances to a .txt file
        with open(f'{fig_dir}/{metrics}.txt', 'a') as f:
            f.write(f'{prefix}up{upgrade_tag}_{str(bldg_id)}_{energy_out}\n')
            f.write(f'MAE: {mae}\n')
            f.write(f'MSE: {mse}\n')
            f.write(f'R2 Score: {r2}\n')
            f.write('Feature Importances:\n')
            for feature, importance in zip(X_train.columns,
                                           feature_importances):
                f.write(f'    {feature}: {importance}\n')
            f.write('\n')

        ## Output metrics to a CSV file
        # Create a dictionary with metrics and feature importances
        data = {'Building ID': [
                    f'{prefix}up{upgrade_tag}_{str(bldg_id)}_{energy_out}'],
                'MAE': [mae],
                'MSE': [mse],
                'R2 Score': [r2]}
        for feature, importance in zip(X_train.columns, feature_importances):
            data[f'FI - {feature}'] = [importance]
        # Convert the dictionary to a DataFrame and round the values
        df = pd.DataFrame(data)
        # Check if the file exists
        file_exists = os.path.isfile(f'{fig_dir}/{metrics}.csv')
        # Append the DataFrame to a CSV file
        df.to_csv(f'{fig_dir}/{metrics}.csv',
                  mode='a', index=False, header=not file_exists)

        ## Output average metrics to a CSV file
        # Check if it's the last iteration of the loop
        if i == len(df_meta.index) - 1:
            # Read the metrics file into a DataFrame
            df = pd.read_csv(f'{fig_dir}/{metrics}.csv')
            # Calculate the average of each column
            averages = df.drop(columns='Building ID').mean().round(6)
            # Convert the Series to a DataFrame and transpose it
            averages_df = pd.DataFrame(averages)
            # Write the DataFrame to a CSV file
            averages_df.to_csv(f'{fig_dir}/averages_{prefix}metrics{upgrade_tag}_{start_index:04}-{end_index:04}.csv', header=False)

    # Output and/or show the fit plot
    if sw_save_fit or sw_show_fit:
        if yr_type == 'targ':
            # Create a new figure with a specified size
            plt.figure(figsize=(100, 10))  # width and height in inches
            # Plot Y and Y_pred
            plt.plot(Y, label='Actual')
            plt.plot(Y_pred, label='Predicted')
            # Set the title and labels
            plt.title('HVAC.elec over time')
            plt.xlabel('Time')
            plt.ylabel('HVAC.elec')
            # Show the legend
            plt.legend()
            if sw_save_fit:
                plt.savefig(f'{fig_dir}/{prefix}up{upgrade_tag}_{str(bldg_id)}.png')
            if sw_show_fit:
                plt.show()

        # Create a scatter plot of the actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(Y_test, Y_pred, color='blue')
        plt.plot([min(Y_test), max(Y_test)], 
                 [min(Y_test), max(Y_test)], color='red')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        if sw_save_fit:
            plt.savefig(f'{fig_dir}/fit_{prefix}up{upgrade_tag}_{str(bldg_id)}.png')
        if sw_show_fit:
            plt.show()

def prediction(base_year, df_eulp, sw_test_base, target_years, sw_test_target,
               energy_type, weather_data_base, weather_data_target, bldg_id,
               df_eulp_targ, target_year_by_row):
    """
    Predict hourly energy consumption for a single building by fitting once on
    base-year weather and predicting over concatenated target-year weather.

    Args:
        base_year (int): The base year for training the prediction model.
        df_eulp (DataFrame): df containing the electricity consumption data.
        sw_test_base (bool): Whether to test/evaluate the random forest model.
        target_years (list[int]): Target years for weather-based prediction.
        sw_test_target (bool): Whether to test/evaluate the random forest model.
        energy_type (str): 'HVAC.elec' or 'natural_gas.heating.energy_consumption'
        weather_data_base (DataFrame): The weather data for the base year.
        weather_data_target (DataFrame): Concatenated weather for all target
            years.
        bldg_id (str): The building ID for which to make the prediction.
        df_eulp_targ (DataFrame): Electricity consumption data for target year.
        target_year_by_row (np.ndarray): Per-row target year labels aligned to
            weather_data_target.

    Returns:
        predictions (DataFrame): Predicted hourly consumption across all target
            years, concatenated in target_years order.
    """

    # Drop the 1st entry (1:00a) of the EULP data bc the weather data starts at
    # 1:00a (instantaneous) and the EULP data for 2:00a is 1:00a to 2:00a
    df_eulp = df_eulp.iloc[1:]
    Y = df_eulp[energy_type].reset_index(drop=True)

    # Predict on one concatenated target-weather matrix.
    X_Predict = weather_data_target

    # Return zeros if sum of base year EULP is near zero
    is_zero_base = Y.sum() <= 0.01
    if is_zero_base:
        predictions = np.zeros(len(X_Predict))
        print(f'{energy_type} is all zeros for {target_years} for {bldg_id}.')
        rf_model = None
    else:
        # Pull in the weather data for the base year
        X = weather_data_base
        # Drop final row so the lengths of the weather and EULP data match
        X = X.iloc[:-1]
        startTime = dt.datetime.now()

        # Train random forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        if sw_cross_val:
            # Perform 5-fold cross-validation
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            results = cross_val_score(rf_model, X, Y, cv=kfold, scoring='neg_mean_squared_error')
            print(f'Cross-validated MSE: {results.mean()}')
            rf_model.fit(X, Y)
            print('Finished Random Forest Model Training: '+ str(dt.datetime.now() - startTime))
            if sw_test_base or sw_save_metrics:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                Y_pred = rf_model.predict(X_test)
                test_fit('base', base_year, prefix, upgrade_tag, bldg_id, rf_model, Y_test,
                        Y_pred, X_train, sw_save_metrics, output_dir, sw_save_fit,
                        sw_show_fit, i, df_meta, Y, start_index, end_index,
                        energy_type)
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
                                                                random_state=42)
            rf_model.fit(X_train, Y_train)
            print('Finished Random Forest Model Training: '+ str(dt.datetime.now() - startTime))
            if sw_test_base or sw_save_metrics:
                # Make predictions on the test data
                Y_pred = rf_model.predict(X_test)
                test_fit('base', base_year, prefix, upgrade_tag, bldg_id, rf_model,
                         Y_test, Y_pred, X_train, sw_save_metrics, output_dir,
                         sw_save_fit, sw_show_fit, i, df_meta, Y, start_index,
                         end_index, energy_type)
        rf_predictions = rf_model.predict(X_Predict)

        # Determine if predictions require extrapolation using the Neural Network
        min_train = X['Dry Bulb Temperature [°C]'].min()
        max_train = X['Dry Bulb Temperature [°C]'].max()
        min_predict = X_Predict['Dry Bulb Temperature [°C]'].min()
        max_predict = X_Predict['Dry Bulb Temperature [°C]'].max()
        needs_extrapolation = min_predict < min_train or max_predict > max_train

        # If hybrid model is on and extrapolation needed, use RFR + NN (v just RFR)
        if sw_hybrid_model and needs_extrapolation:
            print('Extrapolation required, employing hybrid model.')
            scaler = StandardScaler()
            X_scale = scaler.fit_transform(X)

            # Define a simple neural network model
            tf = _ensure_tf()
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    128,
                    activation='relu',
                    input_shape=(X_scale.shape[1],),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
                ),
                tf.keras.layers.Dense(
                    64,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
                ),
                tf.keras.layers.Dense(
                    1,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
                )
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')

            startTime = dt.datetime.now()
            count_rf = 0
            count_nn = 0

            if sw_cross_val:
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                for train_index, test_index in kfold.split(X_scale):
                    X_train, X_test = X_scale[train_index], X_scale[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]
                    model.fit(X_train, Y_train, epochs=50, batch_size=10, verbose=0)
                    mse = model.evaluate(X_test, Y_test, verbose=0)
                    print(f'Fold MSE: {mse}')
                print('Finished Neural Network Training with Cross Validation: ' + str(dt.datetime.now() - startTime))
            else:
                X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.2, random_state=42)
                model.fit(X_train, Y_train, epochs=50, batch_size=10)
                print('Finished Neural Network Training without Cross Validation: ' + str(dt.datetime.now() - startTime))

            X_Predict_scale = scaler.transform(X_Predict)
            nn_predictions = model.predict(X_Predict_scale).flatten()

            predictions = []
            for j in range(len(X_Predict)):
                if ((X_Predict['Dry Bulb Temperature [°C]'].iloc[j] >= min_train) & (X_Predict['Dry Bulb Temperature [°C]'].iloc[j] <= max_train)):
                    predictions.append(rf_predictions[j])
                    count_rf += 1
                else:
                    predictions.append(nn_predictions[j])
                    count_nn += 1
            print('Total extrapolation percentage: ' + str((count_nn / (count_rf + count_nn)) * 100) + '%')

        else:
            predictions = rf_predictions
            print('Using random forest predictions as no extrapolation is needed.')

    predictions = np.asarray(predictions, dtype=float)

    # Align predictions to hour-end load reporting by year block:
    # value at 02:00 uses weather at 01:00, so shift each year forward by 1 hour.
    shifted = predictions.copy()
    for yr in np.unique(target_year_by_row):
        mask = (target_year_by_row == yr)
        vals = predictions[mask]
        if len(vals) == 0:
            continue
        shifted_vals = vals.copy()
        if len(vals) > 1:
            shifted_vals[1:] = vals[:-1]
        # Fill Jan 1 01:00 with the first modeled value for that year.
        shifted_vals[0] = vals[0]
        shifted[mask] = shifted_vals

    # Build one continuous timestamp vector across all requested target years.
    target_timestamps = []
    for yr in target_years:
        n_rows = int((target_year_by_row == yr).sum())
        target_timestamps.extend(
            pd.date_range(
                start=dt.datetime(yr, 1, 1, 1, 0, 0),
                periods=n_rows,
                freq='H'
            )
        )

    predictions = pd.DataFrame({
        'timestamp': pd.to_datetime(target_timestamps),
        energy_type: shifted
    })
    # Energy consumption cannot be negative.
    predictions[energy_type] = predictions[energy_type].clip(lower=0)

    if is_zero_base:
        return predictions

    if sw_test_target:
        if len(target_years) != 1:
            raise ValueError(
                "sw_test_target=True requires exactly one target year."
            )
        # Subset the target year EULP data to the building ID
        df_eulp_targ_bldg = df_eulp_targ.loc[bldg_id].copy()

        Y_test = df_eulp_targ_bldg[energy_type]
        Y = df_eulp_targ_bldg[energy_type].reset_index(drop=True)
        Y_pred = predictions[energy_type]

        test_fit('targ', target_years[0], prefix, upgrade_tag, bldg_id, rf_model, Y_test,
                 Y_pred, X_Predict, sw_save_metrics, output_dir, sw_save_fit,
                 sw_show_fit, i, df_meta, Y, start_index, end_index,
                 energy_type)
    return predictions


# MAIN
# Load the metadata DataFrame
df_meta = pd.read_csv(meta_path)

# Set `county` based on `sw_comstock` value (used for output collapse)
county = 'in.nhgis_county_gisjoin' if sw_comstock else 'in.county'

# Weather-location column: each unique value owns one EPW and gets one
# trained RF per energy type. For ComStock 2025.2 this is the as_sim
# county; for ResStock and older ComStock it's the county itself, in
# which case the per-loc loop degenerates 1-to-1 with the per-county
# loop the old code used.
weather_col = (
    'in.as_simulated_nhgis_county_gisjoin'
    if (sw_comstock and comstock_year == "2025" and comstock_release == "2")
    else county
)

# Subset df_meta to the specified weather locations (passed by B in manifest)
weather_locs = weather_locs_str.split('_')
df_meta = df_meta[df_meta[weather_col].isin(weather_locs)]

# Get the unique states in the metadata DataFrame for process_chunk_agg fxn
chunk_states = df_meta['in.state'].unique().tolist()

# Set index of df_meta to 'bldg_id'
df_meta = df_meta.set_index('bldg_id')

def _sum_across_upgrades(frames):
    """Collapse a list of per-upgrade ts_agg frames into one by summing on
    (bldg_id, timestamp).

    Each frame is bldg_id-indexed with a `timestamp` column plus the
    eight numeric energy columns. Buildings absent from any given
    upgrade's pull simply don't contribute to its sum, which is the
    correct identity. Returns a frame with the same shape and bldg_id
    index as the input frames.
    """
    if len(frames) == 1:
        return frames[0]
    numeric_cols = [c for c in frames[0].columns if c != 'timestamp']
    combined = pd.concat(frames).reset_index()  # bldg_id becomes a column
    summed = (
        combined.groupby(['bldg_id', 'timestamp'], as_index=False)[numeric_cols]
        .sum()
    )
    return summed.set_index('bldg_id')


# Call process_chunk_agg once per upgrade_id in the spec (typically just one;
# a list signals "sum the resulting timeseries across these upgrades"). The
# restrict/avoid predicates from the spec apply identically to every pull.
_base_frames = []
for upg in upgrade_ids:
    _base_frames.append(
        process_chunk_agg(
            base_run, upg, weather_locs, weather_col, bsq_cols, sw_comstock,
            chunk_states, sw_savings_shape, df_meta, applied_only,
            spec_restrict=spec_restrict_filter,
            spec_avoid=spec_avoid_filter,
        )
    )
ts_agg = _sum_across_upgrades(_base_frames)

# Grab the target year AWS data if sw_test_target else set as None. Same
# multi-upgrade summation rule as the base pull.
if sw_test_target and sw_apply_regression:
    _targ_frames = []
    for upg in upgrade_ids:
        _targ_frames.append(
            process_chunk_agg(
                target_run, upg, weather_locs, weather_col, bsq_cols,
                sw_comstock, chunk_states, sw_savings_shape, df_meta,
                applied_only, query_label='target',
                spec_restrict=spec_restrict_filter,
                spec_avoid=spec_avoid_filter,
            )
        )
    df_eulp_targ = _sum_across_upgrades(_targ_frames)
else:
    df_eulp_targ = None

# Restrict/avoid on the spec can shrink BSQ's result set. df_meta still
# carries the full upgrade roster from B's parquet load, so we prune it
# down to bldg_ids that actually came back in ts_agg. Without this, the
# per-loc share denominators (loc_annual_*) would include orphan
# AWS_*=NaN entries and propagate NaN through the share-out math.
_bldg_ids_in_ts = ts_agg.index.unique()
_n_before = len(df_meta)
df_meta = df_meta.loc[df_meta.index.isin(_bldg_ids_in_ts)]
print(
    f'df_meta filtered to bldg_ids present in ts_agg: '
    f'{_n_before} -> {len(df_meta)} (dropped {_n_before - len(df_meta)})'
)
if df_meta.empty:
    raise RuntimeError(
        'No bldg_ids matched BSQ ts_agg AND df_meta. '
        'Check spec restrict/avoid filters and the chunk weather_locs.'
    )

# Error check: Sum AWS cooling/heating/ng timeseries data for each bldg_id.
# AWS_HVAC.elec is the derived sum (cool + heat) — used by the meta_HVAC.elec
# ratio diagnostics further down to catch metering / column-mapping errors.
df_meta['AWS_cooling.elec'] = ts_agg.groupby('bldg_id').apply(
    lambda x: x['cooling.elec'].iloc[:8760].sum()
)
df_meta['AWS_heating.elec'] = ts_agg.groupby('bldg_id').apply(
    lambda x: x['heating.elec'].iloc[:8760].sum()
)
df_meta['AWS_HVAC.elec'] = (
    df_meta['AWS_cooling.elec'] + df_meta['AWS_heating.elec']
)
df_meta['AWS_natural_gas.heating.energy_consumption'] = (
    ts_agg.groupby('bldg_id').apply(
        lambda x: x['natural_gas.heating.energy_consumption'].iloc[:8760].sum()
    )
)
df_meta['AWS_non_hvac.elec'] = ts_agg.groupby('bldg_id').apply(
    lambda x: x['non_hvac.elec'].iloc[:8760].sum()
)

if sw_apply_regression:
    # === Per-weather-location training, then per-bldg_id share-out ===
    #
    # Reference math (per energy type — same shape for cool/heat/NG):
    #     M(a) = sum_{b: loc(b)=a} m(b)            # location annual
    #     share(b) = m(b) / M(a)                    # 0 if M(a) == 0
    #     predicted_bldg(b, h) = predicted_loc(loc(b), h) * share(b)
    # where m(b) is the AWS_* annual for bldg_id b (populated above from
    # the BSQ ts_agg we pulled). Each energy type uses its OWN share
    # (a cooling-heavy bldg gets most of the cool prediction even if it
    # uses no heat, and vice versa).

    # Map bldg_id -> weather_loc once. Used for grouping ts_agg and for the
    # share-out loop. df_meta is indexed by bldg_id and unique.
    bldg_to_loc = df_meta[weather_col].to_dict()

    # Per-loc training Y: sum cool/heat/NG hourly across all bldg_ids that
    # share the location. These are the per-loc aggregate hourly profiles
    # the regression learns to predict (one RF per energy type per loc).
    _y_cols = ['cooling.elec', 'heating.elec',
               'natural_gas.heating.energy_consumption']
    ts_agg_pl = ts_agg.copy()
    ts_agg_pl['__loc'] = ts_agg_pl.index.map(bldg_to_loc)
    ts_agg_per_loc = (
        ts_agg_pl.groupby(['__loc', 'timestamp'], as_index=False)[_y_cols]
        .sum()
        .sort_values(['__loc', 'timestamp'])
        .reset_index(drop=True)
    )

    # Share denominators: the location's annual sums per energy type.
    loc_annual_cool = df_meta.groupby(weather_col)['AWS_cooling.elec'].sum()
    loc_annual_heat = df_meta.groupby(weather_col)['AWS_heating.elec'].sum()
    loc_annual_ng = df_meta.groupby(weather_col)[
        'AWS_natural_gas.heating.energy_consumption'
    ].sum()

    # If sw_test_target is on we also need the actual target-year EULP
    # aggregated to the loc level so prediction()'s test-fit branch can
    # compare apples-to-apples.
    if df_eulp_targ is not None:
        df_eulp_targ_pl = df_eulp_targ.copy()
        df_eulp_targ_pl['__loc'] = df_eulp_targ_pl.index.map(bldg_to_loc)
        df_eulp_targ_per_loc = (
            df_eulp_targ_pl.groupby(['__loc', 'timestamp'], as_index=False)[_y_cols]
            .sum()
            .sort_values(['__loc', 'timestamp'])
            .set_index('__loc')
        )
    else:
        df_eulp_targ_per_loc = None

    unique_locs = df_meta[weather_col].unique().tolist()

    def _build_loc_tasks():
        for loc in unique_locs:
            df_eulp_pred_loc = (
                ts_agg_per_loc[ts_agg_per_loc['__loc'] == loc]
                .drop(columns=['__loc'])
                .reset_index(drop=True)
            )
            df_eulp_targ_for_loc = (
                df_eulp_targ_per_loc.loc[[loc]]
                if df_eulp_targ_per_loc is not None else None
            )
            yield (
                loc,
                df_eulp_pred_loc,
                df_eulp_targ_for_loc,
                base_year,
                target_years,
                sw_test_base,
                sw_test_target,
            )

    # Train + predict per loc (cool/heat/NG all in one worker call).
    pred_per_loc = {}
    if _is_hpc():
        tasks = list(_build_loc_tasks())
        # Pool size: read SLURM_CPUS_PER_TASK so this scales between standard
        # (48) and bigmem (104) profiles set by B at sbatch time. Falls back
        # to os.cpu_count() then 48 if neither is available.
        procs = int(
            os.environ.get('SLURM_CPUS_PER_TASK')
            or os.cpu_count()
            or 48
        )
        print(
            f'Using {procs} processes for {len(tasks)} per-loc regressions '
            f'out of {os.cpu_count()} possible.'
        )
        with ProcessPoolExecutor(max_workers=procs) as ex:
            futures = [ex.submit(_process_one_location, t) for t in tasks]
            for fut in as_completed(futures):
                loc, df_cool, df_heat, df_ng, cool_sum, heat_sum, ng_sum = fut.result()
                pred_per_loc[loc] = (df_cool, df_heat, df_ng,
                                     cool_sum, heat_sum, ng_sum)
    else:
        # Serial path. `i` is referenced inside test_fit (loop counter
        # heuristic for "is this the last building?"); per-loc training
        # makes that heuristic less meaningful, so we set sw_save_metrics
        # off implicitly via the worker.
        for t in _build_loc_tasks():
            loc, df_cool, df_heat, df_ng, cool_sum, heat_sum, ng_sum = (
                _process_one_location(t)
            )
            pred_per_loc[loc] = (df_cool, df_heat, df_ng,
                                 cool_sum, heat_sum, ng_sum)

    # Share-out: distribute predicted hourly per-loc to each bldg_id, with
    # independent shares for cooling, heating, and NG.
    df_bldg_cool = []
    df_bldg_heat = []
    for bldg_id in df_meta.index:
        loc = df_meta.loc[bldg_id, weather_col]
        df_cool_loc, df_heat_loc, df_ng_loc, cool_loc_sum, heat_loc_sum, ng_loc_sum = pred_per_loc[loc]

        m_cool = float(df_meta.loc[bldg_id, 'AWS_cooling.elec'])
        M_cool = float(loc_annual_cool.loc[loc])
        share_cool = (m_cool / M_cool) if M_cool > 0 else 0.0

        m_heat = float(df_meta.loc[bldg_id, 'AWS_heating.elec'])
        M_heat = float(loc_annual_heat.loc[loc])
        share_heat = (m_heat / M_heat) if M_heat > 0 else 0.0

        m_ng = float(df_meta.loc[bldg_id, 'AWS_natural_gas.heating.energy_consumption'])
        M_ng = float(loc_annual_ng.loc[loc])
        share_ng = (m_ng / M_ng) if M_ng > 0 else 0.0

        # Cooling hourly per bldg_id
        df_c = df_cool_loc[['timestamp']].copy()
        df_c[bldg_id] = df_cool_loc['cooling.elec'].values * share_cool
        df_c = df_c.rename(columns={'timestamp': 'timestamp_EST'}).set_index('timestamp_EST')
        df_bldg_cool.append(df_c)

        # Heating hourly per bldg_id
        df_h = df_heat_loc[['timestamp']].copy()
        df_h[bldg_id] = df_heat_loc['heating.elec'].values * share_heat
        df_h = df_h.rename(columns={'timestamp': 'timestamp_EST'}).set_index('timestamp_EST')
        df_bldg_heat.append(df_h)

        # df_meta annuals (used by the diagnostic ratio columns below).
        # HVAC.elec is the derived sum, kept for the existing meta_HVAC.elec
        # ratio diagnostics (metadata has no separate cool/heat reference).
        df_meta.loc[bldg_id, 'cooling.elec'] = round(cool_loc_sum * share_cool, 6)
        df_meta.loc[bldg_id, 'heating.elec'] = round(heat_loc_sum * share_heat, 6)
        df_meta.loc[bldg_id, 'HVAC.elec'] = (
            df_meta.loc[bldg_id, 'cooling.elec']
            + df_meta.loc[bldg_id, 'heating.elec']
        )
        df_meta.loc[bldg_id, 'natural_gas.heating.energy_consumption'] = (
            round(ng_loc_sum * share_ng, 6)
        )

    df_eulp_cool = pd.concat(df_bldg_cool, axis=1)
    df_eulp_heat = pd.concat(df_bldg_heat, axis=1)

    # Non-HVAC electricity passes through unregressed (per spec). Vectorized
    # assignment outside the share-out loop — value is just AWS_non_hvac.elec.
    df_meta['non_hvac.elec'] = df_meta['AWS_non_hvac.elec']

    # Build the per-bldg non_hvac.elec hourly frame from raw ts_agg (base-year
    # EULP), then REPLICATE the 8760-row base frame across every target year
    # so the row layout matches df_eulp_cool / df_eulp_heat (which are
    # target-year RF predictions). Without replication the three chunk CSVs
    # would span different model years and downstream stitching would have
    # to special-case non_hvac.
    ts_agg_nh = ts_agg[ts_agg.index.isin(df_meta.index)].reset_index()
    ts_agg_nh.rename(columns={'timestamp': 'timestamp_EST'}, inplace=True)
    _df_nh_base = ts_agg_nh.pivot(
        index='timestamp_EST', columns='bldg_id', values='non_hvac.elec'
    )
    # Cap at EULP's 8760 hours/year convention (drops leap-day rows when
    # base_year is a leap year; cool/heat targets are likewise capped to
    # 8760 per year by _trim_to_8760_per_year below).
    _df_nh_base = _df_nh_base.iloc[:8760]

    # Roll the base-year hourly profile so its day-of-week pattern aligns
    # with each target year's calendar. Without this, copying e.g. 2018's
    # Mon-Tue-Wed-... block into 2020 (a Wednesday-start year) puts
    # weekday-sensitive loads (lighting / plug / appliance) on the wrong
    # day. Same algorithm as E_combine_nonHVAC.match_day_patterns:
    # shift = (base_jan1_dow - target_jan1_dow) * 24, then np.roll. The
    # 8760-cap above means no leap-day padding is needed (cool/heat are
    # likewise 8760/yr, so they don't have a leap-day to align to).
    # Drift footprint: the wrap caused by np.roll is at most 144 hours
    # (6 days) and lands at year-boundaries, where it interacts with the
    # other end of the rolled profile — small and contained.
    _base_values = _df_nh_base.values
    _base_jan1_dow = _df_nh_base.index[0].dayofweek
    _nh_year_frames = []
    for _yr in target_years:
        _yr_idx = pd.date_range(
            start=dt.datetime(_yr, 1, 1, 1, 0, 0),
            periods=len(_df_nh_base),
            freq='h',
        )
        _shift_hours = (_base_jan1_dow - _yr_idx[0].dayofweek) * 24
        _rolled = np.roll(_base_values, _shift_hours, axis=0)
        _yr_frame = pd.DataFrame(
            _rolled, index=_yr_idx, columns=_df_nh_base.columns
        )
        _nh_year_frames.append(_yr_frame)
    df_eulp_non_hvac = pd.concat(_nh_year_frames)
    df_eulp_non_hvac.index.name = 'timestamp_EST'

else:
    # If not applying regression, duplicate annual cool/heat/NG columns
    df_meta['cooling.elec'] = df_meta['AWS_cooling.elec']
    df_meta['heating.elec'] = df_meta['AWS_heating.elec']
    df_meta['HVAC.elec'] = df_meta['AWS_HVAC.elec']
    df_meta['natural_gas.heating.energy_consumption'] = (
        df_meta['AWS_natural_gas.heating.energy_consumption']
    )
    df_meta['non_hvac.elec'] = df_meta['AWS_non_hvac.elec']

    # Filter ts_agg to include only rows with bldg_id's in df_meta.index
    ts_agg = ts_agg[ts_agg.index.isin(df_meta.index)]

    # Create timeseries x bldg_id DataFrames (cool / heat / non-HVAC).
    ts_agg = ts_agg.reset_index()
    ts_agg.rename(columns={'timestamp': 'timestamp_EST'}, inplace=True)
    df_eulp_cool = ts_agg.pivot(index='timestamp_EST', columns='bldg_id',
                                values='cooling.elec')
    df_eulp_heat = ts_agg.pivot(index='timestamp_EST', columns='bldg_id',
                                values='heating.elec')
    df_eulp_non_hvac = ts_agg.pivot(index='timestamp_EST', columns='bldg_id',
                                    values='non_hvac.elec')

# Collapse bldg_id (county/sim-county) columns to county columns, separately
# for the cooling, heating, and non-HVAC frames. All three share the same
# bldg_id column set so the county_labels lookup is shared.
county_labels = df_meta.loc[df_eulp_cool.columns, county].astype(str)
df_eulp_cool = df_eulp_cool.T.groupby(county_labels).sum().T
df_eulp_heat = df_eulp_heat.T.groupby(county_labels).sum().T
df_eulp_non_hvac = df_eulp_non_hvac.T.groupby(county_labels).sum().T

# Aggregate df_meta to county-level before diagnostics.
# Drop sim-county column to avoid string concatenation during groupby sum.
df_meta = df_meta.drop(columns=['in.as_simulated_nhgis_county_gisjoin'],
                       errors='ignore')
df_meta = (
    df_meta.groupby([county, 'in.county_name', 'in.state'], as_index=False)
    .sum()
    .set_index(county)
)

# Error checking using ratios and percent differences in df_meta
## Ratios (note: small_number is to avoid division by zero)
df_meta['ratio_HVAC_AWS_meta'] = (
    df_meta['AWS_HVAC.elec'] /
    df_meta['meta_HVAC.elec']).round(4)
df_meta['ratio_ng_AWS_meta'] = (
    df_meta['AWS_natural_gas.heating.energy_consumption'] /
    df_meta['meta_natural_gas.heating.energy_consumption']).round(4)

df_meta['ratio_HVAC_reg_meta'] = (
    df_meta['HVAC.elec'] /
    df_meta['meta_HVAC.elec']).round(4)
df_meta['ratio_ng_reg_meta'] = (
    df_meta['natural_gas.heating.energy_consumption'] /
    df_meta['meta_natural_gas.heating.energy_consumption']).round(4)

df_meta['ratio_HVAC_reg_AWS'] = (
    df_meta['HVAC.elec'] /
    df_meta['AWS_HVAC.elec']).round(4)
df_meta['ratio_ng_reg_AWS'] = (
    df_meta['natural_gas.heating.energy_consumption'] /
    df_meta['AWS_natural_gas.heating.energy_consumption']).round(4)

## Percent differences (note: small_number is to avoid division by zero)
df_meta['diff_HVAC_AWS_meta'] = (100 * (
    (df_meta['AWS_HVAC.elec'] - df_meta['meta_HVAC.elec']) /
    df_meta['meta_HVAC.elec'])).round(4)
df_meta['diff_ng_AWS_meta'] = (100 * (
    (df_meta['AWS_natural_gas.heating.energy_consumption'] -
     df_meta['meta_natural_gas.heating.energy_consumption']) /
    df_meta['meta_natural_gas.heating.energy_consumption'])).round(4)

df_meta['diff_HVAC_reg_meta'] = (100 * (
    (df_meta['HVAC.elec'] - df_meta['meta_HVAC.elec']) /
    df_meta['meta_HVAC.elec'])).round(4)
df_meta['diff_ng_reg_meta'] = (100 * (
    (df_meta['natural_gas.heating.energy_consumption'] -
     df_meta['meta_natural_gas.heating.energy_consumption']) /
    df_meta['meta_natural_gas.heating.energy_consumption'])).round(4)

df_meta['diff_HVAC_reg_AWS'] = (100 * (
    (df_meta['HVAC.elec'] - df_meta['AWS_HVAC.elec']) /
    df_meta['AWS_HVAC.elec'])).round(4)
df_meta['diff_ng_reg_AWS'] = (100 * (
    (df_meta['natural_gas.heating.energy_consumption'] -
     df_meta['AWS_natural_gas.heating.energy_consumption']) /
    df_meta['AWS_natural_gas.heating.energy_consumption'])).round(4)

df_meta.rename(columns={
    "in.geometry_building_type_recs": "in.building_type",
    "in.comstock_building_type": "in.building_type",
    "in.county": "in.county_nhgis",
    "in.nhgis_county_gisjoin": "in.county_nhgis",
}, errors='ignore', inplace=True)

# Create 'gas_heating_MWh' column and insert after bsq_cols & sqft columns
df_meta.insert(len(bsq_cols) + 1, 'gas_heating_MWh',
               df_meta['natural_gas.heating.energy_consumption'])

# Save metadata DataFrame to CSV file
df_meta.to_csv(os.path.join(chunks_meta_dir,
    f'{prefix}meta_upgrade{upgrade_tag}_{start_index:04}-{end_index:04}.csv'))

# Round and trim each frame to 8760 hourly rows per model year, then write
# separate chunk CSVs for cooling.elec, heating.elec, and non_hvac.elec.
# Downstream (agg_buildings.py) stitches each enduse independently.
def _trim_to_8760_per_year(df):
    df = df.round(6)
    model_year = (df.index - pd.Timedelta(hours=1)).year
    return df.groupby(model_year, group_keys=False).head(8760)

df_eulp_cool = _trim_to_8760_per_year(df_eulp_cool)
df_eulp_heat = _trim_to_8760_per_year(df_eulp_heat)
df_eulp_non_hvac = _trim_to_8760_per_year(df_eulp_non_hvac)

df_eulp_cool.to_csv(os.path.join(chunks_eulp_dir,
    f'{prefix}eulp_cooling_elec_MWh_upgrade{upgrade_tag}_'
    f'{start_index:04}-{end_index:04}.csv'))
df_eulp_heat.to_csv(os.path.join(chunks_eulp_dir,
    f'{prefix}eulp_heating_elec_MWh_upgrade{upgrade_tag}_'
    f'{start_index:04}-{end_index:04}.csv'))
df_eulp_non_hvac.to_csv(os.path.join(chunks_eulp_dir,
    f'{prefix}eulp_non_hvac_elec_MWh_upgrade{upgrade_tag}_'
    f'{start_index:04}-{end_index:04}.csv'))

print('\nChunk done at:', dt.datetime.now())
print('Total time elapsed:', dt.datetime.now() - script_start_time)
