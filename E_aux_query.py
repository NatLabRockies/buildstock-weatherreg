"""
E_aux_query.py — produce auxiliary files for each baseline-cohort spec.

For each run_spec with `upgrade_id == 0` in switches_agg.json, query baseline
metadata with the spec's restrict/avoid filter and write two CSVs per spec:

  1) aux_coverage_upgrade<spec_tag>.csv
       Columns: <county_key_cols>, sqft, units_count
       One row per county-key with the cohort's weighted sqft and units_count
       (auto-injected by BSQ as sum of per-building sample weights).

  2) aux_samples_upgrade<spec_tag>.csv
       Columns: <county_key_cols>, bldg_id, sqft, weight
       One row per individual building in the cohort. Sum(weight) grouped by
       county-key reproduces units_count in the coverage file.

Specs with non-zero `upgrade_id` are skipped: they share their cohort with a
baseline (`upgrade_id=0`) spec carrying the same restrict/avoid filter, so
their aux files would be exact duplicates.

Output filename convention matches B / D / agg_buildings.py:
  spec_tag = <spec['name']>_<reg|ref>_b<base_year>

Usage:
  python E_aux_query.py <output_dir> [--spec-index N]

  <output_dir>: A run output directory containing inputs/switches_agg.json.
                The aux CSVs are written at the top level of this directory.
  --spec-index N: Process only run_specs[N] (must have upgrade_id == 0).
"""

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time

import certifi

_CA = certifi.where()
os.environ.setdefault("AWS_CA_BUNDLE", _CA)
os.environ.setdefault("CURL_CA_BUNDLE", _CA)
os.environ.setdefault("SSL_CERT_FILE", _CA)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _CA)
import ssl  # noqa: E402
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd  # noqa: E402
from buildstock_query import BuildStockQuery  # noqa: E402


_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9.\-]+$")


def _validate_spec_name(name):
    """Same rule as D_process_chunk_agg / B_building_stock_parallel_agg."""
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


_TRANSIENT_MARKERS = (
    "HIVE_S3_THROTTLING",
    "Status Code: 503",
    "Status Code: 429",
    "SlowDown",
    "TooManyRequests",
    "Throttling",
    "ThrottlingException",
    "RequestLimitExceeded",
    "timed out",
    "TimeoutError",
    "Connection reset",
    "BrokenPipe",
    "ServiceUnavailable",
)


def _execute_with_retry(my_run, query, max_attempts=6, base_delay=30, max_delay=600):
    """Execute an Athena query with backoff+jitter on transient failures.

    Same shape as D_process_chunk_agg.query_execution; transient markers cover
    Athena/S3 throttling and TLS/socket blips. Non-transient errors fail fast.
    """
    for attempt in range(max_attempts):
        try:
            return my_run.execute(query)
        except Exception as e:
            err = str(e).lower()
            is_last = attempt == max_attempts - 1
            is_transient = any(m.lower() in err for m in _TRANSIENT_MARKERS)
            print(f"  attempt {attempt + 1}/{max_attempts} failed: {e}")
            if is_last or not is_transient:
                raise
            base = min(base_delay * (2 ** attempt), max_delay)
            sleep_for = max(1.0, base + base * 0.5 * (2 * random.random() - 1))
            print(f"  transient error; sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)


def _build_filter(my_run, predicate):
    """Translate a spec restrict/avoid dict to BSQ's applied-filter tuple.

    `predicate` is `{"all_of": [...], "any_of": [...]}` or None. Returns a
    RestrictTuple (the kind BSQ accepts in `restrict=`/`avoid=`) or None.
    """
    if not predicate:
        return None
    return my_run.get_applied_buildings_filter(
        any_of=predicate.get("any_of"),
        all_of=predicate.get("all_of"),
    )


# Stock-specific column names. BSQ's `query()` accepts these as bare names in
# group_by; the result frame has them as columns (no 'in.' prefix in output).
# Note: ComStock 2025.2's `as_simulated_nhgis_county_gisjoin` (the weather
# location a sampled building was simulated against) is intentionally NOT
# included — the projection logic cares about source-county only, so buildings
# in the same source-county but assigned to different as-sim weather locations
# aggregate together in coverage.
RES_COUNTY_KEYS = ["county", "county_name", "state"]
COM_COUNTY_KEYS = ["nhgis_county_gisjoin", "county_name", "state"]

# Per-building sqft column. Same enduse-style name used as an "enduse" so BSQ
# applies sample weighting and a SUM aggregator to it.
RES_SQFT_COL = "in.sqft..ft2"
COM_SQFT_COL = "in.sqft..ft2"  # most ComStock releases also expose ..ft2


def process_spec(spec, switch, output_dir):
    """Run the aux query for one spec and write the two CSVs."""
    spec_name = _validate_spec_name(spec["name"])
    regression_tag = "reg" if spec["apply_regression"] else "ref"
    spec_tag = f"{spec_name}_{regression_tag}_b{spec['base_year']}"

    sw_comstock = switch["comstock"]
    county_keys = COM_COUNTY_KEYS if sw_comstock else RES_COUNTY_KEYS
    sqft_col = COM_SQFT_COL if sw_comstock else RES_SQFT_COL
    run_types = switch["run_types"]
    base_run = spec["base_run"]
    aws_run_type = run_types[base_run].copy()

    spec_restrict = spec.get("restrict") or None
    spec_avoid = spec.get("avoid") or None

    print(f"[{dt.datetime.now():%H:%M:%S}] === spec {spec_name!r} | tag {spec_tag} ===")
    print(f"  base_run={base_run}  restrict={spec_restrict}  avoid={spec_avoid}")

    my_run = BuildStockQuery(**aws_run_type)

    applied_restrict = _build_filter(my_run, spec_restrict)
    applied_avoid = _build_filter(my_run, spec_avoid)
    restrict_list = [applied_restrict] if applied_restrict is not None else []
    avoid_list = [applied_avoid] if applied_avoid is not None else []

    # aux_coverage: county-key aggregate. Passing sqft as an "enduse" makes
    # BSQ sum it with sample-weights applied; the aggregator also auto-injects
    # `units_count` (= sum of weights per group).
    cov_sql = my_run.query(
        upgrade_id=0,
        enduses=[sqft_col],
        group_by=county_keys,
        annual_only=True,
        restrict=restrict_list,
        avoid=avoid_list,
        get_query_only=True,
    )
    print(f"  coverage SQL (first 240 chars): {str(cov_sql)[:240]}")
    t0 = time.time()
    cov = _execute_with_retry(my_run, cov_sql)
    print(f"  coverage: {len(cov)} county-key rows in {time.time() - t0:.1f}s")
    print(f"    raw columns returned: {list(cov.columns)}")

    # BSQ also returns `metadata_rows_count` (sum(1)) and `model_count`
    # (distinct bldg_id count); we drop them to keep the output narrow.
    cov = (
        cov.rename(columns={"sqft..ft2": "sqft"})
        .loc[:, county_keys + ["sqft", "units_count"]]
    )

    # aux_samples: per-building rows. Adding `bldg_id` (BSQ's name for the
    # per-building primary key; NOT `building_id`) to group_by makes each
    # output row exactly one building, and units_count becomes that building's
    # sample weight. Rename units_count -> weight to reflect the per-building
    # semantics.
    sample_sql = my_run.query(
        upgrade_id=0,
        enduses=[sqft_col],
        group_by=county_keys + ["bldg_id"],
        annual_only=True,
        restrict=restrict_list,
        avoid=avoid_list,
        get_query_only=True,
    )
    print(f"  samples SQL (first 240 chars): {str(sample_sql)[:240]}")
    t0 = time.time()
    samples = _execute_with_retry(my_run, sample_sql)
    print(f"  samples: {len(samples)} per-building rows in {time.time() - t0:.1f}s")

    samples = (
        samples.rename(columns={"sqft..ft2": "sqft", "units_count": "weight"})
        .loc[:, county_keys + ["bldg_id", "sqft", "weight"]]
    )

    cov_path = os.path.join(output_dir, f"aux_coverage_upgrade{spec_tag}.csv")
    samples_path = os.path.join(output_dir, f"aux_samples_upgrade{spec_tag}.csv")
    cov.to_csv(cov_path, index=False)
    samples.to_csv(samples_path, index=False)

    # Self-consistency: sum(samples.weight) per county-key must equal
    # coverage.units_count. The aggregator is deterministic so the deltas
    # should be exactly zero in practice; we use a small float tolerance to
    # absorb any presto-side ordering artifacts.
    cov_check = (
        samples.groupby(county_keys, observed=True, as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "units_count_from_samples"})
    )
    merged = cov.merge(cov_check, on=county_keys, how="outer")
    bad = merged[
        ~((merged["units_count"] - merged["units_count_from_samples"]).abs() < 1e-6)
    ]
    if not bad.empty:
        print(
            f"  WARNING: coverage.units_count != sum(samples.weight) on {len(bad)} "
            f"of {len(merged)} county-keys. Inspect e.g.:\n{bad.head(3).to_string(index=False)}"
        )
    else:
        print(
            f"  self-check OK: sum(samples.weight) matches coverage.units_count "
            f"on all {len(cov)} county-keys"
        )

    print(f"  wrote {cov_path}")
    print(f"  wrote {samples_path}")
    print(
        f"  totals: sqft={cov['sqft'].sum():,.0f}  "
        f"units_count={cov['units_count'].sum():,.0f}"
    )
    print()
    return cov_path, samples_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="Run output dir (must contain inputs/switches_agg.json).",
    )
    parser.add_argument(
        "--spec-index",
        type=int,
        default=None,
        help="Process only run_specs[N] (default: every spec).",
    )
    args = parser.parse_args(argv)

    snap = os.path.join(args.output_dir, "inputs", "switches_agg.json")
    if not os.path.isfile(snap):
        raise SystemExit(f"switches snapshot not found: {snap}")
    with open(snap) as f:
        switch = json.load(f)

    run_specs = switch["run_specs"]
    if args.spec_index is not None:
        if not 0 <= args.spec_index < len(run_specs):
            raise SystemExit(
                f"--spec-index {args.spec_index} out of range [0, {len(run_specs)})"
            )
        s = run_specs[args.spec_index]
        if s.get("upgrade_id") != 0:
            raise SystemExit(
                f"--spec-index {args.spec_index} ({s.get('name')!r}) has "
                f"upgrade_id={s.get('upgrade_id')!r}; aux files are only "
                "produced for upgrade_id=0 specs."
            )
        specs_to_run = [(args.spec_index, s)]
    else:
        specs_to_run = []
        for i, s in enumerate(run_specs):
            if s.get("upgrade_id") != 0:
                print(
                    f"[skip] spec[{i}] {s.get('name')!r}: "
                    f"upgrade_id={s.get('upgrade_id')!r} (only upgrade_id=0 specs "
                    "produce aux files)"
                )
                continue
            specs_to_run.append((i, s))

    if not specs_to_run:
        raise SystemExit("No specs with upgrade_id=0 found in run_specs.")

    t_all = time.time()
    for i, spec in specs_to_run:
        try:
            process_spec(spec, switch, args.output_dir)
        except Exception as e:
            print(f"  [spec {i}] FAILED: {type(e).__name__}: {e}")
            raise

    print(f"All specs done in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
