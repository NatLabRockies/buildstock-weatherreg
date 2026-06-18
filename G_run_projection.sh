#!/bin/bash
#SBATCH --account=geohc
#SBATCH --time=02:00:00
#SBATCH --qos=high
#SBATCH --partition=bigmem
#SBATCH --mem=1024000
#SBATCH --cpus-per-task=40

# Wall budget: parallelizes (spec, year, group) tasks across 40 workers via
# ProcessPoolExecutor — one task per green box of the projection diagrams,
# each writing that box's 4 enduse files (gap writes 1). ~90 tasks/stock (com)
# / ~66 (res). 40-way parallelism brings the wall to ~10-20 minutes per stock;
# 2h provides ample headroom.
#
# Memory is resolution-dependent:
#   state         → ~49 cols (tiny); memory is a non-issue at 40 workers.
#   county_group  → ~1,038 cols (~1.4 GB/frame); ~6-12 GB peak per worker.
#   county        → ~3,100 cols (~4 GB/frame); ~20-40 GB peak. At 40 workers
#                   that can exceed the 1 TB node — drop to --cpus-per-task=24.
#
# Partition rationale: `bigmem` over `standard` for the memory; over `debug`
# because debug+qos=high enforces 1-job-per-user, preventing res/com parallel.
# The projection is idempotent — re-submitting after a TIMEOUT skips
# already-written projection files.
#
# Usage:
#   sbatch G_run_projection.sh <run_dir>                       # auto stock, state res
#   sbatch G_run_projection.sh <run_dir> res                   # res, state res
#   sbatch G_run_projection.sh <run_dir> com county            # com, county res
#   sbatch G_run_projection.sh <run_dir> "" county_group       # auto stock, county_group
#
# Output:
#   <run_dir>/projections_state/         (49 state cols, gap included)
#   <run_dir>/projections_county/        (~3,100 county-FIPS cols, gap from S3)
#   <run_dir>/projections_county_group/  (~1,038 BuildStock-group cols, gap excluded)
# All resolutions emit CSV.
#
# Post-projection handoffs (run AFTER projections are written, no SLURM needed):
#   python -m projections.reeds <res_run_dir> <com_run_dir>
#       → <res_run_dir>/reeds/<scenario>_y<year>.csv (state-aggregated total)
#   python -m projections.lbl <res_run_dir> <com_run_dir>
#       → <res_run_dir>/lbl/{<scenario>_<sector>_<cohort>_<year>_amy<weather>.csv,
#                            aux_samples_<scenario>_y<year>.csv}

run_dir="$1"
stock_override="$2"
resolution="${3:-state}"

if [ -z "$run_dir" ]; then
  echo "ERROR: usage: sbatch G_run_projection.sh <run_dir> [res|com] [state|county|county_group]" >&2
  exit 1
fi
if [ ! -d "$run_dir" ]; then
  echo "ERROR: run_dir not found: $run_dir" >&2
  exit 1
fi

# Ensure uv is on PATH (adjust if installed elsewhere)
export PATH="$HOME/.local/bin:$PATH"

# Limit per-WORKER BLAS threading to 1 so the ProcessPool doesn't fight itself
# for cores. Total parallelism = (cpus-per-task) workers × 1 BLAS thread.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Unbuffered Python so worker per-file completion messages flush live.
export PYTHONUNBUFFERED=1

# Repo path. Hard-coded because SLURM stages the wrapper into
# /var/spool/slurmd at runtime — `BASH_SOURCE[0]` then points at the staged
# copy, so the auto-resolve trick doesn't work for the projections/ package.
# cd here so `python -m projections` finds the package on the path. Same
# pattern the other wrappers in this repo use.
script_dir=/kfs2/projects/geohc/radhikar/weather_regression/buildstock-weatherreg

cd "$script_dir"

cmd=(uv run python -m projections "$run_dir" --resolution "$resolution")
if [ -n "$stock_override" ]; then
  cmd+=(--stock "$stock_override")
fi
echo "Running: ${cmd[*]}"
"${cmd[@]}"
