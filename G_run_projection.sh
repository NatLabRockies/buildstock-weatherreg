#!/bin/bash
#SBATCH --account=geohc
#SBATCH --time=02:00:00
#SBATCH --qos=high
#SBATCH --partition=standard
#SBATCH --mem=200G
#SBATCH --cpus-per-task=30

# Wall budget: parallelizes (spec, year, group) tasks across `cpus-per-task`
# workers via ProcessPoolExecutor — one task per green box of the projection
# diagrams, each writing that box's 4 enduse files (gap writes 1). ~90
# tasks/stock (com) / ~66 (res). The defaults below assume state resolution.
#
# Per-worker peak memory (input-frame size dominates):
#   state         → reads county-level agg (~4 GB) + state intermediate.
#                   ~4 GB/worker peak; 30 workers × 4 GB = 120 GB safe in 200 GB.
#   county_group  → ~6-12 GB/worker peak; cap at --cpus-per-task=20 → ~200 GB.
#   county        → ~20-40 GB/worker peak; --cpus-per-task=6, plus array fan-out.
#
# Multi-node fan-out via SLURM array. With `-a 0-N%K` sbatch flag, the same
# command is dispatched as N+1 array tasks and projection.py partitions the
# work-list by SLURM_ARRAY_TASK_ID across SLURM_ARRAY_TASK_COUNT — no shared
# state needed, idempotency is the back-stop. Each array task gets one
# standard node; standard has 2,176 nodes so scheduling is much faster than
# bigmem's ~10 nodes. Recommended array sizes:
#   state         → -a 0    (single node; 30 workers comfortably fits)
#   county_group  → -a 0-2  (3 nodes × 20 workers = 60 workers across tasks)
#   county        → -a 0-7  (8 nodes × 6 workers = 48 workers across tasks)
#
# The projection is idempotent — re-submitting after a TIMEOUT or with a
# different array-size skips already-written files. Workers across different
# array tasks process disjoint slices (`tasks[task_id::task_count]`) so
# there's no race even before idempotency kicks in.
#
# Usage:
#   sbatch G_run_projection.sh <run_dir>                       # auto stock, state res
#   sbatch G_run_projection.sh <run_dir> res                   # res, state res
#   sbatch --cpus-per-task=20 -a 0-2 G_run_projection.sh \
#       <run_dir> com county_group                              # com, county_group, 3-node fan
#   sbatch --cpus-per-task=6 --mem=240G -a 0-7 G_run_projection.sh \
#       <run_dir> com county                                    # com, county, 8-node fan
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
