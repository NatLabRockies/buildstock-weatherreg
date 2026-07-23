#!/bin/bash
#SBATCH --account=geohc
#SBATCH --time=01:00:00
#SBATCH --qos=high
#SBATCH --partition=medmem
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# LBL handoff (long-format county-group timeseries + per-cohort samples) for ONE
# run_dir. Reads <run_dir>/projections_county_group/ and writes to
# <run_dir>/LBL/ (override with --out via env if needed). Parallelizes ~260
# source CSVs across 20 workers (~5-7 GB peak per worker).
#
# Usage:
#   sbatch H_run_lbl.sh <run_dir>

run_dir="$1"

if [ -z "$run_dir" ]; then
  echo "ERROR: usage: sbatch H_run_lbl.sh <run_dir>" >&2
  exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# SLURM_SUBMIT_DIR = the directory sbatch was invoked from. The standard
# workflow submits every job from the repo root (see A_start / B / Z), so this
# lands us there. Fall back to PWD for interactive runs.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "Running: uv run python -m projections.lbl $run_dir"
uv run python -m projections.lbl "$run_dir"
