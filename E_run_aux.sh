#!/bin/bash
#SBATCH --account=geohc
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=aux_query

# Auxiliary BSQ-query step. For each baseline spec (upgrade_id == 0) in the
# run_dir's switches snapshot, writes two CSVs at <run_dir>/:
#   aux_coverage_upgrade<spec_tag>.csv  — per-county cohort sqft + units_count
#   aux_samples_upgrade<spec_tag>.csv   — per-sampled-bldg cohort weights
# Both files are read by `python -m projections` and by the LBL handoff.
#
# Usage:
#   sbatch E_run_aux.sh <run_dir>

run_dir="$1"

if [ -z "$run_dir" ]; then
  echo "ERROR: usage: sbatch E_run_aux.sh <run_dir>" >&2
  exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
source /kfs2/shared-projects/buildstock/aws_credentials.sh

# SLURM stages the script into /var/spool/slurmd at runtime, so $0 / BASH_SOURCE
# don't reach the repo. SLURM_SUBMIT_DIR is the directory sbatch was invoked
# from; the standard workflow submits every job from the repo root (A_start
# is run there; B runs there; Z runs there). Fall back to PWD for interactive.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
uv run python E_aux_query.py "$run_dir"
