#!/bin/bash
#SBATCH --account=geohc
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=dash_bake
#SBATCH --output=slurm-%x_%j.out

# Bake the BuildStock projection dashboard from one res run_dir + one com
# run_dir, in two stages:
#   1. aggregate.py        — reads ReEDs/, intermediate/state/, LBL/ and
#                            writes plots/data/main.js + plots/data/state_*.js
#                            (heavy I/O, ~5-7 min).
#   2. build_dashboard.py  — copies the template into plots/dashboard.html
#                            (sub-second). The HTML references
#                            data/main.js via <script src> and lazy-loads
#                            data/state_<postal>.js on state click.
#
# Iterate on plot design by re-running just step 2 from the login node —
# no need to repeat step 1 unless the source run_dirs change.
#
# Usage:
#   sbatch plots/I_run_bake.sh <res_run_dir> <com_run_dir>

set -euo pipefail

res_run_dir="${1:?missing res_run_dir as first arg}"
com_run_dir="${2:?missing com_run_dir as second arg}"

# SLURM runs the staged script from /var/spool/slurmd, so $0 isn't usable to
# locate the repo. $SLURM_SUBMIT_DIR is the directory `sbatch` was invoked
# from — submit from the repo root and we land in the right place.
cd "${SLURM_SUBMIT_DIR:?must be invoked via sbatch from the repo root}"
export PATH="$HOME/.local/bin:$PATH"

# Tell aggregate.py how many processes it can use.
export BAKE_WORKERS="${SLURM_CPUS_PER_TASK:-16}"

echo "res:        $res_run_dir"
echo "com:        $com_run_dir"
echo "workers:    $BAKE_WORKERS"
echo "slurm node: ${SLURMD_NODENAME:-unknown}"
echo
echo "============================================================"
echo "STAGE 1: aggregate.py — reading handoffs, writing payload.json"
echo "============================================================"
uv run python -u plots/aggregate.py \
    --res-run-dir "$res_run_dir" \
    --com-run-dir "$com_run_dir"

echo
echo "============================================================"
echo "STAGE 2: build_dashboard.py — payload + template → HTML"
echo "============================================================"
uv run python -u plots/build_dashboard.py
