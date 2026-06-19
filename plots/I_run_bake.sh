#!/bin/bash
#SBATCH --account=geohc
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=dash_bake
#SBATCH --output=slurm-%x_%j.out

# Bake the BuildStock projection dashboard's data sidecars from one res run_dir
# + one com run_dir. Writes:
#   plots/data/main.js      ~55 MB   (CONUS payload, eager load)
#   plots/data/state_*.js   ~10 MB × 49 (per-state, lazy-loaded on click)
#
# The dashboard itself is `plots/dashboard.html` — committed to git, edit
# directly. It references data/main.js via <script src>. After this bake,
# refresh the browser to pick up new data.
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

uv run python -u plots/aggregate.py \
    --res-run-dir "$res_run_dir" \
    --com-run-dir "$com_run_dir"

# After bake: run dashboard payload tests. Catches schema regressions
# (e.g. peak_gw shape change) + cross-source disagreements before they
# reach the user. Tests load plots/data/main.js + a few state sidecars.
echo
echo "=== running dashboard payload tests ==="
uv run pytest plots/tests/ -v
