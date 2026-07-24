#!/bin/bash
#SBATCH --account=geohc
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=dash_build
#SBATCH --output=slurm-%x_%j.out

# Build the BuildStock projection dashboard from one res run_dir + one com
# run_dir. Produces a self-contained dashboard directory OUTSIDE the code repo:
#
#   <dashboard_dir>/dashboard.html            (copied from plots/)
#   <dashboard_dir>/plotly-*.min.js           (copied from plots/)
#   <dashboard_dir>/pako-*.min.js             (copied from plots/; gzip lib)
#   <dashboard_dir>/data/main.js              ~30 MB (CONUS payload, gzipped+b64)
#   <dashboard_dir>/data/state_*.js           ~2 MB × 49 (per-state, lazy)
#
# The whole directory is portable — zip it, share it, or serve it directly
# with `serve_dashboard.sh <dashboard_dir>`.
#
# <dashboard_dir> defaults to `<parent(res_run_dir)>/dashboard/`; override
# with the optional third arg.
#
# Usage:
#   sbatch plots/I_build_dashboard.sh <res_run_dir> <com_run_dir> [<dashboard_dir>]

set -euo pipefail

res_run_dir="${1:?missing res_run_dir as first arg}"
com_run_dir="${2:?missing com_run_dir as second arg}"
dashboard_dir="${3:-$(dirname "$res_run_dir")/dashboard}"

# SLURM runs the staged script from /var/spool/slurmd, so $0 isn't usable to
# locate the repo. $SLURM_SUBMIT_DIR is the directory `sbatch` was invoked
# from — submit from the repo root and we land in the right place.
cd "${SLURM_SUBMIT_DIR:?must be invoked via sbatch from the repo root}"
export PATH="$HOME/.local/bin:$PATH"

# Tell aggregate.py how many processes it can use.
export DASHBOARD_BUILD_WORKERS="${SLURM_CPUS_PER_TASK:-16}"

# Tests read this to locate the built payload (see plots/tests/conftest.py).
export DASHBOARD_DIR="$dashboard_dir"

# Pin polars to one thread per process. Without this, the parent's polars
# (used by the 2018-baseline path) spawns ~104 threads on a typical node,
# and forked LBL workers inherit pieces of that thread state — combined
# with each worker's own polars threading, this has historically caused
# severe thread-pool oversubscription and contention-induced slowness in
# the LBL stage (single-file polars scans going from ~0.2 s to many
# minutes). One thread per process, parallelism from the ProcessPool.
export POLARS_MAX_THREADS=1
# Same logic for the BLAS family — keep each worker's numpy/scipy on a
# single core so the ProcessPool isn't competing with thread pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "res:        $res_run_dir"
echo "com:        $com_run_dir"
echo "dashboard:  $dashboard_dir"
echo "workers:    $DASHBOARD_BUILD_WORKERS"
echo "slurm node: ${SLURMD_NODENAME:-unknown}"
echo

# Stage the static files (dashboard.html + vendored plotly bundle) so the
# built dashboard directory is self-contained: `python -m http.server
# --directory <dashboard_dir>` alone is enough to view it.
mkdir -p "$dashboard_dir/data"
cp plots/dashboard.html "$dashboard_dir/"
cp plots/plotly-*.min.js "$dashboard_dir/"
cp plots/pako-*.min.js "$dashboard_dir/"

# Aggregate the payload into <dashboard_dir>/data/{main.js, state_*.js}.
uv run python -u plots/aggregate.py \
    --res-run-dir "$res_run_dir" \
    --com-run-dir "$com_run_dir" \
    --out-dir "$dashboard_dir/data"

# After build: run dashboard payload tests. Catches schema regressions
# (e.g. peak_gw shape change) + cross-source disagreements before they
# reach the user. Tests load $DASHBOARD_DIR/data/main.js + a few sidecars.
echo
echo "=== running dashboard payload tests ==="
uv run pytest plots/tests/ -v
