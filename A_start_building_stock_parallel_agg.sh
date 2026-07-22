#!/bin/bash
#SBATCH --account=geohc
#SBATCH --partition=standard
#SBATCH --time=05:00:00
#SBATCH --mem=246064    # launcher only uses 1 CPU; mem here is just for B's Athena queries
#SBATCH --qos=high

# add >>> #SBATCH --qos=high <<< above for quicker launch at double AU cost

# --job-name is passed on the sbatch CLI so the res_/com_ prefix appears in
# both `squeue` and the slurm-<name>_<jobid>.out filename from the start
# (no in-flight rename). Callers:
#   sbatch --job-name=res_building_stock_parallel \
#          A_start_building_stock_parallel_agg.sh switches_agg_resstock.json
#   sbatch --job-name=com_building_stock_parallel \
#          A_start_building_stock_parallel_agg.sh switches_agg_comstock.json
#SBATCH --output=slurm-%x_%j.out

# First arg: path to a switches JSON file (e.g. switches_agg_resstock.json,
# switches_agg_comstock.json).
switches_path="$1"
if [ -z "$switches_path" ]; then
    echo "usage: sbatch --job-name=<res_|com_>building_stock_parallel $0 <switches.json>" >&2
    exit 2
fi

# Ensure uv is on PATH (adjust if installed elsewhere)
export PATH="$HOME/.local/bin:$PATH"
source /kfs2/shared-projects/buildstock/aws_credentials.sh
# Defensive: SLURM precedence is CLI > env > #SBATCH, so any SBATCH_TIMELIMIT
# leaked from the user's shell rc would silently shorten C/F (which ask for 3 h
# / 2 h) on their #SBATCH lines. Unset to make sure that can't happen.
unset SBATCH_TIMELIMIT SLURM_TIMELIMIT

uv run B_building_stock_parallel_agg.py "$switches_path"
