#!/bin/bash
#SBATCH --account=geohc
#SBATCH --time=02:00:00
#SBATCH --qos=high

# Partition / --mem are passed by B at sbatch time (matches the chunk profile
# active on the launcher).

bldg_path="$1"
bldg_type="$2"
upgrade_tag="$3"

# Ensure uv is on PATH (adjust if installed elsewhere)
export PATH="$HOME/.local/bin:$PATH"

# Limit per-process threading to avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

uv run python "$bldg_path/inputs/agg_buildings.py" --bldg-path "$bldg_path" --bldg-type "$bldg_type" --upgrade-tag "$upgrade_tag"
