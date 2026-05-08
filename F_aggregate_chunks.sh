#!/bin/bash
#SBATCH --account=geohc
#SBATCH --time=02:00:00
#SBATCH --mem=246064    # RAM in MB; up to 246064 for normal or 2000000 for bigmem on kestrel
#SBATCH --qos=high

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
