#!/bin/bash
#SBATCH --account=geohc
#SBATCH --time=4:00:00
#SBATCH --mail-user=jmowers@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=246064    # RAM in MB; up to 246064 for normal or 2000000 for bigmem on kestrel
#SBATCH --qos=high

# add >>> #SBATCH --qos=high <<< above for quicker launch at double AU cost

#SBATCH --job-name=building_stock_parallel

# Ensure uv is on PATH (adjust if installed elsewhere)
export PATH="$HOME/.local/bin:$PATH"

aws sso login

uv run B_building_stock_parallel_agg.py
