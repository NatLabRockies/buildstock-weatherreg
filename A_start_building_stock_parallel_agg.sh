#!/bin/bash
#SBATCH --account=geohc
#SBATCH --time=02:00:00
#SBATCH --mail-user=rajendra.adhikari@nrl.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=246064    # RAM in MB; up to 246064 for normal or 2000000 for bigmem on kestrel
#SBATCH --qos=high

# add >>> #SBATCH --qos=high <<< above for quicker launch at double AU cost

#SBATCH --job-name=building_stock_parallel

# Optional first arg: path to a switches JSON file (e.g. switches_resstock.json,
# switches_comstock.json). Defaults to switches_agg.json in the script dir so
# existing one-config workflows keep working.
# Example: sbatch A_start_building_stock_parallel_agg.sh switches_resstock.json
switches_path="${1:-switches_agg.json}"

# Rename the running launcher job with a res_/com_ prefix based on the
# `comstock` switch in the chosen JSON. SLURM has already assigned the name
# from the #SBATCH directive at queue time, so we use scontrol to rename
# the in-progress job. Failures are non-fatal — the run continues.
job_prefix=$(python -c "
import json, sys
try:
    s = json.load(open(sys.argv[1]))
    print('com_' if s.get('comstock') else 'res_')
except Exception:
    print('')
" "$switches_path" 2>/dev/null)
if [ -n "$SLURM_JOB_ID" ] && [ -n "$job_prefix" ]; then
    scontrol update jobid="$SLURM_JOB_ID" name="${job_prefix}building_stock_parallel" 2>/dev/null || true
fi

# Ensure uv is on PATH (adjust if installed elsewhere)
export PATH="$HOME/.local/bin:$PATH"
source /kfs2/shared-projects/buildstock/aws_credentials.sh
# aws sso login

uv run B_building_stock_parallel_agg.py "$switches_path"
