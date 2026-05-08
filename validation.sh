#!/bin/bash
#SBATCH --account=geohc
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --mem=246064    # RAM in MB; up to 246064 for normal or 2000000 for bigmem on kestrel

#SBATCH --job-name=validation
#SBATCH --output=slurm-%x_%j.out

# Optional first arg: path to a switches JSON file. Defaults to switches_agg.json.
# Example: sbatch validation.sh switches_agg_resstock.json
switches_path="${1:-switches_agg.json}"

# Rename the running job with a res_/com_ prefix based on the `comstock` switch
# in the chosen JSON. SLURM has already assigned the name from the #SBATCH
# directive at queue time, so we use scontrol + mv to align the live job and
# its slurm-out file. Failures are non-fatal.
job_prefix=$(python -c "
import json, sys
try:
    s = json.load(open(sys.argv[1]))
    print('com_' if s.get('comstock') else 'res_')
except Exception:
    print('')
" "$switches_path" 2>/dev/null)
if [ -n "$SLURM_JOB_ID" ] && [ -n "$job_prefix" ]; then
    scontrol update jobid="$SLURM_JOB_ID" name="${job_prefix}validation" 2>/dev/null || true
    old_out="slurm-validation_${SLURM_JOB_ID}.out"
    new_out="slurm-${job_prefix}validation_${SLURM_JOB_ID}.out"
    if [ -f "$old_out" ] && [ "$old_out" != "$new_out" ]; then
        mv "$old_out" "$new_out" 2>/dev/null || true
    fi
fi

# Ensure uv is on PATH (adjust if installed elsewhere)
export PATH="$HOME/.local/bin:$PATH"

# Force line-buffered stdout/stderr so progress prints appear in real time
# under srun/sbatch (which pipe stdout, triggering Python's block buffering).
export PYTHONUNBUFFERED=1

# Limit per-process threading to avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /kfs2/shared-projects/buildstock/aws_credentials.sh

# Call the venv's python directly instead of `uv run`. Reason: under srun,
# `uv run` lingers in post-exec lockfile/env teardown on the project Lustre
# mount, which keeps the SLURM step alive long after Python has exited.
# .venv/bin/python is the same interpreter uv would have used anyway.
.venv/bin/python validation.py --switches "$switches_path"
