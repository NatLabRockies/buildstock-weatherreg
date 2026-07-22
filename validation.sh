#!/bin/bash
#SBATCH --account=geohc
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --mem=246064    # RAM in MB; up to 246064 for normal or 2000000 for bigmem on kestrel

# --job-name is passed on the sbatch CLI so the res_/com_ prefix appears in
# both `squeue` and the slurm-<name>_<jobid>.out filename from the start.
# Callers:
#   sbatch --job-name=res_validation validation.sh switches_agg_resstock.json
#   sbatch --job-name=com_validation validation.sh switches_agg_comstock.json
#SBATCH --output=slurm-%x_%j.out

# First arg: path to a switches JSON file.
switches_path="$1"
if [ -z "$switches_path" ]; then
    echo "usage: sbatch --job-name=<res_|com_>validation $0 <switches.json>" >&2
    exit 2
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
