#!/bin/bash
#SBATCH --account=geohc
#SBATCH --time=04:00:00
#SBATCH --mem=246064    # RAM in MB; up to 246064 for normal or 2000000 for bigmem on kestrel
#SBATCH --qos=high

# add >>> #SBATCH --qos=high <<< above for quicker launch at double AU cost

# This script runs as a SLURM array task. The chunk-specific args (start_index,
# end_index, counties_str) are read from a manifest CSV using
# $SLURM_ARRAY_TASK_ID; the rest are passed positionally by B.
manifest_path=$1
meta_path=$2
upgrade=$3
prefix=$4
output_dir=$5
script_dir=$6
spec_index=$7

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID not set. This script must be submitted as an array job (sbatch --array=...)." >&2
  exit 1
fi
if [ ! -f "$manifest_path" ]; then
  echo "ERROR: manifest not found: $manifest_path" >&2
  exit 1
fi

# Manifest format: chunk_idx,start_index,end_index,counties_str (header on row 1).
# counties_str contains underscores but no commas, so f4- safely captures it.
row=$(awk -F, -v id="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == id { print; exit }' "$manifest_path")
if [ -z "$row" ]; then
  echo "ERROR: no row for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $manifest_path" >&2
  exit 1
fi
start_index=$(echo "$row" | cut -d, -f2)
end_index=$(echo "$row" | cut -d, -f3)
counties_str=$(echo "$row" | cut -d, -f4-)

echo "Array task $SLURM_ARRAY_TASK_ID -> chunk $start_index-$end_index"

# Ensure uv is on PATH (adjust if installed elsewhere)
export PATH="$HOME/.local/bin:$PATH"

# Limit per-process threading to avoid oversubscription when using many processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
source /kfs2/shared-projects/buildstock/aws_credentials.sh
uv run python $output_dir/inputs/D_process_chunk_agg.py $start_index $end_index $meta_path $upgrade $prefix $output_dir $script_dir $counties_str $spec_index
