#!/bin/bash
#SBATCH --account=geohc
#SBATCH --time=12:00:00
#SBATCH --qos=high

# Partition / --mem / --cpus-per-task are passed by B at sbatch time so they
# can flex between standard (48c / 246GB) and bigmem (104c / 2TB) profiles.
# add >>> #SBATCH --qos=high <<< above for quicker launch at double AU cost

# This script runs as a SLURM array task. The chunk-specific args (start_index,
# end_index, weather_locs_str) are read from a manifest CSV using
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

# Manifest format: chunk_idx,start_index,end_index,weather_locs_str (header on row 1).
# weather_locs_str contains underscores but no commas, so f4- safely captures it.
row=$(awk -F, -v id="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == id { print; exit }' "$manifest_path")
if [ -z "$row" ]; then
  echo "ERROR: no row for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $manifest_path" >&2
  exit 1
fi
start_index=$(echo "$row" | cut -d, -f2)
end_index=$(echo "$row" | cut -d, -f3)
weather_locs_str=$(echo "$row" | cut -d, -f4-)

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
uv run python $output_dir/inputs/D_process_chunk_agg.py $start_index $end_index $meta_path $upgrade $prefix $output_dir $script_dir $weather_locs_str $spec_index
