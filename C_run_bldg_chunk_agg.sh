#!/bin/bash
#SBATCH --account=geohc
#SBATCH --time=02:00:00
#SBATCH --mail-user=rajendra.adhikari@nrl.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=246064    # RAM in MB; up to 246064 for normal or 2000000 for bigmem on kestrel
#SBATCH --qos=high

# add >>> #SBATCH --qos=high <<< above for quicker launch at double AU cost

start_index=$1
end_index=$2
meta_path=$3
upgrade=$4
prefix=$5
output_dir=$6
script_dir=$7
counties_str=$8

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
uv run python $output_dir/inputs/D_process_chunk_agg.py $start_index $end_index $meta_path $upgrade $prefix $output_dir $script_dir $counties_str
