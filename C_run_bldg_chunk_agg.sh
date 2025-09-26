#!/bin/bash
#SBATCH --account=geohc
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=jmowers@nrel.gov
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
bldg_ids_str=$8

# Set up nodal environment for run
. $HOME/.bashrc
module purge
module use /nopt/nrel/apps/software/gams/modulefiles
source /nopt/nrel/apps/env.sh
module load anaconda3
conda activate geothermal

python $output_dir/inputs/D_process_chunk_agg.py $start_index $end_index $meta_path $upgrade $prefix $output_dir $script_dir $bldg_ids_str