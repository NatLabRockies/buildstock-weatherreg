#!/bin/bash
#SBATCH --account=geohc
#SBATCH --partition=debug
#SBATCH --qos=high
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --job-name=post_pipeline

# Post-aggregation launcher. Submitted by B with --dependency=afterok on every
# F aggregation job, so this only runs once every spec's calibrated agg CSV is
# on disk. From here it dispatches:
#     1. state projection             (G, medmem)
#     2. county_group projection      (G, medmem)
#     3. ReEDs + intermediate handoff (light, debug; depends on both)
#     4. LBL handoff                  (H_run_lbl.sh, medmem; depends on county_group)
#
# Usage (B calls this internally):
#   sbatch --dependency=afterok:<F_IDs...> Z_post_pipeline.sh <run_dir> <stock>

run_dir="$1"
stock="$2"

if [ -z "$run_dir" ] || [ -z "$stock" ]; then
  echo "ERROR: usage: sbatch Z_post_pipeline.sh <run_dir> <stock>" >&2
  exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
cd /kfs2/projects/geohc/radhikar/weather_regression/buildstock-weatherreg
script_dir="$PWD"

# Projection: state resolution (small frames, 20 workers, ~480 GB peak)
state_id=$(sbatch --parsable \
                   --partition=medmem --qos=high \
                   --mem=480G --cpus-per-task=20 --time=01:00:00 \
                   G_run_projection.sh "$run_dir" "$stock" state)
echo "submitted state projection: $state_id"

# Projection: county_group resolution (1,038-col frames; fewer workers, more mem)
cg_id=$(sbatch --parsable \
                --partition=medmem --qos=high \
                --mem=800G --cpus-per-task=16 --time=02:00:00 \
                G_run_projection.sh "$run_dir" "$stock" county_group)
echo "submitted county_group projection: $cg_id"

# Light handoffs (ReEDs + intermediate). ReEDs needs projections_state/;
# intermediate needs both. Single debug job runs them sequentially.
handoff_light_id=$(sbatch --parsable \
                           --account=geohc --partition=debug --qos=high \
                           --mem=8G --cpus-per-task=2 --time=00:30:00 \
                           --dependency=afterok:${state_id}:${cg_id} \
                           --job-name=handoff_light \
                           --wrap="cd $script_dir && \
                                   uv run python -m projections.reeds        '$run_dir' && \
                                   uv run python -m projections.intermediate '$run_dir'")
echo "submitted ReEDs+intermediate: $handoff_light_id"

# LBL handoff (heavier — county-group long-format melt + per-cohort samples).
# Only needs projections_county_group/ and the aux_samples_*.csv files.
lbl_id=$(sbatch --parsable \
                 --dependency=afterok:${cg_id} \
                 H_run_lbl.sh "$run_dir")
echo "submitted LBL: $lbl_id"

echo "post-pipeline dispatched: state=$state_id, cg=$cg_id, "\
"handoff_light=$handoff_light_id, lbl=$lbl_id"
