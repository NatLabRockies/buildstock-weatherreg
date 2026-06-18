"""Future-year GWh projection.

Modules (each sits on the one below):
    projection      the projection components + the parallel driver
    factors         efficiency and cohort-growth multipliers
    gap             ComStock gap-model loader (state CSV / per-county S3)
    growth_factors  AEO 2025 cohort splits
    common          shared types, config, geography, and input loaders
    reeds           state-aggregated total-electricity handoff (long format)
    lbl             county-group timeseries + per-cohort sample lists
    intermediate    relabeled per-component view for publishing/debugging

Run the projection (submit as SLURM batch — never the login node):
    sbatch G_run_projection.sh <run_dir> <stock> <state|county|county_group>
Run the handoffs (light, login node is fine):
    python -m projections.reeds        <res_run_dir> <com_run_dir> --out DIR
    python -m projections.lbl          <res_run_dir> <com_run_dir> --out DIR
    python -m projections.intermediate <res_run_dir> <com_run_dir> --out DIR
"""

from .projection import (
    get_gap,
    get_new_adoption_upgrade,
    get_new_construction,
    get_new_construction_baseline,
    get_surviving_adoption_upgrade,
    get_surviving_baseline,
    get_surviving_non_adoption_upgrade,
    project_run_dir,
)

__all__ = [
    'get_new_adoption_upgrade',
    'get_surviving_adoption_upgrade',
    'get_surviving_non_adoption_upgrade',
    'get_gap',
    'get_new_construction_baseline',
    'get_surviving_baseline',
    'get_new_construction',
    'project_run_dir',
]
