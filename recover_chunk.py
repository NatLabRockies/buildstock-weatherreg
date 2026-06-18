#!/usr/bin/env python
"""Re-split a single failed chunk into smaller sub-chunks and resubmit.

Recovery tool for when one chunk in an array timed out due to a heavy-tail
of bldg_ids (e.g., a "popular" set of source counties that produces N×
average bldg_id count). Re-uses the already-completed chunk files in
chunks_<reg|ref>_b<year>/; only the missing chunk is re-computed,
sub-divided into smaller pieces that finish in parallel.

Mechanism:
  1. Read the run's switches snapshot to identify the spec by index.
  2. Read the original manifest, locate the failed chunk row, and split
     its source-county list into N sub-chunks (default: one per county).
  3. Write a NEW manifest at inputs/manifest_resplit_<tag>_chunk<idx>.csv.
  4. Submit the existing C_run_bldg_chunk_agg.sh as an array job pointing
     at the new manifest (filenames the chunks produce will use the
     sub-chunk's start-end indices, so they slot neatly alongside the
     existing chunk files in chunks_<reg|ref>_b<year>/).
  5. Submit a dependent F_aggregate_chunks.sh agg job. Once the missing
     pieces land, agg_buildings.py picks up the full set (existing 124 +
     new 25) and produces the final aggregated GWh CSV.

Usage:
    python recover_chunk.py <output_dir> <spec_idx> <chunk_idx>

Example:
    # ComStock run; reg_b2018 (spec 0) chunk 81 failed:
    python recover_chunk.py /projects/.../comstock_cross_val_may8_2_2026 0 81
    # Then for reg_b2012 (spec 2):
    python recover_chunk.py /projects/.../comstock_cross_val_may8_2_2026 2 81
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('output_dir', type=Path, help='Run output dir')
    p.add_argument('spec_idx', type=int, help='Index into run_specs')
    p.add_argument('chunk_idx', type=int, help='Manifest chunk_idx that failed')
    p.add_argument('--n-splits', type=int, default=None,
                   help='Number of sub-chunks. Default: one per source county.')
    p.add_argument('--script-dir', type=Path,
                   default=Path('/kfs2/projects/geohc/radhikar/weather_regression/buildstock-weatherreg'),
                   help='Path to the buildstock-weatherreg checkout (for C/F shell wrappers).')
    p.add_argument('--concurrency', type=int, default=200,
                   help='Array %% concurrency cap.')
    args = p.parse_args()

    out = args.output_dir
    if not out.is_dir():
        raise SystemExit(f"output_dir does not exist: {out}")

    # Load the canonical switches snapshot to identify the spec. upgrade_tag
    # now uses `spec['name']` rather than the integer upgrade id (so multiple
    # specs can share an upgrade_id without colliding on filenames). The
    # upgrade arg passed through to C/D is a comma-joined token of the
    # spec's upgrade_id(s); D reads the authoritative list from the snapshot
    # via spec_idx, so this token is informational only.
    snap = out / 'inputs' / 'switches_agg.json'
    switches = json.load(open(snap))
    spec = switches['run_specs'][args.spec_idx]
    spec_name = spec['name']
    _raw_upgrade = spec['upgrade_id']
    upgrade_ids = (
        list(_raw_upgrade)
        if isinstance(_raw_upgrade, (list, tuple))
        else [_raw_upgrade]
    )
    upgrade_argv_token = ','.join(str(u) for u in upgrade_ids)
    base_year = spec['base_year']
    apply_regression = spec['apply_regression']
    regression_tag = 'reg' if apply_regression else 'ref'
    upgrade_tag = f'{spec_name}_{regression_tag}_b{base_year}'

    bldg_type = 'com' if switches.get('comstock') else 'res'
    prefix = f'{bldg_type}_'

    print(f'spec[{args.spec_idx}] = {upgrade_tag}')

    # Locate the failed chunk's row in the original manifest.
    orig_manifest = out / 'inputs' / f'manifest_upgrade{upgrade_tag}.csv'
    if not orig_manifest.exists():
        raise SystemExit(f"manifest not found: {orig_manifest}")
    with open(orig_manifest) as f:
        lines = f.readlines()
    header = lines[0]  # chunk_idx,start_index,end_index,counties_str
    target_row = None
    for line in lines[1:]:
        idx = int(line.split(',', 1)[0])
        if idx == args.chunk_idx:
            target_row = line.rstrip('\n')
            break
    if target_row is None:
        raise SystemExit(f"chunk_idx {args.chunk_idx} not in {orig_manifest}")

    parts = target_row.split(',', 3)
    orig_start = int(parts[1])
    orig_end = int(parts[2])
    counties = parts[3].split('_')
    n_counties = len(counties)
    if n_counties != (orig_end - orig_start):
        print(f'WARNING: counties_str has {n_counties} entries but range is '
              f'{orig_end - orig_start}; using counties_str length.')

    n_splits = args.n_splits if args.n_splits is not None else n_counties
    n_splits = min(n_splits, n_counties)

    # Build sub-chunks. Source-county indices preserved so output filenames
    # don't collide with any existing chunk files.
    sub_chunks = []
    per_split = (n_counties + n_splits - 1) // n_splits  # ceil div
    for i in range(n_splits):
        s_off = i * per_split
        e_off = min((i + 1) * per_split, n_counties)
        if s_off >= e_off:
            break
        sub_start = orig_start + s_off
        sub_end = orig_start + e_off
        sub_counties = '_'.join(counties[s_off:e_off])
        sub_chunks.append((sub_start, sub_end, sub_counties))

    # Write the resplit manifest. chunk_idx in this manifest is fresh
    # 0..N-1; SLURM_ARRAY_TASK_ID indexes into it.
    resplit_manifest = (out / 'inputs' /
                        f'manifest_resplit_upgrade{upgrade_tag}_chunk{args.chunk_idx}.csv')
    with open(resplit_manifest, 'w') as f:
        f.write(header)
        for i, (s, e, cs) in enumerate(sub_chunks):
            f.write(f'{i},{s},{e},{cs}\n')
    print(f'Wrote {resplit_manifest} ({len(sub_chunks)} sub-chunks, '
          f'~{n_counties / max(len(sub_chunks), 1):.1f} counties/sub-chunk)')

    # Submit the array against the resplit manifest.
    meta_path = out / f'{prefix}meta_master_upgrade{upgrade_tag}.csv'
    slurm_out_dir = out / 'slurm-out'
    slurm_out_dir.mkdir(parents=True, exist_ok=True)

    array_cmd = [
        'sbatch',
        f'--job-name={prefix}chunk_{upgrade_tag}_recover',
        f'--array=0-{len(sub_chunks) - 1}%{args.concurrency}',
        f'--output={slurm_out_dir}/slurm-%x_%A_%a.out',
        str(args.script_dir / 'C_run_bldg_chunk_agg.sh'),
        str(resplit_manifest), str(meta_path), upgrade_argv_token, prefix,
        str(out), str(args.script_dir), str(args.spec_idx),
    ]
    print('Submitting array:', ' '.join(array_cmd))
    result = subprocess.run(array_cmd, check=True, capture_output=True, text=True)
    array_job_id = result.stdout.strip().split()[-1]
    print(f'  -> array job {array_job_id}')

    # Submit dependent agg job.
    agg_cmd = [
        'sbatch',
        f'--job-name={prefix}agg_{upgrade_tag}_recover',
        f'--dependency=afterok:{array_job_id}',
        '--kill-on-invalid-dep=yes',
        f'--output={slurm_out_dir}/slurm-%x_%j.out',
        str(args.script_dir / 'F_aggregate_chunks.sh'),
        str(out), bldg_type, upgrade_tag,
    ]
    print('Submitting agg:  ', ' '.join(agg_cmd))
    result = subprocess.run(agg_cmd, check=True, capture_output=True, text=True)
    agg_job_id = result.stdout.strip().split()[-1]
    print(f'  -> agg job {agg_job_id} (depends on array {array_job_id})')


if __name__ == '__main__':
    main()
