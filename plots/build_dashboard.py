"""Stage 2 of the bake pipeline.

Reads a pre-computed payload.json (from aggregate.py) and the HTML template,
string-substitutes the payload into the __PAYLOAD__ token, writes the final
self-contained dashboard.html. Sub-second on its own — re-run this whenever
you change CSS, layout, or plot logic in dashboard_template.html. Re-run
aggregate.py only when the source run_dirs change.

CLI:
  # iterate on plot design with no re-aggregation:
  uv run python plots/build_dashboard.py
  # output: plots/dashboard.html (open in browser).

  # full bake (aggregate.py + build_dashboard.py) via SLURM:
  sbatch plots/I_run_bake.sh <res_run_dir> <com_run_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--payload',  type=Path, default=here / 'payload.json',
                    help='Pre-computed payload (default: plots/payload.json).')
    ap.add_argument('--template', type=Path, default=here / 'dashboard_template.html',
                    help='HTML template containing the __PAYLOAD__ token.')
    ap.add_argument('--out',      type=Path, default=here / 'dashboard.html',
                    help='Output HTML path (default: plots/dashboard.html).')
    args = ap.parse_args()

    if not args.payload.exists():
        sys.exit(f'ERROR: payload not found at {args.payload}. '
                 f'Run aggregate.py first (or `sbatch plots/I_run_bake.sh <res> <com>`).')
    payload_json = args.payload.read_text()
    tpl = args.template.read_text()
    if '__PAYLOAD__' not in tpl:
        sys.exit(f'ERROR: template {args.template} is missing the __PAYLOAD__ token.')
    args.out.write_text(tpl.replace('__PAYLOAD__', payload_json))
    print(f'Wrote {args.out}  '
          f'(payload {args.payload.stat().st_size/1e6:.2f} MB, '
          f'html {args.out.stat().st_size/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
