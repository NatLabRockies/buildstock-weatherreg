"""Stage 2 of the bake pipeline.

Copies the HTML template to dashboard.html. The template references
`data/main.js` (and lazy-loads `data/state_<postal>.js`) so this step has
no payload-substitution to do — it's a near-noop kept for symmetry with
the build pipeline and for any future inline injections.

CLI:
  # iterate on plot design without re-aggregation:
  uv run python plots/build_dashboard.py
  # output: plots/dashboard.html (open in browser).

  # full bake (aggregate.py → data/*.js + build_dashboard.py → HTML):
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
    ap.add_argument('--template', type=Path, default=here / 'dashboard_template.html',
                    help='HTML template path.')
    ap.add_argument('--out',      type=Path, default=here / 'dashboard.html',
                    help='Output HTML path (default: plots/dashboard.html).')
    args = ap.parse_args()

    tpl = args.template.read_text()
    args.out.write_text(tpl)
    data_dir = args.out.parent / 'data'
    main_js = data_dir / 'main.js'
    if main_js.exists():
        sz = main_js.stat().st_size
        print(f'Wrote {args.out}  (template → HTML; data/main.js exists, {sz/1e6:.2f} MB)')
    else:
        print(f'Wrote {args.out}  (template → HTML)')
        print(f'  WARNING: {main_js} not found. Run aggregate.py first.', file=sys.stderr)


if __name__ == '__main__':
    main()
