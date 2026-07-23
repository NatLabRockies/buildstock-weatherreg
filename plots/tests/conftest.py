"""Fixtures: parse JSON out of the dashboard's JS payload files.

Tests run against the live data/ produced by
`sbatch plots/I_build_dashboard.sh`. That script writes the payload to a
dashboard directory outside the code repo (default
`<parent(res_run_dir)>/dashboard/`) and exports `DASHBOARD_DIR` so these
tests can find it. If `DASHBOARD_DIR` isn't set or the data isn't
present, every test in this folder skips with a clear message — keeps
`pytest plots/tests/` from breaking CI on fresh clones.
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import re
from pathlib import Path

import pytest

DATA_DIR = (Path(os.environ["DASHBOARD_DIR"]) / "data"
            if os.environ.get("DASHBOARD_DIR") else None)
SAMPLE_STATES = ["TX", "CA", "ME"]   # summer-peaking, mixed, winter-peaking


def _skip_no_data(reason: str) -> None:
    pytest.skip(
        f"{reason} — set $DASHBOARD_DIR to a built dashboard directory, or "
        f"run `sbatch plots/I_build_dashboard.sh <res> <com>` first"
    )


def _extract_json(js_path: Path, var_pattern: str) -> dict:
    """Decode the payload embedded in a data/*.js file. The file is an IIFE
    of the shape `var b64 = "…"; …; window.X = payload;` where `b64` is a
    base64-encoded gzipped JSON blob (see plots/aggregate.py). `var_pattern`
    is kept for compatibility with the old regex-based extractor but
    ignored — the b64 line is the same shape for every payload."""
    text = js_path.read_text()
    m = re.search(r'var b64\s*=\s*"([^"]+)"', text)
    if not m:
        raise ValueError(
            f"could not find base64 payload in {js_path.name} "
            f"(expected `var b64 = \"…\";`)"
        )
    del var_pattern  # unused; kept in the signature for callers
    gz = base64.b64decode(m.group(1))
    return json.loads(gzip.decompress(gz))


@pytest.fixture(scope="session")
def payload():
    if DATA_DIR is None:
        _skip_no_data("$DASHBOARD_DIR not set")
    main_js = DATA_DIR / "main.js"
    if not main_js.exists():
        _skip_no_data(f"{main_js} not found")
    return _extract_json(main_js, r"window\.PAYLOAD")


@pytest.fixture(scope="session", params=SAMPLE_STATES)
def state_sidecar(request):
    """Yield (postal, parsed_sidecar) for each sample state."""
    if DATA_DIR is None:
        _skip_no_data("$DASHBOARD_DIR not set")
    postal = request.param
    side = DATA_DIR / f"state_{postal}.js"
    if not side.exists():
        _skip_no_data(f"{side} not found")
    data = _extract_json(side, rf'window\.STATE_DATA\["{postal}"\]')
    return postal, data


def _walk(tree, depth):
    """Yield (path_tuple, leaf) pairs at exactly `depth` nesting levels."""
    if depth == 0:
        yield (), tree
        return
    for k, v in tree.items():
        for sub_path, leaf in _walk(v, depth - 1):
            yield (k,) + sub_path, leaf
