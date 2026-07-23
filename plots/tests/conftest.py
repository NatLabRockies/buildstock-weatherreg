"""Fixtures: parse JSON out of the dashboard's JS payload files.

Tests run against the live data/ produced by
`sbatch plots/I_build_dashboard.sh`. If data/ is missing, every test in
this folder skips with a clear message rather than failing — keeps
`pytest plots/tests/` from breaking CI on fresh clones.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_STATES = ["TX", "CA", "ME"]   # summer-peaking, mixed, winter-peaking


def _extract_json(js_path: Path, var_pattern: str) -> dict:
    """Pull the JSON object assigned to a window.* variable in a JS file."""
    text = js_path.read_text()
    m = re.search(var_pattern + r"\s*=\s*(\{.*?\});\s*(?:if|\Z)", text, re.DOTALL)
    if not m:
        raise ValueError(f"could not find {var_pattern} in {js_path.name}")
    return json.loads(m.group(1))


@pytest.fixture(scope="session")
def payload():
    main_js = DATA_DIR / "main.js"
    if not main_js.exists():
        pytest.skip(
            f"{main_js} not found — run `sbatch plots/I_build_dashboard.sh <res> <com>` first"
        )
    return _extract_json(main_js, r"window\.PAYLOAD")


@pytest.fixture(scope="session", params=SAMPLE_STATES)
def state_sidecar(request):
    """Yield (postal, parsed_sidecar) for each sample state."""
    postal = request.param
    side = DATA_DIR / f"state_{postal}.js"
    if not side.exists():
        pytest.skip(f"{side} not found — build the dashboard payload first")
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
