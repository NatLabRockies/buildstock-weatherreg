"""Explicit anti-regressions for bugs we've shipped and fixed.

Each test references the commit/phase that introduced the bug — if
someone re-introduces the same shape in a future refactor, this test
fails before it reaches the user.
"""
from __future__ import annotations

import pytest

from .conftest import _walk


def test_panel3_winter_and_summer_peaks_differ(payload):
    """REGRESSION: an early _peak_week_task used wy_total.max() for BOTH
    winter and summer, so winter and summer peak_gw came back identical.
    They should differ for most (scen, year, wy) buckets (CONUS rarely
    peaks at exactly the same magnitude winter vs summer)."""
    total = 0
    identical = 0
    for path, by_season in _walk(payload["panel3"], 3):
        w = by_season.get("winter", {}).get("peak_gw")
        s = by_season.get("summer", {}).get("peak_gw")
        if w is None or s is None:
            continue
        total += 1
        if abs(w - s) < 1e-6:
            identical += 1
    # Allow rare genuine ties but flag if it's pervasive
    assert identical < max(1, total * 0.05), (
        f"{identical}/{total} (scen,year,wy) have identical winter and "
        f"summer peak_gw — the wy_total.max() bug may have returned"
    )


def test_state_sector_peak_summer_winter_le_annual(payload):
    """REGRESSION: each state's seasonal peak must be ≤ its annual peak
    (the annual max is by definition the maximum of all seasons). If a
    refactor reversed semantics, this breaks immediately."""
    for path, by_state in _walk(payload["state_by_sector"]["peak_gw"], 4):
        for st, v in by_state.items():
            if not isinstance(v, dict):
                continue
            ann = v["annual"]
            assert v["summer"] <= ann + 0.001, (
                f"peak_gw{path}[{st}].summer={v['summer']} > .annual={ann}"
            )
            assert v["winter"] <= ann + 0.001, (
                f"peak_gw{path}[{st}].winter={v['winter']} > .annual={ann}"
            )


def test_state_peak_does_not_exceed_conus_peak(payload):
    """REGRESSION: per-state seasonal peak must be ≤ CONUS peak for the
    same (scen, year, wy, season). Anti-physics if it ever isn't."""
    for path, by_state in _walk(payload["state_by_sector"]["peak_gw"], 4):
        scalar = lambda v: v["annual"] if isinstance(v, dict) else v
        conus = by_state.get("CONUS")
        if conus is None:
            continue
        for st, v in by_state.items():
            if st == "CONUS":
                continue
            if not isinstance(v, dict):
                continue
            for season in ("annual", "summer", "winter"):
                state_v = v[season]
                conus_v = conus[season] if isinstance(conus, dict) else conus
                assert state_v <= conus_v + 0.1, (
                    f"peak_gw{path}[{st}].{season}={state_v} > "
                    f"CONUS.{season}={conus_v}"
                )


def test_panel3_cohort_sum_at_peak_matches_claimed(payload):
    """REGRESSION for the Phase A.19 fix: the cohort decomposition at the
    claimed peak hour must sum to the claimed peak_gw. (Note: other hours
    in the window can exceed peak_gw because the ±3-day window may spill
    out of the seasonal mask used to find peak_gw, so max-over-window is
    the wrong invariant — sum-at-peak-hour is the right one.)"""
    for path, by_season in _walk(payload["panel3"], 3):
        for season, win in by_season.items():
            ts = win["timestamps"]
            peak_idx = ts.index(win["peak_iso"])
            cohorts = win["cohorts"]
            cohort_sum_at_peak = sum(
                cohorts[k][peak_idx]
                for k in cohorts
                if isinstance(cohorts[k], list)
            )
            claimed = win["peak_gw"]
            assert abs(cohort_sum_at_peak - claimed) < 0.5, (
                f"panel3{path}[{season}] claims peak={claimed} but "
                f"sum(cohorts at peak_iso={win['peak_iso']})={cohort_sum_at_peak}"
            )


def test_panel3_peak_iso_inside_timestamps(payload):
    """The annotated peak hour must be one of the window's timestamps."""
    for path, by_season in _walk(payload["panel3"], 3):
        for season, win in by_season.items():
            assert win["peak_iso"] in win["timestamps"], (
                f"panel3{path}[{season}]: peak_iso={win['peak_iso']} not in "
                f"timestamps [{win['timestamps'][0]} ... {win['timestamps'][-1]}]"
            )


def test_2018_baseline_is_present_in_state_by_sector(payload):
    """REGRESSION GATE for Phase A.26: stock year 2018 must appear in
    state_by_sector + panel1 (derived from agg_*_b2018 county files rolled
    to state). cohort_daily and panel3 can legitimately omit 2018 (no
    cohort split available at the baseline year)."""
    assert 2018 in payload["stock_years"], "2018 missing from STOCK_YEARS"
    # panel1 — at least Baseline scenario must have 2018
    p1_ann_baseline = payload["panel1"]["annual_gwh"].get("Baseline", {}).get("2018", {})
    p1_peak_baseline = payload["panel1"]["peak_gw"].get("Baseline", {}).get("2018", {})
    assert p1_ann_baseline, "panel1.annual_gwh[Baseline][2018] missing"
    assert p1_peak_baseline, "panel1.peak_gw[Baseline][2018] missing"
    # state_by_sector — at least one scenario must have 2018 with full sector dims
    found_2018 = False
    for scen, by_y in payload["state_by_sector"]["annual_gwh"].items():
        if "2018" not in by_y:
            continue
        for wy, by_sec in by_y["2018"].items():
            assert {"residential", "commercial", "gap", "total"} <= set(by_sec.keys()), (
                f"state_by_sector[{scen}][2018][{wy}] missing sectors: "
                f"{set(by_sec.keys())}"
            )
            assert "CONUS" in by_sec["total"], (
                f"state_by_sector[{scen}][2018][{wy}].total missing CONUS"
            )
            found_2018 = True
            break
        if found_2018:
            break
    assert found_2018, "state_by_sector has no 2018 cells"


def test_savings_path_does_not_yield_nan(payload):
    """REGRESSION for the Saved/% Saved breakage after peak_gw → dict.
    Simulates the JS computeValue helper: pulls scalar via .annual unwrap,
    subtracts ref - cur. Must produce finite numbers (not NaN) for every
    (cur_scen, ref_scen, year, wy, sector, state) combination."""
    sbs = payload["state_by_sector"]["peak_gw"]
    scalar = lambda v: v["annual"] if isinstance(v, dict) else v
    scenarios = list(sbs.keys())
    if len(scenarios) < 2:
        pytest.skip("need ≥2 scenarios to test savings path")
    ref = scenarios[0]
    cur = scenarios[1]
    nan_count = 0
    checked = 0
    for year, by_wy in sbs[cur].items():
        for wy, by_sec in by_wy.items():
            for sector, by_state in by_sec.items():
                for st, v_cur in by_state.items():
                    if st == "CONUS":
                        continue
                    v_ref = sbs[ref].get(year, {}).get(wy, {}).get(sector, {}).get(st)
                    if v_ref is None:
                        continue
                    saved = scalar(v_ref) - scalar(v_cur)
                    checked += 1
                    if saved != saved:  # NaN check
                        nan_count += 1
    assert checked > 0
    assert nan_count == 0, (
        f"{nan_count}/{checked} savings computations produced NaN — "
        f"the scalar unwrap is broken somewhere"
    )
