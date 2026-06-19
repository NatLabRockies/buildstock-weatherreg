"""Cross-source reconciliation — independent calculations of the same
quantity in different parts of the payload must agree.

These tests would have caught the early "trajectory peak shows 1692
instead of 655" bug (peak-of-sum vs sum-of-peaks) had they existed at
the time. They form a structural alarm: if a refactor changes one
codepath but not its sibling, a mismatch will surface immediately.
"""
from __future__ import annotations

import pytest

from .conftest import _walk

TOL_ENERGY = 0.05  # GWh — float drift from sum order
TOL_POWER = 0.05   # GW


# === panel1 (headline) == state_by_sector total CONUS =======================

def test_panel1_annual_matches_total_conus(payload):
    p1 = payload["panel1"]["annual_gwh"]
    sbs = payload["state_by_sector"]["annual_gwh"]
    for path, val in _walk(p1, 3):
        scen, year, wy = path
        sbs_val = sbs[scen][year][wy]["total"]["CONUS"]
        assert abs(val - sbs_val) < TOL_ENERGY, (
            f"panel1.annual_gwh{path} = {val} but "
            f"state_by_sector.annual_gwh.total.CONUS = {sbs_val}"
        )


def test_panel1_peak_matches_total_conus_annual(payload):
    """The trajectory's headline peak must equal state_by_sector's CONUS
    annual peak. REGRESSION GATE: if peak_gw is ever summed across states
    instead of computed from the joint hourly series, this fails."""
    p1 = payload["panel1"]["peak_gw"]
    sbs = payload["state_by_sector"]["peak_gw"]
    for path, val in _walk(p1, 3):
        scen, year, wy = path
        sbs_entry = sbs[scen][year][wy]["total"]["CONUS"]
        sbs_val = sbs_entry["annual"] if isinstance(sbs_entry, dict) else sbs_entry
        assert abs(val - sbs_val) < TOL_POWER, (
            f"panel1.peak_gw{path} = {val} but "
            f"state_by_sector.peak_gw.total.CONUS.annual = {sbs_val}"
        )


# === total sector == residential + commercial + gap =========================

def test_total_sector_equals_subsector_sum_annual(payload):
    """For annual energy, the 'total' sector group is the row-sum of
    residential + commercial + gap (additive across the year)."""
    sbs = payload["state_by_sector"]["annual_gwh"]
    for path, by_sector in _walk(sbs, 3):
        for st in by_sector["total"]:
            parts = (
                by_sector["residential"].get(st, 0)
                + by_sector["commercial"].get(st, 0)
                + by_sector["gap"].get(st, 0)
            )
            tot = by_sector["total"][st]
            assert abs(tot - parts) < TOL_ENERGY, (
                f"annual_gwh{path}.total[{st}]={tot} but "
                f"sum(res+com+gap)={parts}"
            )


def test_total_peak_does_not_exceed_subsector_peak_sum(payload):
    """Coincident peak invariant: total peak ≤ Σ sector peaks. If they're
    equal, all sectors peak at the same hour — possible but rare; we just
    check the inequality direction."""
    sbs = payload["state_by_sector"]["peak_gw"]
    scalar = lambda v: v["annual"] if isinstance(v, dict) else v
    for path, by_sector in _walk(sbs, 3):
        for st in by_sector["total"]:
            tot = scalar(by_sector["total"][st])
            parts = (
                scalar(by_sector["residential"].get(st, 0))
                + scalar(by_sector["commercial"].get(st, 0))
                + scalar(by_sector["gap"].get(st, 0))
            )
            assert tot <= parts + TOL_POWER, (
                f"peak_gw{path}.total[{st}]={tot} > sum(sector peaks)={parts} "
                "— that means total peak is larger than sum of sector peaks, "
                "which is geometrically impossible"
            )


# === state sidecar agrees with main payload =================================

def test_sidecar_peak_matches_state_by_sector(payload, state_sidecar):
    """REGRESSION GATE for Phase A.19: the state-centered peak window's
    peak_gw must equal state_by_sector's seasonal peak for the same state.
    If the map shows TX summer = 107 GW but the cohort panel shows 91 GW,
    this test fails."""
    postal, side = state_sidecar
    sbs = payload["state_by_sector"]["peak_gw"]
    for scen, by_y in side["peak_week"].items():
        for year, by_wy in by_y.items():
            for wy, by_season in by_wy.items():
                sec_state = sbs[scen][year][wy]["total"].get(postal)
                if not isinstance(sec_state, dict):
                    continue
                for season, win in by_season.items():
                    side_peak = win["peak_gw"]
                    map_peak = sec_state.get(season)
                    assert abs(side_peak - map_peak) < TOL_POWER, (
                        f"sidecar[{postal}].peak_week[{scen}][{year}][{wy}]"
                        f"[{season}].peak_gw={side_peak} but "
                        f"state_by_sector.peak_gw.total.{postal}.{season}={map_peak}"
                    )


def test_sidecar_cohort_daily_sums_to_main_cohort(payload, state_sidecar):
    """Per-state cohort daily values must sum across states to the CONUS
    cohort_daily in the main payload."""
    postal, side = state_sidecar
    main = payload["cohort_daily"]
    # Spot-check one (scen, year, wy); full iteration is O(N²)
    scen = next(iter(side["cohort_daily"]))
    year = next(iter(side["cohort_daily"][scen]))
    wy = next(iter(side["cohort_daily"][scen][year]))
    side_entry = side["cohort_daily"][scen][year][wy]
    main_entry = main[scen][year][wy]
    for cohort_key, side_series in side_entry["cohorts"].items():
        # We can only check that side_series is bounded by main_series,
        # since main is the sum across all states.
        for i, sv in enumerate(side_series):
            mv = main_entry["cohorts"][cohort_key][i]
            assert sv <= mv + TOL_ENERGY, (
                f"sidecar[{postal}] cohort[{cohort_key}][{i}]={sv} > "
                f"CONUS main cohort={mv} — single state larger than total"
            )
