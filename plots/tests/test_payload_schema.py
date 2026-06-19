"""Schema invariants on the dashboard payload.

These tests would have caught the dict-vs-scalar regression that broke
Saved / % Saved: if any downstream code path assumes peak_gw is a number
when it's now a dict (or vice versa), the test fails before the user
hits it in the UI.
"""
from __future__ import annotations

import pytest

from .conftest import _walk


# === panel1: top trajectory =================================================

def test_panel1_has_required_metrics(payload):
    p1 = payload["panel1"]
    assert set(p1.keys()) >= {"annual_gwh", "peak_gw"}


def test_panel1_values_are_scalars(payload):
    """panel1 stores the headline totals — must be plain numbers, not dicts."""
    for metric in ("annual_gwh", "peak_gw"):
        for path, leaf in _walk(payload["panel1"][metric], 3):
            assert isinstance(leaf, (int, float)), (
                f"panel1[{metric}]{path} is {type(leaf).__name__}, expected number"
            )


# === state_by_sector: choropleth + breakdown ================================

def test_state_by_sector_metrics_present(payload):
    sbs = payload["state_by_sector"]
    assert set(sbs.keys()) >= {"annual_gwh", "peak_gw"}


def test_state_by_sector_sectors(payload):
    """Each (scen, year, wy) bucket must carry the four sector groups."""
    required = {"residential", "commercial", "gap", "total"}
    sample_metric = "annual_gwh"
    for path, by_sector in _walk(payload["state_by_sector"][sample_metric], 3):
        assert required <= set(by_sector.keys()), (
            f"sectors missing at {path}: have {set(by_sector.keys())}"
        )


def test_annual_gwh_state_values_are_scalars(payload):
    """state_by_sector.annual_gwh leaves are plain numbers per state."""
    for path, by_state in _walk(payload["state_by_sector"]["annual_gwh"], 4):
        assert isinstance(by_state, dict)
        for st, v in by_state.items():
            assert isinstance(v, (int, float)), (
                f"annual_gwh{path}[{st}] is {type(v).__name__}, expected number"
            )


def test_peak_gw_state_values_are_seasonal_dicts(payload):
    """REGRESSION GATE for the Phase A.17 unification — peak_gw values
    must be {annual, summer, winter} dicts. If this fails, every
    downstream JS scalar reader needs a corresponding unwrap pass
    (see the scalar() helper in renderMap.computeValue)."""
    required_keys = {"annual", "summer", "winter"}
    for path, by_state in _walk(payload["state_by_sector"]["peak_gw"], 4):
        assert isinstance(by_state, dict)
        for st, v in by_state.items():
            assert isinstance(v, dict), (
                f"peak_gw{path}[{st}] is {type(v).__name__}, expected dict — "
                f"a downstream JS reader will silently NaN if this regresses"
            )
            assert required_keys <= set(v.keys()), (
                f"peak_gw{path}[{st}] missing keys: have {set(v.keys())}"
            )
            for season_key in required_keys:
                assert isinstance(v[season_key], (int, float))


# === cohort_daily: bottom-left panel ========================================

def test_cohort_daily_dates_and_values_align(payload):
    """Date array length must equal each cohort series length."""
    for path, entry in _walk(payload["cohort_daily"], 3):
        dates = entry["dates"]
        cohorts = entry["cohorts"]
        assert isinstance(dates, list)
        for cohort_key, series in cohorts.items():
            assert len(series) == len(dates), (
                f"cohort_daily{path}: cohort {cohort_key} has "
                f"{len(series)} values vs {len(dates)} dates"
            )


# === panel3: peak-week hourly ===============================================

def test_panel3_has_both_seasons(payload):
    """Each (scen, year, wy) must have both winter and summer peak windows."""
    for path, by_season in _walk(payload["panel3"], 3):
        assert {"winter", "summer"} <= set(by_season.keys()), (
            f"panel3{path} missing seasons: have {set(by_season.keys())}"
        )


def test_panel3_window_lengths(payload):
    """Peak windows are ±3 days at hourly resolution = 168 hours."""
    for path, by_season in _walk(payload["panel3"], 3):
        for season, w in by_season.items():
            ts = w["timestamps"]
            assert len(ts) == 168, (
                f"panel3{path}[{season}]: {len(ts)} hours, expected 168"
            )
            for cohort_key, series in w["cohorts"].items():
                assert len(series) == 168


# === axis: pinned-axis maxes ================================================

def test_axis_choropleth_max_exists(payload):
    """The choropleth color-scale pins live here. Absent → UI uses fallback 1."""
    assert "choropleth_max" in payload["axis"]
    for sector in ("residential", "commercial", "gap", "total"):
        sec = payload["axis"]["choropleth_max"].get(sector)
        assert sec is not None, f"axis.choropleth_max[{sector}] missing"
        assert {"annual_gwh", "peak_gw"} <= set(sec.keys())
