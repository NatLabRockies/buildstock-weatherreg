"""Browser-side tests against the actual dashboard.html.

These catch what the Python payload tests can't: bugs in the JS render
code that consume correct data but produce broken output. The user-
reported "Saved / % Saved went blank after schema change" is the
canonical case — payload was fine, JS was broken.

Each test:
  1. Loads file://plots/dashboard.html in headless Chromium
  2. Drives an interaction (click chip, toggle slider, click state)
  3. Reads back Plotly's actual trace data via page.evaluate()
  4. Asserts the rendered output is non-degenerate

If chromium-headless-shell isn't available (no `playwright install`
chromium), tests skip gracefully.

NOTE: Tests share a single browser context per session for speed
(opening Chromium + loading the dashboard with all sidecars is the
slow part; subsequent tests reuse the loaded page).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, sync_playwright  # noqa: E402

# I_build_dashboard.sh exports $DASHBOARD_DIR to the built directory
# (containing dashboard.html, plotly-*.min.js, and data/*.js). Skip
# gracefully if it's not set.
_DASHBOARD_DIR = Path(os.environ["DASHBOARD_DIR"]) if os.environ.get("DASHBOARD_DIR") else None
DASHBOARD = _DASHBOARD_DIR / "dashboard.html" if _DASHBOARD_DIR else None
DATA_DIR = _DASHBOARD_DIR / "data" if _DASHBOARD_DIR else None

# A few representative states for the click-a-state test. TX is summer-
# peaking; ME winter-peaking — both exercise the seasonal code paths.
STATES_TO_CLICK = ["TX", "ME"]


@pytest.fixture(scope="session")
def browser_session():
    if _DASHBOARD_DIR is None:
        pytest.skip("$DASHBOARD_DIR not set — run `sbatch plots/I_build_dashboard.sh <res> <com>` first")
    if not DASHBOARD.exists() or not (DATA_DIR / "main.js").exists():
        pytest.skip(f"dashboard.html or data/ not found under {_DASHBOARD_DIR}")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def page(browser_session):
    """A fresh page per test, with a console-error and page-error collector."""
    ctx = browser_session.new_context()
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
    page.on(
        "console",
        lambda msg: errors.append(f"console.{msg.type}: {msg.text}")
        if msg.type == "error" else None,
    )
    page.errors = errors  # attach for tests to inspect
    page.goto(DASHBOARD.as_uri())
    # Wait for the dashboard's JS to bootstrap. The first plot render writes
    # data into #plot-trajectory; once that's there, the page is interactive.
    page.wait_for_function(
        "document.getElementById('plot-trajectory')?.data?.length > 0",
        timeout=15000,
    )
    yield page
    ctx.close()


# === Smoke: dashboard loads without errors =================================

def test_dashboard_loads_without_js_errors(page: Page):
    """No JS exceptions and no console.error during initial render."""
    # Allow a tick for any deferred errors
    page.wait_for_timeout(300)
    assert page.errors == [], f"page errors: {page.errors}"


def test_all_four_plots_render(page: Page):
    """All four panels have non-empty Plotly trace arrays."""
    for plot_id in ("plot-trajectory", "plot-map", "plot-cohort", "plot-cohort-pk"):
        n = page.evaluate(
            f"document.getElementById('{plot_id}')?.data?.length || 0"
        )
        assert n > 0, f"{plot_id} has no traces"


# === Choropleth: all three modes produce non-null z values =================

@pytest.mark.parametrize("mode", ["absolute", "savings", "pct_savings"])
def test_choropleth_mode_renders_non_null(page: Page, mode):
    """REGRESSION GATE for the Saved/% Saved breakage after peak_gw → dict.
    Each mode must produce at least one non-null state value. If a JS
    consumer of state_by_sector ever assumes the wrong shape, this fails."""
    page.click(f'[data-map-mode="{mode}"]')
    page.wait_for_timeout(300)
    z = page.evaluate("document.getElementById('plot-map').data[0].z")
    non_null = [v for v in (z or []) if v is not None]
    assert len(non_null) > 0, (
        f"choropleth mode={mode}: all-null z values — render code broken"
    )
    assert page.errors == [], f"errors in mode={mode}: {page.errors}"


@pytest.mark.parametrize("metric", ["annual_gwh", "peak_gw"])
def test_choropleth_metric_renders(page: Page, metric):
    page.click(f'[data-map-metric="{metric}"]')
    page.wait_for_timeout(300)
    z = page.evaluate("document.getElementById('plot-map').data[0].z")
    non_null = [v for v in (z or []) if v is not None]
    assert len(non_null) > 0
    assert page.errors == []


def test_choropleth_color_scale_pinned_across_year_slide(page: Page):
    """REGRESSION GATE: the absolute Annual GWh color scale should be
    globally pinned. Slide the stock-year slider and verify zmax stays
    constant."""
    page.click('[data-map-metric="annual_gwh"]')
    page.click('[data-map-mode="absolute"]')
    page.wait_for_timeout(200)
    zmax_a = page.evaluate("document.getElementById('plot-map').data[0].zmax")
    # Move the year slider to a different value
    page.evaluate("""
      const s = document.getElementById('ctl-stockyear');
      s.value = Number(s.max);
      s.dispatchEvent(new Event('input', {bubbles: true}));
    """)
    page.wait_for_timeout(300)
    zmax_b = page.evaluate("document.getElementById('plot-map').data[0].zmax")
    assert abs(zmax_a - zmax_b) < 1e-6, (
        f"color scale changed: {zmax_a} → {zmax_b} after year slide"
    )


def test_choropleth_savings_color_scale_pinned(page: Page):
    """REGRESSION GATE for A.24: Saved-mode color scale must be pinned."""
    page.click('[data-map-mode="savings"]')
    page.wait_for_timeout(200)
    zmax_a = page.evaluate("document.getElementById('plot-map').data[0].zmax")
    page.evaluate("""
      const s = document.getElementById('ctl-stockyear');
      s.value = Number(s.max);
      s.dispatchEvent(new Event('input', {bubbles: true}));
    """)
    page.wait_for_timeout(300)
    zmax_b = page.evaluate("document.getElementById('plot-map').data[0].zmax")
    assert abs(zmax_a - zmax_b) < 1e-6, (
        f"savings color scale changed: {zmax_a} → {zmax_b} after year slide"
    )


# === Cohort panel: granularity toggle =====================================

def test_cohort_daily_has_about_365_points(page: Page):
    page.click('[data-cohort-gran="daily"]')
    page.wait_for_timeout(300)
    x = page.evaluate("document.getElementById('plot-cohort').data[0].x")
    assert 360 <= len(x) <= 372, f"daily cohort length unexpected: {len(x)}"


def test_cohort_monthly_has_12_points(page: Page):
    page.click('[data-cohort-gran="monthly"]')
    page.wait_for_timeout(300)
    x = page.evaluate("document.getElementById('plot-cohort').data[0].x")
    assert len(x) == 12, f"monthly cohort length should be 12, got {len(x)}"


def test_cohort_has_total_trace(page: Page):
    """Both bottom panels should include a 'Total' trace for hover readability."""
    names_daily = page.evaluate(
        "document.getElementById('plot-cohort').data.map(t => t.name)"
    )
    assert "Total" in names_daily, f"cohort daily missing Total: {names_daily}"
    names_pk = page.evaluate(
        "document.getElementById('plot-cohort-pk').data.map(t => t.name)"
    )
    assert "Total" in names_pk, f"cohort peak-week missing Total: {names_pk}"


def test_cohort_yaxis_pinned_across_year_slide(page: Page):
    """REGRESSION GATE for A.22: cohort y-axis must be pinned per state."""
    page.click('[data-cohort-gran="daily"]')
    page.wait_for_timeout(200)
    yrange_a = page.evaluate(
        "document.getElementById('plot-cohort').layout.yaxis.range"
    )
    page.evaluate("""
      const s = document.getElementById('ctl-stockyear');
      s.value = Number(s.max);
      s.dispatchEvent(new Event('input', {bubbles: true}));
    """)
    page.wait_for_timeout(300)
    yrange_b = page.evaluate(
        "document.getElementById('plot-cohort').layout.yaxis.range"
    )
    assert yrange_a == yrange_b, (
        f"cohort y-axis range changed: {yrange_a} → {yrange_b} after year slide"
    )


# === Click-a-state: state selection updates the cohort panel ==============

@pytest.mark.parametrize("postal", STATES_TO_CLICK)
def test_select_state_loads_sidecar_and_rerenders(page: Page, postal):
    """Simulate the same code path the choropleth click handler invokes.
    The dashboard should fetch the state's sidecar (lazy load) and
    update the cohort + trajectory panels to that state."""
    page.evaluate(f"setSelectedState('{postal}');")
    # Wait for sidecar load
    page.wait_for_function(
        f"window.STATE_DATA && window.STATE_DATA['{postal}'] !== undefined",
        timeout=10000,
    )
    page.wait_for_timeout(500)
    # Cohort title should now reference this state
    title = page.evaluate(
        "document.getElementById('plot-cohort').layout.title.text"
    )
    assert postal in (title or ""), (
        f"cohort title doesn't reference {postal}: '{title}'"
    )
    assert page.errors == [], f"errors after selecting {postal}: {page.errors}"
