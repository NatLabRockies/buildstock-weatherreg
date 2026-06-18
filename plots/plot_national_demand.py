#!/usr/bin/env python3
"""Plot national residential and commercial hourly demand from two data sources."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
PLOT_DIR = SCRIPT_DIR / "plots"
SECTOR_TOTALS_DIR = OUTPUT_DIR / "sector_totals"
COMPARISON_CSV = SCRIPT_DIR / "inputs" / "conus_profiles_comparison.csv"
DEFAULT_SCENARIO = "Baseline"
DEFAULT_SOURCE = "20250512_eer_load_2018.hyper/Extract"
DEFAULT_SCENARIO_LABEL = "Reference Demand"


def read_sector_totals(scenario_name=DEFAULT_SCENARIO):
    path = SECTOR_TOTALS_DIR / f"{scenario_name}_national_sector_totals.csv"
    if not path.exists():
        available = sorted(p.name for p in SECTOR_TOTALS_DIR.glob("*_national_sector_totals.csv")) if SECTOR_TOTALS_DIR.exists() else []
        raise FileNotFoundError(
            f"Could not find sector totals for scenario '{scenario_name}'.\n"
            f"Expected file: {path}\n"
            f"Available files: {available}"
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    required = ["residential_MWh", "commercial_MWh"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in sector totals file: {missing}")
    return df[["residential_MWh", "commercial_MWh"]]


def read_comparison_csv(source=DEFAULT_SOURCE, scenario=DEFAULT_SCENARIO_LABEL):
    df = pd.read_csv(COMPARISON_CSV)
    df = df[df["Source (group)"] == "ESS 2025"]
    df = df[df["Source1"] == source]
    df = df[df["Scenario1"] == scenario]
    df = df[df["Sector groups"].isin(["Residential", "Commercial"])]
    if df.empty:
        raise ValueError(f"No comparison CSV rows found for Source (group)='ESS 2025', source={source}, scenario={scenario}")

    # Parse datetime from "Hour of Date time" column only
    timestamp = (
        df["Hour of Date time"]
        .astype(str)
        .str.replace("\u202f", " ", regex=False)
    )
    df = df.assign(timestamp=pd.to_datetime(timestamp, format="%B %d, %Y at %I %p", errors="raise"))
    
    # Verify we have 2018 data
    if not all(df['timestamp'].dt.year == 2018):
        raise ValueError("Expected all data to be from 2018, but found other years in 'Hour of Date time'")
    
    df = df.set_index("timestamp")
    pivot = df.pivot_table(index=df.index, columns="Sector groups", values="GWh, mod", aggfunc="sum")
    if "Residential" not in pivot.columns or "Commercial" not in pivot.columns:
        raise ValueError("Comparison CSV does not contain both Residential and Commercial series.")
    result = pivot[["Residential", "Commercial"]].rename(columns={
        "Residential": "residential_MWh",
        "Commercial": "commercial_MWh",
    }) * 1000.0
    return result.sort_index()


def find_peak_week(df, months):
    mask = df.index.month.isin(months)
    if not mask.any():
        raise ValueError(f"No data found in months {months}")
    total = df["residential_MWh"] + df["commercial_MWh"]
    peak_timestamp = total[mask].idxmax()
    # Find the calendar week starting on Sunday and ending on Saturday.
    days_to_sunday = (peak_timestamp.weekday() + 1) % 7
    start = peak_timestamp - pd.Timedelta(days=days_to_sunday)
    end = start + pd.Timedelta(hours=167)
    week = df.loc[start:end]
    if len(week) != 168:
        full_index = pd.date_range(start=start, end=end, freq="H")
        week = week.reindex(full_index, fill_value=0.0)
    return week


def plot_weekly_stacked_area(df, title, ax):
    stacked_df = df[["residential_MWh", "commercial_MWh"]].copy() / 1000.0  # Convert MWh to GWh
    stacked_df = stacked_df.fillna(0.0)  # Fill NaN values with 0
    ax.stackplot(
        stacked_df.index,
        stacked_df["residential_MWh"],
        stacked_df["commercial_MWh"],
        labels=["Residential", "Commercial"],
        colors=["#1f77b4", "#ff7f0e"],
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_ylabel("Load (GWh)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax.set_xlim(stacked_df.index[0], stacked_df.index[-1])
    xticks = pd.date_range(stacked_df.index[0], stacked_df.index[-1], freq="24H")
    ax.set_xticks(xticks)
    ax.set_xticklabels([t.strftime("%a %b %d") for t in xticks], rotation=45, ha="right")


def plot_peak_weeks_by_source(df_combined, df_comparison, season, output_path):
    """Plot peak weeks from each source side-by-side for a given season, with year labels.
    
    Creates one figure per season with combined outputs on the left and comparison CSV on the right.
    """
    months = [12, 1, 2] if season == "winter" else [6, 7, 8]
    
    # Find peak weeks from each source
    week_combined = find_peak_week(df_combined, months)
    week_comparison = find_peak_week(df_comparison, months)
    
    yyyy_combined = week_combined.index[0].year
    yyyy_comparison = week_comparison.index[0].year
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=False, sharey=True)
    
    title_combined = f"Buildstock ({DEFAULT_SCENARIO}): {season.title()} Peak Week ({yyyy_combined})"
    title_comparison = f"EER: {season.title()} Peak Week ({yyyy_comparison})"
    
    plot_weekly_stacked_area(week_combined, title_combined, axes[0])
    plot_weekly_stacked_area(week_comparison, title_comparison, axes[1])
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    combined = read_sector_totals(DEFAULT_SCENARIO)
    comparison = read_comparison_csv()
    
    winter_output = PLOT_DIR / f"peak_winter_demand_{DEFAULT_SCENARIO}.png"
    summer_output = PLOT_DIR / f"peak_summer_demand_{DEFAULT_SCENARIO}.png"
    
    plot_peak_weeks_by_source(combined, comparison, "winter", winter_output)
    plot_peak_weeks_by_source(combined, comparison, "summer", summer_output)


if __name__ == "__main__":
    main()
