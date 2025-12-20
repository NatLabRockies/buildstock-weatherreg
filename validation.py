import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REF_PATH = Path(r"C:\ReEDS\geo_predict\validation_2025-12-19\com_0_ref_outputs_2025-12-10-15-53-30\agg_com_eulp_hvac_elec_GWh_upgrade0.csv")
REG_PATH = Path(r"C:\ReEDS\geo_predict\validation_2025-12-19\com_0_reg_outputs_2025-12-02-09-29-48\agg_com_eulp_hvac_elec_GWh_upgrade0.csv")
OUTPUT_DIR = Path("validation_outputs")

SEASON_LOOKUP = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
PCTL = [90, 95, 99]


def load_csv(path: Path):
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"No data in {path}")
    ts_col = df.columns[0]
    counties = list(df.columns[1:])
    # Parse timestamps robustly: allow mixed date-only and datetime strings.
    ts_parsed = pd.to_datetime(df[ts_col], format="ISO8601", errors="coerce")
    if ts_parsed.isna().any():
        ts_parsed = pd.to_datetime(df[ts_col], errors="coerce")
    if ts_parsed.isna().any():
        bad_samples = df.loc[ts_parsed.isna(), ts_col].head(5).tolist()
        raise SystemExit(f"Unparseable timestamps in {path}: {bad_samples}")
    df[ts_col] = ts_parsed
    df = df.set_index(ts_col)
    # If there are duplicate timestamps, keep the last occurrence to match prior dict overwrite behavior.
    df = df[~df.index.duplicated(keep="last")]
    return counties, df[counties]


def align_frames(df_ref: pd.DataFrame, df_reg: pd.DataFrame):
    common_idx = df_ref.index.intersection(df_reg.index).sort_values()
    miss_ref_only = df_ref.index.difference(df_reg.index)
    miss_reg_only = df_reg.index.difference(df_ref.index)
    if common_idx.empty:
        raise SystemExit("No common timestamps")
    df_ref_aligned = df_ref.loc[common_idx]
    df_reg_aligned = df_reg.loc[common_idx]
    return df_ref_aligned, df_reg_aligned, common_idx, miss_ref_only, miss_reg_only


def monthly_totals(obj):
    return obj.groupby(obj.index.month).sum().reindex(range(1, 13), fill_value=0.0)


def seasonal_totals(series: pd.Series):
    return series.groupby(series.index.month.map(SEASON_LOOKUP)).sum().reindex(SEASON_ORDER, fill_value=0.0)


def diurnal_corr_by_month(nat_ref: pd.Series, nat_reg: pd.Series):
    df = pd.DataFrame({"ref": nat_ref, "reg": nat_reg})
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    out = []
    for m in range(1, 13):
        month_slice = df[df["month"] == m]
        if month_slice.empty:
            out.append((m, float("nan")))
            continue
        prof = month_slice.groupby("hour").mean().reindex(range(24), fill_value=np.nan)
        out.append((m, float(prof["ref"].corr(prof["reg"]))))
    return out


def plot_monthly_pct(monthly_pct: pd.Series, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="k", linewidth=0.8)
    ax.bar(monthly_pct.index, monthly_pct.values, color="#3b82f6")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Percent diff (reg - ref) / ref")
    ax.set_title("Monthly percent difference (national)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_top_county_pct(county_pct: pd.Series, out_path: Path):
    series = pd.concat(
        [
            county_pct.dropna().sort_values(ascending=False).head(5),
            county_pct.dropna().sort_values().head(5),
        ]
    )
    if series.empty:
        return
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in series.values]
    fig, ax = plt.subplots(figsize=(8, 4))
    series[::-1].plot.barh(ax=ax, color=colors[::-1])
    ax.set_xlabel("Percent diff (reg - ref) / ref")
    ax.set_title("Top +/- county annual percent differences")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_daily_scatter(daily_ref: pd.Series, daily_reg: pd.Series, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(daily_ref.values, daily_reg.values, alpha=0.6, color="#0ea5e9", edgecolor="white", linewidth=0.5)
    lims = [
        min(daily_ref.min(), daily_reg.min()),
        max(daily_ref.max(), daily_reg.max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Daily ref total (GWh)")
    ax.set_ylabel("Daily reg total (GWh)")
    ax.set_title("Daily totals: reg vs ref")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hourly_scatter(nat_ref: pd.Series, nat_reg: pd.Series, out_path: Path):
    aligned = pd.concat([nat_ref, nat_reg], axis=1, join="inner").dropna()
    aligned.columns = ["ref", "reg"]
    if aligned.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(aligned["ref"].values, aligned["reg"].values, alpha=0.3, color="#6366f1", edgecolor="white", linewidth=0.3, s=8)
    lims = [
        min(aligned["ref"].min(), aligned["reg"].min()),
        max(aligned["ref"].max(), aligned["reg"].max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Hourly ref total (GWh)")
    ax.set_ylabel("Hourly reg total (GWh)")
    ax.set_title("Hourly totals: reg vs ref")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    counties_ref, df_ref_raw = load_csv(REF_PATH)
    counties_reg, df_reg_raw = load_csv(REG_PATH)
    if counties_ref != counties_reg:
        raise SystemExit("Header mismatch")
    counties = counties_ref
    n = len(counties)

    df_ref, df_reg, common_idx, miss_ref_only, miss_reg_only = align_frames(df_ref_raw, df_reg_raw)

    nat_ref = df_ref.sum(axis=1)
    nat_reg = df_reg.sum(axis=1)

    monthly_nat_ref = monthly_totals(nat_ref)
    monthly_nat_reg = monthly_totals(nat_reg)
    season_ref = seasonal_totals(nat_ref)
    season_reg = seasonal_totals(nat_reg)

    county_ann_ref = df_ref.sum(axis=0)
    county_ann_reg = df_reg.sum(axis=0)
    monthly_county_ref = monthly_totals(df_ref)
    monthly_county_reg = monthly_totals(df_reg)

    annual_ref = float(nat_ref.sum())
    annual_reg = float(nat_reg.sum())
    annual_pct = (annual_reg - annual_ref) / annual_ref * 100 if annual_ref else float("nan")

    monthly_pct = (monthly_nat_reg - monthly_nat_ref).div(monthly_nat_ref.replace(0, np.nan)) * 100
    seasonal_pct = (season_reg - season_ref).div(season_ref.replace(0, np.nan)) * 100

    county_pct = (county_ann_reg - county_ann_ref).div(county_ann_ref.replace(0, np.nan)) * 100
    abs_pct = county_pct.dropna().abs()
    mean_abs = float(abs_pct.mean()) if not abs_pct.empty else float("nan")
    median_abs = float(abs_pct.median()) if not abs_pct.empty else float("nan")
    max_pos = county_pct.idxmax() if not county_pct.dropna().empty else None
    max_neg = county_pct.idxmin() if not county_pct.dropna().empty else None

    pos5 = county_pct.dropna().sort_values(ascending=False).head(5)
    neg5 = county_pct.dropna().sort_values().head(5)

    mape_cm = (monthly_county_reg - monthly_county_ref).abs().div(monthly_county_ref.replace(0, np.nan))
    mape_cm_flat = mape_cm.stack().dropna()
    mean_mape_cm = float(mape_cm_flat.mean() * 100) if not mape_cm_flat.empty else float("nan")
    median_mape_cm = float(mape_cm_flat.median() * 100) if not mape_cm_flat.empty else float("nan")

    hourly_corr = float(nat_ref.corr(nat_reg))

    daily_ref = nat_ref.resample("D").sum()
    daily_reg = nat_reg.resample("D").sum()
    daily_corr = float(daily_ref.corr(daily_reg))

    monthly_diurnal_corr = diurnal_corr_by_month(nat_ref, nat_reg)

    peak_ref_val = float(nat_ref.max())
    peak_reg_val = float(nat_reg.max())
    peak_ref_ts = nat_ref.idxmax()
    peak_reg_ts = nat_reg.idxmax()
    peak_diff_pct = (peak_reg_val - peak_ref_val) / peak_ref_val * 100 if peak_ref_val else float("nan")
    reg_at_ref_peak = float(nat_reg.loc[peak_ref_ts])
    ref_at_reg_peak = float(nat_ref.loc[peak_reg_ts])

    ref_pct = nat_ref.quantile([p / 100 for p in PCTL])
    reg_pct = nat_reg.quantile([p / 100 for p in PCTL])

    lf_ref = annual_ref / (peak_ref_val * len(nat_ref)) if peak_ref_val else float("nan")
    lf_reg = annual_reg / (peak_reg_val * len(nat_ref)) if peak_reg_val else float("nan")

    nat_rmse = float(np.sqrt(((nat_reg - nat_ref) ** 2).mean()))
    county_rmse = np.sqrt(((df_reg - df_ref) ** 2).mean())
    mean_rmse = float(county_rmse.mean())
    median_rmse = float(county_rmse.median())
    max_rmse_idx = county_rmse.idxmax()
    min_rmse_idx = county_rmse.idxmin()

    county_peak_ref = df_ref.max()
    county_peak_reg = df_reg.max()
    county_peak_pct = (county_peak_reg - county_peak_ref).div(county_peak_ref.replace(0, np.nan)) * 100
    finite_peak = county_peak_pct.dropna()
    peak_mean = float(finite_peak.mean()) if not finite_peak.empty else float("nan")
    peak_median = float(finite_peak.median()) if not finite_peak.empty else float("nan")
    if finite_peak.empty:
        peak_max_idx = None
        peak_min_idx = None
    else:
        peak_max_idx = county_peak_pct.idxmax()
        peak_min_idx = county_peak_pct.idxmin()

    print("Ref rows:", len(df_ref_raw), "Reg rows:", len(df_reg_raw), "Common:", len(common_idx))
    print("Missing in reg:", len(miss_ref_only), "Missing in ref:", len(miss_reg_only))

    print("Hours used:", len(common_idx), "Counties:", n)
    print("Annual GWh ref={:,.2f}, reg={:,.2f}, pct diff={:.2f}%".format(annual_ref, annual_reg, annual_pct))
    print("\nSeasonal pct diff (reg-ref)/ref:")
    for k in SEASON_ORDER:
        print(f"  {k}: {seasonal_pct[k]:.2f}% (ref {season_ref[k]:,.2f} GWh, reg {season_reg[k]:,.2f} GWh)")

    print("\nMonthly pct diff (reg-ref)/ref:")
    for m in range(1, 13):
        print(f"  {m:02d}: {monthly_pct[m]:.2f}%")

    print("\nCounty annual pct diff stats:")
    print(f"  Mean abs pct diff: {mean_abs:.2f}%")
    print(f"  Median abs pct diff: {median_abs:.2f}%")
    if max_pos is not None and max_neg is not None:
        print(f"  Max positive pct diff: {county_pct[max_pos]:.2f}% (FIPS {max_pos})")
        print(f"  Max negative pct diff: {county_pct[max_neg]:.2f}% (FIPS {max_neg})")
    print("  Top +5 counties (pct diff):")
    for fips, pct in pos5.items():
        print(f"    {fips}: {pct:.2f}%")
    print("  Top -5 counties (pct diff):")
    for fips, pct in neg5.items():
        print(f"    {fips}: {pct:.2f}%")

    print("\nCounty-month MAPE (ref>0): mean {:.2f}%, median {:.2f}%".format(mean_mape_cm, median_mape_cm))
    print("National hourly correlation: {:.4f}".format(hourly_corr))
    print("Daily-total correlation: {:.4f}".format(daily_corr))
    print("Load factor ref={:.3f}, reg={:.3f}, delta={:.3f}".format(lf_ref, lf_reg, lf_reg - lf_ref))

    print("\nDiurnal profile correlation by month (0-23 avg hour):")
    for m, c in monthly_diurnal_corr:
        print(f"  Month {m:02d}: {c:.4f}")

    print("\nNational hourly RMSE (GWh): {:.3f}".format(nat_rmse))
    print(
        "County hourly RMSE stats (GWh): mean {:.3f}, median {:.3f}, max {:.3f} (FIPS {}), min {:.3f} (FIPS {})".format(
            mean_rmse, median_rmse, county_rmse[max_rmse_idx], max_rmse_idx, county_rmse[min_rmse_idx], min_rmse_idx
        )
    )

    print("\nPeak demand (national totals, GWh):")
    print("  Ref peak: {:.3f} at {}".format(peak_ref_val, peak_ref_ts))
    print("  Reg peak: {:.3f} at {}".format(peak_reg_val, peak_reg_ts))
    print("  Peak percent diff: {:.2f}%".format(peak_diff_pct))
    print(
        "  Reg at ref-peak hour: {:.3f} (pct diff vs ref {:.2f}%)".format(
            reg_at_ref_peak, (reg_at_ref_peak - peak_ref_val) / peak_ref_val * 100 if peak_ref_val else float("nan")
        )
    )
    print(
        "  Ref at reg-peak hour: {:.3f} (pct diff vs reg {:.2f}%)".format(
            ref_at_reg_peak, (ref_at_reg_peak - peak_reg_val) / peak_reg_val * 100 if peak_reg_val else float("nan")
        )
    )

    print("\nNational high-load thresholds (GWh):")
    for p in PCTL:
        pdiff = (reg_pct.loc[p / 100] - ref_pct.loc[p / 100]) / ref_pct.loc[p / 100] * 100
        print("  {:>3}%: ref {:.3f}, reg {:.3f}, pct diff {:.2f}%".format(p, ref_pct.loc[p / 100], reg_pct.loc[p / 100], pdiff))

    print(
        "\nCounty peak percent diff (reg-ref)/ref: mean {:.2f}%, median {:.2f}%".format(
            peak_mean, peak_median
        )
    )
    if peak_max_idx is not None and peak_min_idx is not None:
        print(
            "  Max {:.2f}% (FIPS {}), Min {:.2f}% (FIPS {})".format(
                county_peak_pct[peak_max_idx], peak_max_idx, county_peak_pct[peak_min_idx], peak_min_idx
            )
        )

    plot_monthly_pct(monthly_pct, OUTPUT_DIR / "monthly_percent_diff.png")
    plot_top_county_pct(county_pct, OUTPUT_DIR / "county_percent_diff_top.png")
    plot_daily_scatter(daily_ref, daily_reg, OUTPUT_DIR / "daily_totals_scatter.png")
    plot_hourly_scatter(nat_ref, nat_reg, OUTPUT_DIR / "hourly_totals_scatter.png")
    print(f"\nCharts written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
