import csv
import numpy as np
from math import sqrt
from collections import OrderedDict, defaultdict

ref_path = '/projects/geohc/geo_predict/outputs/outputs_2025-12-10-15-53-30/agg_com_eulp_hvac_elec_GWh_upgrade0.csv'
reg_path = '/projects/geohc/geo_predict/outputs/outputs_2025-12-02-09-29-48/agg_com_eulp_hvac_elec_GWh_upgrade0.csv'

# Load into dicts keyed by timestamp

def load(path):
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        counties = header[1:]
        data = OrderedDict()
        for row in reader:
            ts = row[0]
            data[ts] = np.array([float(x) for x in row[1:]], dtype=float)
    return counties, data

counties_ref, data_ref = load(ref_path)
counties_reg, data_reg = load(reg_path)
if counties_ref != counties_reg:
    raise SystemExit('Header mismatch')
counties = counties_ref
n = len(counties)

# Align timestamps intersection
keys_ref = set(data_ref.keys())
keys_reg = set(data_reg.keys())
common = sorted(keys_ref & keys_reg)
miss_ref_only = sorted(keys_ref - keys_reg)
miss_reg_only = sorted(keys_reg - keys_ref)

if not common:
    raise SystemExit('No common timestamps')

nat_ref = []
nat_reg = []
months = []
hours = []
dates = []
county_hour_ref = []
county_hour_reg = []

for ts in common:
    vals_ref = data_ref[ts]
    vals_reg = data_reg[ts]
    nat_ref.append(float(vals_ref.sum()))
    nat_reg.append(float(vals_reg.sum()))
    county_hour_ref.append(vals_ref)
    county_hour_reg.append(vals_reg)
    months.append(int(ts[5:7]))
    hours.append(int(ts[11:13]))
    dates.append(ts[:10])

nat_ref = np.array(nat_ref)
nat_reg = np.array(nat_reg)
months_arr = np.array(months)
hours_arr = np.array(hours)
dates_arr = np.array(dates)
county_hour_ref = np.vstack(county_hour_ref)
county_hour_reg = np.vstack(county_hour_reg)

# Aggregations
monthly_nat_ref = np.zeros(12)
monthly_nat_reg = np.zeros(12)
season_names = ['DJF','MAM','JJA','SON']
season_ref = dict.fromkeys(season_names, 0.0)
season_reg = dict.fromkeys(season_names, 0.0)
season_lookup = {12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'}

for m, nr, ng in zip(months_arr, nat_ref, nat_reg):
    monthly_nat_ref[m-1] += nr
    monthly_nat_reg[m-1] += ng
    season_ref[season_lookup[m]] += nr
    season_reg[season_lookup[m]] += ng

county_ann_ref = county_hour_ref.sum(axis=0)
county_ann_reg = county_hour_reg.sum(axis=0)
monthly_county_ref = np.zeros((12, n))
monthly_county_reg = np.zeros((12, n))
for m in range(1,13):
    mask = months_arr == m
    if mask.any():
        monthly_county_ref[m-1] = county_hour_ref[mask].sum(axis=0)
        monthly_county_reg[m-1] = county_hour_reg[mask].sum(axis=0)

annual_ref = county_ann_ref.sum()
annual_reg = county_ann_reg.sum()
annual_pct = (annual_reg - annual_ref) / annual_ref * 100
monthly_pct = (monthly_nat_reg - monthly_nat_ref) / monthly_nat_ref * 100
seasonal_pct = {k: (season_reg[k]-season_ref[k]) / season_ref[k] * 100 for k in season_ref}

county_pct = np.full(n, np.nan)
nonzero = county_ann_ref != 0
county_pct[nonzero] = (county_ann_reg[nonzero] - county_ann_ref[nonzero]) / county_ann_ref[nonzero] * 100
abs_pct = np.abs(county_pct[np.isfinite(county_pct)])
mean_abs = float(abs_pct.mean())
median_abs = float(np.median(abs_pct))
max_pos_idx = int(np.nanargmax(county_pct))
max_neg_idx = int(np.nanargmin(county_pct))
max_pos = (counties[max_pos_idx], float(county_pct[max_pos_idx]))
max_neg = (counties[max_neg_idx], float(county_pct[max_neg_idx]))

sorted_pos = np.argsort(-county_pct)
sorted_neg = np.argsort(county_pct)

def top_list(indices, k=5):
    out = []
    for i in indices[:k]:
        if np.isnan(county_pct[i]):
            continue
        out.append((counties[i], float(county_pct[i])))
    return out
pos5 = top_list(sorted_pos)
neg5 = top_list(sorted_neg)

with np.errstate(divide='ignore', invalid='ignore'):
    mape_cm = np.abs((monthly_county_reg - monthly_county_ref) / monthly_county_ref)

mape_cm_flat = mape_cm[np.isfinite(mape_cm) & (monthly_county_ref != 0)]
mean_mape_cm = float(mape_cm_flat.mean()*100)
median_mape_cm = float(np.median(mape_cm_flat)*100)

def corr(a,b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_mean = a.mean()
    b_mean = b.mean()
    numer = ((a - a_mean)*(b - b_mean)).sum()
    denom = sqrt(((a - a_mean)**2).sum() * ((b - b_mean)**2).sum())
    return float(numer/denom) if denom else float('nan')

hourly_corr = corr(nat_ref, nat_reg)

daily_ref = defaultdict(float)
daily_reg = defaultdict(float)
for d, r, g in zip(dates_arr, nat_ref, nat_reg):
    daily_ref[d] += r
    daily_reg[d] += g

days_sorted = sorted(daily_ref.keys())
daily_ref_arr = np.array([daily_ref[d] for d in days_sorted])
daily_reg_arr = np.array([daily_reg[d] for d in days_sorted])
daily_corr = corr(daily_ref_arr, daily_reg_arr)

monthly_diurnal_corr = []
for m in range(1,13):
    mask = months_arr == m
    if not mask.any():
        monthly_diurnal_corr.append((m, float('nan')))
        continue
    hours_m = hours_arr[mask]
    weights_ref = nat_ref[mask]
    weights_reg = nat_reg[mask]
    counts = np.bincount(hours_m, minlength=24)
    prof_ref = np.bincount(hours_m, weights=weights_ref, minlength=24) / counts
    prof_reg = np.bincount(hours_m, weights=weights_reg, minlength=24) / counts
    monthly_diurnal_corr.append((m, corr(prof_ref, prof_reg)))

peak_ref_idx = int(np.argmax(nat_ref))
peak_reg_idx = int(np.argmax(nat_reg))
peak_ref_val = float(nat_ref[peak_ref_idx])
peak_reg_val = float(nat_reg[peak_reg_idx])
peak_ref_ts = common[peak_ref_idx]
peak_reg_ts = common[peak_reg_idx]
peak_diff_pct = (peak_reg_val - peak_ref_val) / peak_ref_val * 100
reg_at_ref_peak = float(nat_reg[peak_ref_idx])
ref_at_reg_peak = float(nat_ref[peak_reg_idx])

pctl = [90,95,99]
ref_pct = {p: float(np.percentile(nat_ref, p)) for p in pctl}
reg_pct = {p: float(np.percentile(nat_reg, p)) for p in pctl}

lf_ref = annual_ref / (peak_ref_val * len(nat_ref))
lf_reg = annual_reg / (peak_reg_val * len(nat_ref))

nat_rmse = sqrt(np.mean((nat_reg - nat_ref)**2))
county_rmse = np.sqrt(np.mean((county_hour_reg - county_hour_ref)**2, axis=0))
mean_rmse = float(np.mean(county_rmse))
median_rmse = float(np.median(county_rmse))
max_rmse_idx = int(np.argmax(county_rmse))
min_rmse_idx = int(np.argmin(county_rmse))

county_peak_ref = county_hour_ref.max(axis=0)
county_peak_reg = county_hour_reg.max(axis=0)
with np.errstate(divide='ignore', invalid='ignore'):
    county_peak_pct = (county_peak_reg - county_peak_ref) / county_peak_ref * 100
finite_peak = county_peak_pct[np.isfinite(county_peak_pct)]
peak_mean = float(np.mean(finite_peak))
peak_median = float(np.median(finite_peak))
peak_max_idx = int(np.nanargmax(county_peak_pct))
peak_min_idx = int(np.nanargmin(county_peak_pct))

print('Ref rows:', len(keys_ref), 'Reg rows:', len(keys_reg), 'Common:', len(common))
print('Missing in reg:', len(miss_ref_only), 'Missing in ref:', len(miss_reg_only))

print('Hours used:', len(nat_ref), 'Counties:', n)
print('Annual GWh ref={:,.2f}, reg={:,.2f}, pct diff={:.2f}%'.format(annual_ref, annual_reg, annual_pct))
print('\nSeasonal pct diff (reg-ref)/ref:')
for k in ['DJF','MAM','JJA','SON']:
    print(f'  {k}: {seasonal_pct[k]:.2f}% (ref {season_ref[k]:,.2f} GWh, reg {season_reg[k]:,.2f} GWh)')

print('\nMonthly pct diff (reg-ref)/ref:')
for i,pct in enumerate(monthly_pct,1):
    print(f'  {i:02d}: {pct:.2f}%')

print('\nCounty annual pct diff stats:')
print(f'  Mean abs pct diff: {mean_abs:.2f}%')
print(f'  Median abs pct diff: {median_abs:.2f}%')
print(f'  Max positive pct diff: {max_pos[1]:.2f}% (FIPS {max_pos[0]})')
print(f'  Max negative pct diff: {max_neg[1]:.2f}% (FIPS {max_neg[0]})')
print('  Top +5 counties (pct diff):')
for fips, pct in pos5:
    print(f'    {fips}: {pct:.2f}%')
print('  Top -5 counties (pct diff):')
for fips, pct in neg5:
    print(f'    {fips}: {pct:.2f}%')

print('\nCounty-month MAPE (ref>0): mean {:.2f}%, median {:.2f}%'.format(mean_mape_cm, median_mape_cm))
print('National hourly correlation: {:.4f}'.format(hourly_corr))
print('Daily-total correlation: {:.4f}'.format(daily_corr))
print('Load factor ref={:.3f}, reg={:.3f}, delta={:.3f}'.format(lf_ref, lf_reg, lf_reg - lf_ref))

print('\nDiurnal profile correlation by month (0-23 avg hour):')
for m,c in monthly_diurnal_corr:
    print(f'  Month {m:02d}: {c:.4f}')

print('\nNational hourly RMSE (GWh): {:.3f}'.format(nat_rmse))
print('County hourly RMSE stats (GWh): mean {:.3f}, median {:.3f}, max {:.3f} (FIPS {}), min {:.3f} (FIPS {})'.format(mean_rmse, median_rmse, county_rmse[max_rmse_idx], counties[max_rmse_idx], county_rmse[min_rmse_idx], counties[min_rmse_idx]))

print('\nPeak demand (national totals, GWh):')
print('  Ref peak: {:.3f} at {}'.format(peak_ref_val, peak_ref_ts))
print('  Reg peak: {:.3f} at {}'.format(peak_reg_val, peak_reg_ts))
print('  Peak percent diff: {:.2f}%'.format(peak_diff_pct))
print('  Reg at ref-peak hour: {:.3f} (pct diff vs ref {:.2f}%)'.format(reg_at_ref_peak, (reg_at_ref_peak-peak_ref_val)/peak_ref_val*100))
print('  Ref at reg-peak hour: {:.3f} (pct diff vs reg {:.2f}%)'.format(ref_at_reg_peak, (ref_at_reg_peak-peak_reg_val)/peak_reg_val*100))

print('\nNational high-load thresholds (GWh):')
for p in pctl:
    pdiff = (reg_pct[p]-ref_pct[p])/ref_pct[p]*100
    print('  {:>3}%: ref {:.3f}, reg {:.3f}, pct diff {:.2f}%'.format(p, ref_pct[p], reg_pct[p], pdiff))

print('\nCounty peak percent diff (reg-ref)/ref: mean {:.2f}%, median {:.2f}%, max {:.2f}% (FIPS {}), min {:.2f}% (FIPS {})'.format(peak_mean, peak_median, county_peak_pct[peak_max_idx], counties[peak_max_idx], county_peak_pct[peak_min_idx], counties[peak_min_idx]))
