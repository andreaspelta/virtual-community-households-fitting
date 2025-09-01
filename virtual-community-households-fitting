#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
households_fitting_notebook.py
--------------------------------
Small, self-contained script to read a household raw-data workbook (15-minute readings),
perform the full "Option B" fitting pipeline, print diagnostics, and output the three
CSV files ready for the Monte Carlo engine / Streamlit app.

Supported inputs (auto-detected):
1) WIDE template (timestamps in first column; first data row contains meter IDs for value columns).
2) LONG template (columns: 'id', 'timestamp', and either 'kW_15min' or 'kwh_15min' or 'value').

Core assumptions:
- Raw values are average power in kilowatt over fifteen minutes by default (configurable with --units).
- Quarters are converted to energy as kWh = kW * 0.25, then aggregated to hourly by summing four quarters.
- Six clusters: winter/summer × weekday/saturday/holiday (Italy, Sundays and public holidays treated as holiday).
- Day-scaler D is lognormal with median 1 (ln D ~ Normal(0, sigma^2)); hour-of-day baseline derives from
  median shares; residuals X = ln(E / (mu * D)) are Normal(0, sigma_eta^2).

Outputs (CSV):
- profiles_households.csv        columns: cluster,hour,mu
- day_scalers_households.csv     columns: cluster,shape_sigma,scale
- residuals_households.csv       columns: cluster,phi,sigma_eta,count_hours

Optional artifacts when --plots is set:
- A 'diagnostics' directory containing histograms and QQ plots for ln D and residual X per cluster.
"""

import argparse
import io
import math
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---- Normal inverse CDF (Acklam approximation) so we avoid SciPy dependency ----
def norm_ppf(p: np.ndarray) -> np.ndarray:
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]
    plow = 0.02425
    phigh = 1 - plow
    p = np.asarray(p, dtype=float)
    q = np.zeros_like(p)
    # lower region
    mask = p < plow
    if np.any(mask):
        ql = np.sqrt(-2*np.log(p[mask]))
        q[mask] = (((((c[0]*ql + c[1])*ql + c[2])*ql + c[3])*ql + c[4])*ql + c[5]) / \
                   ((((d[0]*ql + d[1])*ql + d[2])*ql + d[3])*ql + 1)
    # central region
    mask = (p >= plow) & (p <= phigh)
    if np.any(mask):
        r = p[mask] - 0.5
        t = r*r
        q[mask] = (((((a[0]*t + a[1])*t + a[2])*t + a[3])*t + a[4])*t + a[5])*r / \
                   (((((b[0]*t + b[1])*t + b[2])*t + b[3])*t + b[4])*t + 1)
    # upper region
    mask = p > phigh
    if np.any(mask):
        qu = np.sqrt(-2*np.log(1-p[mask]))
        q[mask] = -(((((c[0]*qu + c[1])*qu + c[2])*qu + c[3])*qu + c[4])*qu + c[5]) / \
                    ((((d[0]*qu + d[1])*qu + d[2])*qu + d[3])*qu + 1)
    return q

# ---- Holidays for Italy ----
def compute_easter_monday(year: int):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = ((h + l - 7*m + 114) % 31) + 1
    easter_sunday = datetime(year, month, day)
    return (easter_sunday + timedelta(days=1)).date()

def italian_fixed_holidays(year: int):
    fixed = [(1,1),(1,6),(4,25),(5,1),(6,2),(8,15),(11,1),(12,8),(12,25),(12,26)]
    return {datetime(year, m, d).date() for (m, d) in fixed}

def build_holiday_set(dates: np.ndarray):
    years = sorted({d.year for d in dates})
    hol = set()
    for y in years:
        hol |= italian_fixed_holidays(y)
        hol.add(compute_easter_monday(y))
    return hol

# ---- Clustering helpers ----
def season_from_month(m: int) -> str:
    return "winter" if m in (10,11,12,1,2,3) else "summer"

def daytype_from_weekday_and_holiday(weekday_monday0: int, is_holiday: bool) -> str:
    if is_holiday or weekday_monday0 == 6:
        return "holiday"
    elif weekday_monday0 == 5:
        return "saturday"
    else:
        return "weekday"

def make_cluster(month: int, weekday_monday0: int, is_holiday: bool) -> str:
    return f"{season_from_month(month)}_{daytype_from_weekday_and_holiday(weekday_monday0, is_holiday)}"

# ---- Readers ----
def read_wide_workbook(path: str, sheet_name: Optional[str]=None) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sh = sheet_name or xls.sheet_names[0]
    df0 = pd.read_excel(xls, sheet_name=sh, header=0)
    time_col_name = df0.columns[0]
    # first data row (row index 0) contains meter IDs
    meter_ids = df0.iloc[0, 1:].astype(str).tolist()
    df_clean = df0.iloc[1:, :].copy()
    df_clean.columns = [time_col_name] + meter_ids
    df_clean[time_col_name] = pd.to_datetime(df_clean[time_col_name], errors="coerce", utc=False)
    df_clean = df_clean.dropna(subset=[time_col_name])
    long_df = df_clean.melt(id_vars=[time_col_name], value_vars=meter_ids, var_name="id", value_name="value")
    long_df = long_df.rename(columns={time_col_name: "timestamp"})
    return long_df

def read_long_workbook(path: str, sheet_name: Optional[str]=None) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sh = sheet_name or xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sh, header=0)
    cols = {c.lower(): c for c in df.columns}
    # must have 'timestamp'
    ts_col = None
    for key in ["timestamp", "date time", "datetime", "date"]:
        if key in cols:
            ts_col = cols[key]; break
    if ts_col is None:
        raise ValueError("Cannot find a timestamp column.")
    # id column (optional)
    id_col = None
    for key in ["id", "meter", "meter_id", "pod", "contatore"]:
        if key in cols:
            id_col = cols[key]; break
    if id_col is None:
        df["id"] = "METER_1"
    else:
        df = df.rename(columns={id_col: "id"})
    # value column
    val_col = None
    for key in ["kwh_15min", "kwh", "kw_15min", "kw", "value", "power"]:
        if key in cols:
            val_col = cols[key]; break
    if val_col is None:
        # take the last column as a fallback
        val_col = df.columns[-1]
    df = df.rename(columns={ts_col: "timestamp", val_col: "value"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df = df.dropna(subset=["timestamp", "value"])
    return df[["id","timestamp","value"]]

# ---- Core pipeline ----
def pipeline_from_dataframe(df: pd.DataFrame, units: str="kw", require_full_hours: bool=False, verbose: bool=True, make_plots: bool=False, plot_dir: Optional[str]=None):
    # Clean and coerce
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["value"] >= 0]
    # Convert to kWh per 15-min
    if units.lower() == "kw":
        df["kwh_15min"] = df["value"] * 0.25
    elif units.lower() == "kwh":
        df["kwh_15min"] = df["value"]
    else:
        raise ValueError("--units must be 'kw' or 'kwh'")
    df = df[["id","timestamp","kwh_15min"]]
    # Aggregate to hourly
    hour = df.copy()
    hour["timestamp"] = hour["timestamp"].dt.floor("H")
    if require_full_hours:
        # count quarters and keep only hours with exactly 4 quarters per id,timestamp
        qc = df.copy()
        qc["timestamp_hour"] = qc["timestamp"].dt.floor("H")
        countq = qc.groupby(["id","timestamp_hour"]).size().reset_index(name="nq")
        ok = countq[countq["nq"]==4][["id","timestamp_hour"]]
        hour = hour.merge(ok, left_on=["id","timestamp"], right_on=["id","timestamp_hour"], how="inner").drop(columns=["timestamp_hour"])
    hour = hour.groupby(["id","timestamp"], as_index=False)["kwh_15min"].sum().rename(columns={"kwh_15min":"kwh"})
    # Calendar
    cal = hour.copy()
    cal["date"] = cal["timestamp"].dt.date
    cal["month"] = cal["timestamp"].dt.month
    cal["weekday_monday0"] = cal["timestamp"].dt.weekday
    holset = build_holiday_set(cal["date"].unique())
    cal["is_holiday"] = cal["date"].isin(holset)
    cal["season"] = cal["month"].apply(season_from_month)
    cal["daytype"] = [daytype_from_weekday_and_holiday(w, h) for w,h in zip(cal["weekday_monday0"], cal["is_holiday"])]
    cal["cluster"] = [make_cluster(m, w, h) for m,w,h in zip(cal["month"], cal["weekday_monday0"], cal["is_holiday"])]
    # Day totals & D
    day = cal.groupby(["id","date","cluster"], as_index=False)["kwh"].sum().rename(columns={"kwh":"kwh_day"})
    med_day = day.groupby("cluster", as_index=False)["kwh_day"].median().rename(columns={"kwh_day":"median_day_kwh"})
    day = day.merge(med_day, on="cluster", how="left")
    day = day[day["median_day_kwh"] > 0]
    day["D"] = day["kwh_day"] / day["median_day_kwh"]
    day = day[day["D"] > 0]
    day["lnD"] = np.log(day["D"])
    # Fit ln D
    fitD = day.groupby("cluster")["lnD"].agg(['count','std']).reset_index().rename(columns={"std":"shape_sigma","count":"count_days"})
    fitD["scale"] = 1.0
    fitD = fitD[["cluster","shape_sigma","scale","count_days"]]
    # Hour shares and baseline mu
    h = cal.merge(day[["id","date","cluster","kwh_day","median_day_kwh"]], on=["id","date","cluster"], how="left")
    h = h[(h["kwh_day"] > 0) & (h["kwh"] >= 0)]
    h["share"] = h["kwh"] / h["kwh_day"]
    h["hour"] = h["timestamp"].dt.hour
    share_med = h.groupby(["cluster","hour"], as_index=False)["share"].median()
    mu_rows = []
    for cl in sorted(share_med["cluster"].unique()):
        sub = share_med[share_med["cluster"]==cl]
        shares = np.zeros(24, dtype=float)
        for _, r in sub.iterrows():
            shares[int(r["hour"])] = float(r["share"]) if pd.notnull(r["share"]) else 0.0
        s = shares.sum()
        if s > 0: shares = shares / s
        med = float(med_day.loc[med_day["cluster"]==cl, "median_day_kwh"].iloc[0]) if cl in med_day["cluster"].values else 0.0
        mu = shares * med
        for hh in range(24):
            mu_rows.append((cl, hh, float(mu[hh])))
    profiles_households = pd.DataFrame(mu_rows, columns=["cluster","hour","mu"]).sort_values(["cluster","hour"])
    # Residuals
    mu_map = {(r["cluster"], int(r["hour"])): float(r["mu"]) for _, r in profiles_households.iterrows()}
    h = h[(h["kwh_day"] > 0)]
    h["mu_h"] = [mu_map.get((cl, hr), np.nan) for cl,hr in zip(h["cluster"], h["hour"])]
    h = h[(h["mu_h"] > 0) & (h["kwh"] > 0)]
    h["X"] = np.log(h["kwh"] / (h["mu_h"] * (h["kwh_day"]/h["median_day_kwh"])))
    residual_fit = h.groupby("cluster")["X"].agg(['count','std']).reset_index().rename(columns={"std":"sigma_eta","count":"count_hours"})
    residual_fit = residual_fit[["cluster","sigma_eta","count_hours"]]
    residuals_households = residual_fit.copy()
    residuals_households["phi"] = 0.0
    residuals_households = residuals_households[["cluster","phi","sigma_eta","count_hours"]]
    # Diagnostics printing
    if verbose:
        print("\n=== PIPELINE SUMMARY ===")
        print(f"Raw quarters: {len(df):,}")
        print(f"Hourly rows:  {len(hour):,}")
        print("\nDays per cluster (for ln D):")
        print(fitD[["cluster","count_days","shape_sigma"]].to_string(index=False))
        print("\nResiduals per cluster (X):")
        print(residuals_households[["cluster","count_hours","sigma_eta"]].to_string(index=False))
        # simple goodness metrics for QQ linear fit (R^2), no SciPy
        def qq_r2(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            n = arr.size
            if n < 5: return np.nan
            arr = np.sort(arr)
            probs = (np.arange(1, n+1) - 0.5) / n
            z = norm_ppf(probs)
            xs = (arr - arr.mean()) / (arr.std(ddof=1) if arr.std(ddof=1)>0 else 1.0)
            # linear fit xs ~ a + b z
            A = np.vstack([np.ones_like(z), z]).T
            coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
            yhat = A @ coef
            ss_res = np.sum((xs - yhat)**2)
            ss_tot = np.sum((xs - xs.mean())**2)
            r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
            return float(r2)
        print("\nQQ R^2 (Normal) for ln D by cluster:")
        r2_lnD = []
        for cl in sorted(day["cluster"].unique()):
            arr = day.loc[day["cluster"]==cl, "lnD"].values
            r2 = qq_r2(arr)
            r2_lnD.append((cl, r2))
        for cl, r2 in r2_lnD:
            print(f"  {cl:20s}  R^2 = {r2:.3f}" if not math.isnan(r2) else f"  {cl:20s}  R^2 = NA")
        print("\nQQ R^2 (Normal) for residual X by cluster:")
        r2_X = []
        for cl in sorted(h["cluster"].unique()):
            arr = h.loc[h["cluster"]==cl, "X"].values
            r2 = qq_r2(arr)
            r2_X.append((cl, r2))
        for cl, r2 in r2_X:
            print(f"  {cl:20s}  R^2 = {r2:.3f}" if not math.isnan(r2) else f"  {cl:20s}  R^2 = NA")
    # Optional plots
    if make_plots:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(plot_dir or "diagnostics", exist_ok=True)
        pdir = plot_dir or "diagnostics"
        def save_hist(arr, title, fname):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size < 5: return
            plt.figure(figsize=(6,3.5))
            plt.hist(arr, bins=32, edgecolor="black")
            plt.title(title); plt.xlabel("Value"); plt.ylabel("Frequency")
            plt.tight_layout(); plt.savefig(f"{pdir}/{fname}", dpi=120); plt.close()
        def save_qq(arr, title, fname):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size < 5: return
            arr = np.sort(arr)
            probs = (np.arange(1, arr.size+1) - 0.5) / arr.size
            z = norm_ppf(probs)
            xs = (arr - arr.mean()) / (arr.std(ddof=1) if arr.std(ddof=1)>0 else 1.0)
            lo = float(min(z.min(), xs.min())); hi = float(max(z.max(), xs.max()))
            plt.figure(figsize=(6,3.5))
            plt.scatter(z, xs, s=8)
            plt.plot([lo, hi], [lo, hi])
            plt.title(title); plt.xlabel("Theoretical quantiles"); plt.ylabel("Sample quantiles")
            plt.tight_layout(); plt.savefig(f"{pdir}/{fname}", dpi=120); plt.close()
        # ln D diagnostics
        for cl in sorted(day["cluster"].unique()):
            arr = day.loc[day["cluster"]==cl, "lnD"].values
            save_hist(arr, f"{cl} — ln D histogram", f"{cl}_lnD_hist.png")
            save_qq(arr, f"{cl} — ln D QQ (Normal)", f"{cl}_lnD_qq.png")
        # residual X diagnostics
        for cl in sorted(h["cluster"].unique()):
            arr = h.loc[h["cluster"]==cl, "X"].values
            save_hist(arr, f"{cl} — residual X histogram", f"{cl}_X_hist.png")
            save_qq(arr, f"{cl} — residual X QQ (Normal)", f"{cl}_X_qq.png")
    # Export CSVs
    day_scalers_households = fitD[["cluster","shape_sigma","scale"]].copy()
    profiles_csv = profiles_households[["cluster","hour","mu"]].copy()
    residuals_csv = residuals_households.copy()
    return profiles_csv, day_scalers_households, residuals_csv

def run_pipeline(input_path: str,
                 sheet: Optional[str],
                 units: str,
                 outdir: str,
                 require_full_hours: bool=False,
                 make_plots: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Try WIDE first, fall back to LONG
    try:
        df = read_wide_workbook(input_path, sheet_name=sheet)
        mode = "WIDE"
    except Exception:
        df = read_long_workbook(input_path, sheet_name=sheet)
        mode = "LONG"
    print(f"Loaded input as {mode} format with {len(df):,} rows.")
    profiles, day_scalers, residuals = pipeline_from_dataframe(
        df, units=units, require_full_hours=require_full_hours, verbose=True, make_plots=make_plots
    )
    # Save CSVs
    import os
    os.makedirs(outdir, exist_ok=True)
    profiles.to_csv(f"{outdir}/profiles_households.csv", index=False)
    day_scalers.to_csv(f"{outdir}/day_scalers_households.csv", index=False)
    residuals.to_csv(f"{outdir}/residuals_households.csv", index=False)
    print("\nSaved CSVs to:", outdir)
    print(" - profiles_households.csv")
    print(" - day_scalers_households.csv")
    print(" - residuals_households.csv")
    return profiles, day_scalers, residuals

def main():
    ap = argparse.ArgumentParser(description="Fit households Option B parameters from 15-min raw data workbook.")
    ap.add_argument("--input", required=True, help="Path to raw workbook (.xlsx)")
    ap.add_argument("--sheet", default=None, help="Sheet name (default: first sheet)")
    ap.add_argument("--units", default="kw", choices=["kw","kwh"], help="Units of the 15-min values (kw=average kW over 15 minutes; kwh=energy per 15 minutes)")
    ap.add_argument("--outdir", default="out_households", help="Output directory for CSVs (and plots if requested)")
    ap.add_argument("--strict-hours", action="store_true", help="Require exactly four quarters per hour; else drop the hour")
    ap.add_argument("--plots", action="store_true", help="Save histograms and QQ plots into ./diagnostics")
    args = ap.parse_args()
    run_pipeline(args.input, args.sheet, args.units, args.outdir, require_full_hours=args.strict_hours, make_plots=args.plots)

if __name__ == "__main__":
    main()
