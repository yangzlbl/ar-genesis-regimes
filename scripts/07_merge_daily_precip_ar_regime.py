#!/usr/bin/env python3
"""
Merge daily precipitation metrics with daily AR/regime labels.

Definitions:
  wet day: precip metric > threshold
  non-AR precip day: wet day and Tar_day == 0
  AR precip day: wet day and Tar_day == 1

Regime-specific primary sample:
  precip_day_type == ar_precip and daily_AR_regime_status == single_known_regime
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

REGIME_ORDER = ["High-moisture", "Coupled-cyclonic", "Frontal", "Ridge"]


def bootstrap_ci_ratio(x, y, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan
    base = np.nanmedian(x) / np.nanmedian(y) if np.nanmedian(y) != 0 else np.nan
    vals = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        den = np.nanmedian(yb)
        vals.append(np.nanmedian(xb) / den if den != 0 else np.nan)
    return base, np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daily-precip-file", required=True)
    p.add_argument("--daily-ar-regime-file", required=True)
    p.add_argument("--out-file", required=True)
    p.add_argument("--summary-file", default=None)
    p.add_argument("--wet-metric", default="precip_mean_land_mm_day")
    p.add_argument("--wet-threshold-mm-day", type=float, default=2.0)
    p.add_argument("--ratio-metric", default="precip_mean_land_mm_day")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    precip = pd.read_csv(args.daily_precip_file, parse_dates=["date"])
    ar = pd.read_csv(args.daily_ar_regime_file, parse_dates=["date"])
    out = precip.merge(ar, on=["region", "date"], how="left", suffixes=("", "_ar"))

    out["Tar_day"] = out["Tar_day"].fillna(0).astype(int)
    out["daily_AR_regime"] = out["daily_AR_regime"].fillna("No_AR")
    out["daily_AR_regime_status"] = out["daily_AR_regime_status"].fillna("no_ar")

    out["is_wet_day"] = out[args.wet_metric] > args.wet_threshold_mm_day
    out["precip_day_type"] = "dry_or_weak_precip"
    out.loc[out["is_wet_day"] & (out["Tar_day"] == 0), "precip_day_type"] = "non_ar_precip"
    out.loc[out["is_wet_day"] & (out["Tar_day"] == 1), "precip_day_type"] = "ar_precip"

    out["primary_regime_attribution"] = (
        (out["precip_day_type"] == "ar_precip")
        & (out["daily_AR_regime_status"] == "single_known_regime")
        & (out["daily_AR_regime"].isin(REGIME_ORDER))
    )

    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_file, index=False, compression="gzip")
    print(f"[SAVE] {args.out_file} rows={len(out)}", flush=True)
    print("[SUMMARY] precip_day_type")
    print(out["precip_day_type"].value_counts(dropna=False).to_string(), flush=True)
    print("[SUMMARY] AR regime status among wet AR days")
    print(out.loc[out["precip_day_type"] == "ar_precip", "daily_AR_regime_status"].value_counts(dropna=False).to_string(), flush=True)

    summary_rows = []
    for region, g in out.groupby("region"):
        non_ar = g.loc[g["precip_day_type"] == "non_ar_precip", args.ratio_metric]
        for reg in REGIME_ORDER:
            sub = g.loc[g["primary_regime_attribution"] & (g["daily_AR_regime"] == reg), args.ratio_metric]
            ratio, lo, hi = bootstrap_ci_ratio(sub.values, non_ar.values, args.n_boot, seed=123)
            summary_rows.append({
                "region": region,
                "regime": reg,
                "metric": args.ratio_metric,
                "n_regime_days": int(sub.notna().sum()),
                "n_non_ar_precip_days": int(non_ar.notna().sum()),
                "median_regime": float(np.nanmedian(sub.values)) if len(sub) else np.nan,
                "median_non_ar": float(np.nanmedian(non_ar.values)) if len(non_ar) else np.nan,
                "ratio_regime_to_non_ar": ratio,
                "ratio_ci_low": lo,
                "ratio_ci_high": hi,
            })
    summary = pd.DataFrame(summary_rows)
    if args.summary_file:
        Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_file, index=False, compression="gzip")
        print(f"[SAVE] {args.summary_file} rows={len(summary)}", flush=True)
    else:
        print(summary.to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
